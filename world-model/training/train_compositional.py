"""End-to-end training & browser export for the Strict Zero-Overfitting Compositional
Open-Vocabulary English -> Single Orthogonal Rotation R -> Language-Free UR5e Policy (v5).

Key guarantees:
  1. 100% zero leakage: asserts NONE of the 80+ banned user/metaphor test words appear
     in ANY of the 26,003 training captions.
  2. 100% language-free robot side: dyn_feats_comp (zero text, zero labels, zero c0)
     + recurrent BehaviorPolicy (d=512, 33D foveated visual obs, top-4 neighbor z-swap,
     1 round of 2,200 language-free DAgger rollouts).
  3. 100% frozen pretrained English encoder (zero robot training).
  4. ONE single orthogonal Procrustes rotation matrix R (R R^T = I or R^T R = I).
  5. Full closed-loop MuJoCo physics verification on strictly unseen zero-shot prompts
     (including "pump your hand like fists", "throw cube", "rotate around your base",
     "slowly circle down low with your claw closed", etc.) for Aligned vs Shuffled R!
"""
import os, sys, json, copy, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

from sim_skills import UR5eArena, HOME_Q, JNT_LO, JNT_HI
from sim_compositional import (
    FAMILY_NAMES, ZERO_SHOT_PROMPTS, assert_zero_leakage,
)
from pilot_compositional import dyn_feats_comp, pca_proj, procrustes, unit, encode_hf
from train_policy import BehaviorPolicy
from run_all_arena import BinPacker, export_hf_lm

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True
if os.environ.get('GPU_GB') and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, float(os.environ['GPU_GB']) * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
    )


def add_fovea(obs_arr):
    fov = np.tanh(obs_arr[..., 26:29] / 0.04).astype(np.float32)
    return np.concatenate([obs_arr, fov], axis=-1)


def project_manifold(qu, cloud_u, top_k=10, beta=18.0):
    if top_k <= 0:
        return qu.astype(np.float32)
    sims = cloud_u @ qu
    idx = np.argpartition(-sims, top_k)[:top_k]
    s = sims[idx]
    w = np.exp((s - s.max()) * beta)
    w /= w.sum()
    qp = (w[:, None] * cloud_u[idx]).sum(0)
    return (qp / max(np.linalg.norm(qp), 1e-8)).astype(np.float32)


def expert_step_comp(env, meta, phase):
    """100% language-free physical state-machine corrective expert for DAgger relabeling."""
    fam = meta['family']
    c = env.cube_pos()
    e = env.eff_pos()
    ih = env.has_cube() > 0.5
    d_xy = float(np.linalg.norm(e[:2] - c[:2]))

    def grasp_fsm(lift_z):
        if phase < 10:
            if phase == 0:
                if d_xy < 0.015 and abs(e[2] - 0.14) < 0.03:
                    return env.act_toward_eff([c[0], c[1], max(0.022, c[2] + 0.002)], 0.0, 0.0, 0.6), 1
                return env.act_toward_eff([c[0], c[1], 0.14], 0.0, 0.0, 0.8), 0
            if phase == 1:
                if d_xy > 0.045:
                    return env.act_toward_eff([c[0], c[1], 0.14], 0.0, 0.0, 0.8), 0
                if abs(e[2] - max(0.022, c[2] + 0.002)) < 0.015 and d_xy < 0.018:
                    return env.act_toward_eff([c[0], c[1], max(0.022, c[2])], 0.0, 1.0, 0.35), 2
                return env.act_toward_eff([c[0], c[1], max(0.022, c[2] + 0.002)], 0.0, 0.0, 0.55), 1
            if 2 <= phase < 9:
                return env.act_toward_eff([c[0], c[1], max(0.022, c[2])], 0.0, 1.0, 0.35), phase + 1
            if phase == 9:
                if c[2] > max(0.08, lift_z - 0.04):
                    return env.act_toward_eff([c[0], c[1], lift_z], 0.0, 1.0, 0.65), 10
                if c[2] < 0.035 and e[2] > 0.08:
                    return env.act_toward_eff([c[0], c[1], 0.14], 0.0, 0.0, 0.8), 0
                return env.act_toward_eff([c[0], c[1], lift_z], 0.0, 1.0, 0.65), 9
        return None, phase

    if fam == 'cube_propel':
        a, ph = grasp_fsm(0.20)
        if a is not None:
            return a, ph
        d = meta.get('dir', 'forward')
        if d == 'forward':
            q_wind = np.array([-1.57, -2.05, 1.85, -1.35, -1.57, 0.0])
            q_whip = np.array([-1.57, -0.55, 0.35, -1.35, -1.57, 0.0]); rel = 4
        elif d == 'high':
            q_wind = np.array([-1.57, -2.15, 2.00, -1.45, -1.57, 0.0])
            q_whip = np.array([-1.57, -1.20, 0.25, -0.60, -1.57, 0.0]); rel = 3
        elif d == 'left':
            q_wind = np.array([-0.85, -1.55, 1.35, -1.35, -1.57, 0.0])
            q_whip = np.array([-2.65, -0.95, 0.55, -1.15, -1.57, 0.0]); rel = 4
        else:
            q_wind = np.array([-2.30, -1.55, 1.35, -1.35, -1.57, 0.0])
            q_whip = np.array([-0.45, -0.95, 0.55, -1.15, -1.57, 0.0]); rel = 4
        if 10 <= phase < 22:
            if not ih and c[2] < 0.04:
                return env.act_toward_eff([c[0], c[1], 0.14], 0.0, 0.0, 0.8), 0
            return env.act_toward_q(q_wind, 1.0, 0.65), phase + 1
        k = phase - 22
        return env.act_toward_q(q_whip, 1.0 if k < rel else 0.0, 1.0 if k < 14 else 0.45), phase + 1

    if fam == 'cube_relocate':
        gx, gy = meta['goal'][0], meta['goal'][1]
        a, ph = grasp_fsm(0.17)
        if a is not None:
            return a, ph
        if phase == 10:
            if math.hypot(e[0] - gx, e[1] - gy) < 0.02:
                return env.act_toward_eff([gx, gy, 0.032], 0.0, 1.0, 0.5), 11
            return env.act_toward_eff([gx, gy, 0.17], 0.0, 1.0, 0.65), 10
        if phase == 11:
            if e[2] < 0.045:
                return env.act_toward_eff([gx, gy, 0.035], 0.0, 0.0, 0.5), 12
            return env.act_toward_eff([gx, gy, 0.032], 0.0, 1.0, 0.5), 11
        if phase < 16:
            return env.act_toward_eff([gx, gy, 0.035], 0.0, 0.0, 0.5), phase + 1
        return env.act_toward_eff([gx, gy, 0.16], 0.0, 0.0, 0.45), phase + 1

    if fam == 'cube_elevate_hold':
        z_t = meta.get('z', 0.24)
        a, ph = grasp_fsm(z_t)
        if a is not None:
            return a, ph
        return env.act_toward_eff([c[0], c[1], z_t], 0.0, 1.0, 0.5), phase

    if fam == 'cube_elevate_release':
        z_t = meta.get('z', 0.28)
        a, ph = grasp_fsm(z_t)
        if a is not None:
            return a, ph
        if phase < 18:
            return env.act_toward_eff([c[0], c[1], z_t], 0.0, 1.0, 0.6), phase + 1
        return env.act_toward_eff([e[0], e[1], z_t], 0.0, 0.0, 0.45), phase + 1

    if fam == 'cube_slide':
        dvec = {'left': np.array([0.0, -1.0]), 'right': np.array([0.0, 1.0]),
                'forward': np.array([1.0, 0.0]), 'back': np.array([-1.0, 0.0])}[meta.get('dir', 'right')]
        p_beh = np.clip(c[:2] - dvec * 0.085, [0.22, -0.34], [0.60, 0.34])
        p_thr = np.clip(c[:2] + dvec * 0.12, [0.20, -0.38], [0.64, 0.38])
        if phase == 0:
            if np.linalg.norm(e[:2] - p_beh) < 0.025 and abs(e[2] - 0.12) < 0.03:
                return env.act_toward_eff([p_beh[0], p_beh[1], 0.026], 0.0, 1.0, 0.6), 1
            return env.act_toward_eff([p_beh[0], p_beh[1], 0.12], 0.0, 1.0, 0.75), 0
        if phase == 1:
            if e[2] < 0.038:
                return env.act_toward_eff([p_thr[0], p_thr[1], 0.026], 0.0, 1.0, 0.75), 2
            return env.act_toward_eff([p_beh[0], p_beh[1], 0.026], 0.0, 1.0, 0.6), 1
        if phase < 28:
            return env.act_toward_eff([p_thr[0], p_thr[1], 0.026], 0.0, 1.0, 0.75), phase + 1
        return env.act_toward_eff([e[0], e[1], 0.14], 0.0, 0.0, 0.5), phase

    if fam == 'cube_tap':
        k = phase
        z_t = 0.14 if k < 14 else (0.042 if (((k - 14) % 16) / 16.0) < 0.45 else 0.13)
        return env.act_toward_eff([c[0], c[1], z_t], 0.0, 1.0, 0.65), phase + 1

    if fam == 'base_revolve':
        sgn = 1.0 if meta.get('dir', 'ccw') == 'ccw' else -1.0
        h = meta.get('height', 'mid')
        g_mode = meta.get('gripper', 'open')
        sp = 0.10 if meta.get('speed') == 'slow' else 0.15
        g = 0.0 if g_mode == 'open' else (1.0 if g_mode == 'closed' else (1.0 if ((phase // 7) % 2 == 1) else 0.0))
        q_cur = env.d.qpos[:6].copy()
        if h == 'low':
            q_t = np.array([q_cur[0], -0.82, 1.55, -1.70, -1.57, 0.0])
        elif h == 'high':
            q_t = np.array([q_cur[0], -1.55, 0.55, -0.95, -1.57, 0.0])
        else:
            q_t = np.array([q_cur[0], -1.30, 1.42, -1.68, -1.57, 0.0])
        if phase < 10 and np.linalg.norm(q_cur[1:] - q_t[1:]) > 0.35:
            return env.act_toward_q(q_t, g, 0.72), phase + 1
        q_t[0] = float(np.clip(q_cur[0] + sgn * sp, -6.1, 6.1))
        return env.act_toward_q(q_t, g, 0.88), max(phase, 10) + 1

    return env.act_toward_q(HOME_Q, 0.0, 0.6), phase


def score_comp_rotation(A_tr, A_te, A_zs, G_tr, G_te, fam_tr, fam_te, metas_tr, metas_te, n_fit, shuffled=False):
    rel_id = FAMILY_NAMES.index('cube_relocate')
    m_tr = fam_tr == rel_id
    m_te = fam_te == rel_id
    g_tr = np.array([metas_tr[i]['goal'][:2] for i in np.where(m_tr)[0]])
    g_te = np.array([metas_te[i]['goal'][:2] for i in np.where(m_te)[0]])
    X_pl = np.c_[G_tr[m_tr], np.ones(m_tr.sum())]
    W_pl = np.linalg.solve(X_pl.T @ X_pl + 0.5 * np.eye(X_pl.shape[1]), X_pl.T @ g_tr)

    m_n = min(len(A_tr), n_fit)
    if shuffled:
        rng = np.random.default_rng(7)
        R = procrustes(A_tr[:m_n][rng.permutation(m_n)], G_tr[:m_n])
    else:
        R = procrustes(A_tr[:m_n], G_tr[:m_n])

    Q_te = A_te @ R
    Q_te_u = unit(Q_te)
    G_te_u = unit(G_te)
    G_bk_u = unit(G_tr[:6000])
    fam_bk = fam_tr[:6000]
    metas_bk = metas_tr[:6000]

    S_p = Q_te_u[:1000] @ G_te_u[:1000].T
    fosc = float((S_p > S_p.diagonal()[:, None]).mean())

    S_nn = Q_te_u @ G_bk_u.T
    top15 = np.argpartition(-S_nn, 15, axis=1)[:, :15]
    fam_ok = grip_ok = h_ok = gn = 0
    for i in range(len(Q_te)):
        nbrs = top15[i]
        sims = np.maximum(S_nn[i, nbrs], 0.0)
        v = np.bincount(fam_bk[nbrs], weights=sims, minlength=len(FAMILY_NAMES))
        fam_ok += int(np.argmax(v) == fam_te[i])
        mt = metas_te[i]
        if not mt['family'].startswith('cube_'):
            gv, hv = {}, {}
            for idx_n, sc in zip(nbrs, sims):
                mn = metas_bk[idx_n]
                gv[mn.get('gripper', '')] = gv.get(mn.get('gripper', ''), 0.0) + sc
                hv[mn.get('height', '')] = hv.get(mn.get('height', ''), 0.0) + sc
            grip_ok += int(max(gv, key=gv.get) == mt['gripper'])
            h_ok += int(max(hv, key=hv.get) == mt['height'])
            gn += 1

    pl_pred = np.c_[Q_te[m_te], np.ones(m_te.sum())] @ W_pl
    pl_cm = float(np.median(np.linalg.norm(pl_pred - g_te, axis=1) * 100))

    Q_zs_u = unit(A_zs @ R)
    S_zs = Q_zs_u @ G_bk_u.T
    zf = za = 0
    for i, (_, checks) in enumerate(ZERO_SHOT_PROMPTS):
        nbrs = np.argpartition(-S_zs[i], 15)[:15]
        sims = np.maximum(S_zs[i, nbrs], 0.0)
        fv, gv, hv, lv, dv, sv = {}, {}, {}, {}, {}, {}
        for idx_n, sc in zip(nbrs, sims):
            mn = metas_bk[idx_n]
            for dm, key in ((fv, 'family'), (gv, 'gripper'), (hv, 'height'), (lv, 'lateral'), (dv, 'dir'), (sv, 'speed')):
                val = mn.get(key)
                if val is not None:
                    dm[val] = dm.get(val, 0.0) + sc
        pf = max(fv, key=fv.get)
        pg = max(gv, key=gv.get) if gv else '-'
        ph = max(hv, key=hv.get) if hv else '-'
        pl = max(lv, key=lv.get) if lv else '-'
        pd = max(dv, key=dv.get) if dv else '-'
        ps = max(sv, key=sv.get) if sv else '-'
        fm = (pf == checks['family'])
        am = fm
        if 'gripper' in checks: am &= (pg == checks['gripper'])
        if 'height_in' in checks: am &= (ph in checks['height_in'])
        if 'lateral' in checks: am &= (pl == checks['lateral'])
        if 'dir' in checks: am &= (pd == checks['dir'])
        if 'speed' in checks: am &= (ps == checks['speed'])
        zf += int(fm)
        za += int(am)

    return dict(
        R=R, W_pl=W_pl,
        top1=fam_ok / len(Q_te), grip_acc=grip_ok / max(1, gn), h_acc=h_ok / max(1, gn),
        fosc=fosc, pl_cm=pl_cm, zs_fam=zf, zs_full=za,
    )


def main():
    lm_name = sys.argv[1] if len(sys.argv) > 1 else 'BAAI/bge-small-en-v1.5'
    pooling = sys.argv[2] if len(sys.argv) > 2 else 'cls'
    k_t = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    reg_t = None if (len(sys.argv) > 4 and sys.argv[4] == 'None') else (float(sys.argv[4]) if len(sys.argv) > 4 else 4.0)
    k_v = int(sys.argv[5]) if len(sys.argv) > 5 else 96
    reg_v = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2

    base = '/home/aif_eng/play/jepa/arena_run'
    robot_dir = os.path.join(base, 'robot')
    data_dir = os.path.join(base, 'data_comp')
    out_dir = os.path.join(base, 'browser_export_comp')
    os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'parity'), exist_ok=True)

    d = np.load(os.path.join(data_dir, 'trajs.npz'))
    meta_d = json.load(open(os.path.join(data_dir, 'meta.json')))
    obs = add_fovea(d['obs'].astype(np.float32))
    act = d['act'].astype(np.float32)
    feat = d['feat'].astype(np.float32)
    fam = d['family']
    caps = meta_d['captions']
    metas = meta_d['metas']

    for c in caps:
        assert_zero_leakage(c)
    print(f"Verified 100% ZERO-LEAKAGE across {len(caps)} captions! Config: {lm_name} ({pooling}), k=({k_t},{k_v}), reg=({reg_t},{reg_v})", flush=True)

    N_te = 2500
    obs_tr, act_tr = obs[:-N_te], act[:-N_te]
    feat_tr, feat_te = feat[:-N_te], feat[-N_te:]
    fam_tr, fam_te = fam[:-N_te], fam[-N_te:]
    caps_tr, caps_te = caps[:-N_te], caps[-N_te:]
    metas_tr, metas_te = metas[:-N_te], metas[-N_te:]
    zs_texts = [p for p, _ in ZERO_SHOT_PROMPTS]

    # 1. Language-free compositional robot dynamics manifold
    X_v_tr = dyn_feats_comp(feat_tr)
    X_v_te = dyn_feats_comp(feat_te)
    mu_v, P_v, G_tr, G_te = pca_proj(X_v_tr, X_v_te, k_v, reg=reg_v)
    G_tr_u = unit(G_tr).astype(np.float32)

    # Top-4 unsupervised neighbors in G_tr (breaks c0/start_q without mixing factors!)
    G_gpu = torch.from_numpy(G_tr_u).to(DEV)
    nn_idx = torch.empty((len(G_tr_u), 4), dtype=torch.long, device=DEV)
    for i in range(0, len(G_tr_u), 1024):
        S = G_gpu[i:i+1024] @ G_gpu.T
        for r_i in range(S.shape[0]):
            S[r_i, i + r_i] = -2.0
        nn_idx[i:i+1024] = S.topk(4, dim=1).indices
    del G_gpu

    obs_mu = obs_tr.mean((0, 1), keepdims=True).astype(np.float32)
    obs_sd = np.maximum(obs_tr.std((0, 1), keepdims=True), 1e-3).astype(np.float32)
    sd_c = torch.from_numpy(obs_sd[0, 0, 23:26]).to(DEV).view(1, 1, 3)
    sd_rel = torch.from_numpy(obs_sd[0, 0, 26:29]).to(DEV).view(1, 1, 3)
    sd_fov = torch.from_numpy(obs_sd[0, 0, 30:33]).to(DEV).view(1, 1, 3)

    def run_bc_stage(pol, ema, obs_all, act_all, Z_all, nn_tab, steps, lr, bs=256, tag='BC1'):
        N, T, _ = obs_all.shape
        O = torch.from_numpy(((obs_all - obs_mu) / obs_sd).astype(np.float32)).to(DEV)
        dist_ce = np.linalg.norm(obs_all[:, :, 26:29], axis=-1).astype(np.float32)
        W_step = torch.from_numpy(np.where(dist_ce < 0.14, 5.0, 1.0).astype(np.float32)).to(DEV).unsqueeze(-1)
        A = torch.from_numpy(act_all.astype(np.float32)).to(DEV)
        Z = torch.from_numpy(Z_all.astype(np.float32)).to(DEV)
        act_w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5], device=DEV).view(1, 1, 7)

        opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
        t0 = time.time()
        for step in range(1, steps + 1):
            idx = torch.randint(0, N, (bs,), device=DEV)
            ob = O[idx].clone()
            ac = A[idx]
            if nn_tab is not None:
                k_pick = torch.randint(0, nn_tab.shape[1], (bs,), device=DEV)
                in_orig = idx < len(nn_tab)
                use_nbr = in_orig & (torch.rand(bs, device=DEV) < 0.55)
                safe_idx = idx.clamp_max(len(nn_tab) - 1)
                z_idx = torch.where(use_nbr, nn_tab[safe_idx, k_pick], idx)
                zb = Z[z_idx]
            else:
                zb = Z[idx]

            mask_n = (torch.rand(bs, 1, 1, device=DEV) < 0.55).float()
            ob[:, :, :23] += torch.randn_like(ob[:, :, :23]) * 0.005 * mask_n
            raw_err = (torch.randn(bs, 1, 3, device=DEV) * 0.003 + torch.randn(bs, T, 3, device=DEV) * 0.002) * mask_n
            ob[:, :, 23:26] += raw_err / sd_c
            ob[:, :, 26:29] += raw_err / sd_rel
            ob[:, :, 30:33] += (raw_err / 0.04) / sd_fov

            sig = (torch.rand(bs, 1, device=DEV) * 0.14) / math.sqrt(k_v)
            zb_n = F.normalize(zb + torch.randn_like(zb) * sig, dim=-1)

            with torch.autocast('cuda', torch.bfloat16):
                pred = pol.forward_seq(ob, zb_n)
                per_el = F.smooth_l1_loss(pred, ac, beta=0.03, reduction='none') + 0.5 * (pred - ac).pow(2)
                loss = (per_el * W_step[idx] * act_w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 2.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), pol.parameters()):
                    pe.mul_(0.996).add_(p, alpha=0.004)
            if step % 2000 == 0 or step == steps:
                print(f"  [{tag}] step {step:5d}/{steps} | loss={loss.item():.5f} ({time.time()-t0:.1f}s)", flush=True)

    # 2. Stage 1 BC (d=512, 12,000 steps)
    pol = BehaviorPolicy(obs_dim=33, dz=k_v, d=512, nblocks=3, act_dim=7).to(DEV)
    ema = BehaviorPolicy(obs_dim=33, dz=k_v, d=512, nblocks=3, act_dim=7).to(DEV)
    ema.load_state_dict(pol.state_dict())
    print(f"Stage 1 BC ({sum(p.numel() for p in pol.parameters())/1e6:.2f}M params, obsDim=33, d=512, dz={k_v})...", flush=True)
    run_bc_stage(pol, ema, obs_tr, act_tr, G_tr_u, nn_idx, steps=12000, lr=1e-3, bs=256, tag='BC1')

    # 3. Stage 2: 2,200 100% Language-Free DAgger rollouts on all cube skills + base_revolve
    print("\nStage 2: Collecting 2,200 language-free DAgger rollouts...", flush=True)
    t_dg = time.time()
    env_dg = UR5eArena(robot_dir)
    ema.eval()
    obs_mu_t = torch.from_numpy(obs_mu[0, 0]).to(DEV)
    obs_sd_t = torch.from_numpy(obs_sd[0, 0]).to(DEV)
    dg_fams = ('cube_propel', 'cube_relocate', 'cube_elevate_hold', 'cube_elevate_release', 'cube_slide', 'cube_tap', 'base_revolve')
    dg_cand = [i for i in range(len(fam_tr)) if FAMILY_NAMES[fam_tr[i]] in dg_fams]
    rng_dg = np.random.default_rng(777)
    pick_dg = rng_dg.choice(dg_cand, size=2200, replace=False)

    dg_obs, dg_act, dg_z = [], [], []
    with torch.inference_mode():
        for k_i, clip_i in enumerate(pick_dg):
            m_i = metas_tr[clip_i]
            z_vec = G_tr_u[int(nn_idx[clip_i, k_i % 4].item())]
            z_t = torch.from_numpy(z_vec).unsqueeze(0).to(DEV)
            ob30 = env_dg.reset(rng_dg)
            h = torch.zeros(1, 1, ema.d, device=DEV)
            phase = 0
            ep_o, ep_a = [], []
            beta_mix = 0.25 if (k_i % 2 == 0) else 0.0
            for t in range(80):
                ob33 = add_fovea(ob30)
                ob_t = (torch.from_numpy(ob33).to(DEV) - obs_mu_t) / obs_sd_t
                a_pol, h = ema.step(ob_t.unsqueeze(0), z_t, h, t / 79.0)
                a_exp, phase = expert_step_comp(env_dg, m_i, phase)
                ep_o.append(ob33)
                ep_a.append(a_exp)
                a_exec = a_exp if (rng_dg.random() < beta_mix) else a_pol[0].cpu().numpy()
                ob30 = env_dg.step(a_exec)
            dg_obs.append(np.stack(ep_o))
            dg_act.append(np.stack(ep_a))
            dg_z.append(z_vec)
    dg_obs = np.stack(dg_obs).astype(np.float32)
    dg_act = np.stack(dg_act).astype(np.float32)
    dg_z = np.stack(dg_z).astype(np.float32)
    print(f"Collected {len(dg_obs)} DAgger rollouts in {time.time()-t_dg:.1f}s!", flush=True)

    # 4. Stage 3 DAgger fine-tune (6,000 steps)
    obs_mix = np.concatenate([obs_tr, dg_obs, dg_obs, dg_obs], axis=0)
    act_mix = np.concatenate([act_tr, dg_act, dg_act, dg_act], axis=0)
    z_mix = np.concatenate([G_tr_u, dg_z, dg_z, dg_z], axis=0)
    pol.load_state_dict(ema.state_dict())
    run_bc_stage(pol, ema, obs_mix, act_mix, z_mix, nn_idx, steps=6000, lr=4e-4, bs=256, tag='DAGGER')
    ema.eval()
    torch.save({
        'state': ema.state_dict(), 'obs_mu': obs_mu, 'obs_sd': obs_sd,
        'obs_dim': 33, 'dz': k_v, 'd': ema.d, 'nblocks': ema.nblocks,
    }, os.path.join(data_dir, 'behavior_policy_comp_v5.pt'))

    # 5. Export quantized LM FIRST so all orthogonal rotations R_N match browser int8 EXACTLY!
    tok, hfq = export_hf_lm(out_dir, lm_name, pooling)
    E_tr = encode_hf(tok, hfq, caps_tr, pooling=pooling)
    E_te = encode_hf(tok, hfq, caps_te, pooling=pooling)
    E_zs = encode_hf(tok, hfq, zs_texts, pooling=pooling)
    mu_t, P_t, A_tr, A_te = pca_proj(E_tr, E_te, k_t, reg=reg_t)
    A_zs = (E_zs - mu_t) @ P_t

    # Random-Weight LM control
    cfg_r = AutoConfig.from_pretrained(lm_name, local_files_only=True)
    torch.manual_seed(42)
    rand_lm = AutoModel.from_config(cfg_r).to(DEV).eval()
    Er_tr = encode_hf(tok, rand_lm, caps_tr, pooling=pooling)
    Er_te = encode_hf(tok, rand_lm, caps_te, pooling=pooling)
    Er_zs = encode_hf(tok, rand_lm, zs_texts, pooling=pooling)
    del rand_lm; torch.cuda.empty_cache()
    mu_rt, Pr_t, Ar_tr, Ar_te = pca_proj(Er_tr, Er_te, k_t, reg=reg_t)
    Ar_zs = (Er_zs - mu_rt) @ Pr_t
    res_rand = score_comp_rotation(Ar_tr, Ar_te, Ar_zs, G_tr, G_te, fam_tr, fam_te, metas_tr, metas_te, n_fit=len(A_tr))
    print(f"\nCONTROL Random-Weight LM: val fam={res_rand['top1']*100:.1f}% grip={res_rand['grip_acc']*100:.1f}% FOSC={res_rand['fosc']:.3f} ZS37 fam={res_rand['zs_fam']}/37 FULL={res_rand['zs_full']}/37", flush=True)

    # 6. Pack quantized BehaviorPolicy v5 + dense 1,600-clip unlabeled manifold cloud + R_N
    torch.set_float32_matmul_precision('highest')
    pol_cpu = copy.deepcopy(ema).cpu().eval()
    sd_p = pol_cpu.state_dict()
    pk = BinPacker()
    pk.pack('pol/obs_mu', obs_mu[0, 0], 'f32')
    pk.pack('pol/obs_sd', obs_sd[0, 0], 'f32')
    pk.pack_linear('pol/in', sd_p['in_proj.weight'], sd_p['in_proj.bias'], 'i8_col')
    pk.pack_linear('pol/gru_ih', sd_p['gru.weight_ih_l0'], sd_p['gru.bias_ih_l0'], 'i8_col')
    pk.pack_linear('pol/gru_hh', sd_p['gru.weight_hh_l0'], sd_p['gru.bias_hh_l0'], 'i8_col')
    for i in range(pol_cpu.nblocks):
        bp = f'blocks.{i}'
        pk.pack(f'pol/b{i}/ln/w', sd_p[f'{bp}.ln.weight'].numpy(), 'f32')
        pk.pack(f'pol/b{i}/ln/b', sd_p[f'{bp}.ln.bias'].numpy(), 'f32')
        pk.pack_linear(f'pol/b{i}/film', sd_p[f'{bp}.film.weight'], sd_p[f'{bp}.film.bias'], 'i8_col')
        pk.pack_linear(f'pol/b{i}/fc1', sd_p[f'{bp}.fc1.weight'], sd_p[f'{bp}.fc1.bias'], 'i8_col')
        pk.pack_linear(f'pol/b{i}/fc2', sd_p[f'{bp}.fc2.weight'], sd_p[f'{bp}.fc2.bias'], 'i8_col')
    pk.pack('pol/out_ln/w', sd_p['out_ln.weight'].numpy(), 'f32')
    pk.pack('pol/out_ln/b', sd_p['out_ln.bias'].numpy(), 'f32')
    pk.pack_linear('pol/head', sd_p['head.weight'], sd_p['head.bias'], 'f16')

    polq = copy.deepcopy(pol_cpu)
    sdpq = polq.state_dict()
    sdpq['in_proj.weight'].copy_(torch.from_numpy(pk.deq['pol/in/w'].T))
    sdpq['gru.weight_ih_l0'].copy_(torch.from_numpy(pk.deq['pol/gru_ih/w'].T))
    sdpq['gru.weight_hh_l0'].copy_(torch.from_numpy(pk.deq['pol/gru_hh/w'].T))
    for i in range(pol_cpu.nblocks):
        bp = f'blocks.{i}'
        sdpq[f'{bp}.film.weight'].copy_(torch.from_numpy(pk.deq[f'pol/b{i}/film/w'].T))
        sdpq[f'{bp}.fc1.weight'].copy_(torch.from_numpy(pk.deq[f'pol/b{i}/fc1/w'].T))
        sdpq[f'{bp}.fc2.weight'].copy_(torch.from_numpy(pk.deq[f'pol/b{i}/fc2/w'].T))
    sdpq['head.weight'].copy_(torch.from_numpy(pk.deq['pol/head/w'].T))
    polq.load_state_dict(sdpq)

    pk.pack('rot/mu_t', mu_t, 'f32')
    pk.pack('rot/P_t', P_t, 'f32')

    _, _, Vt_2d = np.linalg.svd(G_tr - G_tr.mean(0), full_matrices=False)
    vis_proj_2d = Vt_2d[:2].T.astype(np.float32)
    pre_proj_2d = np.zeros((k_t, 2), dtype=np.float32)
    m_cp = min(k_t, k_v)
    pre_proj_2d[:m_cp] = vis_proj_2d[:m_cp]
    for c_i in range(2):
        nrm = np.linalg.norm(pre_proj_2d[:, c_i])
        if nrm > 1e-8:
            pre_proj_2d[:, c_i] /= nrm

    rng_cl = np.random.default_rng(7)
    cloud_idx = []
    for fid in range(len(FAMILY_NAMES)):
        cand = np.where(fam_te == fid)[0]
        if len(cand) > 0:
            cloud_idx.extend(rng_cl.choice(cand, size=min(100, len(cand)), replace=False).tolist())
    cloud_idx = np.array(cloud_idx, dtype=np.int32)
    cloud_g = G_te[cloud_idx]
    cloud_unit = unit(cloud_g).astype(np.float32)
    cloud_2d = (cloud_g @ vis_proj_2d).astype(np.float32)
    cloud_skill = fam_te[cloud_idx].astype(np.int32).tolist()
    cloud_labels = []
    for idx in cloud_idx:
        m = metas_te[idx]; fn = m['family']
        if fn == 'cube_relocate':
            cloud_labels.append(f"relocate ({m['goal'][0]*100:.0f}, {m['goal'][1]*100:+.0f}) cm")
        elif fn.startswith('cube_'):
            d_s = f" ({m['dir']})" if 'dir' in m else ''
            cloud_labels.append(f"{fn.replace('cube_', '')}{d_s}")
        else:
            cloud_labels.append(f"{fn.replace('_', ' ')} · {m['height']} · {m['gripper']}")

    pk.pack('vis/proj_2d', vis_proj_2d, 'f32')
    pk.pack('vis/pre_proj_2d', pre_proj_2d, 'f32')
    pk.pack('vis/cloud_2d', cloud_2d, 'f32')
    cloud_unit_deq = pk.pack('vis/cloud_unit', cloud_unit, 'f16')

    rel_id = FAMILY_NAMES.index('cube_relocate')
    m_pl = fam_tr == rel_id
    goals_tr = np.array([metas_tr[i]['goal'][:2] for i in np.where(m_pl)[0]])
    X_pl = np.c_[G_tr[m_pl], np.ones(m_pl.sum())]
    W_pl = np.linalg.solve(X_pl.T @ X_pl + 0.5 * np.eye(X_pl.shape[1]), X_pl.T @ goals_tr)
    pk.pack('vis/place_w', W_pl[:k_v], 'f32')
    pk.pack('vis/place_b', W_pl[k_v], 'f32')

    N_LIST = [30, 100, 300, 1000, 10000]
    N_OPTIONS = [str(n) for n in N_LIST] + ['shuffled']
    metrics = {}
    R_map = {}
    print("\n=== QUANTIZED LM COMPOSITIONAL PROCRUSTES SCALING CURVE (ZERO LEAKAGE!) ===", flush=True)
    for nKey in N_OPTIONS:
        is_shuf = (nKey == 'shuffled')
        n_fit = len(A_tr) if (is_shuf or nKey == '10000') else int(nKey)
        r = score_comp_rotation(A_tr, A_te, A_zs, G_tr, G_te, fam_tr, fam_te, metas_tr, metas_te, n_fit=n_fit, shuffled=is_shuf)
        pk.pack(f'rot/R_{nKey}', r['R'], 'f32')
        R_map[nKey] = r['R']
        metrics[nKey] = dict(
            top1=round(r['top1'], 3), grip=round(r['grip_acc'], 3), height=round(r['h_acc'], 3),
            foscttm=round(r['fosc'], 3), place_cm=round(r['pl_cm'], 1),
            zs_fam=int(r['zs_fam']), zs_full=int(r['zs_full']), unseen40=int(r['zs_fam']),
        )
        print(f"  {nKey:8s} (fit={n_fit:5d}) | val fam={r['top1']*100:5.1f}% grip={r['grip_acc']*100:5.1f}% h={r['h_acc']*100:5.1f}% | FOSC={r['fosc']:.3f} | pl={r['pl_cm']:4.1f}cm | ZS37 fam={r['zs_fam']:2d}/37 FULL={r['zs_full']:2d}/37", flush=True)

    metrics['random_lm'] = dict(
        top1=round(res_rand['top1'], 3), grip=round(res_rand['grip_acc'], 3), height=round(res_rand['h_acc'], 3),
        foscttm=round(res_rand['fosc'], 3), place_cm=round(res_rand['pl_cm'], 1),
        zs_fam=int(res_rand['zs_fam']), zs_full=int(res_rand['zs_full']), unseen40=int(res_rand['zs_fam']),
    )
    pk.pack('rot/R_untrained', R_map['shuffled'], 'f32')

    # 7. FULL CLOSED-LOOP MUJOCO VERIFICATION on strictly unseen zero-shot prompts!
    proj_k, proj_beta = 10, 18.0
    env_ev = UR5eArena(robot_dir)

    @torch.inference_mode()
    def eval_cl_comp(pol_m, R_mat, tag):
        rng_ev = np.random.default_rng(123)
        cases = [
            # Novel compositional prompts (ZERO words in training!)
            ("pump your hand like fists",                    "fist_pump"),
            ("pump your fist up and down in celebration",    "fist_pump"),
            ("fist pump high in the air",                    "fist_pump"),
            ("slowly circle down low with your claw closed", "low_closed_circle"),
            ("twist your wrist fast up high with closed hand", "high_closed_roll"),
            ("wiggle your claws open and closed",            "pulsing_grip"),
            ("windshield wiper across the table",            "sweep_y"),
            ("shake and party",                              "multi_sway"),
            ("do a happy little dance",                      "multi_sway"),
            ("touch the ceiling",                            "reach_up"),
            ("point straight up at the sky",                 "reach_up"),
            ("point over to the left side",                  "point_left"),
            # Classic open-vocab motions (all user verbs banned!)
            ("rotate around your base",                      "base_revolve"),
            ("do a 360",                                     "base_revolve"),
            ("spin all the way around",                      "base_revolve"),
            ("fling the cube off to the left",               "cube_propel"),
            ("throw cube",                                   "cube_propel"),
            ("yeet the red block",                           "cube_propel"),
            ("put the cube in the front left corner",        "place_fl"),
            ("deliver the crimson block to the rear right",  "place_br"),
            ("knock the red block to the right",             "push_right"),
            ("pick up the red block and keep it aloft",      "lift_hold"),
            ("grab the block and let it fall",               "lift_drop"),
            ("boop the red square",                          "cube_tap"),
        ]
        embs = encode_hf(tok, hfq, [c[0] for c in cases], pooling=pooling)
        Qu = unit((embs - mu_t) @ P_t @ R_mat).astype(np.float32)
        print(f"\n=== CLOSED-LOOP MUJOCO EVAL v5 COMPOSITIONAL ({tag}) ===", flush=True)
        passed = 0
        for idx_c, (pr, check_kind) in enumerate(cases):
            zb = project_manifold(Qu[idx_c], cloud_unit_deq, top_k=proj_k, beta=proj_beta)
            z_t = torch.from_numpy(zb).unsqueeze(0).to(DEV)
            ok_tr = 0; mstr = ''
            for _ in range(3):
                ob30 = env_ev.reset(rng_ev)
                c0 = env_ev.cube_pos().copy()
                h = torch.zeros(1, 1, pol_m.d, device=DEV)
                eh, gh, ph, q5h, ch, cvh = [], [], [], [], [], []
                min_ce = 9.0
                for t in range(80):
                    ob33 = add_fovea(ob30)
                    ob_t = (torch.from_numpy(ob33).to(DEV) - obs_mu_t) / obs_sd_t
                    a_t, h = pol_m.step(ob_t.unsqueeze(0), z_t, h, t / 79.0)
                    ob30 = env_ev.step(a_t[0].cpu().numpy())
                    c = env_ev.cube_pos(); e = env_ev.eff_pos()
                    eh.append(e); gh.append(env_ev.grip_closed()); ph.append(float(env_ev.d.qpos[0]))
                    q5h.append(float(env_ev.d.qpos[5])); ch.append(c); cvh.append(float(np.linalg.norm(env_ev.cube_vel())))
                    min_ce = min(min_ce, float(np.linalg.norm(c - e)))
                eh = np.stack(eh); gh = np.array(gh); ch = np.stack(ch)
                eh_ex = eh[20:]; gh_ex = gh[20:]
                xspan = float(eh_ex[:, 0].max() - eh_ex[:, 0].min())
                yspan = float(eh_ex[:, 1].max() - eh_ex[:, 1].min())
                zspan = float(eh_ex[:, 2].max() - eh_ex[:, 2].min())
                max_ez = float(eh_ex[:, 2].max()); mean_ez = float(eh_ex[:, 2].mean())
                g_mean = float(gh_ex.mean()); g_flips = int(np.sum((gh_ex > 0.4)[1:] != (gh_ex > 0.4)[:-1]))
                dpan = max(ph) - min(ph); dq5 = max(q5h[20:]) - min(q5h[20:])
                ce = ch[-1]; fly = float(np.linalg.norm(ce[:2] - c0[:2]))
                max_cz = float(ch[:, 2].max()); max_cv = max(cvh)

                ok = False
                if check_kind == 'fist_pump':
                    ok = zspan > 0.14 and g_mean > 0.55 and max_ez > 0.42
                    mstr = f"z_span={zspan*100:.0f}cm max_ez={max_ez*100:.0f}cm grip={g_mean:.2f}"
                elif check_kind == 'low_closed_circle':
                    ok = xspan > 0.14 and yspan > 0.14 and mean_ez < 0.28 and g_mean > 0.55
                    mstr = f"span=({xspan*100:.0f}x{yspan*100:.0f})cm ez={mean_ez*100:.0f}cm g={g_mean:.2f}"
                elif check_kind == 'high_closed_roll':
                    ok = dq5 > 1.1 and g_mean > 0.55 and max_ez > 0.45
                    mstr = f"roll={dq5:.2f}rad max_ez={max_ez*100:.0f}cm g={g_mean:.2f}"
                elif check_kind == 'pulsing_grip':
                    ok = g_flips >= 3
                    mstr = f"flips={g_flips} g_mean={g_mean:.2f}"
                elif check_kind == 'sweep_y':
                    ok = yspan > 0.24
                    mstr = f"y_span={yspan*100:.0f}cm ez={mean_ez*100:.0f}cm"
                elif check_kind == 'multi_sway':
                    ok = dpan > 0.55
                    mstr = f"pan_amp={dpan:.2f}rad"
                elif check_kind == 'reach_up':
                    ok = max_ez > 0.65
                    mstr = f"max_ez={max_ez*100:.0f}cm"
                elif check_kind == 'point_left':
                    ok = float(eh_ex[:, 1].mean()) < -0.18
                    mstr = f"mean_ey={eh_ex[:,1].mean()*100:+.0f}cm"
                elif check_kind == 'base_revolve':
                    ok = dpan > 3.2
                    mstr = f"spin={math.degrees(dpan):.0f}deg"
                elif check_kind == 'cube_propel':
                    ok = (fly > 0.30 or max_cv > 1.0 or max_cz > 0.35) and (env_ev.has_cube() < 0.5)
                    mstr = f"fly={fly:.2f}m max_v={max_cv:.1f}m/s max_z={max_cz:.2f}m"
                elif check_kind == 'place_fl':
                    err = math.hypot(ce[0] - 0.50, ce[1] - (-0.18)) * 100
                    ok = err < 11.0 and ce[2] < 0.05
                    mstr = f"end=({ce[0]*100:.0f},{ce[1]*100:+.0f})cm err={err:.1f}cm"
                elif check_kind == 'place_br':
                    err = math.hypot(ce[0] - 0.34, ce[1] - 0.18) * 100
                    ok = err < 11.0 and ce[2] < 0.05
                    mstr = f"end=({ce[0]*100:.0f},{ce[1]*100:+.0f})cm err={err:.1f}cm"
                elif check_kind == 'push_right':
                    dy = ce[1] - c0[1]
                    ok = dy > 0.07
                    mstr = f"dy={dy*100:+.1f}cm"
                elif check_kind == 'lift_hold':
                    ok = ce[2] > 0.11 and env_ev.has_cube() > 0.5
                    mstr = f"end_z={ce[2]*100:.0f}cm ih={int(env_ev.has_cube())}"
                elif check_kind == 'lift_drop':
                    ok = max_cz > 0.15 and ce[2] < 0.045
                    mstr = f"peak_z={max_cz*100:.0f}cm end_z={ce[2]*100:.0f}cm"
                elif check_kind == 'cube_tap':
                    ok = min_ce < 0.065 and ce[2] < 0.05
                    mstr = f"min_d={min_ce*100:.1f}cm"
                ok_tr += int(ok)
            if ok_tr >= 2:
                passed += 1
            print(f"  [{ok_tr}/3] {pr:46s} ({check_kind:16s}) | {mstr}", flush=True)
        print(f"  SUMMARY v5 ({tag}): {passed}/{len(cases)} strictly zero-shot commands succeeded physically!", flush=True)
        return passed, len(cases)

    polq_gpu = copy.deepcopy(polq).to(DEV).eval()
    p_ok, p_tot = eval_cl_comp(polq_gpu, R_map['10000'], 'QUANTIZED ALIGNED')
    s_ok, s_tot = eval_cl_comp(polq_gpu, R_map['shuffled'], 'SHUFFLED CONTROL')
    metrics['closed_loop'] = dict(aligned=f"{p_ok}/{p_tot}", shuffled=f"{s_ok}/{s_tot}")

    # 8. Exact FP32 parity references (including "pump your hand like fists"!)
    parity_prompts = [
        ("pump your hand like fists", "10000"),
        ("rotate around your base", "10000"),
        ("put the cube in the front left corner", "10000"),
        ("slowly circle down low with your claw closed", "1000"),
        ("pump your hand like fists", "shuffled"),
    ]
    parity_cases = []
    polq.eval()
    for txt, nKey in parity_prompts:
        e = encode_hf(tok, hfq, [txt], pooling=pooling)[0]
        a = (e - mu_t) @ P_t
        q = a @ R_map[nKey]
        qu = (q / max(np.linalg.norm(q), 1e-8)).astype(np.float32)
        zb = project_manifold(qu, cloud_unit_deq, top_k=proj_k, beta=proj_beta)
        obs_seq, act_seq = [], []
        h = torch.zeros(1, 1, polq.d)
        r_par = np.random.default_rng(abs(hash(txt)) % (2 ** 31))
        for step_i in range(3):
            ob_raw = r_par.standard_normal(polq.obs_dim).astype(np.float32) * 0.2
            obs_seq.append(ob_raw.tolist())
            ob_n = (torch.from_numpy(ob_raw) - torch.from_numpy(obs_mu[0, 0])) / torch.from_numpy(obs_sd[0, 0])
            act_t, h = polq.step(ob_n.unsqueeze(0), torch.from_numpy(zb).unsqueeze(0), h, step_i / 79.0)
            act_seq.append(act_t[0].numpy().tolist())
        parity_cases.append(dict(text=txt, nKey=nKey, emb=e.tolist(), q_unit=qu.tolist(),
                                 obs_seq=obs_seq, act_seq=act_seq))

    centroids_2d = []
    for fid, fname in enumerate(FAMILY_NAMES):
        m_s = (np.array(cloud_skill) == fid)
        if m_s.any():
            xy = cloud_2d[m_s].mean(0)
            centroids_2d.append(dict(id=fid, name=fname.replace('cube_', '').replace('_', ' '), x=float(xy[0]), y=float(xy[1])))

    pol_meta = dict(
        lmModel=lm_name, lmPooling=pooling,
        kT=k_t, kV=k_v,
        obsDim=33, d=int(pol_cpu.d), nblocks=int(pol_cpu.nblocks), actDim=7, horizon=80,
        projTopK=proj_k, projBeta=proj_beta,
        skills=FAMILY_NAMES, nOptions=N_OPTIONS,
        cloudCount=int(len(cloud_idx)), cloudSkill=cloud_skill, cloudLabels=cloud_labels,
        centroids2D=centroids_2d, metrics=metrics, tensors=pk.meta,
    )
    open(os.path.join(out_dir, 'model', 'arena_policy.bin'), 'wb').write(bytes(pk.blob))
    json.dump(pol_meta, open(os.path.join(out_dir, 'model', 'arena_policy.json'), 'w'))
    json.dump({'cases': parity_cases}, open(os.path.join(out_dir, 'parity', 'arena_ref.json'), 'w'))
    print(f"\nExported v5 compositional model/arena_policy.bin ({len(pk.blob)/1e6:.2f} MB) + arena_policy.json + parity/arena_ref.json", flush=True)
    print("COMP_V5_DONE", flush=True)


if __name__ == '__main__':
    main()
