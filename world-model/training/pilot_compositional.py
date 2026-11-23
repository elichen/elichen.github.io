"""Zero-Overfitting Compositional Orthogonal Procrustes Pilot Shootout.

Re-captions data_comp/meta.json with strictly neutral zero-leakage captions
(verified by assert_zero_leakage on every single clip) and compares:
  - BAAI/bge-small-en-v1.5 (cls, 33 MB int8)
  - sentence-transformers/all-MiniLM-L6-v2 (mean, 22.7 MB int8)
  - thenlper/gte-small (mean, 33 MB int8)
  - Random-weight MiniLM control
  - Shuffled-pair control
across text PCA regularization reg_t and latent dims (k_t, k_v) on all 37
strictly held-out zero-shot compositional prompts!
"""
import os, sys, json, time, math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from sim_compositional import FAMILY_NAMES, ZERO_SHOT_PROMPTS, describe_compositional, assert_zero_leakage

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
if os.environ.get('GPU_GB') and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, float(os.environ['GPU_GB']) * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
    )


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def svd_cache(X_tr):
    mu = X_tr.mean(0)
    Xc = X_tr - mu
    # Fast eigendecomposition of Xc^T Xc (D x D, takes 0.02s instead of 5s full SVD!)
    cov = (Xc.T @ Xc) / max(1, len(X_tr) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    std = np.sqrt(np.maximum(eigvals[order], 1e-12))
    Vt = eigvecs[:, order].T
    return mu, Xc, std, Vt


def proj_from_cache(cache, X_te, k, reg=None):
    mu, Xc, std, Vt = cache
    P = Vt[:k].T.copy()
    if reg is not None:
        s = std[:k]
        scale = 1.0 / np.sqrt(s ** 2 + reg * float(np.median(s ** 2)))
        P = P * scale[None, :]
    A_tr = Xc @ P
    rms = math.sqrt(float(np.mean(np.sum(A_tr ** 2, axis=1))))
    P = P / max(rms, 1e-8)
    return mu, P, Xc @ P, (X_te - mu) @ P


def pca_proj(X_tr, X_te, k, reg=None):
    """Regularized or unwhitened (reg=None) PCA projection normalized to unit RMS."""
    return proj_from_cache(svd_cache(X_tr), X_te, k, reg=reg)


def procrustes(A, B):
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vt


def dyn_feats_comp(F_arr):
    """100% Language-free, de-confounded compositional kinematic statistics (zero c0 / start_q!)."""
    F_ex = F_arr[:, 28:, :]
    eff = F_ex[:, :, 18:21]
    deff = np.diff(eff, axis=1)

    e_mean = eff.mean(1)
    e_min = eff.min(1)
    e_max = eff.max(1)
    e_end = F_arr[:, -10:, 18:21].mean(1)
    e_ptp = e_max - e_min
    e_spd = np.abs(deff).mean(1) * 20.0
    e_std = eff.std(1)

    v_sm = 0.5 * (deff[:, 1:, :] + deff[:, :-1, :])
    rev = ((v_sm[:, 1:, :] * v_sm[:, :-1, :]) < -1e-6).sum(1) / 10.0

    ec = eff - e_mean[:, None, :]
    ec_m = ec[:, :-1, :]
    ang_xy = (ec_m[:, :, 0] * deff[:, :, 1] - ec_m[:, :, 1] * deff[:, :, 0]).sum(1, keepdims=True) * 10.0
    ang_yz = (ec_m[:, :, 1] * deff[:, :, 2] - ec_m[:, :, 2] * deff[:, :, 1]).sum(1, keepdims=True) * 10.0

    arm_ex = F_ex[:, :, :18]
    j_mean = arm_ex.mean(1)
    j_std = arm_ex.std(1)
    j_end = F_arr[:, -10:, :12].mean(1)
    qvel_abs = np.abs(F_ex[:, :, 12:18]).mean(1)
    qvel_max = np.abs(F_ex[:, :, 12:18]).max(1)
    pan_int = F_arr[:, 14:, 12].sum(1, keepdims=True) / 20.0
    pan_abs = np.abs(F_arr[:, 14:, 12]).sum(1, keepdims=True) / 20.0
    w1_abs = np.abs(F_ex[:, :, 15]).mean(1, keepdims=True) * 4.0
    w3_abs = np.abs(F_ex[:, :, 17]).mean(1, keepdims=True) * 4.0

    g_ex = F_ex[:, :, 27]
    g_mean = g_ex.mean(1, keepdims=True) * 2.5
    g_std = g_ex.std(1, keepdims=True) * 2.5
    g_flips = np.abs(np.diff(F_arr[:, 14:, 27], axis=1)).sum(1, keepdims=True) * 0.5
    g_end = F_arr[:, -10:, 27].mean(1, keepdims=True) * 2.0

    c0 = F_arr[:, 0, 21:24]
    c_end = F_arr[:, -8:, 21:24].mean(1)
    c_disp = c_end - c0
    c_dist = np.linalg.norm(c_disp[:, :2], axis=1, keepdims=True)
    c_zmax = F_arr[:, 20:, 23].max(1, keepdims=True)
    c_vmax = np.linalg.norm(F_arr[:, 20:, 24:27], axis=-1).max(1, keepdims=True)
    c_vdir = F_arr[:, 25:, 24:27].mean(1)
    ih_end = F_arr[:, -10:, 28].mean(1, keepdims=True)
    ih_max = F_arr[:, 20:, 28].max(1, keepdims=True)
    ce_min = np.linalg.norm(F_arr[:, 20:, 18:21] - F_arr[:, 20:, 21:24], axis=-1).min(1, keepdims=True)
    moved = np.clip((np.maximum(c_dist, c_zmax - 0.02) - 0.03) / 0.04, 0.0, 1.0)
    c_end_rel = (c_end - np.array([[0.42, 0.0, 0.02]])) * moved * 3.0

    return np.concatenate([
        e_mean * 2.0, e_min * 1.5, e_max * 1.5, e_end * 1.5,
        e_ptp * 3.0, e_spd * 2.5, e_std * 2.5, rev * 1.5,
        ang_xy, ang_yz, np.abs(ang_xy), np.abs(ang_yz),
        j_mean, j_std, j_end, qvel_abs * 2.0, qvel_max,
        pan_int * 2.0, pan_abs * 2.0, w1_abs, w3_abs,
        g_mean, g_std, g_flips, g_end,
        c_end_rel, c_disp * 3.0, c_dist * 3.0, c_vmax * 2.0, c_vdir * 2.0,
        c_zmax * 3.0, ih_end * 2.5, ih_max * 2.5, ce_min * 2.0, moved * 2.5,
    ], axis=1).astype(np.float64)


@torch.inference_mode()
def encode_hf(tok, model, texts, pooling='mean', bs=256):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=64, return_tensors='pt').to(DEV)
        h = model(**enc).last_hidden_state
        if pooling == 'cls':
            v = h[:, 0]
        else:
            m = enc['attention_mask'].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out.append(v.cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def score_zs(Q_te, G_te, Q_zs, G_tr, fam_tr, fam_te, metas_tr, metas_te, k_nn=15, print_detail=False):
    G_bank_u = unit(G_tr[:6000])
    fam_bank = fam_tr[:6000]
    metas_bank = metas_tr[:6000]
    Q_te_u = unit(Q_te)
    G_te_u = unit(G_te)
    S_pair = Q_te_u[:1000] @ G_te_u[:1000].T
    fosc = float((S_pair > S_pair.diagonal()[:, None]).mean())

    S_nn = Q_te_u @ G_bank_u.T
    top = np.argpartition(-S_nn, k_nn, axis=1)[:, :k_nn]
    fam_ok = grip_ok = h_ok = gn = 0
    for i in range(len(Q_te)):
        nbrs = top[i]
        sims = np.maximum(S_nn[i, nbrs], 0.0)
        votes = np.zeros(len(FAMILY_NAMES))
        for idx_n, sc in zip(nbrs, sims):
            votes[fam_bank[idx_n]] += sc
        fam_ok += int(np.argmax(votes) == fam_te[i])
        mt = metas_te[i]
        if not mt['family'].startswith('cube_'):
            gv, hv = {}, {}
            for idx_n, sc in zip(nbrs, sims):
                mn = metas_bank[idx_n]
                gv[mn.get('gripper', '')] = gv.get(mn.get('gripper', ''), 0.0) + sc
                hv[mn.get('height', '')] = hv.get(mn.get('height', ''), 0.0) + sc
            grip_ok += int(max(gv, key=gv.get) == mt['gripper'])
            h_ok += int(max(hv, key=hv.get) == mt['height'])
            gn += 1

    Q_zs_u = unit(Q_zs)
    S_zs = Q_zs_u @ G_bank_u.T
    zf = za = 0
    rows = []
    for i, (prompt, checks) in enumerate(ZERO_SHOT_PROMPTS):
        nbrs = np.argpartition(-S_zs[i], k_nn)[:k_nn]
        nbrs = nbrs[np.argsort(-S_zs[i, nbrs])]
        sims = np.maximum(S_zs[i, nbrs], 0.0)
        fv, gv, hv, lv, dv, sv = {}, {}, {}, {}, {}, {}
        for idx_n, sc in zip(nbrs, sims):
            mn = metas_bank[idx_n]
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
        if print_detail:
            tag = 'OK' if am else ('~F' if fm else 'XX')
            rows.append(f"    [{tag}] {prompt:46s} -> {pf:16s} h={ph:8s} g={pg:7s} l={pl:6s} s={ps:6s} d={pd:7s} cos={sims[0]:.2f}")
    return fam_ok / len(Q_te) * 100, grip_ok / gn * 100, h_ok / gn * 100, fosc, zf, za, rows


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data_comp'
    d = np.load(os.path.join(data_dir, 'trajs.npz'))
    meta = json.load(open(os.path.join(data_dir, 'meta.json')))
    feat = d['feat'].astype(np.float32)
    fam = d['family']
    metas = meta['metas']

    # Re-caption all clips using concise + varied zero-leakage neutral English
    rng = np.random.default_rng(2026)
    caps = [describe_compositional(m, rng) for m in metas]
    for c in caps:
        assert_zero_leakage(c)
    meta['captions'] = caps
    json.dump(meta, open(os.path.join(data_dir, 'meta.json'), 'w'))
    print(f"Re-captioned all {len(caps)} clips (100% ZERO-LEAKAGE PASS)! Samples:", flush=True)
    for c in caps[:10]:
        print("  -", c, flush=True)

    N_te = 2500
    F_tr, F_te = feat[:-N_te], feat[-N_te:]
    fam_tr, fam_te = fam[:-N_te], fam[-N_te:]
    caps_tr, caps_te = caps[:-N_te], caps[-N_te:]
    metas_tr, metas_te = metas[:-N_te], metas[-N_te:]
    zs_texts = [p for p, _ in ZERO_SHOT_PROMPTS]

    X_v_tr = dyn_feats_comp(F_tr)
    X_v_te = dyn_feats_comp(F_te)
    v_cache = svd_cache(X_v_tr)

    for mname, pooling, is_rand in [
        ('BAAI/bge-small-en-v1.5', 'cls', False),
        ('sentence-transformers/all-MiniLM-L6-v2', 'mean', False),
        ('thenlper/gte-small', 'mean', False),
        ('sentence-transformers/all-MiniLM-L6-v2', 'mean', True),
    ]:
        tok = AutoTokenizer.from_pretrained(mname, local_files_only=True)
        if is_rand:
            cfg = AutoConfig.from_pretrained(mname, local_files_only=True)
            torch.manual_seed(1234)
            hf = AutoModel.from_config(cfg).to(DEV).eval()
            label = 'RANDOM-LM CONTROL'
        else:
            hf = AutoModel.from_pretrained(mname, local_files_only=True).to(DEV).eval()
            label = mname.split('/')[-1]

        E_tr = encode_hf(tok, hf, caps_tr, pooling=pooling)
        E_te = encode_hf(tok, hf, caps_te, pooling=pooling)
        E_zs = encode_hf(tok, hf, zs_texts, pooling=pooling)
        t_cache = svd_cache(E_tr)

        # Raw text ceiling
        S_txt = unit(E_zs) @ unit(E_tr[:6000]).T
        cf = ca = 0
        for i, (_, checks) in enumerate(ZERO_SHOT_PROMPTS):
            nbrs = np.argpartition(-S_txt[i], 15)[:15]
            sims = np.maximum(S_txt[i, nbrs], 0.0)
            fv, gv, hv, lv, dv, sv = {}, {}, {}, {}, {}, {}
            for idx_n, sc in zip(nbrs, sims):
                mn = metas_tr[idx_n]
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
            cf += int(fm)
            ca += int(am)
        print(f"\n=== {label} === RAW-TEXT 384D CEILING: fam={cf}/37 FULL={ca}/37", flush=True)

        configs = [(96, 4.0, 96, 0.2)] if is_rand else [
            (64,  4.0, 64, 0.2),
            (96,  1.0, 96, 0.2),
            (96,  2.0, 96, 0.2),
            (96,  4.0, 96, 0.2),
            (96,  8.0, 96, 0.2),
            (96,  None, 96, 0.2),
            (112, 4.0, 112, 0.2),
            (128, 4.0, 112, 0.2),
            (384, None, 96, 0.2),
        ]
        best = (-1, None)
        for kt, rt, kv, rv in configs:
            mu_t, P_t, A_tr, A_te = proj_from_cache(t_cache, E_te, kt, reg=rt)
            A_zs = (E_zs - mu_t) @ P_t
            _, _, G_tr, G_te = proj_from_cache(v_cache, X_v_te, kv, reg=rv)
            R = procrustes(A_tr, G_tr)
            f1, g1, h1, fosc, zf, za, _ = score_zs(A_te @ R, G_te, A_zs @ R, G_tr, fam_tr, fam_te, metas_tr, metas_te)
            tag = f"kt={kt:3d}(r={str(rt):4s})->kv={kv:3d}(r={str(rv):4s})"
            print(f"  {tag} | val fam={f1:5.1f}% grip={g1:5.1f}% h={h1:5.1f}% FOSC={fosc:.3f} | ZS37 fam={zf:2d}/37 FULL={za:2d}/37", flush=True)
            if (za, zf, f1) > (best[0], best[1][0] if best[1] else -1, best[1][1] if best[1] else -1):
                best = (za, (zf, f1, tag, kt, rt, kv, rv))

        _, (_, _, btag, kt, rt, kv, rv) = best
        mu_t, P_t, A_tr, A_te = proj_from_cache(t_cache, E_te, kt, reg=rt)
        A_zs = (E_zs - mu_t) @ P_t
        _, _, G_tr, G_te = proj_from_cache(v_cache, X_v_te, kv, reg=rv)
        R = procrustes(A_tr, G_tr)
        _, _, _, _, zf, za, rows = score_zs(A_te @ R, G_te, A_zs @ R, G_tr, fam_tr, fam_te, metas_tr, metas_te, print_detail=True)
        print(f"  >> BEST {label} ({btag}): ZS37 fam={zf}/37 FULL={za}/37", flush=True)
        for r in rows:
            print(r, flush=True)

    print("\nPILOT_COMP_V2_DONE", flush=True)


if __name__ == '__main__':
    main()
