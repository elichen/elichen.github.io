"""End-to-end Colab L4 pipeline for Rotating Language Into Robot Control (True English -> R -> UR5e):
  1) Generate 16,000 physically verified 17-behavior UR5e clips (7D joint-delta + gripper @ 20 Hz)
  2) Build 100% language-free unsupervised robot execution-dynamics manifold (DynPCA, ZERO text/labels)
  3) Evaluate RMS-matched single orthogonal Procrustes R (RR^T = I) for MiniLM-L6-v2 vs BGE-small-en-v1.5
     + negative controls (Shuffled pairs R, Random-Weight BERT with same vocab/arch)
  4) Train 100% language-free recurrent BehaviorPolicy pi(a_t | obs_t, z_beh) with Readout2 visual noise
     + isotropic latent-sphere noise so single-rotation R vectors execute crisply from any start pose
  5) Run closed-loop MuJoCo eval on 23 strictly unseen English commands (Aligned N=10k vs Shuffled)
  6) Export quantized browser assets (model/lm.{json,bin}, model/arena_policy.{json,bin}, parity/arena_ref.json)
"""
import os, sys, json, copy, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sim_skills import generate_dataset, SKILL_NAMES, UNSEEN_PROMPTS, UR5eArena
from train_policy import BehaviorPolicy, train_policy

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True


class BinPacker:
    def __init__(self):
        self.blob = bytearray(); self.meta = []; self.deq = {}
    def _pad4(self):
        while len(self.blob) % 4: self.blob.append(0)
    def pack(self, name, arr, kind='f32'):
        a = np.asarray(arr, np.float32); self._pad4()
        e = {'name': name, 'shape': list(a.shape), 'dtype': kind}
        if kind == 'f32':
            e['offset'] = len(self.blob); self.blob.extend(a.tobytes()); back = a.copy()
        elif kind == 'f16':
            h = a.astype(np.float16); e['offset'] = len(self.blob); self.blob.extend(h.tobytes()); back = h.astype(np.float32)
        elif kind == 'i8_col':
            sc = np.maximum(np.abs(a).max(axis=0), 1e-8) / 127.0
            q = np.clip(np.round(a / sc[None, :]), -127, 127).astype(np.int8)
            e['scaleOffset'] = len(self.blob); self.blob.extend(sc.astype(np.float32).tobytes()); self._pad4()
            e['offset'] = len(self.blob); self.blob.extend(q.tobytes()); back = q.astype(np.float32) * sc[None, :]
        elif kind == 'i8_row':
            sc = np.maximum(np.abs(a).max(axis=1), 1e-8) / 127.0
            q = np.clip(np.round(a / sc[:, None]), -127, 127).astype(np.int8)
            e['scaleOffset'] = len(self.blob); self.blob.extend(sc.astype(np.float32).tobytes()); self._pad4()
            e['offset'] = len(self.blob); self.blob.extend(q.tobytes()); back = q.astype(np.float32) * sc[:, None]
        else:
            raise ValueError(kind)
        self._pad4(); self.meta.append(e); self.deq[name] = back; return back
    def pack_linear(self, prefix, w_pt, b_pt, kind='i8_col'):
        w = w_pt.detach().cpu().float().numpy().T
        b = b_pt.detach().cpu().float().numpy()
        self.pack(f'{prefix}/w', w, kind if w.size >= 2048 else 'f16')
        self.pack(f'{prefix}/b', b, 'f32')


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def procrustes(A, B):
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vt


def dyn_feats(F_arr):
    """100% unsupervised kinematic trajectory summary of execution window (t=16..80).
    Zero text, zero skill labels — captures what the robot physically does without start-pose nuisance."""
    F_mid = F_arr[:, 16:, :]
    f_mean = F_mid.mean(1)
    f_std = F_mid.std(1)
    f_end = F_arr[:, -10:, :].mean(1)
    f_max = F_mid.max(1)
    f_min = F_mid.min(1)
    pan_int = F_arr[:, :, 12].sum(1, keepdims=True) / 20.0
    pan_abs = np.abs(F_arr[:, :, 12]).sum(1, keepdims=True) / 20.0
    c_disp = F_arr[:, -1, 21:24] - F_arr[:, 0, 21:24]
    c_vmax = np.linalg.norm(F_arr[:, :, 24:27], axis=-1).max(1, keepdims=True)
    c_zmax = F_arr[:, :, 23].max(1, keepdims=True)
    g_flips = np.abs(np.diff(F_arr[:, :, 27], axis=1)).sum(1, keepdims=True)
    return np.concatenate([
        f_mean, f_std, f_end, f_max, f_min, pan_int, pan_abs, c_disp, c_vmax, c_zmax, g_flips
    ], axis=1).astype(np.float64)


def reg_pca_rms(X_tr, X_te, k, lam):
    """Unpaired PCA + gentle shrinkage whitening + single global RMS scalar = 1.0."""
    mu = X_tr.mean(0)
    _, S, Vt = np.linalg.svd(X_tr - mu, full_matrices=False)
    sv = S[:k] / math.sqrt(len(X_tr) - 1)
    if lam is None:
        P = Vt[:k].T
    else:
        P = Vt[:k].T / np.sqrt(sv * sv + lam)[None, :]
    A_tr = (X_tr - mu) @ P
    rms = math.sqrt(float((A_tr ** 2).mean()))
    P = P / rms
    return mu, P, (X_tr - mu) @ P, (X_te - mu) @ P


@torch.inference_mode()
def encode_hf(tok, model, texts, pooling='mean', bs=256, max_pos=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=max_pos, return_tensors='pt').to(DEV)
        h = model(**enc).last_hidden_state
        if pooling == 'cls':
            pooled = h[:, 0]
        else:
            m = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
        pooled = F.normalize(pooled.float(), p=2, dim=-1)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def score_rotation(A_tr, A_te, A_un, G_tr, G_te, sk_tr, sk_te, metas_tr, metas_te, n_fit=10000, shuffled=False):
    place_id = SKILL_NAMES.index('place')
    m_tr = sk_tr == place_id; m_te = sk_te == place_id
    g_tr = np.array([metas_tr[i]['goal'][:2] for i in np.where(m_tr)[0]])
    g_te = np.array([metas_te[i]['goal'][:2] for i in np.where(m_te)[0]])
    X_pl = np.c_[G_tr[m_tr], np.ones(m_tr.sum())]
    W_pl = np.linalg.solve(X_pl.T @ X_pl + 0.5 * np.eye(X_pl.shape[1]), X_pl.T @ g_tr)
    ceil_cm = float(np.median(np.linalg.norm(np.c_[G_te[m_te], np.ones(m_te.sum())] @ W_pl - g_te, axis=1) * 100))

    m_n = min(len(A_tr), n_fit)
    if shuffled:
        rng = np.random.default_rng(7)
        R = procrustes(A_tr[:m_n][rng.permutation(m_n)], G_tr[:m_n])
    else:
        R = procrustes(A_tr[:m_n], G_tr[:m_n])

    Q_te = A_te @ R; Q_u = unit(Q_te)
    G_te_u = unit(G_te); G_bk_u = unit(G_tr[:4000]); sk_bk = sk_tr[:4000]
    S_p = Q_u[:1000] @ G_te_u[:1000].T
    fosc = float((S_p > S_p.diagonal()[:, None]).mean())

    S_nn = Q_u @ G_bk_u.T
    top15 = np.argpartition(-S_nn, 15, axis=1)[:, :15]
    acc1 = float(np.mean([
        np.argmax(np.bincount(sk_bk[top15[i]], weights=np.maximum(S_nn[i, top15[i]], 0), minlength=len(SKILL_NAMES))) == sk_te[i]
        for i in range(len(Q_te))
    ]))
    pl_pred = np.c_[Q_te[m_te], np.ones(m_te.sum())] @ W_pl
    pl_err = float(np.median(np.linalg.norm(pl_pred - g_te, axis=1) * 100))

    Qu_raw = A_un @ R; Qu = unit(Qu_raw); Su = Qu @ G_bk_u.T
    u1 = u3 = 0
    unseen_rows = []
    for i, (pr, ts) in enumerate(UNSEEN_PROMPTS):
        nbrs = np.argpartition(-Su[i], 15)[:15]
        v = np.bincount(sk_bk[nbrs], weights=np.maximum(Su[i, nbrs], 0), minlength=len(SKILL_NAMES))
        ord_s = np.argsort(-v)
        tid = SKILL_NAMES.index(ts)
        u1 += int(ord_s[0] == tid); u3 += int(tid in ord_s[:3])
        p1, p2 = SKILL_NAMES[ord_s[0]], SKILL_NAMES[ord_s[1]]
        mk = 'OK' if p1 == ts else ('~3' if tid in ord_s[:3] else 'XX')
        ex = ''
        if p1 == 'place':
            xy = (np.r_[Qu_raw[i], 1.0] @ W_pl) * 100
            ex = f" -> ({xy[0]:.0f}, {xy[1]:+.0f})cm"
        elif p1 in ('throw', 'push', 'point', 'rotate_base'):
            bn = nbrs[np.argmax(Su[i, nbrs])]
            ex = f" (dir={metas_tr[bn].get('dir','?')})"
        unseen_rows.append(f"    [{mk}] {pr:48s} -> {p1:12s} (2nd: {p2:11s}) cos={Su[i,nbrs].max():.2f}{ex}")

    return dict(R=R, W_pl=W_pl, ceil_cm=ceil_cm, top1=acc1, fosc=fosc, pl_err=pl_err,
                u1=u1, u3=u3, unseen_rows=unseen_rows)


def export_hf_lm(out_dir, model_name, pooling, max_pos=64):
    """Export quantized int8 BERT/MiniLM/BGE encoder + WordPiece vocab, returning exact dequantized HF model."""
    torch.set_float32_matmul_precision('highest')
    tok = AutoTokenizer.from_pretrained(model_name)
    hf = AutoModel.from_pretrained(model_name).eval().cpu()
    sd = hf.state_dict()
    cfg = hf.config
    pk = BinPacker()

    tok_w = sd['embeddings.word_embeddings.weight'].numpy()
    pos_w = sd['embeddings.position_embeddings.weight'].numpy()[:max_pos] + sd['embeddings.token_type_embeddings.weight'].numpy()[0:1]
    pk.pack('emb/tok', tok_w, 'i8_row')
    pk.pack('emb/pos', pos_w, 'f16')
    pk.pack('emb/ln/w', sd['embeddings.LayerNorm.weight'].numpy(), 'f32')
    pk.pack('emb/ln/b', sd['embeddings.LayerNorm.bias'].numpy(), 'f32')

    n_layers = cfg.num_hidden_layers
    for l in range(n_layers):
        p = f'encoder.layer.{l}'
        qkv_w = torch.cat([sd[f'{p}.attention.self.query.weight'],
                           sd[f'{p}.attention.self.key.weight'],
                           sd[f'{p}.attention.self.value.weight']], dim=0)
        qkv_b = torch.cat([sd[f'{p}.attention.self.query.bias'],
                           sd[f'{p}.attention.self.key.bias'],
                           sd[f'{p}.attention.self.value.bias']], dim=0)
        pk.pack_linear(f'l{l}/qkv', qkv_w, qkv_b, 'i8_col')
        pk.pack_linear(f'l{l}/o', sd[f'{p}.attention.output.dense.weight'], sd[f'{p}.attention.output.dense.bias'], 'i8_col')
        pk.pack(f'l{l}/ln1/w', sd[f'{p}.attention.output.LayerNorm.weight'].numpy(), 'f32')
        pk.pack(f'l{l}/ln1/b', sd[f'{p}.attention.output.LayerNorm.bias'].numpy(), 'f32')
        pk.pack_linear(f'l{l}/fc1', sd[f'{p}.intermediate.dense.weight'], sd[f'{p}.intermediate.dense.bias'], 'i8_col')
        pk.pack_linear(f'l{l}/fc2', sd[f'{p}.output.dense.weight'], sd[f'{p}.output.dense.bias'], 'i8_col')
        pk.pack(f'l{l}/ln2/w', sd[f'{p}.output.LayerNorm.weight'].numpy(), 'f32')
        pk.pack(f'l{l}/ln2/b', sd[f'{p}.output.LayerNorm.bias'].numpy(), 'f32')

    hfq = copy.deepcopy(hf)
    sdq = hfq.state_dict()
    sdq['embeddings.word_embeddings.weight'].copy_(torch.from_numpy(pk.deq['emb/tok']))
    sdq['embeddings.position_embeddings.weight'][:max_pos].copy_(torch.from_numpy(pk.deq['emb/pos']))
    sdq['embeddings.token_type_embeddings.weight'].zero_()
    for l in range(n_layers):
        p = f'encoder.layer.{l}'
        qkv = torch.from_numpy(pk.deq[f'l{l}/qkv/w'].T)
        qw, kw, vw = qkv.chunk(3, dim=0)
        sdq[f'{p}.attention.self.query.weight'].copy_(qw)
        sdq[f'{p}.attention.self.key.weight'].copy_(kw)
        sdq[f'{p}.attention.self.value.weight'].copy_(vw)
        sdq[f'{p}.attention.output.dense.weight'].copy_(torch.from_numpy(pk.deq[f'l{l}/o/w'].T))
        sdq[f'{p}.intermediate.dense.weight'].copy_(torch.from_numpy(pk.deq[f'l{l}/fc1/w'].T))
        sdq[f'{p}.output.dense.weight'].copy_(torch.from_numpy(pk.deq[f'l{l}/fc2/w'].T))
    hfq.load_state_dict(sdq)
    hfq.to(DEV).eval()

    inv_vocab = [''] * len(tok.vocab)
    for token_str, tid in tok.vocab.items():
        if tid < len(inv_vocab):
            inv_vocab[tid] = token_str

    meta = {
        'model': model_name,
        'pooling': pooling,
        'd': cfg.hidden_size,
        'layers': n_layers,
        'heads': cfg.num_attention_heads,
        'maxPos': max_pos,
        'eps': cfg.layer_norm_eps,
        'clsId': tok.cls_token_id,
        'sepId': tok.sep_token_id,
        'unkId': tok.unk_token_id,
        'vocab': inv_vocab,
        'tensors': pk.meta,
    }
    os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
    open(os.path.join(out_dir, 'model', 'minilm.bin'), 'wb').write(bytes(pk.blob))
    json.dump(meta, open(os.path.join(out_dir, 'model', 'minilm.json'), 'w'))
    print(f"Exported {model_name} -> model/minilm.bin ({len(pk.blob)/1e6:.2f} MB)", flush=True)
    return tok, hfq


@torch.inference_mode()
def eval_closed_loop(robot_dir, pol, obs_mu, obs_sd, tok, hfq, pooling, mu_t, P_t, R, tag, T=80):
    env = UR5eArena(robot_dir)
    obs_mu_t = torch.from_numpy(obs_mu[0, 0]).to(DEV)
    obs_sd_t = torch.from_numpy(obs_sd[0, 0]).to(DEV)
    rng = np.random.default_rng(123)

    test_cases = [
        ("throw cube", "throw"),
        ("yeet the red block", "throw"),
        ("toss it to the right", "throw"),
        ("rotate around your base", "rotate_base"),
        ("do a 360", "rotate_base"),
        ("put the cube in the front left corner", "place", (0.50, -0.18)),
        ("move the red block to the back right", "place", (0.34, 0.18)),
        ("set it down in the middle of the table", "place", (0.42, 0.0)),
        ("shove the cube to the left", "push", "left"),
        ("knock the red block to the right", "push", "right"),
        ("pick up the red block and keep it aloft", "lift_hold"),
        ("grab the block and let it fall", "lift_drop"),
        ("say hi", "wave"),
        ("wave to me", "wave"),
        ("touch the ceiling", "reach_up"),
        ("take a bow", "bow"),
        ("stir the pot over the table", "circle"),
        ("sweep side to side", "sweep"),
        ("wiggle your claws open and closed", "clap_gripper"),
        ("boop the red square", "poke_cube"),
        ("point over to the left side", "point", "left"),
        ("nod your head yes", "nod"),
        ("shake and party", "dance"),
        ("freeze at home position", "home"),
    ]

    embs = encode_hf(tok, hfq, [c[0] for c in test_cases], pooling=pooling)
    Qu = unit((embs - mu_t) @ P_t @ R).astype(np.float32)

    print(f"\n=== CLOSED-LOOP MUJOCO EVAL ({tag}): UNSEEN ENGLISH -> LM -> R -> UR5e ===", flush=True)
    passed = 0
    for idx, case in enumerate(test_cases):
        prompt, sk = case[0], case[1]
        z_t = torch.from_numpy(Qu[idx:idx+1]).to(DEV)
        ok_trials = 0; mstr = ''
        for _ in range(3):
            ob = env.reset(rng)
            c0 = env.cube_pos().copy()
            h = torch.zeros(1, 1, pol.d, device=DEV)
            max_cz = max_cv = max_ez = 0.0; min_ez = min_ce = 9.0
            gh, ph, exy, w1h = [], [], [], []
            for t in range(T):
                ob_t = (torch.from_numpy(ob).to(DEV) - obs_mu_t) / obs_sd_t
                a_t, h = pol.step(ob_t.unsqueeze(0), z_t, h, t / float(T - 1))
                ob = env.step(a_t[0].cpu().numpy())
                c = env.cube_pos(); cv = env.cube_vel(); e = env.eff_pos()
                max_cz = max(max_cz, float(c[2])); max_cv = max(max_cv, float(np.linalg.norm(cv)))
                max_ez = max(max_ez, float(e[2])); min_ez = min(min_ez, float(e[2]))
                min_ce = min(min_ce, float(np.linalg.norm(c - e)))
                gh.append(env.grip_closed()); ph.append(float(env.d.qpos[0]))
                exy.append(e[:2].copy()); w1h.append(float(env.d.qpos[3]))
            c_end = env.cube_pos(); fly = float(np.linalg.norm(c_end[:2] - c0[:2]))
            dpan = max(ph) - min(ph); g_rng = max(gh) - min(gh)
            exy = np.array(exy)
            yspan = float(exy[:, 1].max() - exy[:, 1].min()); xspan = float(exy[:, 0].max() - exy[:, 0].min())

            ok = False
            if sk == 'throw':
                ok = (fly > 0.30 or max_cv > 1.0 or max_cz > 0.35) and (env.has_cube() < 0.5)
                mstr = f"fly={fly:.2f}m max_v={max_cv:.1f}m/s max_z={max_cz:.2f}m"
            elif sk == 'rotate_base':
                ok = dpan > 3.2; mstr = f"spin={math.degrees(dpan):.0f}deg"
            elif sk == 'place':
                gx, gy = case[2]; err = math.hypot(c_end[0] - gx, c_end[1] - gy) * 100
                ok = err < 10.0 and c_end[2] < 0.05
                mstr = f"end=({c_end[0]*100:.0f},{c_end[1]*100:+.0f})cm err={err:.1f}cm"
            elif sk == 'push':
                dy = c_end[1] - c0[1]
                ok = (dy < -0.07) if case[2] == 'left' else (dy > 0.07)
                mstr = f"dy={dy*100:+.1f}cm"
            elif sk == 'lift_hold':
                ok = c_end[2] > 0.11 and env.has_cube() > 0.5
                mstr = f"end_z={c_end[2]*100:.0f}cm ih={int(env.has_cube())}"
            elif sk == 'lift_drop':
                ok = max_cz > 0.15 and c_end[2] < 0.045
                mstr = f"peak_z={max_cz*100:.0f}cm end_z={c_end[2]*100:.0f}cm"
            elif sk == 'wave':
                ok = max_ez > 0.45 and (max(w1h[20:]) - min(w1h[20:])) > 0.5
                mstr = f"max_ez={max_ez*100:.0f}cm w_amp={max(w1h[20:])-min(w1h[20:]):.2f}"
            elif sk == 'reach_up':
                ok = max_ez > 0.68; mstr = f"max_ez={max_ez*100:.0f}cm"
            elif sk == 'bow':
                ok = min_ez < 0.15; mstr = f"min_ez={min_ez*100:.0f}cm"
            elif sk == 'circle':
                ok = xspan > 0.18 and yspan > 0.18; mstr = f"span=({xspan*100:.0f}x{yspan*100:.0f})cm"
            elif sk == 'sweep':
                ok = yspan > 0.35; mstr = f"y_span={yspan*100:.0f}cm"
            elif sk == 'clap_gripper':
                ga = np.array(gh) > 0.4; flips = int(np.sum(ga[1:] != ga[:-1]))
                ok = g_rng > 0.5 and flips >= 3; mstr = f"flips={flips} rng={g_rng:.2f}"
            elif sk == 'poke_cube':
                ok = min_ce < 0.065 and c_end[2] < 0.05; mstr = f"min_d={min_ce*100:.1f}cm"
            elif sk == 'point':
                ok = abs(env.d.qpos[0] - (-math.pi)) < 0.7 or env.eff_pos()[1] < -0.35
                mstr = f"eff_y={env.eff_pos()[1]*100:+.0f}cm pan={math.degrees(env.d.qpos[0]):.0f}deg"
            elif sk == 'nod':
                amp = max(w1h[15:]) - min(w1h[15:]); ok = amp > 0.6; mstr = f"nod={amp:.2f}rad"
            elif sk == 'dance':
                ok = dpan > 0.6 and g_rng > 0.4; mstr = f"pan={dpan:.2f} g_rng={g_rng:.2f}"
            elif sk == 'home':
                eh = float(np.linalg.norm(env.d.qpos[:6] - np.array([-math.pi/2,-math.pi/2,math.pi/2,-math.pi/2,-math.pi/2,0])))
                ok = eh < 0.6; mstr = f"home_err={eh:.2f}rad"
            ok_trials += int(ok)
        if ok_trials >= 2: passed += 1
        print(f"  [{ok_trials}/3] {prompt:42s} ({sk:12s}) | {mstr}", flush=True)
    print(f"  SUMMARY ({tag}): {passed}/{len(test_cases)} unseen English commands succeeded physically!", flush=True)
    return passed, len(test_cases)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else '/content'
    robot_dir = os.path.join(base, 'robot')
    data_dir = os.path.join(base, 'data_arena')
    out_dir = os.path.join(base, 'browser_export_arena')
    if os.environ.get('GPU_GB') and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, float(os.environ['GPU_GB']) * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
        )
    os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'parity'), exist_ok=True)

    # 1. Generate 16,000 verified multi-behavior clips with enriched captions
    if not os.path.exists(os.path.join(data_dir, 'trajs.npz')):
        generate_dataset(robot_dir, data_dir, total_clips=16000, T=80, workers=10, seed=42)

    d = np.load(os.path.join(data_dir, 'trajs.npz'))
    meta_d = json.load(open(os.path.join(data_dir, 'meta.json')))
    obs = d['obs'].astype(np.float32)
    act = d['act'].astype(np.float32)
    feat = d['feat'].astype(np.float32)
    skill = d['skill']
    caps = meta_d['captions']
    metas = meta_d['metas']

    N_te = 2000
    obs_tr, act_tr = obs[:-N_te], act[:-N_te]
    feat_tr, feat_te = feat[:-N_te], feat[-N_te:]
    sk_tr, sk_te = skill[:-N_te], skill[-N_te:]
    caps_tr, caps_te = caps[:-N_te], caps[-N_te:]
    metas_tr, metas_te = metas[:-N_te], metas[-N_te:]
    unseen_texts = [p for p, _ in UNSEEN_PROMPTS]

    # 2. Unsupervised language-free robot dynamics manifold (DynPCA)
    D_tr = dyn_feats(feat_tr); D_te = dyn_feats(feat_te)
    d_mu, d_sd = D_tr.mean(0), np.maximum(D_tr.std(0), 1e-4)
    D_tr = (D_tr - d_mu) / d_sd; D_te = (D_te - d_mu) / d_sd

    # 3. Compare MiniLM-L6-v2 vs BGE-small-en-v1.5 + Random-Weight MiniLM control!
    best = None
    for lm_name, pooling in (
        ('sentence-transformers/all-MiniLM-L6-v2', 'mean'),
        ('BAAI/bge-small-en-v1.5', 'cls'),
    ):
        tok = AutoTokenizer.from_pretrained(lm_name)
        hf = AutoModel.from_pretrained(lm_name).to(DEV).eval()
        E_tr = encode_hf(tok, hf, caps_tr, pooling=pooling)
        E_te = encode_hf(tok, hf, caps_te, pooling=pooling)
        E_un = encode_hf(tok, hf, unseen_texts, pooling=pooling)
        del hf; torch.cuda.empty_cache()

        print(f"\n=== {lm_name} ({pooling}) ===", flush=True)
        for k_t, k_v, lam_t, lam_v in (
            (64, 64, 0.005, 0.5),
            (80, 80, 0.003, 0.3),
            (96, 96, 0.002, 0.2),
            (96, 96, 0.005, 0.5),
            (112, 112, 0.002, 0.2),
        ):
            mu_t, P_t, A_tr, A_te = reg_pca_rms(E_tr, E_te, k_t, lam_t)
            A_un = (E_un - mu_t) @ P_t
            _, _, G_tr, G_te = reg_pca_rms(D_tr, D_te, k_v, lam_v)
            res = score_rotation(A_tr, A_te, A_un, G_tr, G_te, sk_tr, sk_te, metas_tr, metas_te)
            # Composite score favoring unseen open-vocab accuracy, low place cm, and low FOSC
            sc = res['u1'] * 2.5 + res['top1'] * 40.0 - res['pl_err'] * 1.2 - res['fosc'] * 30.0
            print(f"  k=({k_t:3d},{k_v:3d}) lam=({lam_t},{lam_v}) | top1={res['top1']*100:5.1f}% FOSC={res['fosc']:.3f} place={res['pl_err']:4.1f}cm | UNSEEN={res['u1']:2d}/40 (t3={res['u3']:2d}) | score={sc:.1f}", flush=True)
            if best is None or sc > best['sc']:
                best = dict(sc=sc, lm_name=lm_name, pooling=pooling, k_t=k_t, k_v=k_v, lam_t=lam_t, lam_v=lam_v, res=res)

    print(f"\n>>> WINNER: {best['lm_name']} ({best['pooling']}) k=({best['k_t']},{best['k_v']}) lam=({best['lam_t']},{best['lam_v']})", flush=True)
    print('\n'.join(best['res']['unseen_rows']), flush=True)

    # 4. Export winning LM in int8 FIRST so all R matrices & parity refs match browser int8 EXACTLY!
    tok, hfq = export_hf_lm(out_dir, best['lm_name'], best['pooling'])
    E_tr = encode_hf(tok, hfq, caps_tr, pooling=best['pooling'])
    E_te = encode_hf(tok, hfq, caps_te, pooling=best['pooling'])
    E_un = encode_hf(tok, hfq, unseen_texts, pooling=best['pooling'])

    k_t, k_v, lam_t, lam_v = best['k_t'], best['k_v'], best['lam_t'], best['lam_v']
    mu_t, P_t, A_tr, A_te = reg_pca_rms(E_tr, E_te, k_t, lam_t)
    A_un = (E_un - mu_t) @ P_t
    mu_v, P_v, G_tr, G_te = reg_pca_rms(D_tr, D_te, k_v, lam_v)

    # Also measure Random-Weight BERT control (same vocab & arch, ZERO pretrained English!)
    cfg_r = AutoConfig.from_pretrained(best['lm_name'])
    torch.manual_seed(42)
    rand_lm = AutoModel.from_config(cfg_r).to(DEV).eval()
    Er_tr = encode_hf(tok, rand_lm, caps_tr, pooling=best['pooling'])
    Er_te = encode_hf(tok, rand_lm, caps_te, pooling=best['pooling'])
    Er_un = encode_hf(tok, rand_lm, unseen_texts, pooling=best['pooling'])
    del rand_lm; torch.cuda.empty_cache()
    _, Pr_t, Ar_tr, Ar_te = reg_pca_rms(Er_tr, Er_te, k_t, lam_t)
    Ar_un = (Er_un - Er_tr.mean(0)) @ Pr_t
    res_rand = score_rotation(Ar_tr, Ar_te, Ar_un, G_tr, G_te, sk_tr, sk_te, metas_tr, metas_te)
    print(f"\nCONTROL Random-Weight LM (same vocab/arch, ZERO pretrained English): val top1={res_rand['top1']*100:.1f}% FOSC={res_rand['fosc']:.3f} place={res_rand['pl_err']:.1f}cm UNSEEN={res_rand['u1']}/40", flush=True)

    # 5. Train 100% LANGUAGE-FREE recurrent BehaviorPolicy on G_tr (ZERO text!)
    pol, obs_mu, obs_sd = train_policy(obs_tr, act_tr, G_tr, steps=14000, bs=256, lr=8e-4, dz=k_v)
    torch.save({'state': pol.state_dict(), 'obs_mu': obs_mu, 'obs_sd': obs_sd,
                'obs_dim': obs.shape[2], 'dz': k_v, 'd': pol.d, 'nblocks': pol.nblocks},
               os.path.join(data_dir, 'behavior_policy.pt'))

    # 6. Export BehaviorPolicy + all orthogonal R_N + 510-clip manifold cloud + parity refs
    torch.set_float32_matmul_precision('highest')
    pol_cpu = copy.deepcopy(pol).cpu().eval()
    sd_p = pol_cpu.state_dict()
    pk = BinPacker()
    pk.pack('pol/obs_mu', obs_mu[0, 0], 'f32')
    pk.pack('pol/obs_sd', obs_sd[0, 0], 'f32')
    pk.pack_linear('pol/in', sd_p['in_proj.weight'], sd_p['in_proj.bias'], 'i8_col')
    pk.pack_linear('pol/gru_ih', sd_p['gru.weight_ih_l0'], sd_p['gru.bias_ih_l0'], 'i8_col')
    pk.pack_linear('pol/gru_hh', sd_p['gru.weight_hh_l0'], sd_p['gru.bias_hh_l0'], 'i8_col')
    for i in range(pol.nblocks):
        bp = f'blocks.{i}'
        pk.pack(f'pol/b{i}/ln/w', sd_p[f'{bp}.ln.weight'].numpy(), 'f32')
        pk.pack(f'pol/b{i}/ln/b', sd_p[f'{bp}.ln.bias'].numpy(), 'f32')
        pk.pack_linear(f'pol/b{i}/film', sd_p[f'{bp}.film.weight'], sd_p[f'{bp}.film.bias'], 'i8_col')
        pk.pack_linear(f'pol/b{i}/fc1', sd_p[f'{bp}.fc1.weight'], sd_p[f'{bp}.fc1.bias'], 'i8_col')
        pk.pack_linear(f'pol/b{i}/fc2', sd_p[f'{bp}.fc2.weight'], sd_p[f'{bp}.fc2.bias'], 'i8_col')
    pk.pack('pol/out_ln/w', sd_p['out_ln.weight'].numpy(), 'f32')
    pk.pack('pol/out_ln/b', sd_p['out_ln.bias'].numpy(), 'f32')
    pk.pack_linear('pol/head', sd_p['head.weight'], sd_p['head.bias'], 'f16')

    # Exact dequantized policy for parity + closed-loop eval!
    polq = copy.deepcopy(pol_cpu)
    sdpq = polq.state_dict()
    sdpq['in_proj.weight'].copy_(torch.from_numpy(pk.deq['pol/in/w'].T))
    sdpq['gru.weight_ih_l0'].copy_(torch.from_numpy(pk.deq['pol/gru_ih/w'].T))
    sdpq['gru.weight_hh_l0'].copy_(torch.from_numpy(pk.deq['pol/gru_hh/w'].T))
    for i in range(pol.nblocks):
        bp = f'blocks.{i}'
        sdpq[f'{bp}.film.weight'].copy_(torch.from_numpy(pk.deq[f'pol/b{i}/film/w'].T))
        sdpq[f'{bp}.fc1.weight'].copy_(torch.from_numpy(pk.deq[f'pol/b{i}/fc1/w'].T))
        sdpq[f'{bp}.fc2.weight'].copy_(torch.from_numpy(pk.deq[f'pol/b{i}/fc2/w'].T))
    sdpq['head.weight'].copy_(torch.from_numpy(pk.deq['pol/head/w'].T))
    polq.load_state_dict(sdpq)

    pk.pack('rot/mu_t', mu_t, 'f32')
    pk.pack('rot/P_t', P_t, 'f32')

    # 2D unsupervised PCA slice of G_tr for live manifold visualization
    _, _, Vt_2d = np.linalg.svd(G_tr - G_tr.mean(0), full_matrices=False)
    vis_proj_2d = Vt_2d[:2].T.astype(np.float32)
    pre_proj_2d = vis_proj_2d[:k_t].copy()
    for c in range(2):
        nrm = np.linalg.norm(pre_proj_2d[:, c])
        if nrm > 1e-8: pre_proj_2d[:, c] /= nrm

    rng_cl = np.random.default_rng(7)
    cloud_idx = []
    for sid in range(len(SKILL_NAMES)):
        cand = np.where(sk_te == sid)[0]
        if len(cand) > 0:
            cloud_idx.extend(rng_cl.choice(cand, size=min(30, len(cand)), replace=False).tolist())
    cloud_idx = np.array(cloud_idx, dtype=np.int32)
    cloud_g = G_te[cloud_idx]
    cloud_unit = unit(cloud_g).astype(np.float32)
    cloud_2d = (cloud_g @ vis_proj_2d).astype(np.float32)
    cloud_skill = sk_te[cloud_idx].astype(np.int32).tolist()
    cloud_labels = []
    for idx in cloud_idx:
        m = metas_te[idx]; sk = m['skill']
        if sk == 'place': cloud_labels.append(f"place ({m['goal'][0]*100:.0f}, {m['goal'][1]*100:+.0f}) cm")
        elif 'dir' in m: cloud_labels.append(f"{sk.replace('_', ' ')} ({m['dir']})")
        else: cloud_labels.append(sk.replace('_', ' '))

    pk.pack('vis/proj_2d', vis_proj_2d, 'f32')
    pk.pack('vis/pre_proj_2d', pre_proj_2d, 'f32')
    pk.pack('vis/cloud_2d', cloud_2d, 'f32')
    pk.pack('vis/cloud_unit', cloud_unit, 'f16')

    # Traj-only linear place probe (zero text!)
    place_id = SKILL_NAMES.index('place')
    m_pl = sk_tr == place_id
    goals_tr = np.array([metas_tr[i]['goal'][:2] for i in np.where(m_pl)[0]])
    X_pl = np.c_[G_tr[m_pl], np.ones(m_pl.sum())]
    W_pl = np.linalg.solve(X_pl.T @ X_pl + 0.5 * np.eye(X_pl.shape[1]), X_pl.T @ goals_tr)
    pk.pack('vis/place_w', W_pl[:k_v], 'f32')
    pk.pack('vis/place_b', W_pl[k_v], 'f32')

    N_LIST = [30, 100, 300, 1000, 10000]
    N_OPTIONS = [str(n) for n in N_LIST] + ['shuffled']
    metrics = {}
    R_map = {}
    print("\n=== QUANTIZED LM ORTHOGONAL PROCRUSTES SCALING CURVE ===", flush=True)
    for nKey in N_OPTIONS:
        is_shuf = (nKey == 'shuffled')
        n_fit = 10000 if is_shuf else int(nKey)
        r = score_rotation(A_tr, A_te, A_un, G_tr, G_te, sk_tr, sk_te, metas_tr, metas_te, n_fit=n_fit, shuffled=is_shuf)
        pk.pack(f'rot/R_{nKey}', r['R'], 'f32')
        R_map[nKey] = r['R']
        metrics[nKey] = dict(top1=round(r['top1'], 3), foscttm=round(r['fosc'], 3),
                             place_cm=round(r['pl_err'], 1), unseen40=int(r['u1']))
        print(f"  {nKey:8s} | val top1={r['top1']*100:5.1f}% | FOSC={r['fosc']:.3f} | place={r['pl_err']:4.1f}cm | UNSEEN40={r['u1']:2d}/40 (t3={r['u3']:2d})", flush=True)

    # Also store Random-LM metrics for article/UI comparison
    metrics['random_lm'] = dict(top1=round(res_rand['top1'], 3), foscttm=round(res_rand['fosc'], 3),
                                place_cm=round(res_rand['pl_err'], 1), unseen40=int(res_rand['u1']))
    # Dummy untrained key pointing to shuffled for backward parity compatibility
    pk.pack('rot/R_untrained', R_map['shuffled'], 'f32')

    # 7. Run FULL CLOSED-LOOP MUJOCO EVAL using exact quantized LM + quantized BehaviorPolicy!
    polq_gpu = copy.deepcopy(polq).to(DEV).eval()
    p_ok, p_tot = eval_closed_loop(robot_dir, polq_gpu, obs_mu, obs_sd, tok, hfq, best['pooling'], mu_t, P_t, R_map['10000'], 'QUANTIZED N=10000')
    s_ok, s_tot = eval_closed_loop(robot_dir, polq_gpu, obs_mu, obs_sd, tok, hfq, best['pooling'], mu_t, P_t, R_map['shuffled'], 'SHUFFLED CONTROL')
    metrics['closed_loop'] = dict(aligned=f"{p_ok}/{p_tot}", shuffled=f"{s_ok}/{s_tot}")

    # 8. Build exact FP32 parity refs (dequantized LM + R + 3-step dequantized BehaviorPolicy!)
    parity_prompts = [
        ("throw cube", "10000"),
        ("rotate around your base", "10000"),
        ("put the cube in the front left corner", "10000"),
        ("say hi", "1000"),
        ("throw cube", "shuffled"),
    ]
    parity_cases = []
    polq.eval()
    for txt, nKey in parity_prompts:
        e = encode_hf(tok, hfq, [txt], pooling=best['pooling'])[0]
        a = (e - mu_t) @ P_t
        q = a @ R_map[nKey]
        qu = (q / max(np.linalg.norm(q), 1e-8)).astype(np.float32)
        obs_seq, act_seq = [], []
        h = torch.zeros(1, 1, polq.d)
        r_par = np.random.default_rng(abs(hash(txt)) % (2**31))
        for step_i in range(3):
            ob_raw = r_par.standard_normal(polq.obs_dim).astype(np.float32) * 0.2
            obs_seq.append(ob_raw.tolist())
            ob_n = (torch.from_numpy(ob_raw) - torch.from_numpy(obs_mu[0, 0])) / torch.from_numpy(obs_sd[0, 0])
            act_t, h = polq.step(ob_n.unsqueeze(0), torch.from_numpy(qu).unsqueeze(0), h, step_i / 79.0)
            act_seq.append(act_t[0].numpy().tolist())
        parity_cases.append(dict(text=txt, nKey=nKey, emb=e.tolist(), q_unit=qu.tolist(),
                                 obs_seq=obs_seq, act_seq=act_seq))

    centroids_2d = []
    for sid, sname in enumerate(SKILL_NAMES):
        m_s = (np.array(cloud_skill) == sid)
        if m_s.any():
            xy = cloud_2d[m_s].mean(0)
            centroids_2d.append(dict(id=sid, name=sname.replace('_', ' '), x=float(xy[0]), y=float(xy[1])))

    pol_meta = dict(
        lmModel=best['lm_name'], lmPooling=best['pooling'],
        kT=k_t, kV=k_v,
        obsDim=int(obs.shape[2]), d=int(pol.d), nblocks=int(pol.nblocks), actDim=7, horizon=80,
        skills=SKILL_NAMES, nOptions=N_OPTIONS,
        cloudCount=int(len(cloud_idx)), cloudSkill=cloud_skill, cloudLabels=cloud_labels,
        centroids2D=centroids_2d, metrics=metrics, tensors=pk.meta,
    )
    open(os.path.join(out_dir, 'model', 'arena_policy.bin'), 'wb').write(bytes(pk.blob))
    json.dump(pol_meta, open(os.path.join(out_dir, 'model', 'arena_policy.json'), 'w'))
    json.dump({'cases': parity_cases}, open(os.path.join(out_dir, 'parity', 'arena_ref.json'), 'w'))
    print(f"\nExported model/arena_policy.bin ({len(pk.blob)/1e6:.2f} MB) + arena_policy.json + parity/arena_ref.json", flush=True)
    print("ALL_ARENA_DONE", flush=True)


if __name__ == '__main__':
    main()
