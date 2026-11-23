"""Pilot experiment on Colab Blackwell RTX PRO 6000:
Test the core scientific claim BEFORE building the full browser demo:

  1) Train a 100% LANGUAGE-FREE Trajectory JEPA (E_traj) purely on unlabeled UR5e
     kinematic trajectories (masked temporal sub-clip JEPA + SIGReg isotropy — ZERO text,
     ZERO skill labels!).
  2) Encode captions with a REAL frozen pretrained English sentence encoder
     (all-MiniLM-L6-v2 — ZERO robot training!).
  3) Connect them with ONE SINGLE ORTHOGONAL PROCRUSTES ROTATION MATRIX R (R R^T = I)!
  4) Measure held-out retrieval (Top-1, Top-3, FOSCTTM), continuous place error (cm),
     N-pair scaling (30 -> 10,000 + shuffled control), AND inspect all 40 strictly unseen
     free-form English prompts ("throw cube", "rotate around your base", "yeet the red block",
     "do a 360", "say hi", "put the cube in the front left corner", etc.)!
"""
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sim_skills import SKILL_NAMES, UNSEEN_PROMPTS

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True


def sigreg(z, n_proj=128, n_t=17):
    """Exact Epps-Pulley / SIGReg isotropic Gaussian regularizer (zero collapse, unit sphere)."""
    B, D = z.shape
    a = torch.randn(D, n_proj, device=z.device)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-6)
    proj = z @ a                                           # (B, n_proj)
    t = torch.linspace(-5.0, 5.0, n_t, device=z.device)    # (n_t,)
    ang = proj.unsqueeze(-1) * t                           # (B, n_proj, n_t)
    ecf_r = ang.cos().mean(0)
    ecf_i = ang.sin().mean(0)
    phi = torch.exp(-0.5 * t * t)                          # target N(0, 1) CF
    err = (ecf_r - phi).pow(2) + ecf_i.pow(2)
    return (err * phi).mean() * B


class TrajJEPA(nn.Module):
    """Language-free self-supervised Trajectory JEPA.
    Encodes a variable/masked sequence of robot trajectory frames (T x F) -> z_beh in R^D.
    """
    def __init__(self, fdim=29, d=256, layers=4, heads=4, dim=64, max_t=80):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Linear(fdim, d)
        self.pos = nn.Parameter(torch.randn(1, max_t, d) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)
        # Attentive pooling + MLP projector -> R^dim
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.BatchNorm1d(d),
            nn.GELU(),
            nn.Linear(d, dim),
        )
        # Lightweight JEPA predictor: predicts target view embedding from context view + view metadata
        self.pred = nn.Sequential(
            nn.Linear(dim + 4, d),
            nn.GELU(),
            nn.Linear(d, dim),
        )

    def encode(self, x, mask=None):
        """x: (B, T, F), mask: optional bool (B, T) where True = keep."""
        B, T, _ = x.shape
        h = self.in_proj(x) + self.pos[:, :T]
        if mask is not None:
            pad_mask = ~mask
            h = self.tf(h, src_key_padding_mask=pad_mask)
            h = self.norm(h)
            m = mask.float().unsqueeze(-1)
            h_mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
            w = (h * self.pool_q).sum(-1) / math.sqrt(h.shape[-1])
            w = w.masked_fill(pad_mask, -1e9).softmax(-1).unsqueeze(-1)
            h_attn = (h * w).sum(1)
        else:
            h = self.norm(self.tf(h))
            h_mean = h.mean(1)
            w = ((h * self.pool_q).sum(-1) / math.sqrt(h.shape[-1])).softmax(-1).unsqueeze(-1)
            h_attn = (h * w).sum(1)
        return self.proj(torch.cat([h_mean, h_attn], dim=-1))


def sample_subclip_mask(B, T, device, min_frac=0.45, max_frac=0.80):
    """Sample random contiguous + strided temporal views of the same trajectory (pure JEPA views!)."""
    idx = torch.arange(T, device=device)[None, :].expand(B, T)
    lens = torch.randint(int(T * min_frac), int(T * max_frac) + 1, (B,), device=device)
    starts = (torch.rand(B, device=device) * (T - lens).float()).long()
    ends = starts + lens
    mask = (idx >= starts[:, None]) & (idx < ends[:, None])
    # Also drop 20% random frames inside window so encoder captures smooth global dynamics
    keep = torch.rand(B, T, device=device) > 0.20
    mask = mask & keep
    # Guarantee at least 4 tokens kept
    empty = mask.sum(1) < 4
    if empty.any():
        mask[empty, ::10] = True
    meta = torch.stack([starts.float() / T, ends.float() / T], dim=-1)
    return mask, meta


def train_traj_jepa(feat_tr, dim=64, steps=6000, bs=512, lr=1e-3):
    """Train E_traj with ZERO language and ZERO skill labels."""
    N, T, Fdim = feat_tr.shape
    mu = feat_tr.mean((0, 1), keepdims=True)
    sd = np.maximum(feat_tr.std((0, 1), keepdims=True), 1e-4)
    X = torch.from_numpy(((feat_tr - mu) / sd).astype(np.float32)).to(DEV)

    model = TrajJEPA(fdim=Fdim, d=256, layers=4, heads=4, dim=dim, max_t=T).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    t0 = time.time()
    for step in range(1, steps + 1):
        i = torch.randint(0, N, (bs,), device=DEV)
        xb = X[i]
        # Mild sensor noise matching browser Readout2 jitter on cube coords (cols 21..24)
        xb = xb + torch.randn_like(xb) * 0.03
        m1, meta1 = sample_subclip_mask(bs, T, DEV, 0.45, 0.85)
        m2, meta2 = sample_subclip_mask(bs, T, DEV, 0.45, 0.85)
        with torch.autocast('cuda', torch.bfloat16):
            z1 = model.encode(xb, m1)
            z2 = model.encode(xb, m2)
            # Symmetric JEPA view prediction + invariance + SIGReg isotropy
            p12 = model.pred(torch.cat([z1, meta1, meta2], dim=-1))
            p21 = model.pred(torch.cat([z2, meta2, meta1], dim=-1))
            loss_jepa = 0.5 * (F.mse_loss(p12, z2.detach()) + F.mse_loss(p21, z1.detach()))
            loss_inv = F.mse_loss(z1, z2)
            loss_reg = 0.5 * (sigreg(z1.float()) + sigreg(z2.float()))
            loss = loss_jepa + 0.5 * loss_inv + 0.05 * loss_reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        if step % 1500 == 0 or step == steps:
            print(f"  [E_traj] step {step:5d}/{steps} | jepa={loss_jepa.item():.4f} inv={loss_inv.item():.4f} reg={loss_reg.item():.3f} ({time.time()-t0:.1f}s)", flush=True)

    model.eval()
    return model, mu, sd


@torch.inference_mode()
def encode_all_trajs(model, feat, mu, sd, bs=512):
    X = ((feat - mu) / sd).astype(np.float32)
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEV)
        out.append(model.encode(xb).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def whiten_pca(X_tr, X_te, k):
    mu = X_tr.mean(0)
    _, S, Vt = np.linalg.svd(X_tr - mu, full_matrices=False)
    P = Vt[:k].T / np.maximum(S[:k] / math.sqrt(len(X_tr) - 1), 1e-6)
    return mu, P, (X_tr - mu) @ P, (X_te - mu) @ P


def procrustes(A, B):
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vt


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def eval_alignment(label, A_tr, G_tr, A_te, G_te, sk_tr, sk_te, metas_tr, metas_te,
                   A_unseen, N_list=(30, 100, 300, 1000, 5000, 10000)):
    print(f"\n================ {label} (D_t={A_tr.shape[1]} -> D_v={G_tr.shape[1]}) ================", flush=True)
    # Build language-free kNN bank on held-out visual/traj test clips (or train clips)
    G_bank_u = unit(G_tr[:4000])
    sk_bank = sk_tr[:4000]
    G_te_u = unit(G_te)

    # Also fit a strictly traj-only linear probe on place clips (NEVER sees text!)
    place_id = SKILL_NAMES.index('place')
    m_pl_tr = sk_tr == place_id
    m_pl_te = sk_te == place_id
    goals_tr = np.array([metas_tr[i]['goal'][:2] for i in np.where(m_pl_tr)[0]])
    goals_te = np.array([metas_te[i]['goal'][:2] for i in np.where(m_pl_te)[0]])
    X_pl = np.c_[G_tr[m_pl_tr], np.ones(m_pl_tr.sum())]
    W_pl = np.linalg.solve(X_pl.T @ X_pl + 1.0 * np.eye(X_pl.shape[1]), X_pl.T @ goals_tr)
    ceil_pl = float(np.median(np.linalg.norm(np.c_[G_te[m_pl_te], np.ones(m_pl_te.sum())] @ W_pl - goals_te, axis=1) * 100))
    print(f"  Traj-only place probe ceiling (zero text): {ceil_pl:.2f} cm", flush=True)

    rng = np.random.default_rng(7)
    R_best = None
    for n in list(N_list) + ['shuffled']:
        if n == 'shuffled':
            m_n = min(len(A_tr), 10000)
            R = procrustes(A_tr[:m_n][rng.permutation(m_n)], G_tr[:m_n])
            tag = 'SHUFFLED'
        else:
            m_n = min(len(A_tr), int(n))
            R = procrustes(A_tr[:m_n], G_tr[:m_n])
            tag = f'N={m_n:5d}'
            if n == max(N_list):
                R_best = R
        Q_te = A_te @ R
        Q_te_u = unit(Q_te)
        # FOSCTTM on held-out 1000 clips
        S_pair = Q_te_u[:1000] @ G_te_u[:1000].T
        fosc = float((S_pair > S_pair.diagonal()[:, None]).mean())

        # Behavior accuracy via pure cosine nearest-neighbor in traj space (k=15 weighted vote)
        S_nn = Q_te_u @ G_bank_u.T
        top15 = np.argpartition(-S_nn, 15, axis=1)[:, :15]
        preds1, preds3 = [], []
        for i in range(len(Q_te)):
            nbrs = top15[i]
            sims = np.maximum(S_nn[i, nbrs], 0.0)
            votes = np.zeros(len(SKILL_NAMES))
            for idx_n, sc in zip(nbrs, sims):
                votes[sk_bank[idx_n]] += sc
            order = np.argsort(-votes)
            preds1.append(order[0] == sk_te[i])
            preds3.append(sk_te[i] in order[:3])
        acc1 = float(np.mean(preds1))
        acc3 = float(np.mean(preds3))

        # Place continuous xy error via traj-only linear probe
        pl_pred = np.c_[Q_te[m_pl_te], np.ones(m_pl_te.sum())] @ W_pl
        pl_err = float(np.median(np.linalg.norm(pl_pred - goals_te, axis=1) * 100))

        # Unseen 40 human prompts accuracy!
        Q_u = unit(A_unseen @ R)
        S_u = Q_u @ G_bank_u.T
        u_top1 = u_top3 = 0
        for i, (_, true_sk) in enumerate(UNSEEN_PROMPTS):
            nbrs = np.argpartition(-S_u[i], 15)[:15]
            votes = np.zeros(len(SKILL_NAMES))
            for idx_n in nbrs:
                votes[sk_bank[idx_n]] += max(0.0, float(S_u[i, idx_n]))
            order = np.argsort(-votes)
            tid = SKILL_NAMES.index(true_sk)
            u_top1 += int(order[0] == tid)
            u_top3 += int(tid in order[:3])
        print(f"  {tag:8s} | val top1={acc1*100:5.1f}% top3={acc3*100:5.1f}% | FOSC={fosc:.3f} | place={pl_err:5.1f}cm | UNSEEN40 top1={u_top1:2d}/40 ({u_top1*2.5:4.1f}%) top3={u_top3:2d}/40", flush=True)

    # Show detailed predictions for ALL 40 strictly unseen prompts at max N!
    print("\n  --- Strictly Unseen Free-Form English Prompts -> Single Orthogonal R ---", flush=True)
    Q_u_raw = A_unseen @ R_best
    Q_u = unit(Q_u_raw)
    S_u = Q_u @ G_bank_u.T
    for i, (prompt, true_sk) in enumerate(UNSEEN_PROMPTS):
        nbrs = np.argpartition(-S_u[i], 15)[:15]
        votes = np.zeros(len(SKILL_NAMES))
        for idx_n in nbrs:
            votes[sk_bank[idx_n]] += max(0.0, float(S_u[i, idx_n]))
        order = np.argsort(-votes)
        p1, p2 = SKILL_NAMES[order[0]], SKILL_NAMES[order[1]]
        mark = 'OK' if p1 == true_sk else ('~3' if SKILL_NAMES.index(true_sk) in order[:3] else 'XX')
        extra = ''
        if p1 == 'place':
            xy = (np.r_[Q_u_raw[i], 1.0] @ W_pl) * 100
            extra = f" -> ({xy[0]:.0f}, {xy[1]:+.0f})cm"
        elif p1 in ('throw', 'push', 'point', 'rotate_base'):
            best_nn = nbrs[np.argmax(S_u[i, nbrs])]
            extra = f" (dir={metas_tr[best_nn].get('dir', '?')})"
        print(f"    [{mark}] {prompt:48s} -> {p1:12s} (2nd: {p2:11s}) cos={S_u[i, nbrs].max():.2f}{extra}", flush=True)


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data_arena'
    d = np.load(os.path.join(data_dir, 'trajs.npz'))
    meta = json.load(open(os.path.join(data_dir, 'meta.json')))
    feat = d['feat'].astype(np.float32)
    skill = d['skill']
    caps = meta['captions']
    metas = meta['metas']

    N_te = 2000
    feat_tr, feat_te = feat[:-N_te], feat[-N_te:]
    sk_tr, sk_te = skill[:-N_te], skill[-N_te:]
    caps_tr, caps_te = caps[:-N_te], caps[-N_te:]
    metas_tr, metas_te = metas[:-N_te], metas[-N_te:]
    print(f"Loaded {len(feat_tr)} train + {len(feat_te)} val trajectories ({feat.shape[1]}x{feat.shape[2]}), GPU={DEV}", flush=True)

    # 1. Train language-free Trajectory JEPA (ZERO text, ZERO skill labels!)
    model, mu_f, sd_f = train_traj_jepa(feat_tr, dim=64, steps=6000, bs=512, lr=1e-3)
    Z_v_tr = encode_all_trajs(model, feat_tr, mu_f, sd_f)
    Z_v_te = encode_all_trajs(model, feat_te, mu_f, sd_f)
    torch.save({'state': model.state_dict(), 'mu': mu_f, 'sd': sd_f, 'dim': 64},
               os.path.join(data_dir, 'e_traj_pilot.pt'))

    # Compare two real browser-friendly pretrained English sentence encoders!
    unseen_texts = [p for p, _ in UNSEEN_PROMPTS]
    for model_name in ('sentence-transformers/all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5'):
        t0 = time.time()
        st = SentenceTransformer(model_name, device=DEV)
        E_tr = st.encode(caps_tr, batch_size=512, show_progress_bar=False, normalize_embeddings=False).astype(np.float64)
        E_te = st.encode(caps_te, batch_size=512, show_progress_bar=False, normalize_embeddings=False).astype(np.float64)
        E_un = st.encode(unseen_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=False).astype(np.float64)
        print(f"\nEncoded {len(caps_tr)+len(caps_te)} captions with {model_name} in {time.time()-t0:.1f}s (dim={E_tr.shape[1]})", flush=True)

        # Test D=48 and D=64 orthogonal Procrustes rotations!
        for k_t, k_v in ((48, 64), (64, 64)):
            mu_t, P_t, A_tr, A_te = whiten_pca(E_tr, E_te, k_t)
            A_un = (E_un - mu_t) @ P_t
            _, _, G_tr, G_te = whiten_pca(Z_v_tr, Z_v_te, k_v)
            eval_alignment(f"{model_name.split('/')[-1]} (k_t={k_t}, k_v={k_v})",
                           A_tr, G_tr, A_te, G_te, sk_tr, sk_te, metas_tr, metas_te, A_un)

    # Also test a STRICT UNALIGNED CONTROL: randomly initialized untrained TrajJEPA!
    rand_m = TrajJEPA(fdim=feat.shape[2], d=256, layers=4, heads=4, dim=64, max_t=feat.shape[1]).to(DEV).eval()
    Z_r_tr = encode_all_trajs(rand_m, feat_tr, mu_f, sd_f)
    Z_r_te = encode_all_trajs(rand_m, feat_te, mu_f, sd_f)
    st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEV)
    E_tr = st.encode(caps_tr, batch_size=512, show_progress_bar=False).astype(np.float64)
    E_te = st.encode(caps_te, batch_size=512, show_progress_bar=False).astype(np.float64)
    E_un = st.encode(unseen_texts, batch_size=64, show_progress_bar=False).astype(np.float64)
    mu_t, P_t, A_tr, A_te = whiten_pca(E_tr, E_te, 48)
    A_un = (E_un - mu_t) @ P_t
    _, _, G_r_tr, G_r_te = whiten_pca(Z_r_tr, Z_r_te, 64)
    eval_alignment("CONTROL: Untrained Random E_traj + MiniLM", A_tr, G_r_tr, A_te, G_r_te, sk_tr, sk_te, metas_tr, metas_te, A_un, N_list=(10000,))
    print("\nPILOT_DONE", flush=True)


if __name__ == '__main__':
    main()
