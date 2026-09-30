"""Distil the exact alignment error into the curve encoder.

For each query drawing, candidates are its current nearest pool curves in
embedding space (re-mined every --refresh steps) plus random pool curves. The
teacher ranks them by the exact alignment error; the student's softmax over
dot products is trained to match it (listwise KL). Queries mix real Quick,
Draw! doodles, jittered mechanism curves and distorted test shapes.

usage: python train_encoder.py --pool POOLDIR --doodles qd.npz --out RUNDIR
"""

import argparse
import glob
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from encoder import Encoder, exact_err, normalise, fourier_descriptor, variants, N
from targets import targets


def load_pool(files, dev):
    zs = []
    for f in files:
        z = np.load(f)['z'].astype(np.float32)
        zs.append(torch.from_numpy(z[..., 0] + 1j * z[..., 1]))
    return torch.cat(zs).to(dev)


def resample_t(z, n=N):
    """Arc-length resample closed complex curves (B, m) -> (B, n)."""
    B, m = z.shape
    closed = torch.cat([z, z[:, :1]], 1)
    seg = (closed[:, 1:] - closed[:, :-1]).abs()
    cum = torch.cat([torch.zeros(B, 1, device=z.device), seg.cumsum(1)], 1)
    total = cum[:, -1:].clamp_min(1e-9)
    s = torch.arange(n, device=z.device)[None] / n * total
    idx = (torch.searchsorted(cum.contiguous(), s.contiguous(), right=True) - 1).clamp(0, m - 1)
    c0, c1 = cum.gather(1, idx), cum.gather(1, idx + 1)
    w = (s - c0) / (c1 - c0).clamp_min(1e-9)
    return closed.gather(1, idx) * (1 - w) + closed.gather(1, idx + 1) * w


def augment(z, gen, jitter=0.04, affine=0.0):
    """Hand-drawn distortions of normalised curves (B, n)."""
    B, n = z.shape
    dev = z.device
    R = lambda *s: torch.rand(*s, generator=gen, device=dev)
    if affine > 0:   # non-similarity stretch and shear make new shapes
        a = (1 + affine * (2 * R(B) - 1))[:, None]
        sh = (0.5 * affine * (2 * R(B) - 1))[:, None]
        x, y = z.real * a, z.imag / a + sh * z.real
        z = torch.complex(x, y)
    k = torch.arange(1, 7, device=dev)
    spec = torch.zeros(B, n, dtype=torch.complex64, device=dev)
    amp = (1 / (1 + (k / 2.0) ** 2))[None]
    rnd = lambda: torch.complex(torch.randn(B, 6, generator=gen, device=dev), torch.randn(B, 6, generator=gen, device=dev))
    spec[:, 1:7] = rnd() * amp
    spec[:, -6:] = torch.flip(rnd() * amp, [-1])
    noise = torch.fft.ifft(spec) * n
    noise = noise / (noise.abs().pow(2).mean(-1, keepdim=True).sqrt() + 1e-9)
    z = z + noise * (jitter * R(B))[:, None]
    z = resample_t(z)
    v = (R(B) * 4).long().clamp_max(3)
    vs = variants(z)
    z = torch.where(v[:, None] == 0, vs[0], torch.where(v[:, None] == 1, vs[1], torch.where(v[:, None] == 2, vs[2], vs[3])))
    z = torch.roll(z, int(R(1).item() * n), -1)
    return normalise(z)


@torch.no_grad()
def embed_all(model, Z, chunk=4096):
    out = []
    for i in range(0, len(Z), chunk):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=Z.is_cuda):
            out.append(model(Z[i:i + chunk]).half())
    return torch.cat(out)


@torch.no_grad()
def topk_sim(Q, E, k, chunk=1 << 20):
    """Top-k rows of E (N, d) by dot product with Q (B, d)."""
    vals, idxs = [], []
    for i in range(0, len(E), chunk):
        s = Q.half() @ E[i:i + chunk].T
        v, ix = s.float().topk(min(k, s.shape[1]), dim=1)
        vals.append(v)
        idxs.append(ix + i)
    v = torch.cat(vals, 1)
    ix = torch.cat(idxs, 1)
    top = v.topk(k, dim=1).indices
    return ix.gather(1, top)


@torch.no_grad()
def exact_best_all(q, P, qc=8, pc=65536):
    """Exact best error of each query over the whole pool P, and its index."""
    n = q.shape[-1]
    best = torch.full((len(q),), 2.0, device=q.device)
    arg = torch.zeros(len(q), dtype=torch.long, device=q.device)
    FPv = None
    for j in range(0, len(P), pc):
        Pj = P[j:j + pc]
        FPv = [torch.fft.fft(v).conj() for v in variants(Pj)]
        for i in range(0, len(q), qc):
            Fq = torch.fft.fft(q[i:i + qc])[:, None, :]
            m = None
            for F_ in FPv:
                mm = torch.fft.ifft(Fq * F_[None]).abs().amax(-1) / n
                m = mm if m is None else torch.maximum(m, mm)
            err = (1 - m.clamp(max=1) ** 2).clamp_min(0).sqrt()
            e, a = err.min(1)
            upd = e < best[i:i + qc]
            best[i:i + qc] = torch.where(upd, e, best[i:i + qc])
            arg[i:i + qc] = torch.where(upd, a + j, arg[i:i + qc])
    return best, arg


@torch.no_grad()
def eval_retrieval(model, Qe, Pe, best, Ks=(1, 10, 100, 1000), E=None):
    """Mean exact error of the best candidate within the embedding top-K."""
    if E is None:
        E = embed_all(model, Pe)
    Eq = embed_all(model, Qe).float() if model is not None else None
    return _eval_with(Eq, E, Qe, Pe, best, Ks)


@torch.no_grad()
def _eval_with(Eq, E, Qe, Pe, best, Ks):
    kmax = max(Ks)
    out = {'exact': best.mean().item()}
    for i in range(0, len(Qe), 64):
        idx = topk_sim(Eq[i:i + 64], E, kmax)
        err = exact_err(Qe[i:i + 64], Pe[idx])
        for K in Ks:
            out.setdefault(f'top{K}', []).append(err[:, :K].min(1).values)
    for K in Ks:
        out[f'top{K}'] = torch.cat(out[f'top{K}']).mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--doodles', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-shards', type=int, default=3)
    ap.add_argument('--pool-limit', type=int, default=0)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--bq', type=int, default=256)
    ap.add_argument('--hard', type=int, default=24)
    ap.add_argument('--rand', type=int, default=24)
    ap.add_argument('--refresh', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--tau-t', type=float, default=0.02)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--n-eval', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    gen = torch.Generator(device=dev).manual_seed(0)

    files = sorted(glob.glob(os.path.join(args.pool, 'shard_*.npz')))
    P = load_pool(files[:args.train_shards], dev)
    Pe = load_pool(files[-1:], dev)             # held-out eval pool
    if args.pool_limit:
        P, Pe = P[:args.pool_limit], Pe[:args.pool_limit]
    print(f'train pool {len(P)} curves, eval pool {len(Pe)}', flush=True)

    qd = np.load(args.doodles)
    Zd = torch.from_numpy(qd['z']).to(dev)
    perm = torch.randperm(len(Zd), generator=torch.Generator().manual_seed(1)).to(dev)
    n_hold = 4000
    Zd_eval, Zd_train = Zd[perm[:n_hold]], Zd[perm[n_hold:]]
    T = targets()
    from linkage import resample as np_resample, to_complex, normalise as np_norm
    Zs = np.concatenate([np_norm(to_complex(np_resample(T[k][None])))[0] for k in T])
    Zs = torch.from_numpy(Zs.astype(np.complex64)).to(dev)

    # eval queries: held-out doodles + jittered test shapes (fixed)
    eg = torch.Generator(device=dev).manual_seed(123)
    Qe = torch.cat([Zd_eval[:args.n_eval - 10 * len(Zs)], augment(torch.cat([Zs] * 10), eg, jitter=0.02)])
    t0 = time.time()
    best, _ = exact_best_all(Qe, Pe)
    print(f'exact brute force over eval pool: mean best err {best.mean():.4f} ({time.time() - t0:.0f}s)', flush=True)
    Ef = fourier_descriptor(Pe).half()
    base = _eval_with(fourier_descriptor(Qe), Ef, Qe, Pe, best, (1, 10, 100, 1000))
    print('fourier baseline', json.dumps({k: round(v, 4) for k, v in base.items()}), flush=True)
    del Ef

    model = Encoder(args.width, args.dim).to(dev)
    fast = torch.compile(model) if dev == 'cuda' else model
    log_tau = torch.nn.Parameter(torch.tensor(math.log(0.05), device=dev))
    opt = torch.optim.AdamW(list(model.parameters()) + [log_tau], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    print('params', sum(p.numel() for p in model.parameters()), flush=True)
    E = None
    hist = []
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.refresh == 0 or (step < args.refresh and step in (100, 300, 600)):
            model.eval()
            E = embed_all(fast, P)
            model.train()
        if step % args.eval_every == 0 and step > 0:
            model.eval()
            res = eval_retrieval(fast, Qe, Pe, best)
            model.train()
            res['step'] = step
            hist.append(res)
            print('eval', json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
            torch.save({'model': model.state_dict(), 'args': vars(args), 'hist': hist}, os.path.join(args.out, 'encoder.pt'))
        if step == args.steps:
            break
        nb = args.bq
        n_d, n_s = nb // 2, nb // 8
        n_p = nb - n_d - n_s
        ids_d = torch.randint(0, len(Zd_train), (n_d,), device=dev, generator=gen)
        ids_s = torch.randint(0, len(Zs), (n_s,), device=dev, generator=gen)
        ids_p = torch.randint(0, len(P), (n_p,), device=dev, generator=gen)
        q = torch.cat([augment(Zd_train[ids_d], gen, 0.03, 0.15),
                       augment(Zs[ids_s], gen, 0.03, 0.35),
                       augment(P[ids_p], gen, 0.05, 0.1)])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            eq = fast(q).float()
        hard = topk_sim(eq.detach(), E, args.hard)
        rand = torch.randint(0, len(P), (nb, args.rand), device=dev, generator=gen)
        idx = torch.cat([hard, rand], 1)
        C = P[idx]                                             # (B, K, n)
        with torch.no_grad():
            err = exact_err(q, C)
        # candidates are embedded without gradients (like MoCo's keys): the
        # shared weights learn through the queries, at a fraction of the memory
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            ec = fast(C.reshape(-1, N)).float().reshape(nb, idx.shape[1], -1)
        logits = (eq[:, None, :] * ec).sum(-1) / log_tau.exp()
        pt = F.softmax(-err / args.tau_t, dim=1)
        loss = (pt * (torch.log(pt + 1e-12) - F.log_softmax(logits, 1))).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            # how often the student's top pick is the teacher's top pick
            agree = (logits.argmax(1) == err.argmin(1)).float().mean().item()
            print(f'step {step} loss {loss.item():.4f} agree {agree:.3f} tau {log_tau.exp().item():.4f} '
                  f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
