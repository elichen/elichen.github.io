"""Train the listener: a network that hears a drum's notes and draws its outline.

Input: the first K eigenvalue ratios log(λ_k / λ_1), k = 2..K (K random from 3
to 40 per example, the rest masked), with a little noise. Output: M candidate
outlines (Fourier coefficients -> 64 boundary points) and a confidence for
each. The notes can't tell rotation, mirror image, position or where the
outline starts, so the loss aligns each candidate to the true outline over all
of those first (as Kempe's Machine does) and scores what's left. Winner takes
all across candidates, so the network can say "it's one of these".

usage: python train_hear.py SPECTRA.npz OUT_DIR [--steps 20000]
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KMAX = 40
N = 64
NF = 32          # Fourier coefficients per outline: k = -16 .. 15
M = 4            # candidate outlines


def resample_np(p, n=N):
    closed = np.concatenate([p, p[:, :1]], 1)
    seg = np.linalg.norm(np.diff(closed, axis=1), axis=-1)
    cum = np.concatenate([np.zeros((len(p), 1)), np.cumsum(seg, 1)], 1)
    out = np.zeros((len(p), n), complex)
    for i in range(len(p)):
        s = np.linspace(0, cum[i, -1], n, endpoint=False)
        out[i] = np.interp(s, cum[i], closed[i, :, 0]) + 1j * np.interp(s, cum[i], closed[i, :, 1])
    return out


def unit(z):
    z = z - z.mean(-1, keepdim=True)
    return z / (z.abs().pow(2).mean(-1, keepdim=True).sqrt() + 1e-9)


def variants(z):
    rev = torch.roll(torch.flip(z, [-1]), 1, -1)
    return [z, rev, z.conj(), rev.conj()]


def align_err(pred, target):
    """1 - best |correlation| over start shift, direction and mirror; both unit
    RMS. pred (..., n), target (..., n) broadcastable. Returns (...,)."""
    n = target.shape[-1]
    Ft = torch.fft.fft(target)
    best = None
    for v in variants(pred):
        m = torch.fft.ifft(Ft * torch.fft.fft(v).conj()).abs().amax(-1) / n
        best = m if best is None else torch.maximum(best, m)
    return 1 - best


def features(lam, K, noise=0.0, gen=None):
    """lam (B, KMAX) sorted eigenvalues; K (B,) notes heard. -> (B, 2*(KMAX-1))."""
    if noise > 0:
        lam = lam * (1 + noise * torch.randn(lam.shape, generator=gen, device=lam.device))
        lam = torch.sort(lam, dim=1).values
    r = torch.log(lam[:, 1:] / lam[:, :1])
    mask = (torch.arange(1, KMAX, device=lam.device)[None] < K[:, None]).float()
    return torch.cat([r * mask, mask], 1)


class Listener(nn.Module):
    def __init__(self, width=768, depth=4):
        super().__init__()
        self.inp = nn.Linear(2 * (KMAX - 1), width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(approximate='tanh'),
                                                   nn.Linear(width, width)) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, M * 2 * NF + M)
        k = torch.cat([torch.arange(0, NF // 2), torch.arange(-NF // 2, 0)])
        t = torch.arange(N) * 2 * np.pi / N
        self.register_buffer('basis', torch.exp(1j * k[:, None] * t[None]).to(torch.complex64))   # (NF, N)
        self.register_buffer('circle', (k == 1).to(torch.complex64))        # start from a circle

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        o = self.out(self.norm(h))
        coef = o[:, :M * 2 * NF].reshape(-1, M, NF, 2)
        c = torch.complex(coef[..., 0], coef[..., 1])
        c = c + self.circle
        z = c @ self.basis                                   # (B, M, N)
        return unit(z), o[:, M * 2 * NF:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('out')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--width', type=int, default=768)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--noise', type=float, default=0.002)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = np.load(args.data)
    lam = d['lam'].astype(np.float64)
    target = resample_np(d['outline'].astype(np.float64))
    test = (d['seed'] % 10) == 0
    L = torch.tensor(lam, dtype=torch.float32, device=dev)
    T = unit(torch.tensor(target, dtype=torch.complex64, device=dev))
    tr, te = np.flatnonzero(~test), np.flatnonzero(test)
    print(f'{len(tr)} train, {len(te)} test drums', flush=True)
    torch.manual_seed(0)
    gen = torch.Generator(device=dev).manual_seed(0)
    model = Listener(args.width, args.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    tr_t = torch.tensor(tr, device=dev)
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                res = {}
                for K in (5, 10, 20, 40):
                    idx = torch.tensor(te[:4000], device=dev)
                    Kt = torch.full((len(idx),), K, device=dev)
                    z, conf = model(features(L[idx], Kt))
                    e = align_err(z, T[idx][:, None])
                    best = e.min(1).values.mean().item()
                    top = e.gather(1, conf.argmax(1, keepdim=True)).mean().item()
                    res[K] = (round(best, 4), round(top, 4))
            print(f'step {step} {time.time() - t0:.0f}s  err (best of {M}, most confident) by notes heard: {res}', flush=True)
            torch.save({'model': model.state_dict(), 'args': vars(args)}, os.path.join(args.out, 'listener.pt'))
            model.train()
        if step == args.steps:
            break
        b = tr_t[torch.randint(0, len(tr), (args.batch,), device=dev, generator=gen)]
        K = torch.randint(3, KMAX + 1, (args.batch,), device=dev, generator=gen)
        z, conf = model(features(L[b], K, args.noise, gen))
        e = align_err(z, T[b][:, None])                          # (B, M)
        win = e.argmin(1)
        loss = e.min(1).values.mean() + 0.05 * e.mean() + 0.1 * F.cross_entropy(conf, win)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()


if __name__ == '__main__':
    main()
