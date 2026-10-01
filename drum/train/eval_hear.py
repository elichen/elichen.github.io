"""How well does the listener hear shapes? Overlap (intersection over union)
between each test drum and the network's outline, after turning and
mirroring the guess to fit best (the notes can't tell orientation), by
number of notes heard. Also draws a grid of examples.

usage: python eval_hear.py SPECTRA.npz listener.pt OUT.png [--n 600]
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from train_hear import Listener, features, resample_np, unit, variants, KMAX, N

G = 160
XS = np.linspace(-2.2, 2.2, G)
GRID = np.stack(np.meshgrid(XS, XS), -1).reshape(-1, 2)


def to_unit_area(z):
    x, y = z.real, z.imag
    a = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    z = z - z.mean()
    return z / np.sqrt(max(a, 1e-9))


def aligned(pred, target):
    """Turn/mirror pred (n,) complex to best match target (n,), both unit RMS."""
    best = (-1, None)
    for v, pv in enumerate(variants(torch.from_numpy(pred)[None])):
        pv = pv[0].resolve_conj().numpy()
        corr = np.fft.ifft(np.fft.fft(target) * np.conj(np.fft.fft(pv)))
        s = int(np.abs(corr).argmax())
        if abs(corr[s]) > best[0]:
            a = corr[s] / abs(corr[s])
            best = (abs(corr[s]), a * np.roll(pv, s))
    return best[1]


def overlap(a, b):
    A = Path(np.stack([a.real, a.imag], 1)).contains_points(GRID)
    B = Path(np.stack([b.real, b.imag], 1)).contains_points(GRID)
    return (A & B).sum() / max((A | B).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=600)
    args = ap.parse_args()
    d = np.load(args.data)
    test = np.flatnonzero((d['seed'] % 10) == 0)
    rng = np.random.default_rng(0)
    idx = rng.choice(test, args.n, replace=False)
    ck = torch.load(args.ckpt, map_location='cpu')
    model = Listener(ck['args']['width'], ck['args'].get('depth', 4))
    model.load_state_dict(ck['model'])
    model.eval()
    lam = torch.tensor(d['lam'][idx], dtype=torch.float32)
    tgt = unit(torch.tensor(resample_np(d['outline'][idx].astype(float)), dtype=torch.complex64)).numpy()
    kinds = d['kind'][idx]
    Ks = (3, 5, 10, 20, 40)
    res = {}
    shown = {}
    for K in Ks:
        with torch.no_grad():
            z, conf = model(features(lam, torch.full((len(idx),), K)))
        z = z.numpy()
        top = conf.argmax(1).numpy()
        ious = []
        for i in range(len(idx)):
            g = aligned(z[i, top[i]], tgt[i])
            ious.append(overlap(to_unit_area(g), to_unit_area(tgt[i])))
            shown[(K, i)] = g
        ious = np.array(ious)
        res[K] = ious
        by = '  '.join(f'{k} {ious[kinds == k].mean():.3f}' for k in np.unique(kinds))
        print(f'K={K:2d} notes: mean overlap {ious.mean():.3f}, median {np.median(ious):.3f}   by family: {by}')
    # baseline: always guessing a circle
    circ = np.exp(2j * np.pi * np.arange(N) / N)
    base = np.mean([overlap(to_unit_area(circ), to_unit_area(tgt[i])) for i in range(len(idx))])
    print(f'baseline (always a circle): {base:.3f}')
    # figure: 12 drums x 5 note counts
    pick = rng.choice(len(idx), 12, replace=False)
    fig, axs = plt.subplots(12, len(Ks), figsize=(2.2 * len(Ks), 2.2 * 12))
    for r, i in enumerate(pick):
        t = to_unit_area(tgt[i])
        for c, K in enumerate(Ks):
            ax = axs[r, c]
            g = to_unit_area(shown[(K, i)])
            ax.fill(t.real, t.imag, color='#ccc')
            ax.plot(np.r_[g.real, g.real[:1]], np.r_[g.imag, g.imag[:1]], 'k-', lw=1.2)
            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title(f'{kinds[i]} K={K} {res[K][i]:.2f}', fontsize=7)
    plt.tight_layout()
    plt.savefig(args.out, dpi=70)


if __name__ == '__main__':
    main()
