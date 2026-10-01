"""Compute the notes of many drums: outlines from shapes.py families and
closed Quick, Draw! doodles, each scaled to unit area, with the lowest
K_MODES eigenvalues from fem.py.

usage: python gen_spectra.py OUT.npz --doodles qd.npz [--scale 1.0] [--families 1.0] [--doodle-aug 1]

Doodle seeds are the doodle's index; a stretched copy gets 10,000,000 + index,
so it lands in the same train/test split (seed % 10) as its original.
"""

import argparse
import multiprocessing as mp
import numpy as np
from fem import drum_spectrum, normalise, K_MODES
import shapes as S

FAMILIES = {'blob': 60000, 'star': 40000, 'polygon': 30000, 'ellipse': 5000}


def work(job):
    kind, seed, payload = job
    rng = np.random.default_rng(seed)
    if kind == 'doodle':
        p = S.doodle(payload) if len(payload) != S.N_OUT else payload
    elif kind == 'doodle_aff':     # a stretched and sheared copy of a doodle
        p = S.doodle(payload) if len(payload) != S.N_OUT else payload
        a = np.exp(rng.normal(0, 0.25))
        sh = rng.normal(0, 0.2)
        p = p @ np.array([[a, sh], [0, 1 / a]])
    elif kind == 'blob':
        p = S.fourier_blob(rng, kmax=rng.integers(3, 10), decay=rng.uniform(1.2, 2.2), amp=rng.uniform(0.15, 0.5))
    elif kind == 'star':
        p = S.star_shaped(rng)
    elif kind == 'polygon':
        p = S.polygon(rng)
    else:
        p = S.ellipse(rng)
    if not S.usable(p):
        return None
    p = normalise(p)
    try:
        lam = drum_spectrum(p)
    except Exception:
        return None
    if not np.all(np.isfinite(lam)) or lam[0] <= 0:
        return None
    return kind, seed, p.astype(np.float32), lam.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--doodles', required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--families', type=float, default=1.0, help='scale of the synthetic families')
    ap.add_argument('--doodle-aug', type=int, default=0, help='stretched copies per doodle')
    ap.add_argument('--only-aug', action='store_true', help='skip the original doodles')
    ap.add_argument('--part', type=int, nargs=2, default=None, metavar=('START', 'STOP'),
                    help='only doodles START..STOP (positions among the closed ones)')
    ap.add_argument('--procs', type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()
    d = np.load(args.doodles)
    if 'pts' in d.files:            # already cleaned up by prep_doodles.py
        raw, idx = d['pts'].astype(np.float64), d['index']
        pos = {int(i): k for k, i in enumerate(idx)}
        raw = {int(i): raw[pos[int(i)]] for i in idx}
    else:
        raw, closed = d['raw'], d['closed']
        idx = np.flatnonzero(closed)
    if args.scale < 1:
        idx = idx[:: int(round(1 / args.scale))]
    if args.part:
        idx = idx[args.part[0]:args.part[1]]
    jobs = [] if args.only_aug else [('doodle', int(i), raw[i]) for i in idx]
    for k in range(args.doodle_aug):
        jobs += [('doodle_aff', 10_000_000 * (k + 1) + int(i), raw[i]) for i in idx]
    seed = 1
    for kind, n in FAMILIES.items():
        for _ in range(int(n * args.scale * args.families)):
            jobs.append((kind, seed, None))
            seed += 1
    # results are written in shards as they come (OUT.0.npz, OUT.1.npz, ...)
    # so a long run can be used before it finishes; jobs keep their order,
    # originals before stretched copies
    res, shard = [], 0

    def flush():
        nonlocal res, shard
        if not res:
            return
        kinds = np.array([r[0] for r in res])
        np.savez(args.out.replace('.npz', f'.{shard}.npz'), kind=kinds, seed=np.array([r[1] for r in res]),
                 outline=np.stack([r[2] for r in res]), lam=np.stack([r[3] for r in res]))
        print(f'shard {shard}: ' + ', '.join(f'{k} {int((kinds == k).sum())}' for k in np.unique(kinds)), flush=True)
        res, shard = [], shard + 1

    with mp.Pool(args.procs) as pool:
        for i, r in enumerate(pool.imap(work, jobs, chunksize=64)):
            if r is not None:
                res.append(r)
            if (i + 1) % 50000 == 0:
                print(f'{i + 1}/{len(jobs)} done', flush=True)
                flush()
    flush()


if __name__ == '__main__':
    main()
