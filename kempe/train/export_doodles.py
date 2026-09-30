"""Training doodles (everything except the held-out evaluation set), plus
stretched and sheared test shapes, as float32 [N][256][2] closed paths for the
offline tuner (kempe/tools/tune_doodles.mjs). Paths are centred and scaled to
a radius of about 3, the size people draw on the plate.

usage: python export_doodles.py qd.npz OUT.bin OUT.json [--per-cat 1200] [--shapes 60]
"""

import argparse
import json
import numpy as np
import torch
from targets import targets, jitter


def densify(p, n=256):
    closed = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(u, cum, closed[:, 0]), np.interp(u, cum, closed[:, 1])], 1)


def fit3(p):
    p = p - p.mean(0)
    return p / np.abs(p).max() * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('qd')
    ap.add_argument('out_bin')
    ap.add_argument('out_json')
    ap.add_argument('--per-cat', type=int, default=1200)
    ap.add_argument('--shapes', type=int, default=60)
    ap.add_argument('--holdout', type=int, default=4000)
    args = ap.parse_args()
    d = np.load(args.qd)
    raw, cat_ids, cat_names = d['raw'], d['cat'], d['cats']      # load once: npz reads lazily
    perm = torch.randperm(len(d['z']), generator=torch.Generator().manual_seed(1)).numpy()
    train = perm[args.holdout:]
    cats = cat_ids[train]
    paths, labels = [], []
    for c in np.unique(cats):
        for i in train[cats == c][:args.per_cat]:
            paths.append(fit3(raw[i]))
            labels.append(str(cat_names[c]))
    rng = np.random.default_rng(11)
    for name, shape in targets().items():
        for _ in range(args.shapes):
            a = 1 + 0.35 * (2 * rng.random() - 1)
            sh = 0.18 * (2 * rng.random() - 1)
            p = shape @ np.array([[a, sh], [0, 1 / a]])
            paths.append(fit3(densify(jitter(p, rng, 0.012))))
            labels.append('shape:' + name)
    arr = np.stack(paths).astype(np.float32)
    arr.tofile(args.out_bin)
    json.dump(dict(count=len(arr), labels=labels), open(args.out_json, 'w'))
    print(len(arr), 'paths')


if __name__ == '__main__':
    main()
