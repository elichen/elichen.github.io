"""Held-out Quick, Draw! doodles (never used for training or selection) as
JSON, for evaluating the web app end to end (kempe/tools/eval_app.mjs).

usage: python export_heldout.py qd.npz OUT.json [--n 400]
"""

import argparse
import json
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('qd')
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--holdout', type=int, default=4000)
    args = ap.parse_args()
    d = np.load(args.qd)
    perm = torch.randperm(len(d['z']), generator=torch.Generator().manual_seed(1)).numpy()
    held = perm[:args.holdout][:args.n]
    out = []
    raw, cat_ids, cat_names, closed = d['raw'], d['cat'], d['cats'], d['closed']
    for i in held:
        r = raw[i]
        r = (r - r.mean(0)) / np.abs(r - r.mean(0)).max() * 3      # about the size people draw
        out.append(dict(cat=str(cat_names[cat_ids[i]]), closed=bool(closed[i]),
                        pts=np.round(r.ravel(), 4).tolist()))
    json.dump(out, open(args.out, 'w'))
    print(len(out), 'doodles')


if __name__ == '__main__':
    main()
