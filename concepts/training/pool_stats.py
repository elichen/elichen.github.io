# How many validation images the app must ship for every concept to have good examples: for pools of
# n images per class, how many concepts fire on at least 6 / 12 pool images, and how close their best pool
# example comes to their best example in all 50k.
#   python pool_stats.py run2 layer4.2 1e-4
import sys

import numpy as np

from common import ROOT

RUN, LAYER, TH = sys.argv[1], sys.argv[2], float(sys.argv[3])
c = np.load(f'{ROOT}/runs/{RUN}/codes_{LAYER}.npz')
idx, val, labels = c['idx'], c['val'].astype(np.float32), c['labels']
freq = np.load(f'{ROOT}/runs/{RUN}/mine.npz')[f'{LAYER}/freq']
keep = np.flatnonzero(freq > TH)
N, P, K = idx.shape
m = len(freq)
# Per image, the max activation of every latent over positions (sparse -> dense [N, m] float16).
best = np.zeros((N, m), np.float16)
for p in range(P):
    for k in range(K):
        np.maximum.at(best, (np.arange(N), idx[:, p, k]), val[:, p, k].astype(np.float16))
best = best[:, keep].astype(np.float32)
top_all = best.max(0)
rng = np.random.default_rng(0)
for per in [2, 4, 6, 10]:
    pool = np.concatenate([rng.choice(np.flatnonzero(labels == y), per, replace=False) for y in range(1000)])
    b = best[pool]
    fires = (b > 0).sum(0)
    ratio = b.max(0) / np.maximum(top_all, 1e-6)
    print(f'{per}/class = {len(pool)} images: concepts firing on >=6 {np.mean(fires >= 6):.3f}, >=12 {np.mean(fires >= 12):.3f};'
          f' best pool example / best overall: median {np.median(ratio):.2f}, 10th pct {np.percentile(ratio, 10):.2f}')
np.save(f'{ROOT}/runs/{RUN}/best_{LAYER}.npy', best.astype(np.float16))
