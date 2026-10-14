# Pick a small pool of validation images that still gives every concept strong examples: greedily add the
# image that is a strong example (activation >= frac * the concept's best over all 50k) for the most concepts
# that still lack `need` of them. Needs best_{layer}.npy from pool_stats.py.
#   python pool_cover.py run2 layer4.2   ->  runs/run2/pool_{frac}_{need}.npy (image indices, in pick order)
import sys

import numpy as np
import torch

from common import ROOT

RUN, LAYER = sys.argv[1], sys.argv[2]
OUT = f'{ROOT}/runs/{RUN}'
best = torch.from_numpy(np.load(f'{OUT}/best_{LAYER}.npy')).cuda().float()  # [N, m]
top = best.max(0).values
for k in [1, 3, 6]:
    u = torch.unique(best.topk(k, dim=0).indices)
    print(f'union of top-{k} per concept: {len(u)} images', flush=True)
for frac, need in [(0.7, 6), (0.6, 6), (0.5, 6)]:
    strong = (best >= frac * top).half()
    have = torch.zeros(best.shape[1], device='cuda')
    taken = torch.zeros(best.shape[0], dtype=torch.bool, device='cuda')
    chosen = []
    while True:
        lacking = (have < need).half()
        score = (strong @ lacking).float().masked_fill(taken, -1)
        i = int(score.argmax())
        if score[i] <= 0:
            break
        chosen.append(i)
        taken[i] = True
        have += strong[i].float()
        if len(chosen) % 2000 == 0:
            print(' ', len(chosen), f'{(have >= need).float().mean():.3f}', flush=True)
    np.save(f'{OUT}/pool_{frac}_{need}.npy', np.array(chosen))
    print(f'>= {frac} of max, {need} each: {len(chosen)} images, concepts covered {(have >= need).float().mean():.3f}',
          flush=True)
