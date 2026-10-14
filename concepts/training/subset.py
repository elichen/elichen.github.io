# What is lost by shipping only the latents that fire often enough: the SAE restricted to latents with
# validation frequency > threshold (TopK over the kept ones only), on the run's held-out images.
#   python subset.py run2 layer4.2 0 1e-5 3e-5 1e-4
import sys

import numpy as np
import torch

from common import ROOT, batches, center, load_model, load_saes, tokens

RUN, LAYER = sys.argv[1], sys.argv[2]
THRESHOLDS = [float(t) for t in sys.argv[3:]] or [0.0]
OUT = f'{ROOT}/runs/{RUN}'
torch.set_grad_enabled(False)
model = load_model()
sae = load_saes(f'{OUT}/sae.pt')[LAYER]
freq = torch.from_numpy(np.load(f'{OUT}/mine.npz')[f'{LAYER}/freq']).cuda()
imgs = np.load(f'{ROOT}/imgs256.npy', mmap_mode='r')
labels = np.load(f'{ROOT}/labels.npy')
held = np.load(f'{OUT}/held.npy')

stats = {t: torch.zeros(5, device='cuda') for t in THRESHOLDS}  # err, var, correct, agree, kept-k overlap
orig = 0
for u8, y, _ in batches(imgs, labels, held, 250):
    y = y.cuda()
    with torch.autocast('cuda', torch.bfloat16):
        logits, a = model(center(u8), taps={LAYER})
    pred = logits.argmax(1)
    orig += (pred == y).sum().item()
    t = a[LAYER]
    B, C, H, W = t.shape
    x = tokens(t).reshape(-1, C) / sae.scale
    p = sae.pre(x)
    full = p.topk(sae.k, dim=-1).indices
    for th in THRESHOLDS:
        q = p.masked_fill(freq <= th, -1e9) if th > 0 else p
        v, i = q.topk(sae.k, dim=-1)
        v = v.clamp(min=0)
        r = sae.decode(v, i)
        h = (r * sae.scale).reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', torch.bfloat16):
            pn = model(start=LAYER, h=h)[0].argmax(1)
        same = (i[:, :, None] == full[:, None, :]).any(-1).float().mean()
        stats[th] += torch.stack([(x - r).pow(2).sum(), (x - x.mean(0)).pow(2).sum(), (pn == y).sum(),
                                  (pn == pred).sum(), same * B])
N = len(held)
print(f'{RUN} {LAYER}: model top-1 {orig / N:.4f} on {N} held-out images')
for th, s in stats.items():
    print(f'freq > {th:g}: {int((freq > th).sum())} latents  fvu {s[0] / s[1]:.4f}  spliced top-1 {s[2] / N:.4f}'
          f'  agree {s[3] / N:.4f}  same top-k {s[4] / N:.3f}')
