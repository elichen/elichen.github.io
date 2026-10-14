# For every SAE latent and every neuron (channel), find the top-K validation images (center crop)
# by max activation over positions, and count how often each latent fires.
#   python mine.py run1
import sys

import numpy as np
import torch

from common import LAYERS, ROOT, batches, center, load_model, load_saes, tokens

RUN = sys.argv[1] if len(sys.argv) > 1 else 'run1'
K = 16
OUT = f'{ROOT}/runs/{RUN}'
imgs = np.load(f'{ROOT}/imgs256.npy', mmap_mode='r')
labels = np.load(f'{ROOT}/labels.npy')
model = load_model()
saes = load_saes(f'{OUT}/sae.pt')


def empty(m):
    return [torch.full((K, m), -1.0, device='cuda'), torch.zeros((K, m), dtype=torch.long, device='cuda'),
            torch.zeros((K, m), dtype=torch.long, device='cuda')]


def update(top, a, ids):
    """a: [B, P, m] activations; keep the K largest per-image maxima per column."""
    mx, arg = a.max(1)
    v = torch.cat([top[0], mx])
    im = torch.cat([top[1], ids[:, None].expand_as(mx)])
    pos = torch.cat([top[2], arg])
    v, j = v.topk(K, dim=0)
    top[:] = [v, im.gather(0, j), pos.gather(0, j)]


top = {n: empty(s.W_dec.shape[0]) for n, s in saes.items()}
ntop = {n: empty(LAYERS[n]) for n in saes}
freq = {n: torch.zeros(s.W_dec.shape[0], device='cuda') for n, s in saes.items()}
total = {n: 0 for n in saes}
with torch.no_grad():
    for b, (u8, _, j) in enumerate(batches(imgs, labels, np.arange(len(imgs)), 50)):
        ids = torch.from_numpy(j).cuda()
        with torch.autocast('cuda', torch.bfloat16):
            _, a = model(center(u8), taps=set(saes), logits=False)
        for n, t in a.items():
            x = tokens(t)
            sae = saes[n]
            v, i, _ = sae.encode(x / sae.scale)
            z = sae.dense(v, i)
            freq[n] += (z > 0).sum((0, 1))
            total[n] += z.shape[0] * z.shape[1]
            update(top[n], z, ids)
            update(ntop[n], x, ids)
        if b % 100 == 0:
            print(b, flush=True)

out = {}
for n in saes:
    out[f'{n}/freq'] = (freq[n] / total[n]).cpu().numpy()
    for name, t in [('sae', top[n]), ('neuron', ntop[n])]:
        for key, arr in zip(['val', 'img', 'pos'], t):
            out[f'{n}/{name}_{key}'] = arr.cpu().numpy()
np.savez(f'{OUT}/mine.npz', **out)
for n in saes:
    f = out[f'{n}/freq']
    print(n, 'never fired', int((f == 0).sum()), 'median freq', float(np.median(f[f > 0])), 'max', float(f.max()))
