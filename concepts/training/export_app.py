# Export everything the concepts app needs, on the CPU (the GPU may be busy):
#   sae/encoder.bin, sae/decoder.bin  int8 weights of the latents with validation frequency > TH, one scale per
#                                     latent (encoder column / decoder row); sae/sae.json holds the scales, biases,
#                                     k, scale and the original latent numbers
#   concepts.json   per concept: frequency, best activation over all 50k validation images, the ImageNet classes
#                   of its top-20 images, and example pool images at full / half / a fifth of its best activation
#   pool.json, pool/{i}.jpg   the example pool (pool_cover.py), as 224 center crops, sorted by class
#   renders/{i}.jpg  direction x cos^3 renders (render.py all:...)
#   test/*.png + test.json   PyTorch references for the browser parity test, using the int8 weights
#   python export_app.py run2 layer4.2 1e-4 pool_0.5_6 /mnt/c/w/sae/app
import json, os, sys

import numpy as np
import torch
from PIL import Image

from common import ROOT, load_model, load_saes, tokens

RUN, LAYER, TH, POOL, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
RENDER_SIZE, N_TEST, N_QUANT = 128, 6, 300
SRC = f'{ROOT}/runs/{RUN}'
torch.set_num_threads(4)
torch.set_grad_enabled(False)
for d in ['sae', 'pool', 'renders', 'test']:
    os.makedirs(f'{OUT}/{d}', exist_ok=True)
sae = load_saes(f'{SRC}/sae.pt', 'cpu')[LAYER]
model = load_model('cpu').float().to(memory_format=torch.contiguous_format)
imgs = np.load(f'{ROOT}/imgs256.npy', mmap_mode='r')
labels = np.load(f'{ROOT}/labels.npy')
held = np.load(f'{SRC}/held.npy')
freq = np.load(f'{SRC}/mine.npz')[f'{LAYER}/freq']
keep = np.flatnonzero(freq > TH)
m, d = len(keep), sae.W_dec.shape[1]
W_enc, b_enc = sae.W_enc[:, keep], sae.b_enc[keep]
W_dec, b_dec = sae.W_dec[keep], sae.b_dec
W_fc = model.fc.weight  # [1000, d]


def q8(W, dim):
    """Symmetric int8 with one scale per slice along `dim` (0: per column, 1: per row)."""
    s = W.abs().amax(dim=dim, keepdim=True) / 127
    return (W / s).round().clamp(-127, 127).to(torch.int8), s.squeeze(dim)


qe, se = q8(W_enc, 0)
qd, sd = q8(W_dec, 1)
W_enc_q, W_dec_q = qe.float() * se, qd.float() * sd[:, None]


def crop(i):
    return np.array(imgs[i])[16:240, 16:240]


def run(i):
    x = torch.from_numpy(crop(i)).permute(2, 0, 1)[None].float() / 255
    logits, a = model(x, taps={LAYER})
    return logits[0], tokens(a[LAYER])[0] / sae.scale  # [positions, d]


def code(t, We, be, Wd):
    p = (t - b_dec) @ We + be
    v, i = p.topk(sae.k, dim=-1)
    z = torch.zeros(t.shape[0], m).scatter_(-1, i, v.clamp(min=0))
    return i, z, z @ Wd + b_dec


# What int8 costs, on held-out images: same top-k, reconstruction error, top contributions to the top class.
same, err, var, overlap = 0.0, 0.0, 0.0, 0.0
for i in held[:N_QUANT]:
    logits, t = run(i)
    c = int(logits.argmax())
    i32, z32, r32 = code(t, W_enc, b_enc, W_dec)
    i8, z8, r8 = code(t, W_enc_q, b_enc, W_dec_q)
    same += (i8[:, :, None] == i32[:, None, :]).any(-1).float().mean().item()
    err += ((t - r8) ** 2).sum().item() - ((t - r32) ** 2).sum().item()
    var += ((t - t.mean(0)) ** 2).sum().item()
    con = lambda z, Wd: (z.mean(0) * (Wd @ W_fc[c])).topk(10).indices.tolist()
    overlap += len(set(con(z32, W_dec)) & set(con(z8, W_dec_q))) / 10
print(f'int8 vs float32 on {N_QUANT} held-out images: same top-k {same / N_QUANT:.4f}, FVU +{err / var:.5f}, '
      f'top-10 contributions overlap {overlap / N_QUANT:.3f}', flush=True)

qe.numpy().tofile(f'{OUT}/sae/encoder.bin')  # [d, m] row-major
qd.numpy().tofile(f'{OUT}/sae/decoder.bin')  # [m, d] row-major
r = lambda a, n=6: [float(f'{v:.{n}g}') for v in a.tolist()]
json.dump({'description': f'TopK SAE on the robust ResNet-50 {LAYER} output ({RUN}), latents with validation '
                          f'frequency > {TH:g}; int8 weights with one scale per latent.',
           'layer': LAYER, 'latents': m, 'channels': d, 'k': sae.k, 'scale': sae.scale.item(),
           'ids': keep.tolist(), 'encoderScale': r(se), 'decoderScale': r(sd), 'b_enc': r(b_enc), 'b_dec': r(b_dec)},
          open(f'{OUT}/sae/sae.json', 'w'))

# Parity references with the int8 weights.
cases = []
for i in held[::len(held) // N_TEST][:N_TEST]:
    Image.fromarray(crop(i)).save(f'{OUT}/test/{i}.png')
    logits, t = run(i)
    _, z, _ = code(t, W_enc_q, b_enc, W_dec_q)
    mean_z = z.mean(0)
    top = mean_z.topk(10)
    c = int(logits.argmax())
    contrib = sae.scale * mean_z * (W_dec_q @ W_fc[c])
    tc = contrib.topk(10)
    cases.append({'image': f'{i}.png', 'index': int(i), 'top5': logits.topk(5).indices.tolist(),
                  'concepts': top.indices.tolist(), 'concept_means': top.values.tolist(), 'class': c,
                  'contrib_concepts': tc.indices.tolist(), 'contrib_values': tc.values.tolist(),
                  'contrib_total': contrib.sum().item()})
json.dump(cases, open(f'{OUT}/test/test.json', 'w'), indent=1)

# The example pool, sorted by class, and each concept's examples in it.
best = np.load(f'{SRC}/best_{LAYER}.npy').astype(np.float32)  # [50000, m] max over positions
assert best.shape[1] == m
pool = np.load(f'{SRC}/{POOL}.npy')
pool = pool[np.lexsort((pool, labels[pool]))]
for n, i in enumerate(pool):
    Image.fromarray(crop(i)).save(f'{OUT}/pool/{n}.jpg', quality=82)
json.dump({'images': pool.tolist(), 'labels': labels[pool].tolist()}, open(f'{OUT}/pool.json', 'w'))
top_all = best.max(0)
bp = best[pool]  # [pool, m]
concepts = []
for j in range(m):
    a = bp[:, j]
    order = np.argsort(-a)
    full = [int(n) for n in order[:8] if a[n] > 0]

    def near(frac, count=4):
        target = frac * top_all[j]
        cand = np.flatnonzero((a > 0.6 * target) & (a < 1.4 * target))
        return [int(n) for n in cand[np.argsort(np.abs(a[cand] - target))][:count] if n not in full]

    cls, counts = np.unique(labels[np.argsort(-best[:, j])[:20]], return_counts=True)
    o = np.argsort(-counts)[:3]
    concepts.append({'freq': float(f'{freq[keep[j]]:.3g}'), 'max': round(float(top_all[j]), 2),
                     'classes': [[int(cls[q]), int(counts[q])] for q in o],
                     'examples': [[n, round(float(a[n]), 2)] for n in full + near(0.5) + near(0.2)]})
json.dump(concepts, open(f'{OUT}/concepts.json', 'w'), separators=(',', ':'))

for j, orig in enumerate(keep):
    Image.open(f'{SRC}/renders_all/{LAYER}_sae_{orig}_dircos.png').resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS) \
        .save(f'{OUT}/renders/{j}.jpg', quality=85)
sizes = {dd: sum(os.path.getsize(f'{OUT}/{dd}/{f}') for f in os.listdir(f'{OUT}/{dd}')) / 1e6 for dd in ['sae', 'pool', 'renders']}
print('exported', m, 'concepts,', len(pool), 'pool images; MB', {k: round(v, 1) for k, v in sizes.items()},
      'concepts.json', round(os.path.getsize(f'{OUT}/concepts.json') / 1e6, 2))
