# The SAE code of every validation image (center crop) at one layer, kept sparse: for each position, the
# k latent indices and their activations. Example images at any activation level are picked from these.
#   python codes.py run2 layer4.2   ->  runs/run2/codes_layer4.2.npz: idx uint16 [N, P, k], val float16 [N, P, k],
#                                       top5 int16 [N, 5] (model's classes), labels int16 [N]
import sys

import numpy as np
import torch

from common import ROOT, batches, center, load_model, load_saes, tokens

RUN, LAYER = sys.argv[1], sys.argv[2]
OUT = f'{ROOT}/runs/{RUN}'
torch.set_grad_enabled(False)
model = load_model()
sae = load_saes(f'{OUT}/sae.pt')[LAYER]
imgs = np.load(f'{ROOT}/imgs256.npy', mmap_mode='r')
labels = np.load(f'{ROOT}/labels.npy')
idx, val, top5 = [], [], []
for b, (u8, _, _) in enumerate(batches(imgs, labels, np.arange(len(imgs)), 100)):
    with torch.autocast('cuda', torch.bfloat16):
        logits, a = model(center(u8), taps={LAYER})
    v, i, _ = sae.encode(tokens(a[LAYER]) / sae.scale)
    idx.append(i.to(torch.int32).cpu().numpy().astype(np.uint16))
    val.append(v.clamp(min=0).cpu().numpy().astype(np.float16))
    top5.append(logits.float().topk(5, dim=1).indices.cpu().numpy().astype(np.int16))
    if b % 100 == 0:
        print(b, flush=True)
np.savez(f'{OUT}/codes_{LAYER}.npz', idx=np.concatenate(idx), val=np.concatenate(val), top5=np.concatenate(top5),
         labels=labels.astype(np.int16))
print('codes', np.concatenate(idx).shape)
