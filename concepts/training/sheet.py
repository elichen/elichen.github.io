# Contact sheets: each rendered latent/neuron with its 8 top validation patches. Every render variant
# that exists (see render.py) gets a column; with more than two, sheets have one concept per row.
#   python sheet.py run1
import json, os, sys

import numpy as np
from PIL import Image, ImageDraw

from common import GRID, ROOT

RUN = sys.argv[1] if len(sys.argv) > 1 else 'run1'
OUT = f'{ROOT}/runs/{RUN}'
imgs = np.load(f'{ROOT}/imgs256.npy', mmap_mode='r')
mine = np.load(f'{OUT}/mine.npz')
targets = json.load(open(f'{OUT}/renders/targets.json'))
CROP = {'layer3.5': 128, 'layer4.1': 192, 'layer4.2': 192}
VARIANTS = [('', 'center'), ('_mean', 'whole image'), ('_dircos', 'direction x cos^3'), ('_maco', 'MACO'),
            ('_macoalpha', 'MACO + mask'), ('_diverse', 'diversity (4)')]
l0, k0, j0 = targets[0]
VARIANTS = [v for v in VARIANTS if os.path.exists(f'{OUT}/renders/{l0}_{k0}_{j0}{v[0]}.png')]
WIDE = len(VARIANTS) > 2
R, P = (176, 88) if WIDE else (192, 96)  # render and patch tile sizes
PER_ROW = 1 if WIDE else 2
RW = len(VARIANTS) * (R + 4)
TILE_W, TILE_H = RW + 4 * P + 4, R + 18
HEADER = 16 if WIDE else 0
os.makedirs(f'{OUT}/sheets', exist_ok=True)


def patch(layer, img, pos):
    g = GRID[layer]
    stride = 224 // g
    cy, cx = 16 + stride * (pos // g) + stride // 2, 16 + stride * (pos % g) + stride // 2
    h = CROP[layer] // 2
    cy, cx = min(max(cy, h), 256 - h), min(max(cx, h), 256 - h)
    return Image.fromarray(np.asarray(imgs[img])[cy - h:cy + h, cx - h:cx + h]).resize((P, P), Image.BILINEAR)


def tile(layer, kind, j):
    t = Image.new('RGB', (TILE_W, TILE_H), 'white')
    for c, (suffix, _) in enumerate(VARIANTS):
        t.paste(Image.open(f'{OUT}/renders/{layer}_{kind}_{j}{suffix}.png').resize((R, R), Image.BILINEAR),
                (c * (R + 4), 18))
    vals, ims, poss = (mine[f'{layer}/{kind}_{k}'][:, j] for k in ['val', 'img', 'pos'])
    for q in range(8):
        t.paste(patch(layer, ims[q], poss[q]), (RW + 4 + P * (q % 4), 18 + P * (q // 4)))
    freq = f"  freq {mine[f'{layer}/freq'][j]:.2e}" if kind == 'sae' else ''
    label = '' if WIDE else '   renders: ' + ' / '.join(name for _, name in VARIANTS)
    ImageDraw.Draw(t).text((2, 3), f'{layer} {kind} {j}{freq}  top {vals[0]:.1f} / 8th {vals[7]:.1f}{label}', fill='black')
    return t


groups = {}
for l, kind, j in targets:
    groups.setdefault((l, kind), []).append(j)
for (l, kind), js in groups.items():
    for s in range(0, len(js), 10):
        part = js[s:s + 10]
        rows = (len(part) + PER_ROW - 1) // PER_ROW
        sheet = Image.new('RGB', (PER_ROW * (TILE_W + 16), HEADER + rows * (TILE_H + 12)), 'white')
        d = ImageDraw.Draw(sheet)
        for c, (_, name) in enumerate(VARIANTS if WIDE else []):
            d.text((c * (R + 4) + 2, 2), name, fill='black')
        if WIDE:
            d.text((RW + 6, 2), 'top validation patches', fill='black')
        for q, j in enumerate(part):
            sheet.paste(tile(l, kind, j), ((q % PER_ROW) * (TILE_W + 16), HEADER + (q // PER_ROW) * (TILE_H + 12)))
        sheet.save(f'{OUT}/sheets/{l}_{kind}_{s // 10}.png')
print(sorted(os.listdir(f'{OUT}/sheets')))
