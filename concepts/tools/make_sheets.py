# Contact sheets for naming concepts: per row, a concept's render and its 6 strongest pool examples, each
# example dimmed where the concept is quiet (as in the app). Also hints.json: each concept's top ImageNet classes.
#   python3 make_sheets.py <repo>/concepts out/pool_codes.npz out/naming
import json, os, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP, CODES, OUT = sys.argv[1:4]
T, GAP, PER_SHEET, N_EX = 100, 4, 16, 6
concepts = json.load(open(f'{APP}/data/concepts.json'))
ids = json.load(open(f'{APP}/model/sae.json'))['ids']
labels = json.load(open(f'{APP}/data/labels.json'))
codes = np.load(CODES)
idx, val = codes['idx'], codes['val'].astype(np.float32)
font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 15)
os.makedirs(f'{OUT}/sheets', exist_ok=True)


def spotlight(img, n, orig, best):
    m = np.zeros(49, np.float32)
    hit = idx[n] == orig  # [49, k]
    m[np.nonzero(hit)[0]] = val[n][hit]
    norm = max(m.max(), 0.3 * best, 1e-6)
    a = Image.fromarray((255 * 0.72 * (1 - np.minimum(1, m / norm))).reshape(7, 7).astype(np.uint8)).resize((T, T), Image.BILINEAR)
    return Image.composite(Image.new('RGB', (T, T), 'black'), img, a)


pool_cache = {}
def pool(n):
    if n not in pool_cache:
        pool_cache[n] = Image.open(f'{APP}/pool/{n}.jpg').convert('RGB').resize((T, T), Image.LANCZOS)
    return pool_cache[n]


for s in range(0, len(concepts), PER_SHEET):
    js = range(s, min(s + PER_SHEET, len(concepts)))
    sheet = Image.new('RGB', ((N_EX + 1) * (T + GAP) + 8, len(js) * (T + GAP)), 'white')
    d = ImageDraw.Draw(sheet)
    for r, j in enumerate(js):
        y = r * (T + GAP)
        sheet.paste(Image.open(f'{APP}/renders/{j}.jpg').convert('RGB').resize((T, T), Image.LANCZOS), (0, y))
        for e, (n, _) in enumerate(concepts[j]['examples'][:N_EX]):
            sheet.paste(spotlight(pool(n), n, ids[j], concepts[j]['max']), ((e + 1) * (T + GAP) + 8, y))
        w = d.textlength(f'#{j}', font=font)
        d.rectangle([0, y, w + 6, y + 19], fill='black')
        d.text((3, y + 1), f'#{j}', fill='white', font=font)
    sheet.save(f'{OUT}/sheets/{s // PER_SHEET:03d}.jpg', quality=88)
json.dump({j: ', '.join(f'{labels[c]} {k}/20' for c, k in c['classes']) for j, c in enumerate(concepts)},
          open(f'{OUT}/hints.json', 'w'), indent=0)
print(len(os.listdir(f'{OUT}/sheets')), 'sheets')
