"""Invent writers: unprimed samples of the prompt, for the page's "borrow a hand" row.

Writes a contact sheet of candidates (hands_grid.png) and all candidates to
hands_all.json; pick indices with --pick to write the page's hands.json.
"""
import argparse, sys, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('ckpt')
ap.add_argument('--count', type=int, default=30)
ap.add_argument('--pick', default='')
ap.add_argument('--out', default='../model/hands.json')
hv = ap.parse_args()
sys.argv = ['sample.py', hv.ckpt]
exec(open('sample.py').read().split("if __name__ == '__main__':")[0])
PROMPT = 'The quick brown fox'

if hv.pick:
    allh = json.load(open('hands_all.json'))
    chosen = [allh[int(i)] for i in hv.pick.split(',')]
    out = []
    for h in chosen:
        pts, lift = np.array(h['pts']), np.array(h['lift'])
        # drop tiny marks after the last letter (a stray full stop doesn't belong to the prompt)
        ends = np.where(lift)[0]
        while len(ends) > 1:
            st = ends[-2] + 1
            if np.ptp(pts[st:], axis=0).max() < 0.35:
                pts, lift = pts[:st], lift[:st]; ends = ends[:-1]
            else:
                break
        out.append(dict(pts=np.round(pts, 3).tolist(), lift=lift.astype(int).tolist()))
    json.dump(out, open(hv.out, 'w'))
    print('wrote', hv.out, len(chosen))
    sys.exit()

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cands = []
for i in range(hv.count):
    bias = [0.6, 0.9, 1.2][i % 3]
    pts, pen = generate(None, '', PROMPT, bias, 1000 + i)
    # baseline: the unprimed run starts at the origin; re-estimate it like the page does for a visitor
    lift = np.zeros(len(pts), int); lift[:-1] = pen[1:] > 0.5; lift[-1] = 1
    cands.append(dict(pts=pts, lift=lift, bias=bias))

calib = json.load(open('calib.json'))[PROMPT]
q = np.array(calib['q']); w = np.array(calib['w']); v = np.array(calib['v'])
fig, axs = plt.subplots(10, 3, figsize=(15, 14))
out = []
for i, (c, ax) in enumerate(zip(cands, axs.T.flat)):
    p = c['pts']
    y = np.sort(p[:, 1]); qq = np.quantile(y, q); med = qq[list(q).index(0.5)]
    xh = float((qq - med) @ w); base = float(med + (qq - med) @ v)
    p = np.stack([(p[:, 0] - p[:, 0].min()) / xh, (p[:, 1] - base) / xh], 1)
    out.append(dict(pts=np.round(p, 3).tolist(), lift=c['lift'].tolist(), bias=c['bias']))
    draw(ax, p, np.concatenate([[1], c['lift'][:-1]]))
    ax.set_title(f'{i}  bias {c["bias"]}', fontsize=8, loc='left'); ax.set_aspect('equal'); ax.axis('off')
    ax.axhline(0, color='#c2c8db', lw=0.5)
plt.tight_layout(); plt.savefig('hands_grid.png', dpi=55)
json.dump(out, open('hands_all.json', 'w'))
print('wrote hands_grid.png and hands_all.json')
