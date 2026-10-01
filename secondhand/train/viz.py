import pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
d = pickle.load(open('data/brush_norm.pkl','rb'))
S = d['samples']; rng = np.random.default_rng(1)
ws = rng.choice(sorted(d['writers']), 4, replace=False)
fig, axs = plt.subplots(8, 2, figsize=(16, 14))
for r, w in enumerate(ws):
    mine = [s for s in S if s['w'] == w]
    for k in range(4):
        s = mine[rng.integers(len(mine))]; ax = axs[r*2 + k//2, k%2]
        p, l = s['pts'], s['lift']; st = 0
        for e in np.where(l)[0]:
            seg = p[st:e+1]; ax.plot(seg[:,0], -seg[:,1], 'k-', lw=1.2); st = e+1
        ax.axhline(0, color='b', lw=.4); ax.axhline(1, color='r', lw=.4)
        ax.set_title(f"w{w}: {s['text']}", fontsize=8); ax.set_aspect('equal'); ax.set_xlim(-0.5, 22); ax.set_ylim(-2, 3); ax.axis('off')
plt.tight_layout(); plt.savefig('viz_norm.png', dpi=70)
