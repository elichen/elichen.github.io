# head-to-food distance per run: run.py npz (BAAIWorm) vs loop_run logs (worm-sim)
import re, os, sys, numpy as np
O = sys.argv[1]
for who in ('ref', 'mine'):
    rows = []
    for i in range(10):
        if who == 'ref':
            p = f'{O}/ref_{i}.npz'
            if not os.path.exists(p): continue
            d = np.load(p); dist = np.linalg.norm(d['head'] - d['target_local'], axis=1)[29::30]
        else:
            L = [l for l in open(f'{O}/mine_{i}.log') if l.startswith('t ')]
            dist = np.array([float(re.search(r'food dist ([0-9.]+)', l).group(1)) for l in L])
        rows.append((dist[0], dist.min(), dist[-1], dist.argmin() + 1))
        print(f'{who} dir {i}: 1 s {dist[0]:.2f}  min {dist.min():.2f} (at {dist.argmin()+1} s)  60 s {dist[-1]:.2f}')
    r = np.array(rows)
    if len(r): print(f'{who}: mean at 1 s {r[:,0].mean():.2f}, mean min {r[:,1].mean():.2f}, mean at end {r[:,2].mean():.2f}, closer at end {int((r[:,2] < r[:,0]).sum())}/{len(r)}\n')
