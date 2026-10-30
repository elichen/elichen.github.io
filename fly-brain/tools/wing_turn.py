"""Which way would the brain + cord model turn the fly? Per wing motor neuron type, left vs right rate, on BANC, for one
DNp03 (Croke et al. 2026: raises its own side's wingbeat amplitude, lowers the other) and for one-sided looms.
    python tools/wing_turn.py <banc dir> <work dir>
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

data, work = sys.argv[1:3]
BIN = os.path.join(os.path.dirname(__file__), '../engine/target/release/spikes')
os.makedirs(work, exist_ok=True)
t = pd.read_parquet(f'{data}/banc_neurons.parquet')
ty, si = t.cell_type.astype(str).to_numpy(dtype=object), t.side.astype(str).to_numpy(dtype=object)
N, T = len(t), 1.0
wing = ((t.super_class == 'motor') & (t.body_part_effector.astype(str) == 'wing')).to_numpy()
cells = lambda k, s=None: np.flatnonzero((ty == k) if s is None else (ty == k) & (si == s))


def run(a):
    ix, hz, seed = a
    f = f'{work}/w{abs(hash((tuple(ix[:6]), len(ix), hz, seed)))}'
    open(f + '.txt', 'w').write(' '.join(map(str, ix)))
    subprocess.run([BIN, f'{data}/banc.gz', f + '.txt', str(hz), str(T), str(seed), f + '.bin'], check=True, stderr=subprocess.DEVNULL)
    x = np.fromfile(f + '.bin', dtype='<u4').reshape(-1, 2)
    os.remove(f + '.bin'); os.remove(f + '.txt')
    return np.bincount(x[:, 1], minlength=N) / T


def mean(ix, hz):
    with ThreadPoolExecutor(6) as p:
        return np.mean(list(p.map(run, [(ix, hz, s) for s in range(1, 5)])), axis=0)


loom = lambda s: np.concatenate([cells(k, s) for k in ('LC4', 'LPLC1', 'LPLC4')])
cases = {'DNp03 alone': (lambda s: cells('DNp03', s), 100), 'DNp03 alone 200': (lambda s: cells('DNp03', s), 200), 'loom': (loom, 100)}
names = sorted(set(ty[wing]))
for label, (get, hz) in cases.items():
    L, R = mean(get('left'), hz), mean(get('right'), hz)
    print(f'== {label}, {hz} Hz. Rate (left wing cells / right wing cells) for input on the left, then on the right; ipsi - contra averaged over both')
    for k in names:
        a, b = cells(k, 'left'), cells(k, 'right')
        if not len(a) or not len(b):
            continue
        ll, lr, rl, rr = L[a].mean(), L[b].mean(), R[a].mean(), R[b].mean()
        if max(ll, lr, rl, rr) < 1:
            continue
        print(f'  {k:10s} {ll:5.0f}/{lr:<5.0f} {rl:5.0f}/{rr:<5.0f}  ipsi-contra {((ll - lr) + (rr - rl)) / 2:+6.1f}')
