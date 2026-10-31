"""Does an object straight ahead drive the landing neurons, and one to the side not? BANC, Shiu's LIF model.
Landing: DNp07, DNp10 (Croke et al. 2026 list them; Ache et al. 2019 for DNp07). Dodge: DNp03. Escape: DNp01.
    python tools/land.py <banc dir> <work dir>
Both-eyes input stands for an object ahead, one eye for one to the side. Pass: landing neurons fire for both eyes and much
less for one, on at least one looming set, with the brain not ignited. Reported whichever way it goes.
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
ty, si = t.cell_type.astype(object).where(t.cell_type.notna(), '').map(str).to_numpy(dtype=object), t.side.astype(object).where(t.side.notna(), '').map(str).to_numpy(dtype=object)
N = len(t)
cells = lambda k, s=None: np.flatnonzero((ty == k) if s is None else (ty == k) & (si == s))
def run(a):
    ix, hz, seed = a
    f = f'{work}/l{abs(hash((tuple(ix[:6]), len(ix), hz, seed)))}'
    open(f + '.txt', 'w').write(' '.join(map(str, ix)))
    subprocess.run([BIN, f'{data}/banc.gz', f + '.txt', str(hz), '1', str(seed), f + '.bin'], check=True, stderr=subprocess.DEVNULL)
    x = np.fromfile(f + '.bin', dtype='<u4').reshape(-1, 2)
    os.remove(f + '.bin'); os.remove(f + '.txt')
    return np.bincount(x[:, 1], minlength=N).astype(float)
def mean(ix, hz):
    with ThreadPoolExecutor(4) as p:
        return np.mean(list(p.map(run, [(ix, hz, s) for s in range(1, 5)])), axis=0)
READ = ['DNp07', 'DNp10', 'DNp03', 'DNp01', 'DNp26', 'DNp57']
print({k: (len(cells(k, 'left')), len(cells(k, 'right'))) for k in READ})
fmt = lambda r: '  '.join(f'{d} {r[cells(d, "left")].mean():.0f}/{r[cells(d, "right")].mean():.0f}' for d in READ if len(cells(d)))
for name, sets in (('LC4', ['LC4']), ('LPLC2', ['LPLC2']), ('LPLC1', ['LPLC1']), ('LPLC4', ['LPLC4']), ('LC4+LPLC2', ['LC4', 'LPLC2']), ('all', ['LC4', 'LPLC1', 'LPLC4', 'LPLC2'])):
    L = np.concatenate([cells(k, 'left') for k in sets]); R = np.concatenate([cells(k, 'right') for k in sets])
    for hz in (50, 100):
        one, both = mean(L, hz), mean(np.concatenate([L, R]), hz)
        print(f'{name:10s} {hz:3d} Hz  left eye: brain {int(one.sum()):7d} | {fmt(one)}', flush=True)
        print(f'{"":14s}  both eyes: brain {int(both.sum()):6d} | {fmt(both)}', flush=True)
# what else in the screen: which DN types care most about both eyes vs one
L = np.concatenate([cells(k, 'left') for k in ('LC4', 'LPLC2')]); R = np.concatenate([cells(k, 'right') for k in ('LC4', 'LPLC2')])
one, oneR, both = mean(L, 100), mean(R, 100), mean(np.concatenate([L, R]), 100)
dn = t.index[t.super_class == 'descending'].to_numpy()
rows = []
for k in sorted(set(ty[dn])):
    c = cells(k)
    if k and len(c):
        rows.append((both[c].mean() - max(one[c].mean(), oneR[c].mean()), k, both[c].mean(), one[c].mean(), oneR[c].mean()))
print('DN types firing most for both eyes over either eye (LC4+LPLC2, 100 Hz):')
for r in sorted(rows, reverse=True)[:8]:
    print(f'  {r[1]:12s} both {r[2]:5.1f}  left {r[3]:5.1f}  right {r[4]:5.1f}')
