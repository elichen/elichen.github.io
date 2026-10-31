"""Could real images drive this model? Does activity in early visual neurons reach the looming neurons and DNp03 through the optic
lobe, on one side, without igniting the brain? BANC, Shiu's LIF model. A first test: no images yet, whole populations.
    python tools/vision_reach.py <banc dir> <work dir>
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
data, work = sys.argv[1:3]
BIN = os.path.join(os.path.dirname(__file__), '../engine/target/release/spikes')
os.makedirs(work, exist_ok=True)
t = pd.read_parquet(f'{data}/banc_neurons.parquet')
clean = lambda c: c.astype(object).where(c.notna(), '').map(str).to_numpy(dtype=object)
ty, si = clean(t.cell_type), clean(t.side)
N = len(t)
cells = lambda k, s=None: np.flatnonzero((ty == k) if s is None else (ty == k) & (si == s))
def run(a):
    ix, hz, seed = a
    f = f'{work}/v{abs(hash((tuple(ix[:6]), len(ix), hz, seed)))}'
    open(f + '.txt', 'w').write(' '.join(map(str, ix)))
    subprocess.run([BIN, f'{data}/banc.gz', f + '.txt', str(hz), '0.5', str(seed), f + '.bin'], check=True, stderr=subprocess.DEVNULL)
    x = np.fromfile(f + '.bin', dtype='<u4').reshape(-1, 2); os.remove(f + '.bin'); os.remove(f + '.txt')
    return np.bincount(x[:, 1], minlength=N) / 0.5
def mean(ix, hz):
    with ThreadPoolExecutor(3) as p:
        return np.mean(list(p.map(run, [(ix, hz, s) for s in range(1, 3)])), axis=0)
print({k: (len(cells(k, 'left')), len(cells(k, 'right'))) for k in ['R1-6', 'R7', 'R8', 'L1', 'L2', 'L3', 'Tm1', 'Tm2', 'Mi1', 'T4a', 'T5a', 'T2', 'T2a', 'T3', 'Tm3', 'LC4', 'LPLC1', 'LPLC2', 'LPLC4']})
READ = ['LC4', 'LPLC1', 'LPLC2', 'LPLC4', 'DNp03', 'DNp01']
for sets in (['L1'], ['L2'], ['L3'], ['Tm1'], ['Mi1'], ['T4a', 'T4b', 'T4c', 'T4d'], ['T5a', 'T5b', 'T5c', 'T5d'], ['T2'], ['T2a'], ['T3'], ['L1', 'L2', 'L3']):
    ix = [cells(k, 'left') for k in sets]
    if not all(len(c) for c in ix):
        print(sets, 'none'); continue
    ix = np.concatenate(ix)
    for hz in (20, 60, 150):
        r = mean(ix, hz)
        print(f'{"+".join(sets)[:18]:18s} left ({len(ix)}) {hz:3d} Hz brain {int(r.sum()):7d} | ' + '  '.join(f'{d} {r[cells(d, "left")].mean():.0f}/{r[cells(d, "right")].mean():.0f}' for d in READ), flush=True)
