"""Does a one-sided looming signal reach the flight saccade neurons on that side? Shiu's LIF model, every spike from
engine/target/release/spikes. Croke et al. 2026 (Curr Biol 36:133): DNp03 fires more to looms on its own side; its
main inputs are LPLC1, LPLC4, LC4; one DNp03 raises its own side's wingbeat amplitude and lowers the other's.
    python tools/loom.py flywire <model dir> <work dir> | banc <banc dir> <work dir>
A pass: more firing on the loom's side, it flips with the side, both-sides input is symmetric, and the brain is not ignited.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

which, data, work = sys.argv[1:4]
BIN = os.path.join(os.path.dirname(__file__), '../engine/target/release/spikes')
os.makedirs(work, exist_ok=True)
T, SEEDS = 1.0, 4
if which == 'flywire':
    n = json.load(open(f'{data}/neurons.json'))
    names, ct = np.array(n['cell_type']['names'], dtype=object), np.array(n['cell_type']['index'])
    types = names[ct]
    sides = np.array(n['side']['names'], dtype=object)[np.array(n['side']['index'])]
    conn, N = f'{data}/connectome.gz', n['count']
else:
    import pandas as pd
    t = pd.read_parquet(f'{data}/banc_neurons.parquet')
    types, sides = t.cell_type.astype(str).to_numpy(dtype=object), t.side.astype(str).to_numpy(dtype=object)
    conn, N = f'{data}/banc.gz', len(t)
    effector = t.body_part_effector.astype(str).to_numpy(dtype=object)
    motor = (t.super_class == 'motor').to_numpy()


def cells(t_, s=None):
    m = types == t_
    return np.flatnonzero(m if s is None else m & (sides == s))


def run(a):
    ix, hz, seed = a
    f = f'{work}/{which}_{abs(hash((tuple(ix[:5]), len(ix), hz, seed)))}'
    open(f + '.txt', 'w').write(' '.join(map(str, ix)))
    subprocess.run([BIN, conn, f + '.txt', str(hz), str(T), str(seed), f + '.bin'], check=True, stderr=subprocess.DEVNULL)
    x = np.fromfile(f + '.bin', dtype='<u4').reshape(-1, 2)
    os.remove(f + '.bin'); os.remove(f + '.txt')
    return np.bincount(x[:, 1], minlength=N) / T


def mean(ix, hz):
    with ThreadPoolExecutor(6) as p:
        return np.mean(list(p.map(run, [(ix, hz, s) for s in range(1, SEEDS + 1)])), axis=0)


READ = ['DNp03', 'DNa15', 'DNp26', 'DNp57', 'DNb01', 'DNp01', 'DNp02', 'DNp04', 'DNp06', 'DNp11', 'DNa04', 'DNa05']
READ = [r for r in READ if len(cells(r))]
print(which, 'input cells:', {k: (len(cells(k, 'left')), len(cells(k, 'right'))) for k in ['LC4', 'LPLC1', 'LPLC4', 'LPLC2']}, 'read:', READ)
for label, sets in (('LC4', ['LC4']), ('LPLC2', ['LPLC2']), ('LPLC1', ['LPLC1']), ('LPLC4', ['LPLC4']), ('all loom', ['LC4', 'LPLC1', 'LPLC4'])):
    for hz in (30, 60, 100, 150):
        ix = {s: np.concatenate([cells(k, s) for k in sets]) for s in ('left', 'right')}
        if not len(ix['left']):
            continue
        L, R = mean(ix['left'], hz), mean(ix['right'], hz)
        B = mean(np.concatenate([ix['left'], ix['right']]), hz)
        lr = lambda r, d: (r[cells(d, 'left')].mean(), r[cells(d, 'right')].mean())
        row = '  '.join(f'{d} {lr(L, d)[0]:.0f}/{lr(L, d)[1]:.0f} {lr(R, d)[0]:.0f}/{lr(R, d)[1]:.0f} {lr(B, d)[0]:.0f}/{lr(B, d)[1]:.0f}' for d in READ[:6])
        print(f'{label:9s}{hz:4d} Hz brain {int(L.sum()):7d} {int(R.sum()):7d} {int(B.sum()):7d} | (L/R cell for left loom, right loom, both) {row}', flush=True)
        if which == 'banc' and hz >= 100:
            w = motor & (effector == 'wing')
            ws = lambda r: (r[w & (sides == 'left')].sum() / max(1, (w & (sides == 'left')).sum()), r[w & (sides == 'right')].sum() / max(1, (w & (sides == 'right')).sum()))
            print(f'           wing MNs mean Hz L/R: left loom {ws(L)[0]:.1f}/{ws(L)[1]:.1f}, right loom {ws(R)[0]:.1f}/{ws(R)[1]:.1f}, both {ws(B)[0]:.1f}/{ws(B)[1]:.1f}', flush=True)
