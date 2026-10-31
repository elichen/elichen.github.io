"""Can BANC alone carry the fly's whole life? Feeding and walking results that the page has on FlyWire, on BANC. Shiu's LIF model.
    python tools/banc_feed.py <banc dir> <work dir>
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
data, work = sys.argv[1:3]
BIN = os.path.join(os.path.dirname(__file__), '../engine/target/release/spikes')
os.makedirs(work, exist_ok=True)
t = pd.read_parquet(f'{data}/banc_neurons.parquet')
clean = lambda c: c.astype(object).where(c.notna(), '').map(str).to_numpy(dtype=object)
ty, si, ctype = clean(t.cell_type), clean(t.side), clean(t.fafb_cell_type)
N = len(t)
cells = lambda k, s=None: np.flatnonzero((ty == k) if s is None else (ty == k) & (si == s))
def run(a):
    ix, hz, seed, also = a
    f = f'{work}/f{abs(hash((tuple(ix[:6]), len(ix), hz, seed, also is None)))}'
    open(f + '.txt', 'w').write(' '.join(map(str, ix)))
    env = dict(os.environ)
    if also is not None:
        open(f + 'a.txt', 'w').write(' '.join(map(str, also[0]))); env['ALSO'] = f'{f}a.txt:{also[1]}'
    subprocess.run([BIN, f'{data}/banc.gz', f + '.txt', str(hz), '1', str(seed), f + '.bin'], check=True, stderr=subprocess.DEVNULL, env=env)
    x = np.fromfile(f + '.bin', dtype='<u4').reshape(-1, 2); os.remove(f + '.bin'); os.remove(f + '.txt')
    if also is not None: os.remove(f + 'a.txt')
    return np.bincount(x[:, 1], minlength=N).astype(float)
def mean(ix, hz, also=None):
    with ThreadPoolExecutor(3) as p:
        return np.mean(list(p.map(run, [(ix, hz, s, also) for s in range(1, 5)])), axis=0)
# taste types in BANC: which gustatory neurons exist
sens = t[(t.super_class.astype(str).str.contains('sensory'))]
gr = sens[sens.cell_class.astype(str).str.contains('gust', case=False) | sens.cell_sub_class.astype(str).str.contains('gust|taste', case=False)]
print('gustatory by type', gr.groupby(['cell_type', 'side']).size().sort_values(ascending=False).head(14).to_dict())
MN9 = np.concatenate([cells('MN9')]) if len(cells('MN9')) else np.flatnonzero(clean(t.fafb_cell_type) == 'MN9')
print('MN9 cells', len(MN9), [ty[i] for i in MN9][:3])
for sets in (['LB3b'], ['LB3a'], ['LB3b', 'LB3a'], ['LB3c'], ['LB4a'], ['LB3b', 'LB3a', 'LB3c']):
    for s in ('right', 'left'):
        ix = np.concatenate([cells(k, s) for k in sets]) if all(len(cells(k, s)) for k in sets) else np.array([], int)
        if not len(ix): continue
        for hz in (100, 150):
            r = mean(ix, hz)
            print(f'{"+".join(sets)} {s} ({len(ix)}) {hz} Hz: MN9 {r[MN9].mean():.0f} Hz, brain {int(r.sum())}, DNg97 {r[cells("DNg97")].mean():.1f}, DNg100 {r[cells("DNg100")].mean():.1f}', flush=True)
P9, O = cells('DNp09'), cells('DNg97')
print('DNp09', len(P9), 'DNg97', len(O), 'BPN?', [k for k in ('BPN', 'DNg100') if len(cells(k))])
for hz in (100, 150, 200):
    a = mean(P9, hz)
    print(f'P9 {hz}: oDN1 {a[O].mean():.1f}, DNg100 {a[cells("DNg100")].mean():.1f}, brain {int(a.sum())}', flush=True)
    su = np.concatenate([cells(k, 'right') for k in ('LB3b', 'LB3a')])
    for sh in (100, 150):
        b = mean(P9, hz, (su, sh))
        print(f'   + sugar {sh}: oDN1 {b[O].mean():.1f}, MN9 {b[MN9].mean():.0f}', flush=True)
