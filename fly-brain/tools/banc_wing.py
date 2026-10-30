"""Which descending neurons drive the wing motor neurons in Shiu's LIF model on BANC? From `screen` runs.
    python tools/banc_wing.py <banc_neurons.parquet> <sets file> <screen.bin> <seconds> [skip=0.2]
Groups: power (DLM, DVM; stretch-activated muscles, no spike per wingbeat in real flies), steering (b, i, iii, tp...),
haltere. Prints the types that drive each group, and whether rates show any rhythm is out of scope here.
"""
import sys
import numpy as np
import pandas as pd
tab = pd.read_parquet(sys.argv[1])
names = [l.split()[0] for l in open(sys.argv[2])]
raw = np.fromfile(sys.argv[3], np.uint32).reshape(-1, 3)
T, skip = float(sys.argv[4]), float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
raw = raw[raw[:, 1] >= skip * 1e4]
dur = T - skip
mn = tab[(tab.super_class == 'motor') & tab.body_part_effector.astype(str).isin(['wing', 'haltere'])]
def group(r):
    if r.body_part_effector == 'haltere': return 'haltere'
    return 'power' if str(r.cell_type).startswith(('DLM', 'DVM')) else 'steering'
g = {i: group(r) for i, r in mn.iterrows()}
ty = mn.cell_type.astype(str).to_dict()
size = pd.Series(g).value_counts().to_dict()
print('motor neurons', size)
rows = []
for s in np.unique(raw[:, 0]):
    a = raw[raw[:, 0] == s][:, 2]
    c = pd.Series(a).value_counts()
    tot = {k: 0 for k in size}; active = {k: 0 for k in size}; which = {}
    for i, n in c.items():
        if i not in g: continue
        tot[g[i]] += n / dur; active[g[i]] += 1
        which[ty[i]] = which.get(ty[i], 0) + n / dur
    rows.append((names[s], {k: tot[k] / size[k] for k in size}, active, sorted(which.items(), key=lambda x: -x[1])[:6]))
print(len(rows), 'of', len(names), 'DN types make a wing or haltere motor neuron fire')
for k in ('power', 'steering', 'haltere'):
    print('--', k)
    for n, t, a, w in sorted(rows, key=lambda r: -r[1][k])[:12]:
        print(f'  {n:12s} mean {t[k]:6.1f} Hz, {a[k]:2d}/{size[k]} cells; top {[(x, round(y)) for x, y in w]}')
for q in ('DNg02', 'DNp01', 'DNp03', 'DNp04', 'DNp06', 'DNb01', 'DNa01', 'DNg100', 'DNg06'):
    r = [x for x in rows if x[0] == q]
    print(q, 'nothing' if not r else {k: round(v, 1) for k, v in r[0][1].items()}, r[0][3] if r else '')
