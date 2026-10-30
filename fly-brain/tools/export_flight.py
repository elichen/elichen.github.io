"""flight/model/neurons.json for the flight page, from tools/export_banc.py's table and BANC's metadata.
    python tools/export_flight.py <banc dir> flight/model/neurons.json
pos: x, y, z in um per neuron (BANC's position point, voxels of 4 x 4 x 45 nm). looms: visual inputs per side; named: flight
descending neurons; wing: wing motor neurons by type and side. Sides are BANC's annotation of the fly's side.
"""
import json
import sys
import numpy as np
import pandas as pd

d, out = sys.argv[1:3]
t = pd.read_parquet(f'{d}/banc_neurons.parquet')
m = pd.read_feather(f'{d}/meta.feather', columns=['banc_888_id', 'position']).drop_duplicates('banc_888_id').set_index('banc_888_id')
xyz = np.array([[float(x) for x in s.replace(',', ' ').split()] for s in m.position.reindex(t.banc_888_id).astype(str)], dtype=float)
pos = np.round(xyz * [4, 4, 45] / 1000).astype(int)
clean = lambda c: c.astype(object).where(c.notna(), '').map(str)
ty = clean(t.cell_type)
side = clean(t.side)
by = lambda names: {n: {s: [int(i) for i in t.index[(ty == n) & (side == s)]] for s in ('left', 'right')} for n in names}
wing = t[(t.super_class == 'motor') & (t.body_part_effector.astype(str) == 'wing')]
names, index = np.unique(ty.to_numpy(dtype=str), return_inverse=True)
sc_names, sc_index = np.unique(clean(t.super_class).to_numpy(dtype=str), return_inverse=True)
s_names, s_index = np.unique(side.to_numpy(dtype=str), return_inverse=True)
json.dump({'count': len(t), 'pos': pos.ravel().tolist(),
           'looms': by(['LC4', 'LPLC1', 'LPLC4']), 'named': by(['DNp03', 'DNp26', 'DNp57', 'DNa15', 'DNb01', 'DNp01']), 'wing': by(sorted(set(clean(wing.cell_type)))),
           'cell_type': {'names': names.tolist(), 'index': index.tolist()}, 'super_class': {'names': sc_names.tolist(), 'index': sc_index.tolist()},
           'side': {'names': s_names.tolist(), 'index': s_index.tolist()}}, open(out, 'w'), separators=(',', ':'))
print(pos.min(0), pos.max(0))
