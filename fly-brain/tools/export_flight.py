"""model/neurons.json for the page, from tools/export_banc.py's table and BANC's metadata.
    python tools/export_flight.py <banc dir> model/neurons.json
pos: x, y, z in um per neuron (BANC's position point, voxels of 4 x 4 x 45 nm). looms: visual inputs per side; named: flight
and takeoff descending neurons; wing: wing motor neurons by type and side; feel: the body's own sensory neurons, by BANC's
annotation (cell_sub_class): halteres and wing base per side, and for each leg its chordotonal claw and hook neurons, hair
plates, campaniform sensilla and the touch neurons of its taste bristles; feel.motor: the motor neurons of the legs, neck and
halteres, for tools/feel.mjs. Sides are BANC's annotation of the fly's side.
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
sub = clean(t.cell_sub_class)
sense = lambda pattern, s: [int(i) for i in t.index[sub.str.contains(pattern) & (side == s)]]
sides = lambda pattern: {s: sense(pattern, s) for s in ('left', 'right')}
feel = {'haltere': sides('^haltere_campaniform'), 'wing': sides('^wing_(?:base_|tegula_)?campaniform|^wing_base_orphan'),
        'legs': {f'{T}_{s}': {k: sense(f'^{leg}_leg_{p}', s) for k, p in (('claw', 'claw'), ('hook', 'hook'), ('hairplate', 'hair_plate'), ('campaniform', '(?:bilateral_)?campaniform'), ('touch', 'taste_bristle_tactile'))}
                 for leg, T in (('front', 'T1'), ('middle', 'T2'), ('hind', 'T3')) for s in ('left', 'right')}}
names, index = np.unique(ty.to_numpy(dtype=str), return_inverse=True)
sc_names, sc_index = np.unique(clean(t.super_class).to_numpy(dtype=str), return_inverse=True)
s_names, s_index = np.unique(side.to_numpy(dtype=str), return_inverse=True)
eff = clean(t.body_part_effector)
feel['motor'] = {p: {s: [int(i) for i in t.index[(t.super_class == 'motor') & (eff == p) & (side == s)]] for s in ('left', 'right')} for p in ('front_leg', 'middle_leg', 'hind_leg', 'neck', 'haltere')}
json.dump({'count': len(t), 'pos': pos.ravel().tolist(),
           'looms': by(['LC4', 'LPLC1', 'LPLC4']), 'named': by(['DNp03', 'DNp26', 'DNp57', 'DNa15', 'DNb01', 'DNp01', 'DNp04', 'DNp11']), 'wing': by(sorted(set(clean(wing.cell_type)))), 'feel': feel,
           'life': {'sugar': [int(i) for i in open(f'{d}/sugarR.txt').read().split()], **{k: [int(i) for i in t.index[ty == v]] for k, v in (('MN9', 'MN9'), ('P9', 'DNp09'), ('oDN1', 'DNg97'), ('GF', 'DNp01'))}, 'LPLC2': by(['LPLC2'])['LPLC2'], 'mouth': {k: [int(i) for i in t.index[ty == k]] for k in ('MN9', 'MN1', 'MN2Da', 'MN2Db', 'MN2V', 'MN4a', 'MN4b', 'MN3L', 'MN3M', 'MN7')}},
           'cell_type': {'names': names.tolist(), 'index': index.tolist()}, 'super_class': {'names': sc_names.tolist(), 'index': sc_index.tolist()},
           'side': {'names': s_names.tolist(), 'index': s_index.tolist()}}, open(out, 'w'), separators=(',', ':'))
print(pos.min(0), pos.max(0))
