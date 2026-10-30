"""Neuron table for the page: positions, names and Shiu et al.'s stimulus sets -> model/neurons.json.

    python tools/export_neurons.py <Completeness_783.csv> <Supplemental_file1_neuron_annotations.tsv> \
        <figures.ipynb from philshiu/Drosophila_brain_model> <banc_888_meta.feather> model/neurons.json

Order = the connectome's neuron index. Positions are the soma (or, for neurons without a soma in
the brain, such as sensory neurons, a point on the neuron) in µm, FlyWire (FAFB14.1) space.
Annotations: Schlegel et al. 2024 / Dorkenwald et al. 2024 (flywire_annotations).
Stimulus sets: the FlyWire IDs Shiu et al. list in figures.ipynb (v630), kept where they are
unchanged in v783.
Proboscis and pharynx motor neurons by muscle (MN1-MN12, McKellar et al. 2020): FlyWire has them
only as CB numbers; BANC v888 (Bates et al. 2026) names them and matches each to its FlyWire neuron.
"""
import json
import re
import sys
import numpy as np
import pandas as pd

comp_path, ann_path, nb_path, banc_path, out = sys.argv[1:6]
ids = pd.read_csv(comp_path, index_col=0).index.to_numpy()
index = {f: i for i, f in enumerate(ids)}
ann = pd.read_csv(ann_path, sep='\t', low_memory=False).drop_duplicates('root_id').set_index('root_id')
ann = ann.reindex(ids)
print(f'{ann.super_class.notna().sum()} of {len(ids)} neurons annotated')

has_soma = ann.soma_x.notna()
x = np.where(has_soma, ann.soma_x, ann.pos_x) * 4e-3   # voxels of 4 x 4 x 40 nm -> µm
y = np.where(has_soma, ann.soma_y, ann.pos_y) * 4e-3
z = np.where(has_soma, ann.soma_z, ann.pos_z) * 40e-3
pos = np.round(np.stack([x, y, z], 1), 1)


def column(name, fallback=None):
    s = ann[name]
    if fallback is not None:
        s = s.fillna(ann[fallback])
    s = s.fillna('').astype(str)
    names = sorted(set(s))
    lookup = {k: i for i, k in enumerate(names)}
    return {'names': names, 'index': [lookup[v] for v in s]}


nb = json.load(open(nb_path))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
sets = {}
for key, name in [('neu_sugar', 'sugar'), ('neu_bitter', 'bitter'), ('neu_water', 'water'), ('neu_ir94e', 'Ir94e'),
                  ('neu_JON_CE', 'JO-CE'), ('neu_JON_F', 'JO-F')]:
    m = re.search(key + r' = \[(.*?)\]', src, re.S)
    found = [int(v) for v in re.findall(r'\d{18}', m.group(1))]
    sets[name] = {'paper': len(found), 'index': [index[f] for f in found if f in index]}
    print(name, len(found), '->', len(sets[name]['index']))
# not from the paper: the LB3b taste neurons on the fly's right, which send the model into runaway
# activity through antennal lobe local neurons whose predicted transmitter is excitatory
lb3b = ann.index[(ann.cell_type == 'LB3b') & (ann.side == 'right')]
sets['LB3b-right'] = {'paper': None, 'index': sorted(index[f] for f in lb3b)}
named = {'MN9': 720575940660219265, 'aDN1': 720575940616185531, 'aDN2': 720575940629806974, 'aBN1': 720575940630907434,
         # the neuron whose silencing lowers MN9 most during sugar input at 100 Hz (engine/src/bin/silence_scan.rs)
         'CB0553': 720575940607272649}

banc = pd.read_feather(banc_path, columns=['cell_class', 'cell_type', 'fafb_match'])
banc = banc[banc.cell_class.astype(str).str.contains('proboscis_motor|pharynx_motor') & banc.fafb_match.notna()]
motor = {}
for t, g in banc.groupby('cell_type'):
    found = sorted({index[int(f)] for f in g.fafb_match if int(f) in index})
    if found:
        motor[t] = found
print('proboscis/pharynx motor neurons:', {k: len(v) for k, v in motor.items()})
assert motor['MN9'] == sorted([index[named['MN9']], *[i for i in motor['MN9'] if i != index[named['MN9']]]])

data = {
    'count': len(ids),
    'ids': [str(i) for i in ids],
    'pos': [None if np.isnan(p).any() else p.tolist() for p in pos],
    'super_class': column('super_class'),
    'cell_class': column('cell_class'),
    'cell_sub_class': column('cell_sub_class'),
    'cell_type': column('cell_type', 'hemibrain_type'),
    'side': column('side'),
    'nt': column('top_nt'),
    'sets': sets,
    'named': {k: index[v] for k, v in named.items()},
    'motor': motor,
}
with open(out, 'w') as f:
    json.dump(data, f, separators=(',', ':'))
print('missing positions:', sum(p is None for p in data['pos']))
