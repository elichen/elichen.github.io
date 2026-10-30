"""BANC (brain and nerve cord, Bates et al. 2026) -> the engine's FLY2 file + a neuron table.

    python tools/export_banc.py <banc_888_edgelist_simple_v3.feather> <banc_888_meta.feather> \
        <banc_888_neurotransmitter_prediction_v2.csv> <out dir>

Files from the BANC v888 Dataverse deposit (doi:10.7910/DVN/7WTH1N). Same rule as Shiu et al.:
weight = synapse count, negative when the presynaptic neuron's predicted transmitter is GABA or
glutamate. Histamine is also taken as inhibitory (it opens chloride channels in flies; Shiu's
FlyWire table had no histamine class). Neurons with no prediction are excitatory, as in Shiu's
table, and counted. Glia, trachea and 'not_a_neuron' are dropped.
Writes <out>/banc.gz (FLY2, see export_brain.py) and <out>/banc_neurons.parquet (index order).
"""
import gzip
import os
import sys
import numpy as np
import pandas as pd
from fly2 import fly2

edge_path, meta_path, nt_path, out = sys.argv[1:5]
e = pd.read_feather(edge_path, columns=['pre', 'post', 'count'])
m = pd.read_feather(meta_path).drop_duplicates('banc_888_id').set_index('banc_888_id')
drop = set(m.index[m.super_class.isin(['glia', 'trachea', 'not_a_neuron'])])
e = e[~e.pre.isin(drop) & ~e.post.isin(drop)]
ids = np.array(sorted(set(e.pre) | set(e.post)))
index = pd.Series(np.arange(len(ids)), index=ids)
meta = m.reindex(ids)

nt = meta.neurotransmitter_predicted.copy()
ntv2 = pd.read_csv(nt_path, usecols=['root_id', 'neurotransmitter_predicted'], dtype={'root_id': str})
ntv2 = ntv2.drop_duplicates('root_id').set_index('root_id').neurotransmitter_predicted
nt = nt.fillna(meta.root_id.astype(str).map(ntv2))
print('transmitter known for', nt.notna().sum(), 'of', len(ids))
print(nt.value_counts(dropna=False).to_string())
sign = np.where(nt.isin(['gaba', 'glutamate', 'histamine']), -1, 1)

pre = index[e.pre].to_numpy()
post = index[e.post].to_numpy()
w = e['count'].to_numpy().astype(np.int64) * sign[pre]
os.makedirs(out, exist_ok=True)
data = fly2(len(ids), pre, post, w)
with gzip.open(os.path.join(out, 'banc.gz'), 'wb', compresslevel=6) as f:
    f.write(data)
cols = ['root_id', 'side', 'super_class', 'cell_class', 'cell_sub_class', 'cell_type', 'neuromere',
        'body_part_effector', 'peripheral_target_type', 'fafb_cell_type', 'manc_cell_type']
tab = meta[cols].astype(str).reset_index()
tab['nt'] = nt.fillna('').to_numpy()
tab.to_parquet(os.path.join(out, 'banc_neurons.parquet'))
print(f'{len(ids)} neurons, {len(pre)} connections, {int(np.abs(w).sum())} synapses')
