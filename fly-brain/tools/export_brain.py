"""Shiu et al.'s connectivity table -> model/connectome.gz for the engine (page and tools).

    python tools/export_brain.py <Completeness_783.csv> <Connectivity_783.parquet> model/connectome.gz

Gzipped FLY2 layout (tools/fly2.py). counts = 'Excitatory x Connectivity' (signed synapse
count; the sign comes from the predicted transmitter). Sorting changes only the order in which
Brian would sum simultaneous inputs; tools/parity_ref.py + `parity` check the spikes.
"""
import gzip
import sys
import numpy as np
import pandas as pd
from fly2 import fly2

comp_path, con_path, out = sys.argv[1:4]
n = len(pd.read_csv(comp_path, index_col=0))
con = pd.read_parquet(con_path, columns=['Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory x Connectivity'])
pre = con['Presynaptic_Index'].to_numpy()
post = con['Postsynaptic_Index'].to_numpy()
w = con['Excitatory x Connectivity'].to_numpy().astype(np.int64)
data = fly2(n, pre, post, w)
with gzip.open(out, 'wb', compresslevel=9) as f:
    f.write(data)
print(f'{n} neurons, {len(pre)} connections, {int(np.abs(w).sum())} synapses -> {out} ({len(data) / 1e6:.1f} MB raw)')
