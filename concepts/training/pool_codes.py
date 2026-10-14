# The pool images' rows of codes_{layer}.npz, read with seeks so the 314 MB file never sits in RAM.
#   python pool_codes.py run2 layer4.2 pool_0.5_6 out.npz   (rows in the app's pool order: by class, then index)
import sys, zipfile

import numpy as np

from common import ROOT

RUN, LAYER, POOL, OUT = sys.argv[1:5]
labels = np.load(f'{ROOT}/labels.npy')
pool = np.load(f'{ROOT}/runs/{RUN}/{POOL}.npy')
pool = pool[np.lexsort((pool, labels[pool]))]
out = {}
with zipfile.ZipFile(f'{ROOT}/runs/{RUN}/codes_{LAYER}.npz') as z:
    for name in ['idx', 'val']:
        with z.open(f'{name}.npy') as f:
            major, _ = np.lib.format.read_magic(f)
            read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            shape, _, dtype = read_header(f)
            start, row = f.tell(), int(np.prod(shape[1:])) * dtype.itemsize
            rows = []
            for i in pool:
                f.seek(start + int(i) * row)
                rows.append(np.frombuffer(f.read(row), dtype).reshape(shape[1:]))
            out[name] = np.stack(rows)
np.savez(OUT, **out)
print({k: v.shape for k, v in out.items()})
