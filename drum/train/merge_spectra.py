"""Combine spectra files for training: the synthetic families from one file and
the doodles (and their stretched copies) from another, so that doodle seeds
all index the same Quick, Draw! file and the held-out split (seed % 10 == 0)
stays consistent.

usage: python merge_spectra.py FAMILIES.npz OUT.npz DOODLES.0.npz [DOODLES.1.npz ...]
"""

import sys
import numpy as np


def main():
    a = np.load(sys.argv[1])
    parts = [np.load(f) for f in sys.argv[3:]]
    fam = ~np.isin(a['kind'], ['doodle', 'doodle_aff'])
    out = {k: np.concatenate([a[k][fam]] + [p[k] for p in parts]) for k in ('kind', 'seed', 'outline', 'lam')}
    np.savez(sys.argv[2], **out)
    for k in np.unique(out['kind']):
        print(k, int((out['kind'] == k).sum()))


if __name__ == '__main__':
    main()
