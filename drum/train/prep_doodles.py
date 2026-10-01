"""Clean up closed Quick, Draw! strokes once (shapes.doodle), so the spectra
can be computed elsewhere from a small file.

usage: python prep_doodles.py qd.npz OUT.npz
"""

import sys
import numpy as np
import shapes as S


def main():
    d = np.load(sys.argv[1])
    raw, closed = d['raw'], d['closed']
    idx = np.flatnonzero(closed)
    pts = np.stack([S.doodle(raw[i]) for i in idx]).astype(np.float32)
    np.savez(sys.argv[2], pts=pts, index=idx.astype(np.int64))
    print(len(idx), 'doodles')


if __name__ == '__main__':
    main()
