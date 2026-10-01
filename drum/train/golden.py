"""Reference meshes and spectra for checking drum/fem.js against fem.py.

usage: python golden.py SPECTRA.npz OUT.json
Then: node drum/tools/golden_test.mjs OUT.json
"""

import json
import sys
import numpy as np
from fem import normalise, mesh, modes, crisscross_mesh, GWW1, GWW2


def main():
    d = np.load(sys.argv[1])
    rng = np.random.default_rng(3)
    cases = []
    for kind in ('doodle', 'blob', 'polygon', 'star'):
        i = rng.choice(np.flatnonzero(d['kind'] == kind))
        p = normalise(d['outline'][i].astype(float))
        nodes, tri, nr = mesh(p)
        lam = modes(nodes, tri, nr, 40)
        cases.append(dict(name=kind, outline=p.ravel().tolist(), nodes=len(nodes), tris=len(tri), nRim=nr,
                          lam=lam.tolist()))
    for name, g in (('gww1', GWW1), ('gww2', GWW2)):
        nodes, tri, nr = crisscross_mesh(g, 6)
        lam = modes(nodes, tri, nr, 20)
        cases.append(dict(name=name, crisscross=6, outline=g.ravel().tolist(), nodes=len(nodes), tris=len(tri),
                          nRim=nr, lam=lam.tolist()))
    json.dump(cases, open(sys.argv[2], 'w'))


if __name__ == '__main__':
    main()
