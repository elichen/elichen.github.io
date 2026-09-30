"""Reference outputs for checking the JavaScript port (kempe/mech.js).

usage: python golden.py SHARD.npz OUT.json
Then: node kempe/tools/golden_test.mjs OUT.json
"""

import json
import sys
import numpy as np
from refine import Topo, fit, resample_offset, normalise, best_alignment
from linkage import resample, to_complex
from targets import targets, jitter


def main():
    d = np.load(sys.argv[1])
    rng = np.random.default_rng(5)
    T = targets()
    heart = jitter(T['heart'], rng, 0.015)
    t = normalise(to_complex(resample(heart[None]))[0][None])[0]
    cases = []
    for i in rng.choice(len(d['z']), 6, replace=False):
        n = int(d['n'][i])
        spec = dict(kind=d['kind'][i, :n].tolist(), a=d['par'][i, :n, 0].tolist(),
                    b=d['par'][i, :n, 1].tolist(), gear=d['gear'][i, :n].tolist(),
                    pos=d['pos'][i, :n].astype(float).ravel().tolist())
        topo = Topo(d['pos'][i, :n].astype(float), d['par'][i, :n].astype(int), d['kind'][i, :n],
                    d['gear'][i, :n], n - 1)
        p0 = topo.params_from_pos(d['pos'][i, :n].astype(float))
        traj, ms = topo.simulate(p0[None])
        c = normalise(resample_offset(traj[:, n - 1], [0.0]))[0]
        dist, v, s = best_alignment(t, c)
        p, err, hist, _ = fit(topo, p0, t, iters=30)
        cases.append(dict(spec=spec, params=p0.tolist(), curve=[c.real.tolist(), c.imag.tolist()],
                          min_sin=float(ms[0]), dist=float(dist), variant=int(v), shift=int(s),
                          fit_err=float(err), fit_params=p.tolist(), hist=[float(h) for h in hist]))
    json.dump(dict(target=heart.ravel().tolist(), cases=cases), open(sys.argv[2], 'w'))


if __name__ == '__main__':
    main()
