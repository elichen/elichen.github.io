"""Shiu's stock model (Brian's PoissonInput, its own random numbers): mean rate per neuron over trials.

    python tools/stats_ref.py <Completeness.csv> <Connectivity.parquet> <excited index file> <Hz> <trials> <out.csv>

The excited file lists neuron indices (whitespace separated). Output: index,rate (Hz), active neurons only.
"""
import sys
import numpy as np
from joblib import Parallel, delayed
import brian_model as bm


def trial(comp_path, con_path, exc, hz, seed):
    from brian2 import Network, SpikeMonitor, Hz, ms, prefs, seed as brian_seed
    prefs.codegen.target = 'cython'
    brian_seed(seed)
    comp, con = bm.load(comp_path, con_path)
    neu, syn = bm.build(len(comp), con)
    pois = bm.poisson(neu, exc, hz * Hz)
    mon = SpikeMonitor(neu)
    Network(neu, syn, mon, *pois).run(1000 * ms)
    return np.bincount(np.asarray(mon.i), minlength=len(comp))


if __name__ == '__main__':
    comp_path, con_path, exc_path, hz, trials, out = sys.argv[1:7]
    exc = [int(x) for x in open(exc_path).read().split()]
    counts = Parallel(n_jobs=8)(delayed(trial)(comp_path, con_path, exc, float(hz), s) for s in range(int(trials)))
    rate = np.mean(counts, 0)
    sd = np.std(counts, 0)
    idx = np.nonzero(rate)[0]
    np.savetxt(out, np.stack([idx, rate[idx], sd[idx]], 1), fmt=['%d', '%.4f', '%.4f'], delimiter=',', header='index,rate,sd', comments='')
    print(f'{len(idx)} active neurons, {rate.sum():.0f} spikes/s')
