"""Pugliese et al.'s firing-rate model of the leg circuits (bioRxiv 2025.09.12.675944, MIT code at
github.com/smpuglie/Pugliese_cpg_2025), rebuilt from their Methods on BANC v888, to ask whether
the nerve cord connectome makes a stepping rhythm when a descending neuron is driven.

    python tools/rate_vnc.py <banc dir: edgelist_simple_v3.feather, meta.feather, banc_neurons.parquet> \
        <DN cell type> <input> [replicates=8] [network: t1 (front legs, as the paper) | legs (all six)] [out.npz]

Model (their eq. 1): tau dr/dt = -r + [rmax tanh(a/rmax (I + W r - theta))]+, W = 0.03 x signed
synapse count (ACh +, GABA/Glu -; floor of 5 synapses), per neuron a ~ N(1, 0.1) / size,
theta ~ N(7.5, 0.6) x size, rmax ~ N(200, 10) Hz, tau ~ N(20, 2) ms (truncated at 0), size = volume
over the median volume. Input on from 20 ms for 1 s. Forward Euler, dt 0.1 ms.
Network (their Methods): leg motor neurons, all neurons with synapses onto them (premotor), and the
descending neurons with synapses onto the premotor neurons; only neurons with a transmitter
prediction. Rhythmicity score per their Methods (autocorrelation peak vs a matched sine).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import find_peaks

d, dn_type, stim = sys.argv[1], sys.argv[2], float(sys.argv[3])
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 8
which = sys.argv[5] if len(sys.argv) > 5 else 't1'
FLOOR, MULT, DT, T, ON = 5, 0.03, 1e-4, 1.0, 0.02

e = pd.read_feather(f'{d}/edgelist_simple_v3.feather', columns=['pre', 'post', 'count'])
e = e[e['count'] >= FLOOR]
meta = pd.read_feather(f'{d}/meta.feather').drop_duplicates('banc_888_id').set_index('banc_888_id')
cls = meta.cell_class.astype(str)
nt = meta.neurotransmitter_predicted
if which == 't1':
    mn = set(meta.index[cls.str.contains('leg_motor') & (meta.body_part_effector.astype(str) == 'front_leg')])
else:
    mn = set(meta.index[cls.str.contains('leg_motor')])
pre_mn = set(e.pre[e.post.isin(mn)]) - mn
dns = set(meta.index[meta.super_class == 'descending'])
premotor = pre_mn - dns
dn_in = set(e.pre[e.post.isin(premotor)]) & dns
dn_in |= pre_mn & dns
keep = [i for i in sorted(mn | premotor | dn_in) if i in meta.index and pd.notna(nt.get(i))]
ix = {b: k for k, b in enumerate(keep)}
n = len(keep)
sub = e[e.pre.isin(ix) & e.post.isin(ix)]
sign = np.where(nt.reindex(keep).isin(['gaba', 'glutamate']), -1.0, 1.0)
pre_i, post_i = sub.pre.map(ix).to_numpy(), sub.post.map(ix).to_numpy()
W = sparse.csr_matrix((sub['count'].to_numpy() * sign[pre_i] * MULT, (post_i, pre_i)), shape=(n, n))
vol = meta.volume_nm3.reindex(keep).astype(float).to_numpy()
size = vol / np.nanmedian(vol)
size[~np.isfinite(size) | (size <= 0)] = 1
is_mn = np.array([b in mn for b in keep])
is_dn = np.array([b in dn_in for b in keep])
types = meta.cell_type.reindex(keep).astype(str).to_numpy()
sides = meta.side.reindex(keep).astype(str).to_numpy()
stim_ix = np.flatnonzero(types == dn_type)
print(f'{which}: {n} neurons ({is_mn.sum()} motor, {is_dn.sum()} descending), {len(sub)} connections; '
      f'stimulating {dn_type} {list(sides[stim_ix])} with input {stim}')
if not len(stim_ix):
    sys.exit('no such descending neuron in the network')


def tn(rng, mean, sd, size_):
    x = rng.normal(mean, sd, size_)
    while (bad := x <= 0).any():
        x[bad] = rng.normal(mean, sd, bad.sum())
    return x


def score(trace):
    """Their motor rhythmicity score for one trace (after the transient)."""
    x = trace
    if x.max() - x.min() < 1e-9:
        return 0.0, 0.0
    x = 2 * (x - x.min()) / (x.max() - x.min()) - 1
    def raw(y):
        y = y - y.mean()
        ac = np.correlate(y, y, 'full')[len(y) - 1:]
        ac /= ac[0]
        pk, pr = find_peaks(ac, prominence=0.05)
        if not len(pk):
            return 0.0, None
        k = np.argmax(pr['prominences'])
        return min(ac[pk[k]], pr['prominences'][k]), pk[k]
    s, lag = raw(x)
    if lag is None:
        return 0.0, 0.0
    t = np.arange(len(x))
    ref, _ = raw(np.sin(2 * np.pi * t / lag))
    return min(1.0, s / ref) if ref else 0.0, 1 / (lag * SAVE)


SAVE = 1e-3     # s between saved samples
scores = []
for rep in range(reps):
    rng = np.random.default_rng(rep)
    a = tn(rng, 1, 0.1, n) / size
    th = tn(rng, 7.5, 0.6, n) * size
    rmax = tn(rng, 200, 10, n)
    tau = tn(rng, 0.02, 0.002, n)
    r = np.zeros(n)
    I = np.zeros(n)
    I[stim_ix] = stim
    steps, every = int(T / DT), int(SAVE / DT)
    rec = np.zeros((steps // every, n))
    for k in range(steps):
        inp = (I if k * DT >= ON else 0) + W @ r
        f = np.maximum(rmax * np.tanh(a / rmax * (inp - th)), 0)
        r += DT / tau * (f - r)
        if (k + 1) % every == 0:
            rec[(k + 1) // every - 1] = r
    post = rec[250:]
    active = np.flatnonzero(is_mn & (post.max(0) > 0.01))
    sc = [score(post[:, i]) for i in active]
    s = float(np.mean([x[0] for x in sc])) if sc else float('nan')
    freqs = [x[1] for x in sc if x[0] > 0.5]
    flex = [i for i in active if 'flexor' in types[i]]
    ext = [i for i in active if 'extensor' in types[i]]
    fe = np.corrcoef(post[:, flex].sum(1), post[:, ext].sum(1))[0, 1] if flex and ext and post[:, flex].std() > 0 and post[:, ext].std() > 0 else float('nan')
    scores.append(s)
    if len(sys.argv) > 6 and rep == int(os.environ.get('SAVE_REP', 0)):   # keep one replicate's motor neuron traces
        np.savez(sys.argv[6], rates=post[:, is_mn], types=types[is_mn].astype('U'), sides=sides[is_mn].astype('U'),
                 effector=meta.body_part_effector.reindex(keep).astype(str).to_numpy()[is_mn].astype('U'), dt=SAVE)
    print(f'  replicate {rep}: {int((post.max(0) > 0.01).sum())} neurons active, {len(active)} motor neurons, '
          f'rhythmicity {s:.2f}, rhythm {np.median(freqs) if freqs else float("nan"):.1f} Hz, flexor-extensor r {fe:+.2f}, '
          f'max rate {rec.max():.0f} Hz')
print(f'mean rhythmicity {np.nanmean(scores):.2f}; replicates above 0.5: {sum(s > 0.5 for s in scores)}/{reps}')
