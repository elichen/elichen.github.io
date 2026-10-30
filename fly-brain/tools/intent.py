"""What do food cues do to the brain's walking and steering descending neurons (DNs)? Brain-only
FlyWire, Shiu's model, every spike from engine/target/release/spikes. Compared with recordings:

  odor    one-sided ORN input -> DNa01/DNa02 left vs right. Rayshubskiy et al. 2025 (eLife): flies steer
          toward the stimulated antenna, DNa02 fires more on that side. Gaudry et al. 2013: ipsilateral PNs
          ~50% above contralateral (checked here as the intermediate step).
  sugar   sugar GRNs -> which inhibitory neurons onto DNg100 (BDN2) / DNg97 (oDN1) fire, and does sugar cut
          P9 -> oDN1. Sapkal et al. 2024 (Nature 634:191): sugar recruits Foxglove, which inhibits both.

    python tools/intent.py odor|sugar <model dir> <work dir> [seeds=3] [Connectivity_783.parquet]
Rates are read as ratios and rank orders, not absolute Hz (input rates are near-threshold choices).
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

mode, model, work = sys.argv[1:4]
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
BIN = os.path.join(os.path.dirname(__file__), '../engine/target/release/spikes')
T = 2.0
n = json.load(open(f'{model}/neurons.json'))
names, ct = n['cell_type']['names'], np.array(n['cell_type']['index'])
sn, side = n['side']['names'], np.array(n['side']['index'])
N = n['count']
os.makedirs(work, exist_ok=True)


def cells(t, s=None):
    ix = np.flatnonzero(ct == names.index(t))
    return ix if s is None else ix[side[ix] == sn.index(s)]


def write(name, ix):
    p = f'{work}/{name}.txt'
    open(p, 'w').write(' '.join(map(str, ix)))
    return p


def run(args):
    """mean rate (Hz) of every neuron over the run, one seed"""
    tag, ex, hz, seed, also, silenced = args
    out = f'{work}/{tag}_{seed}.bin'
    env = dict(os.environ)
    if also:
        env['ALSO'] = also
    subprocess.run([BIN, f'{model}/connectome.gz', ex, str(hz), str(T), str(seed), out] + ([silenced] if silenced else []),
                   check=True, stderr=subprocess.DEVNULL, env=env)
    a = np.fromfile(out, dtype='<u4').reshape(-1, 2)
    os.remove(out)
    return np.bincount(a[:, 1], minlength=N) / T


def trials(conds):
    """conds: {name: (index file, Hz, also, silenced)} -> {name: rates, seeds x neurons}"""
    jobs = [(k, *v[:2], s, *v[2:]) for k, v in conds.items() for s in range(1, SEEDS + 1)]
    with ThreadPoolExecutor(6) as pool:
        res = list(pool.map(run, jobs))
    return {k: np.array([r for j, r in zip(jobs, res) if j[0] == k]) for k in conds}


def lr(R, t):
    return R[:, cells(t, 'left')].mean(), R[:, cells(t, 'right')].mean()


if mode == 'steer':
    # can one-sided odor input reach the steering DNs without igniting the brain? ORNs weakly, or PNs directly
    STEER = ['DNa01', 'DNa02']
    GLOM = ['DM1', 'DM4', 'DM2', 'VA2']
    pn = lambda g: np.flatnonzero(np.isin(ct, [i for i, x in enumerate(names) if x.startswith(g + '_') and 'PN' in x]))
    print('whole-brain spikes/s, then (left, right) Hz; a graded response has low spikes/s, a pass flips with the side')
    for label, get, rates in (('ORN', lambda g: cells(f'ORN_{g}'), (1, 2, 5)), ('PN', pn, (10, 25, 50, 100, 150, 200))):
        for g in GLOM:
            ix = get(g)
            for hz in rates:
                c = {}
                for sd in ('left', 'right'):
                    c[sd] = (write(f'st_{sd}', ix[side[ix] == sn.index(sd)]), hz, None, None)
                c['both'] = (write('st_both', ix), hz, None, None)
                R = trials(c)
                print(f'{label} {g} {hz:3d} Hz n={len(ix):3d}: ' + ' | '.join(
                    f'{k} {int(v.sum(1).mean()):7d} ' + ' '.join(f'{d} {lr(v, d)[0]:4.1f} {lr(v, d)[1]:4.1f}' for d in STEER) for k, v in R.items()), flush=True)
elif mode == 'odor':
    # LESION=1 silences the cholinergic antennal lobe local neurons: with them the lobe ignites all-or-none
    LESION = None
    if os.environ.get('LESION'):
        cc, nt = n['cell_class'], n['nt']
        ix = np.flatnonzero((np.array(cc['index']) == cc['names'].index('ALLN')) & (np.array(nt['index']) == nt['names'].index('acetylcholine')))
        print(len(ix), 'cholinergic ALLNs silenced')
        LESION = write('lesion', ix)
    GLOM = ['DM1', 'DM4', 'DM2', 'DM5', 'VA2', 'DL1', 'DA1']
    STEER = ['DNa01', 'DNa02', 'DNa03']
    conds = {}
    for g in GLOM:
        for s in ('left', 'right'):
            conds[f'{g}_{s}'] = (write(f'{g}_{s}', cells(f'ORN_{g}', s)), HZ := float(os.environ.get('HZ', 150)), None, LESION)
        conds[f'{g}_both'] = (write(f'{g}_both', cells(f'ORN_{g}')), HZ, None, None)
    # controls: a non-olfactory one-sided set
    jo = np.array(n['sets']['JO-F']['index'])
    for s in ('left', 'right'):
        conds[f'JO_{s}'] = (write(f'JO_{s}', jo[side[jo] == sn.index(s)]), HZ, None, LESION)
    R = trials(conds)
    print(f'{HZ} Hz to one side; {SEEDS} seeds; rates in Hz as (left cell, right cell)')
    print('whole brain, spikes/s per run:', {k: int(v.sum(1).mean()) for k, v in R.items() if k.startswith('DM1') or k.startswith('JO')})
    for g in GLOM + ['JO']:
        for s in ('left', 'right', 'both'):
            k = f'{g}_{s}'
            if k not in R:
                continue
            row = '  '.join(f'{d} {lr(R[k], d)[0]:5.1f} {lr(R[k], d)[1]:5.1f}' for d in STEER)
            walk = '  '.join(f'{d} {R[k][:, cells(d)].mean():5.1f}' for d in ['DNp09', 'DNg100', 'DNg97', 'MDN'])
            print(f'{k:10s} {row}  | {walk}')
    # the intermediate step: PNs of the same glomerulus, ipsilateral vs contralateral soma side
    for g in ['DM1', 'DM4', 'DM2']:
        pn = [i for i, x in enumerate(names) if x.startswith(g + '_') and 'PN' in x]
        ix = np.flatnonzero(np.isin(ct, pn))
        print(g, 'PNs:', {s: (round(R[f'{g}_{s}'][:, ix[side[ix] == sn.index('left')]].mean(), 1), round(R[f'{g}_{s}'][:, ix[side[ix] == sn.index('right')]].mean(), 1)) for s in ('left', 'right')}, '(left PNs, right PNs)')
    # which descending neurons are most lateralized, unbiased
    dn = np.flatnonzero(np.array(n['super_class']['index']) == n['super_class']['names'].index('descending'))
    for g in ['DM1', 'DM4']:
        d = (R[f'{g}_left'] - R[f'{g}_right']).mean(0)[dn]
        top = dn[np.argsort(-np.abs(d))[:8]]
        print(g, 'most lateralized DNs:', [(names[ct[i]], sn[side[i]], round(R[f'{g}_left'][:, i].mean(), 1), round(R[f'{g}_right'][:, i].mean(), 1)) for i in top])
else:
    import pandas as pd
    e = pd.read_parquet(sys.argv[5])
    ids = {int(x): i for i, x in enumerate(n['ids'])}
    sugar = np.array(n['sets']['sugar']['index'])
    bitter = np.array(n['sets']['bitter']['index'])
    s_f, b_f = write('sugar', sugar), write('bitter', bitter)
    base = trials({'sugar': (s_f, 100, None, None), 'bitter': (b_f, 100, None, None)})
    targets = np.concatenate([cells('DNg100'), cells('DNg97')])
    tid = [int(n['ids'][i]) for i in targets]
    inh = e[e.Postsynaptic_ID.isin(tid) & (e['Excitatory x Connectivity'] < 0)]
    syn = inh.groupby('Presynaptic_ID').Connectivity.sum().sort_values(ascending=False)
    print('inhibitory inputs onto DNg100+DNg97 that fire under sugar (top by synapses):')
    found = []
    for pid, c in syn.items():
        i = ids[int(pid)]
        rs, rb = base['sugar'][:, i].mean(), base['bitter'][:, i].mean()
        if rs >= 2:
            found.append(i)
            print(f'  {pid} {names[ct[i]] or "(no type)":14s} {sn[side[i]]:5s} {n["nt"]["names"][n["nt"]["index"][i]] if isinstance(n["nt"], dict) else "":6s} {c:4d} syn; sugar {rs:5.1f} Hz, bitter {rb:5.1f} Hz')
    print(f'{len(found)} of {len(syn)} inhibitory inputs fire under sugar')
    # as Sapkal 2024 Fig. 5c: drive the walking command neuron P9 (DNp09), read oDN1 (DNg97)
    p9, off = write('p9', cells('DNp09')), write('cb0890', cells('CB0890'))
    print('\nP9 -> oDN1 (mean Hz), with sugar or bitter at the given rate, and with CB0890 silenced:')
    for hz in (50, 100, 150, 200):
        R = trials({'alone': (p9, hz, None, None), 'sugar50': (p9, hz, f'{s_f}:50', None), 'sugar100': (p9, hz, f'{s_f}:100', None),
                    'sugar150': (p9, hz, f'{s_f}:150', None), 'bitter150': (p9, hz, f'{b_f}:150', None), 'sugar150_off': (p9, hz, f'{s_f}:150', off)})
        print(f'  P9 {hz:3d}: ' + '  '.join(f'{k} {v[:, cells("DNg97")].mean():5.1f}' for k, v in R.items()))
