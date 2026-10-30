"""Rank a `screen` run (engine/src/bin/screen.rs) by leg rhythm.

    python tools/banc_screen.py <banc_neurons.parquet> <sets file> <screen.bin> <seconds> [skip, default 0.2]

For every set and leg (neuromere + side): the summed rate of the leg's motor neurons in 5 ms bins,
its strongest spectral peak between 2 and 40 Hz as a share of the power over 0.5-100 Hz, and that
share over the 99th percentile of Poisson trains of the same rate ('x null'; a clean rhythm is
many times the null). Also the correlation of the leg's flexor and extensor rates (all joints;
alternation is negative).
"""
import sys
import numpy as np
import pandas as pd

tab = pd.read_parquet(sys.argv[1])
names = [l.split()[0] for l in open(sys.argv[2])]
raw = np.fromfile(sys.argv[3], np.uint32).reshape(-1, 3)
T, skip = float(sys.argv[4]), float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
BIN = 50
nb = int(round((T - skip) / (BIN * 1e-4)))
raw = raw[raw[:, 1] >= skip * 1e4]
freqs = np.fft.rfftfreq(nb, BIN * 1e-4)
band, whole = (freqs >= 2) & (freqs <= 40), (freqs >= 0.5) & (freqs <= 100)
legmn = tab[tab.cell_class.str.contains('leg_motor')]
legs = legmn.groupby(['neuromere', 'side']).groups
kind = np.zeros(len(tab), np.int8)   # +1 flexor, -1 extensor
for i, ty in legmn.cell_type.items():
    kind[i] = 1 if 'flexor' in ty else -1 if 'extensor' in ty else 0

rng = np.random.default_rng(0)
null_cache = {}


def null99(mean):
    key = round(mean, 2)
    if key not in null_cache:
        v = []
        for _ in range(300):
            y = rng.poisson(mean, nb).astype(float)
            q = np.abs(np.fft.rfft(y - y.mean())) ** 2
            v.append(q[band].max() / q[whole].sum())
        null_cache[key] = np.percentile(v, 99)
    return null_cache[key]


rows = []
for s, name in enumerate(names):
    r = raw[raw[:, 0] == s]
    if len(r) == 0:
        rows.append((name, 0, 0, '', 0, 0, np.nan))
        continue
    b = (r[:, 1] - int(skip * 1e4)) // BIN
    best = (0, '', 0, 0)
    for (nm, sd), members in legs.items():
        sel = np.isin(r[:, 2], members.to_numpy())
        if sel.sum() < 20:
            continue
        x = np.bincount(b[sel], minlength=nb)[:nb].astype(float)
        p = np.abs(np.fft.rfft(x - np.polyval(np.polyfit(np.arange(nb), x, 1), np.arange(nb)))) ** 2
        k = np.argmax(np.where(band, p, 0))
        score = p[k] / p[whole].sum() / null99(x.mean())
        if score > best[0]:
            best = (score, f'{nm} {sd}', freqs[k], sel.sum() / (T - skip))
    fl = np.bincount(b[kind[r[:, 2]] == 1], minlength=nb)[:nb]
    ex = np.bincount(b[kind[r[:, 2]] == -1], minlength=nb)[:nb]
    corr = np.corrcoef(fl, ex)[0, 1] if fl.std() > 0 and ex.std() > 0 else np.nan
    rows.append((name, len(np.unique(r[:, 2])), len(r) / (T - skip), best[1], best[2], best[0], corr))
res = pd.DataFrame(rows, columns=['set', 'leg_mns_firing', 'leg_spikes_per_s', 'best_leg', 'peak_hz', 'x_null', 'flex_ext_r'])
print(f'{(res.leg_mns_firing > 0).sum()} of {len(res)} sets make leg motor neurons fire')
print(res.sort_values('x_null', ascending=False).head(25).to_string(index=False))
print('\nmost leg drive:')
print(res.sort_values('leg_spikes_per_s', ascending=False).head(15).to_string(index=False))
