"""Do leg motor neurons oscillate in a BANC run? Reads a `spikes` dump (engine/src/bin/spikes.rs).

    python tools/banc_rhythm.py <banc_neurons.parquet> <run.bin> [seconds to skip, default 0.2]

Prints the most active neurons by class, then every active leg motor neuron with its rate and
its rhythm: the strongest peak of the spectrum of its 5 ms binned rate between 2 and 40 Hz,
as a share of the power over 0.5-100 Hz (a flat, tonic train gives a small share), plus the
correlation of flexor and extensor rates in each leg (alternation = negative).
"""
import sys
import numpy as np
import pandas as pd

tab = pd.read_parquet(sys.argv[1])
raw = np.fromfile(sys.argv[2], np.uint32).reshape(-1, 2)
skip = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
steps, who = raw[:, 0], raw[:, 1]
T = (steps.max() + 1) * 1e-4 if len(steps) else 0
keep = steps * 1e-4 >= skip
steps, who = steps[keep], who[keep]
dur = T - skip
rate = np.bincount(who, minlength=len(tab)) / dur
active = np.flatnonzero(rate > 0)
print(f'{len(active)} neurons fire, {len(who) / dur:.0f} spikes/s, over {dur:.2f} s')
summary = tab.iloc[active].assign(rate=rate[active]).groupby('super_class').rate.agg(['count', 'mean'])
print(summary.sort_values('count', ascending=False).to_string(), '\n')
top = tab.iloc[active].assign(rate=rate[active]).sort_values('rate', ascending=False)
print(top.head(25)[['cell_type', 'super_class', 'side', 'neuromere', 'nt', 'rate']].to_string(), '\n')

BIN = 50   # 5 ms
nb = int(dur / (BIN * 1e-4))
freqs = np.fft.rfftfreq(nb, BIN * 1e-4)
band = (freqs >= 2) & (freqs <= 40)
whole = (freqs >= 0.5) & (freqs <= 100)
mn = tab.index[tab.cell_class.str.contains('leg_motor')]
rows = []
series = {}
for i in mn:
    if rate[i] < 2:
        continue
    t = (steps[who == i] - int(skip * 1e4)) // BIN
    x = np.bincount(t, minlength=nb)[:nb].astype(float)
    series[i] = x
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    k = np.argmax(np.where(band, p, 0))
    rows.append((i, tab.cell_type[i], tab.side[i], tab.neuromere[i], rate[i], freqs[k], p[k] / p[whole].sum()))
res = pd.DataFrame(rows, columns=['i', 'cell_type', 'side', 'neuromere', 'rate', 'peak_hz', 'peak_share'])
print(f'{len(res)} of {len(mn)} leg motor neurons fire >= 2 Hz')
if len(res):
    print(res.sort_values(['neuromere', 'side', 'rate'], ascending=[True, True, False]).to_string(index=False), '\n')
    # flexor vs extensor of the tibia and trochanter in each leg
    for joint in ['tibia', 'trochanter']:
        for (nm, sd), g in res.groupby(['neuromere', 'side']):
            fl = [series[i] for i in g.i if f'{joint}_flexor' in tab.cell_type[i] or f'{joint}_flexor' in tab.peripheral_target_type[i]]
            ex = [series[i] for i in g.i if f'{joint}_extensor' in tab.cell_type[i] or f'{joint}_extensor' in tab.peripheral_target_type[i]]
            if fl and ex:
                r = np.corrcoef(np.sum(fl, 0), np.sum(ex, 0))[0, 1]
                print(f'{joint} flexor/extensor, {nm} {sd}: r = {r:+.2f} ({len(fl)} flexors, {len(ex)} extensors firing)')

# whole leg: all its motor neurons summed, against Poisson trains of the same rate (95th percentile)
rng = np.random.default_rng(0)
print()
for (nm, sd), g in res.groupby(['neuromere', 'side']):
    x = np.sum([series[i] for i in g.i], 0)
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    k = np.argmax(np.where(band, p, 0))
    share = p[k] / p[whole].sum()
    null = []
    for _ in range(200):
        y = rng.poisson(x.mean(), nb).astype(float)
        q = np.abs(np.fft.rfft(y - y.mean())) ** 2
        null.append(q[band].max() / q[whole].sum())
    print(f'leg {nm} {sd}: {len(g)} MNs, {x.sum() / dur:.0f} spikes/s, peak {freqs[k]:.1f} Hz share {share:.3f} '
          f'(Poisson 95%: {np.percentile(null, 95):.3f})')
