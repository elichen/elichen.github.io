"""Phase structure of a simulated leg motor rhythm (npz files from tools/rate_vnc.py, one per replicate):
do antagonist motor neurons within a leg alternate, and do the legs keep fixed phases to each other
(real headless flies driven by DNg100 alternate left-right and keep a tripod even stepping in air,
Sapkal et al. 2026)?

    python tools/phase.py <rates_*.npz ...>

Per replicate: each leg's peak frequency (4-30 Hz) from its motor neurons' summed power; each
motor neuron's phase and modulation depth at that frequency. Muscle groups as in leg_drive.py.
Across replicates: circular mean and consistency R (1 = same phase every run, ~0 = random) of
antagonist pairs within a leg and of each leg against the left front leg.
"""
import sys
import numpy as np

GROUPS = {   # joint: (one direction, the other)
    'coxa': (['sternal_anterior_rotator', 'tergopleural_promotor'], ['sternal_posterior_rotator', 'pleural_remotor']),
    'femur': (['tergotrochanter_extensor', 'sternotrochanter_extensor', 'trochanter_extensor'], ['trochanter_flexor', 'accessory_trochanter_flexor']),
    'tibia': (['tibia_extensor'], ['tibia_flexor', 'accessory_tibia_flexor']),
    'tarsus': (['tarsus_levator'], ['tarsus_depressor', 'long_tendon_muscle']),
}
LEGS = [(e, s) for e in ('front_leg', 'middle_leg', 'hind_leg') for s in ('left', 'right')]
SHORT = {'front_leg': '1', 'middle_leg': '2', 'hind_leg': '3', 'left': 'L', 'right': 'R'}
name = lambda leg: SHORT[leg[1]] + SHORT[leg[0]]
MIN_DEPTH = 0.2     # a motor neuron counts as rhythmic if its oscillation is >= 20% of its mean rate


def circ(z):
    """circular mean (deg) and resultant length of unit phasors"""
    z = np.asarray([x / abs(x) for x in z if x != 0])
    if not len(z):
        return float('nan'), 0.0
    m = z.mean()
    return np.degrees(np.angle(m)), abs(m)


pairs, legs_vs_L1, summary = {}, {}, []
for f in sys.argv[1:]:
    z = np.load(f)
    R, types, sides, eff, dt = z['rates'], z['types'], z['sides'], z['effector'], float(z['dt'])
    R = R - R.mean(0)
    n = len(R)
    freqs = np.fft.rfftfreq(n, dt)
    F = np.fft.rfft(R * np.hanning(n)[:, None], axis=0)
    band = (freqs >= 4) & (freqs <= 30)
    mean = z['rates'].mean(0)
    line = []
    leg_phase = {}
    for leg in LEGS:
        idx = np.flatnonzero((eff == leg[0]) & (sides == leg[1]))
        if not len(idx):
            continue
        power = (np.abs(F[:, idx]) ** 2).sum(1)
        if power[band].max() < 1e-9:
            line.append(f'{name(leg)} silent')
            continue
        k = np.flatnonzero(band)[np.argmax(power[band])]
        c = F[k, idx]                                # complex amplitude per motor neuron at the leg's frequency
        depth = 2 * np.abs(c) / np.hanning(n).sum() / np.maximum(mean[idx], 1e-9)
        rhythmic = idx[depth > MIN_DEPTH]
        line.append(f'{name(leg)} {freqs[k]:.1f} Hz {len(rhythmic)}/{(mean[idx] > 0.01).sum()} MNs')
        grp = {}
        for joint, (a, b) in GROUPS.items():
            for d, keys in (('+', a), ('-', b)):
                m = [i for i, j in zip(idx, range(len(idx))) if depth[j] > MIN_DEPTH and any(types[i].startswith(x) for x in keys)]
                if m:
                    grp[joint + d] = F[k, m].sum()
            if joint + '+' in grp and joint + '-' in grp:
                pairs.setdefault((name(leg), joint), []).append(grp[joint + '+'] * np.conj(grp[joint + '-']))
        # the leg's phase: its strongest rhythmic component (first principal direction of the complex amplitudes)
        cc = F[k, idx]
        leg_phase[leg] = (freqs[k], grp.get('coxa+', cc[np.argmax(np.abs(cc))]))
    if ('front_leg', 'left') in leg_phase:
        f1, p1 = leg_phase[('front_leg', 'left')]
        for leg, (fl, p) in leg_phase.items():
            if leg != ('front_leg', 'left'):
                legs_vs_L1.setdefault(name(leg), []).append((p * np.conj(p1)) if abs(fl - f1) < 0.6 else 0)
    summary.append(f'{f.split("/")[-1]}: ' + '; '.join(line))

print('\n'.join(summary))
print('\nwithin a leg: phase of + group vs - group at the leg frequency (180 = alternate, 0 = co-fire)')
for (leg, joint), v in sorted(pairs.items()):
    m, r = circ(v)
    print(f'  {leg} {joint:6s} {m:+5.0f} deg, R {r:.2f} (n {len(v)})')
print('\nbetween legs: coxa promotor phase vs left front leg (legs at a different frequency count as uncoupled)')
print('  real flies on DNg100, air-stepping: 1R 180, 2L 180, 2R 0, 3L 0, 3R 180 (tripod)')
for leg, v in legs_vs_L1.items():
    m, r = circ(v)
    same = sum(1 for x in v if x != 0)
    print(f'  {leg}: {m:+5.0f} deg, R {r:.2f}  (same frequency in {same}/{len(v)})')
