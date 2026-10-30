"""Drive flybody's front legs open loop with leg motor neuron rates from tools/rate_vnc.py, to see
what movement a simulated motor rhythm makes. The thorax is held in place (a tethered fly in air).

    python tools/leg_drive.py <flybody>/flybody/fruitfly/assets/fruitfly.xml <rates.npz> <out dir> [seconds]

Same rule as the page's proboscis (body.js): a muscle group's activation is 1 - exp(-r / 40 Hz) of
its motor neurons' mean rate; the joint's flybody position actuator aims at the joint's range limit
times the activation, one direction minus the other. Muscle groups from BANC's motor neuron types
(peripheral targets); directions read off flybody renders: coxa + = promotion (forward),
femur + = trochanter extension (depression), tibia + = extension, tarsus - = depression.
Prints joint ranges of motion and the tarsus tip's path; writes frames for a GIF.
"""
import os
import sys
import numpy as np
import mujoco
from PIL import Image

xml, npz, out = sys.argv[1:4]
T = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
R_HALF = 40.0
GROUPS = {   # joint: (+ direction types, - direction types)
    'coxa': (['sternal_anterior_rotator', 'tergopleural_promotor'], ['sternal_posterior_rotator', 'pleural_remotor']),
    'femur': (['tergotrochanter_extensor', 'sternotrochanter_extensor', 'trochanter_extensor'], ['trochanter_flexor', 'accessory_trochanter_flexor']),
    'tibia': (['tibia_extensor'], ['tibia_flexor', 'accessory_tibia_flexor']),
    'tarsus': (['tarsus_levator'], ['tarsus_depressor', 'long_tendon_muscle']),
}

z = np.load(npz)
rates, types, sides, eff, dt = z['rates'], z['types'], z['sides'], z['effector'], float(z['dt'])
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
os.makedirs(out, exist_ok=True)


LEGS = {'T1': 'front_leg', 'T2': 'middle_leg', 'T3': 'hind_leg'}


def members(side, leg, keys):
    return [i for i, t in enumerate(types) if sides[i] == side and eff[i] == LEGS[leg] and any(t.startswith(k) for k in keys)]


def direction(joint, leg, side):
    """+1 if the joint's positive direction is the GROUPS '+' movement: promotion moves the foot
    forward (coxa), trochanter extension and tarsus levation are judged by foot height, tibia
    extension by the foot's distance from the coxa."""
    name = f'{joint}_{leg}_{side}'
    q = m.jnt_qposadr[m.joint(name).id]
    foot, base = m.body(f'claw_{leg}_{side}').id, m.body(f'coxa_{leg}_{side}').id
    out = []
    for v in (-0.1, 0.1):
        mujoco.mj_resetData(m, d)
        d.qpos[q] = v
        mujoco.mj_forward(m, d)
        out.append((d.xpos[foot].copy(), np.linalg.norm(d.xpos[foot] - d.xpos[base])))
    (p0, r0), (p1, r1) = out
    change = {'coxa': p1[0] - p0[0], 'femur': p0[2] - p1[2], 'tibia': r1 - r0, 'tarsus': p1[2] - p0[2]}[joint]
    return 1 if change > 0 else -1


drive = []
for leg in LEGS:
    for side in ('left', 'right'):
        for joint, (pos, neg) in GROUPS.items():
            name = f'{joint}_{leg}_{side}'
            a = m.actuator(name).id
            lo, hi = m.actuator_ctrlrange[a]
            sgn = direction(joint, leg, side)
            up, down = (hi, lo) if sgn > 0 else (lo, hi)
            drive.append((name, a, up, down, members(side, leg, pos), members(side, leg, neg), m.jnt_qposadr[m.joint(name).id]))
            print(f'{name}: + {len(drive[-1][4])} motor neurons, - {len(drive[-1][5])}, positive angle = {"+" if sgn > 0 else "-"} movement')

mujoco.mj_resetData(m, d)
d.qpos[2] = 0.3
mujoco.mj_forward(m, d)
hold_q, hold_v = d.qpos[:7].copy(), np.zeros(6)
steps = int(T / m.opt.timestep)
FEET = [(leg, side) for leg in LEGS for side in ('left', 'right')]
feet = [m.body(f'claw_{leg}_{side}').id for leg, side in FEET]
renderer = mujoco.Renderer(m, 360, 480)
angles, paths, frame = [], [], 0
for k in range(steps):
    t = k * m.opt.timestep
    row = min(int(t / dt), len(rates) - 1)
    for name, a, up, down, pos, neg, q in drive:
        ap = 1 - np.exp(-rates[row, pos].mean() / R_HALF) if pos else 0
        an = 1 - np.exp(-rates[row, neg].mean() / R_HALF) if neg else 0
        d.ctrl[a] = ap * up + an * down
    mujoco.mj_step(m, d)
    d.qpos[:7], d.qvel[:6] = hold_q, hold_v          # tethered
    if k % 10 == 0:
        angles.append([d.qpos[q] for *_, q in drive])
        paths.append(d.xpos[feet].copy())
    if k % 33 == 0:                                       # ~300 frames/s of sim time
        cam = mujoco.MjvCamera()
        cam.lookat[:] = d.xpos[m.body('thorax').id] + [0.02, 0, -0.06]
        cam.distance, cam.azimuth, cam.elevation = 0.55, -70, -10
        renderer.update_scene(d, cam)
        Image.fromarray(renderer.render()).save(f'{out}/f{frame:04d}.png')
        frame += 1
A = np.array(angles)[len(angles) // 4:]
for j, (name, *_rest) in enumerate(drive):
    print(f'{name:18s} range of motion {np.ptp(A[:, j]):.2f} rad (mean {A[:, j].mean():+.2f})')
# per leg: foot loop direction in the body's x-z plane (negative signed area = forward-walking loop:
# the foot moves back while low, forward while high), and each leg's phase of coxa swing
P = np.array(paths)[len(paths) // 4:]          # samples x legs x 3
sdt = 10 * m.opt.timestep
ref = None
for k, (leg, side) in enumerate(FEET):
    x, zz = P[:, k, 0] - P[:, k, 0].mean(), P[:, k, 2] - P[:, k, 2].mean()
    signed = 0.5 * np.sum(x[:-1] * zz[1:] - x[1:] * zz[:-1])
    F = np.fft.rfft(x); fr = np.fft.rfftfreq(len(x), sdt); kk = np.argmax(np.abs(F[1:])) + 1
    if ref is None:
        ref, f0 = F[kk], fr[kk]
    k0 = int(round(f0 * len(x) * sdt))
    ph = np.degrees(np.angle(np.fft.rfft(x)[k0] / ref))
    print(f'{leg} {side:5s}: foot travel {np.ptp(x) * 10:.2f} mm forward, {np.ptp(zz) * 10:.2f} mm up; main rhythm {fr[kk]:.1f} Hz; '
          f'loop {"forward" if signed < 0 else "backward"} ({signed:+.1e}); phase vs T1 left {ph:+.0f} deg')
