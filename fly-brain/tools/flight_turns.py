"""How hard can flybody's flight policy turn? (1) its own training paths (saccades of real flies), (2) generated banked
turns: yaw rate ramps up and down, roll = atan(v w / g) as in a coordinated turn.
    python tools/flight_turns.py <flight policy dir> <flight data dir: hdf5 + wing_pattern_fmech.npy>
"""
import os
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import tensorflow as tf
from flybody.fly_envs import flight_imitation
from flybody.quaternions import mult_quat

src, data = sys.argv[1:3]
r = tf.train.load_checkpoint(src + '/variables/variables')
v = [r.get_tensor(f'_variables/{i}/.ATTRIBUTES/VARIABLE_VALUE') for i in range(12)]
elu = lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
def policy(x):
    h = x @ v[1] + v[0]
    h = np.tanh((h - h.mean()) / np.sqrt(h.var() + 1e-5) * v[3] + v[2])
    for b, w in ((v[4], v[5]), (v[6], v[7])):
        h = elu(h @ w + b)
    return h @ v[9] + v[8]
concat = lambda o: np.concatenate([np.ravel(o[k]) for k in sorted(o)]).astype(np.float32)
DT = 2e-4
def quat_axis(th, ax):
    q = np.zeros(4); q[0] = np.cos(th / 2); q[ax + 1] = np.sin(th / 2); return q
PITCH = quat_axis(np.deg2rad(-47.5), 1)
def path(speed, w_of_t, T, bank=True):
    n = int(T / DT)
    pos, q, th = np.zeros((n, 3)), np.zeros((n, 7)), 0.0
    p = np.array([0, 0, 1.0])
    for k in range(n):
        w = w_of_t(k * DT)
        roll = -np.arctan2(speed * w, 981.0) if bank else 0.0   # a left turn banks left
        q[k, :3] = p
        q[k, 3:] = mult_quat(quat_axis(th, 2), mult_quat(quat_axis(roll, 0), PITCH))
        p = p + DT * speed * np.array([np.cos(th), np.sin(th), 0]); th += w * DT
    qv = np.zeros((n, 6)); qv[:-1, :3] = np.diff(q[:, :3], axis=0) / DT; qv[-1] = qv[-2]
    return q, qv
def ramp(w, t0=0.1, rise=0.03, hold=0.1):
    def f(t):
        if t < t0 or t > t0 + 2 * rise + hold: return 0.0
        if t < t0 + rise: return w * (t - t0) / rise
        if t > t0 + rise + hold: return w * (t0 + 2 * rise + hold - t) / rise
        return w
    return f
env = flight_imitation(wpg_pattern_path=f'{data}/wing_pattern_fmech.npy', terminal_com_dist=float('inf'))
spec = env.action_spec(); lo, hi = spec.minimum, spec.maximum
def roll(T):
    ts = env.reset(); P = []
    while not ts.last() and len(P) < T / DT:
        a = np.clip(policy(concat(ts.observation)), -1, 1)
        ts = env.step(lo + (a + 1) / 2 * (hi - lo))
        P.append(env.physics.named.data.xpos['walker/thorax'].copy())
    return np.array(P)
env.task._time_limit = env._time_limit = 1.2
print('generated turns, 30 cm/s, then level flight (T 1 s); ok = stays within 0.3 cm')
for w in (10, 15, 20, 30):
    for bank in (True,):
        for rise, hold, name in ((0.03, 0.1, 'saccade'), (0.1, 0.8, 'sustained')):
            q, qv = path(30.0, ramp(w, 0.1, rise, hold), 1.3, bank)
            env.task._traj_generator.set_next_trajectory(q, qv)
            P = roll(1.0)
            e = np.linalg.norm(P - q[:len(P), :3], axis=1)
            turned = np.degrees(np.trapz([ramp(w, 0.1, rise, hold)(k * DT) for k in range(len(P))], dx=DT))
            print(f'  w {w:3d} rad/s {"bank" if bank else "flat"} {name:9s} turned {turned:5.0f} deg: flew {len(P) * DT:.2f} s, err max {e.max():.2f} cm, lowest z {P[:, 2].min():.2f} {"ok" if e.max() < 0.3 and len(P) * DT > 0.95 else "FAIL"}', flush=True)
f = [x for x in os.listdir(data) if x.endswith('.hdf5')][0]
env2 = flight_imitation(ref_path=f'{data}/{f}', wpg_pattern_path=f'{data}/wing_pattern_fmech.npy', terminal_com_dist=float('inf'), randomize_start_step=False, random_state=np.random.RandomState(0))
env = env2; spec = env.action_spec(); lo, hi = spec.minimum, spec.maximum
res = []
gen = env.task._traj_generator
n = getattr(gen, 'num_trajectories', 40)
idx = list(range(0, n, max(1, n // 40)))[:40]
print('their paths:', n, 'testing', len(idx))
for i in idx:
    env.task.set_next_trajectory_index(i)
    P = roll(0.6)
    rq = env.task._ref_qpos[:len(P)]
    e = np.linalg.norm(P - rq[:, :3], axis=1)
    h = np.unwrap(2 * np.arctan2(rq[:, 6], rq[:, 3]))
    yawrate = np.abs(np.diff(np.convolve(h, np.ones(50) / 50, 'valid'))) / DT
    res.append((len(P) * DT, e.max(), yawrate.max() if len(yawrate) else 0))
res = np.array(res)
print(f'  flew {res[:, 0].mean():.2f} s on average of {len(idx)}; err max median {np.median(res[:, 1]):.2f} cm; within 0.5 cm: {(res[:, 1] < 0.5).sum()}/{len(res)}; yaw rate in these paths median {np.median(res[:, 2]):.0f}, top {res[:, 2].max():.0f} rad/s (note: heading from quaternion is approximate)')
