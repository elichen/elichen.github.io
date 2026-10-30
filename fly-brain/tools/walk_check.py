"""flybody's trained walking policy (Vaxenburg et al. 2025) rebuilt from its raw weights in numpy, run on
flybody's own walk_imitation task with generated speed + turn paths: does it walk, does it follow?
    python tools/walk_check.py <trained policy dir: walking/> [out.npz]   # venv with dm_control, flybody, tensorflow
"""
import os
import sys
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from flybody.fly_envs import walk_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

r = tf.train.load_checkpoint(sys.argv[1] + '/variables/variables')
v = [r.get_tensor(f'_variables/{i}/.ATTRIBUTES/VARIABLE_VALUE') for i in range(14)]
b0, w0, ln_off, ln_scale = v[0], v[1], v[2], v[3]   # see check below: which of 2, 3 is scale decides LN
layers = [(v[4], v[5]), (v[6], v[7]), (v[8], v[9])]
mean_b, mean_w = v[10], v[11]
elu = lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def policy(x, tanh_mean=False):
    h = x @ w0 + b0
    m, var = h.mean(), h.var()      # sonnet LayerNorm over the last axis, eps 1e-5
    h = (h - m) / np.sqrt(var + 1e-5) * ln_scale + ln_off
    h = np.tanh(h)
    for b, w in layers:
        h = elu(h @ w + b)
    a = h @ mean_w + mean_b
    return np.tanh(a) if tanh_mean else a


env = walk_imitation(terminal_com_dist=float('inf'))
spec = env.action_spec()
lo, hi = spec.minimum, spec.maximum
ts = env.reset()
print({k: np.shape(x) for k, x in ts.observation.items()})
print('obs', sum(np.size(x) for x in ts.observation.values()), 'control dt', env.control_timestep(), 'physics dt', env.physics.timestep())
concat = lambda o: np.concatenate([np.ravel(o[k]) for k in sorted(o)]).astype(np.float32)
for ln in (0, 1):
    if ln:
        ln_off, ln_scale = v[3], v[2]
    for tm in (False, True):
        for speed, yaw in [(2.0, 0.0), (2.0, 2.0), (1.0, 0.0)]:
            q, qv = constant_speed_trajectory(1200, speed, yaw_speed=yaw)
            env.task._traj_generator.set_next_trajectory(q, qv)
            ts = env.reset()
            P, t0 = [], time.time()
            for k in range(1000):
                a = np.clip(policy(concat(ts.observation), tm), -1, 1)
                ts = env.step(lo + (a + 1) / 2 * (hi - lo))
                P.append(env.physics.named.data.xpos['walker/thorax'].copy())
            P = np.array(P)
            err = np.linalg.norm(P[:, :2] - q[:1000, :2], axis=1)
            print(f'ln {ln} tanh {tm} speed {speed} yaw {yaw}: {time.time() - t0:.0f} s wall for 2 s; path {np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum():.2f} cm (asked {speed * 2}), mean error {err.mean():.3f} cm, final {err[-1]:.3f}, thorax z {P[:, 2].min():.3f}-{P[:, 2].max():.3f}', flush=True)
