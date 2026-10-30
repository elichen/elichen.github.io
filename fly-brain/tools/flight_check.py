"""flybody's trained flight policy (Vaxenburg et al. 2025) rebuilt from its raw weights in numpy, on flybody's own
flight_imitation task with its wing pattern and generated paths: does it hold a path, how much does it cost?
    python tools/flight_check.py <trained policy dir: flight/> <wing_pattern_fmech.npy> [scene dir to export]
"""
import os
import sys
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from flybody.fly_envs import flight_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

r = tf.train.load_checkpoint(sys.argv[1] + '/variables/variables')
v = [r.get_tensor(f'_variables/{i}/.ATTRIBUTES/VARIABLE_VALUE') for i in range(12)]
elu = lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def policy(x):
    h = x @ v[1] + v[0]
    h = np.tanh((h - h.mean()) / np.sqrt(h.var() + 1e-5) * v[3] + v[2])
    for b, w in ((v[4], v[5]), (v[6], v[7])):
        h = elu(h @ w + b)
    return h @ v[9] + v[8]


env = flight_imitation(wpg_pattern_path=sys.argv[2], terminal_com_dist=float('inf'))
spec = env.action_spec()
lo, hi = spec.minimum, spec.maximum
ts = env.reset()
print({k: np.shape(x) for k, x in ts.observation.items()})
print('obs', sum(np.size(x) for x in ts.observation.values()), 'actions', spec.shape, spec.name.replace('\t', ' '), 'control dt', env.control_timestep(), 'physics dt', env.physics.timestep())
concat = lambda o: np.concatenate([np.ravel(o[k]) for k in sorted(o)]).astype(np.float32)
LONG = float(os.environ.get('LONG', 0.6))   # flybody's episodes are 0.6 s; how long does it stay up?
env.task._time_limit = env._time_limit = LONG
cases = ((30.0, 0.0, 1.0), (30.0, 2.0, 1.0), (30.0, 5.0, 1.0), (30.0, 10.0, 1.0), (10.0, 0.0, 1.0), (0.0, 0.0, 1.0), (45.0, 0.0, 1.0))
for speed, yaw, z in cases:
    q, qv = constant_speed_trajectory(int(LONG / env.control_timestep()) + 100, speed, yaw_speed=yaw, init_pos=(0, 0, z), body_rot_angle_y=-47.5, control_timestep=env.control_timestep())
    env.task._traj_generator.set_next_trajectory(q, qv)
    ts = env.reset()
    P, t0 = [], time.time()
    while not ts.last():
        a = np.clip(policy(concat(ts.observation)), -1, 1)
        ts = env.step(lo + (a + 1) / 2 * (hi - lo))
        P.append(env.physics.named.data.xpos['walker/thorax'].copy())
    P = np.array(P)
    err = np.linalg.norm(P - q[:len(P), :3], axis=1)
    sim = len(P) * env.control_timestep()
    print(f'speed {speed} cm/s yaw {yaw} rad/s: {sim:.2f} s sim in {time.time() - t0:.0f} s; path {np.linalg.norm(np.diff(P, axis=0), axis=1).sum():.2f} cm (asked {speed * sim:.2f}); error mean {err.mean():.3f} max {err.max():.3f} cm; z {P[:, 2].min():.2f}-{P[:, 2].max():.2f}', flush=True)
if len(sys.argv) > 3:
    from dm_control import mjcf
    mjcf.export_with_assets(env.task.root_entity.mjcf_model, sys.argv[3], out_file_name='scene.xml')
