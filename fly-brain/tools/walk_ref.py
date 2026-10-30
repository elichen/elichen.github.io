"""Reference for the browser walker: flybody's walk_imitation (mujoco 3.1.6, dm-control 1.0.20) driven by its
trained policy, every control step's observation, action and state, plus the names the port needs.
    python tools/walk_ref.py <walking policy dir> <out dir>
"""
import json
import os
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from flybody.fly_envs import walk_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
r = tf.train.load_checkpoint(src + '/variables/variables')
v = [r.get_tensor(f'_variables/{i}/.ATTRIBUTES/VARIABLE_VALUE').astype(np.float32) for i in range(14)]
# weights as one Float32Array: input layer w, b, LN offset, LN scale, 3 x (w, b), mean w, mean b; matrices are in x out
np.concatenate([x.ravel() for x in (v[1], v[0], v[2], v[3], v[5], v[4], v[7], v[6], v[9], v[8], v[11], v[10])]).tofile(f'{out}/policy.bin')
elu = lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def policy(x):
    h = x @ v[1] + v[0]
    h = np.tanh((h - h.mean()) / np.sqrt(h.var() + 1e-5) * v[3] + v[2])
    for b, w in ((v[4], v[5]), (v[6], v[7]), (v[8], v[9])):
        h = elu(h @ w + b)
    return h @ v[11] + v[10]


env = walk_imitation(terminal_com_dist=float('inf'))
spec = env.action_spec()
lo, hi = spec.minimum, spec.maximum
ts = env.reset()
keys = sorted(ts.observation)
w = env.task._walker
p = env.physics   # replaced on reset
name = lambda e: e.full_identifier
s = w.mjcf_model.sensor
meta = {'keys': keys, 'sizes': [int(np.size(ts.observation[k])) for k in keys],
        'action': spec.name.split('\t'), 'lo': lo.tolist(), 'hi': hi.tolist(),
        'joints': [name(j) for j in w.observable_joints], 'sites': [name(x) for x in w.appendages],
        'force': [name(x) for x in s.force], 'touch': [name(x) for x in s.touch], 'accelerometer': [name(x) for x in s.accelerometer],
        'gyro': [name(x) for x in s.gyro], 'velocimeter': [name(x) for x in s.velocimeter],
        'root': name(w.root_body), 'buffer': w._buffer_size if hasattr(w, '_buffer_size') else None,
        'qpos0': p.data.qpos.tolist(), 'names_q': [p.model.id2name(i, 'joint') for i in range(p.model.njnt)],
        'names_a': [p.model.id2name(i, 'actuator') for i in range(p.model.nu) if p.model.actuator_actadr[i] >= 0]}
json.dump(meta, open(f'{out}/meta.json', 'w'))
print({k: meta[k] for k in ('sites', 'force', 'touch', 'accelerometer', 'gyro', 'velocimeter', 'root', 'buffer')})
print(meta['action'][:8], len(meta['joints']))
for tag, speed, yaw in (('straight', 2.0, 0.0), ('turn', 2.0, 2.0)):
    q, qv = constant_speed_trajectory(1200, speed, yaw_speed=yaw)
    env.task._traj_generator.set_next_trajectory(q, qv)
    ts = env.reset()
    O, A, Q, V, ACT = [], [], [], [], []
    for k in range(600):
        o = np.concatenate([np.ravel(ts.observation[x]) for x in keys]).astype(np.float32)
        a = np.clip(policy(o), -1, 1)
        p = env.physics
        O.append(o); A.append(a); Q.append(p.data.qpos.copy()); V.append(p.data.qvel.copy()); ACT.append(p.data.act.copy())
        ts = env.step(lo + (a + 1) / 2 * (hi - lo))
    np.savez(f'{out}/{tag}.npz', obs=np.array(O), action=np.array(A), qpos=np.array(Q), qvel=np.array(V), act=np.array(ACT), ref_q=q, ref_v=qv)
    for n, x in (('obs', O), ('action', A), ('qpos', Q), ('qvel', V), ('act', ACT)):
        np.array(x, dtype=np.float64).tofile(f'{out}/{tag}_{n}.bin')
    print(tag, 'end', Q[-1][:3].round(3), 'ref', q[599, :3].round(3))
