"""Pack flybody's flying fly for the web: <dir>/{fly.xml, meshes/, manifest.json, meta.json, policy.bin} and
reference rollouts in <ref dir> for tools/flight_parity.mjs.
    python tools/pack_flyer.py <flight policy dir> <wing_pattern_fmech.npy> <dir> <ref dir>
    (venv: mujoco 3.1.6, dm-control 1.0.20, flybody, tensorflow, fast-simplification)
Exported from flybody's flight_imitation task, so the physics is the policy's, except: the ghost fly, cameras, lights and
path markers go, and a looming object (mocap, no contact) is added. Visual meshes are decimated.
"""
import json
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import fast_simplification
import tensorflow as tf
from dm_control import mjcf
from flybody.fly_envs import flight_imitation
from flybody.tasks.constants import _WING_PARAMS
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

policy_dir, pattern, out, ref = sys.argv[1:5]
# Legs. flybody fixes them in flight at each joint's springref, a tucked pose "estimated by visual inspection of videography"
# (Vaxenburg et al. 2025). Photos of free flight show them hanging lower, so we scale those angles toward the zero pose.
LEG_SCALE = float(os.environ.get('LEG_SCALE', 0.6))  # not 0: flybody reads a 0 springref as unset
from flybody.fruitfly import fruitfly as _ff
_orig = _ff.body_quat_from_springrefs
def _scaled(body):
    joints = body.find_all('joint', immediate_children_only=True)
    saved = []
    for j in joints:
        sr = j.springref if j.springref is not None else j.dclass.joint.springref
        saved.append(j.springref)
        j.springref = (sr or 0) * LEG_SCALE
    q = _orig(body)
    for j, v in zip(joints, saved):
        j.springref = v
    return q
_ff.body_quat_from_springrefs = _scaled
KEEP, MIN_FACES = 0.12, 600
os.makedirs(f'{out}/meshes', exist_ok=True)
os.makedirs(ref, exist_ok=True)
r = tf.train.load_checkpoint(policy_dir + '/variables/variables')
v = [r.get_tensor(f'_variables/{i}/.ATTRIBUTES/VARIABLE_VALUE').astype(np.float32) for i in range(12)]
# same order as tools/walk_ref.py: input w, b, LN offset, LN scale, layers (w, b), mean w, mean b
np.concatenate([x.ravel() for x in (v[1], v[0], v[2], v[3], v[5], v[4], v[7], v[6], v[9], v[8])]).tofile(f'{out}/policy.bin')
elu = lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
def policy(x):
    h = x @ v[1] + v[0]
    h = np.tanh((h - h.mean()) / np.sqrt(h.var() + 1e-5) * v[3] + v[2])
    for b, w in ((v[4], v[5]), (v[6], v[7])):
        h = elu(h @ w + b)
    return h @ v[9] + v[8]
FULL = bool(os.environ.get('FULL'))   # legs free and floor solid: the full-body fly, for training takeoff and landing
if FULL:     # the one body: legs, wings and mouth all free, both of flybody's policies run on it
    _build = _ff.FruitFly._build
    _ff.FruitFly._build = lambda self, name='walker', use_mouth=False, **kw: _build(self, name=name, use_mouth=name == 'walker' or use_mouth, **kw)
env = flight_imitation(wpg_pattern_path=pattern, terminal_com_dist=float('inf'), disable_legs=not FULL, joint_filter=0.01 if FULL else 0.)
spec = env.action_spec(); lo, hi = spec.minimum, spec.maximum
ts = env.reset()
w = env.task._walker
s = w.mjcf_model.sensor
keys = sorted(ts.observation)
name = lambda e: e.full_identifier
tmp = tempfile.mkdtemp()
mjcf.export_with_assets(env.task.root_entity.mjcf_model, tmp, out_file_name='scene.xml')
tree = ET.parse(f'{tmp}/scene.xml')
root = tree.getroot()
parent = {c: p for p in root.iter() for c in p}
ghost = lambda el: any((el.get(a) or '').startswith('ghost/') for a in ('name', 'class', 'body1', 'body2', 'childclass', 'joint', 'site', 'mesh', 'material', 'target', 'objname', 'joint1'))
for el in list(root.iter()):
    if el in parent and (ghost(el) or el.tag in ('light', 'camera', 'key') or (el.tag == 'site' and (el.get('name') or '').startswith(('traj', 'cross')))):
        parent[el].remove(el)
size = root.find('size')
if size is not None:
    size.attrib.pop('nkey', None)
asset = root.find('asset')
total = 0
def read_obj(path):
    verts, faces = [], []
    for line in open(path):
        if line.startswith('v '):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(p.split('/')[0]) - 1 for p in line.split()[1:]]
            faces += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    return np.array(verts), np.array(faces)
for mesh in asset.findall('mesh'):
    a, f = read_obj(f'{tmp}/{mesh.get("file")}')
    uniq, inv = np.unique(np.round(a, 7), axis=0, return_inverse=True)
    f = inv.reshape(-1)[f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    target = max(MIN_FACES, int(len(f) * KEEP))
    if len(f) > target:
        uniq, f = fast_simplification.simplify(uniq.astype(np.float32), f.astype(np.int32), 1 - target / len(f))
    nm = mesh.get('name').replace('walker/', '') + '.stl'
    rec = np.zeros(len(f), np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['v'] = np.asarray(uniq)[np.asarray(f)].astype(np.float32)
    with open(f'{out}/meshes/{nm}', 'wb') as fh:
        fh.write(b'\0' * 80 + np.uint32(len(f)).tobytes() + rec.tobytes())
    mesh.set('file', nm)
    total += len(f)
comp = root.find('compiler')
comp.set('meshdir', 'meshes')
comp.attrib.pop('assetdir', None)
ET.SubElement(asset, 'material', name='looming', rgba='0.06 0.06 0.07 1')
b = ET.SubElement(root.find('worldbody'), 'body', name='looming', mocap='true', pos='0 0 -50')
ET.SubElement(b, 'geom', name='looming', type='sphere', size='1', material='looming', contype='0', conaffinity='0', group='1')
opt = root.find('option')
if opt is None:
    opt = ET.SubElement(root, 'option')
fl = opt.find('flag')
if fl is None:
    fl = ET.SubElement(opt, 'flag')
if FULL:
    for g in root.iter('geom'):
        if g.get('name') == 'groundplane':       # as flybody's Walking task
            g.set('contype', '1'); g.set('conaffinity', '1'); g.set('friction', '0.5')
            g.set('solref', '0.001 1'); g.set('solimp', '0.95 0.99 0.01')
    for el in root.iter():
        if el.get('name') in [f'walker/wing_{x}_{y}' for x in ('yaw', 'roll', 'pitch') for y in ('left', 'right')] and el.tag == 'general':
            el.attrib.pop('dyntype', None); el.attrib.pop('dynprm', None)   # wings are direct, as in flight
    d_ = [x for x in root.iter('default') if x.get('class') == 'walker/adhesion-collision']
    for x in d_:
        for g in x.findall('geom'):
            g.set('friction', '1')     # flybody's walking claw friction
else:
    fl.set('contact', 'disable')    # in the air, nothing touches
tree.write(f'{out}/fly.xml')
shutil.rmtree(tmp)
json.dump(sorted(os.listdir(f'{out}/meshes')), open(f'{out}/manifest.json', 'w'))
pat = np.load(pattern)
p = env.physics
meta = {'keys': keys, 'sizes': [int(np.size(ts.observation[k])) for k in keys], 'action': spec.name.split('\t'), 'lo': lo.tolist(), 'hi': hi.tolist(),
        'joints': [name(j) for j in w.observable_joints], 'accelerometer': name(s.accelerometer[0]), 'gyro': name(s.gyro[0]), 'velocimeter': name(s.velocimeter[0]),
        'wing_joints': [name(j) for j in env.task._wing_joints], 'wing_action': [int(i) for i in env.task._wing_inds_action], 'user_action': int(env.task._user_idx_action),
        'base_freq': float(env.task._wbpg.base_beat_freq), 'rel_range': float(env.task._wbpg.rel_freq_range), 'ctrl_filter': float(env.task._wbpg.ctrl_filter),
        'pattern': pat.astype(float).round(6).tolist(), 'names_q': [p.model.id2name(i, 'joint') for i in range(p.model.njnt)], 'qpos0': p.data.qpos.tolist()}
json.dump(meta, open(f'{out}/meta.json', 'w'))
print(meta['action'], meta['sizes'], {k: meta[k] for k in ('base_freq', 'rel_range', 'ctrl_filter')}, len(meta['joints']))
concat = lambda o: np.concatenate([np.ravel(o[k]) for k in keys]).astype(np.float32)
for tag, speed, yaw in (() if os.environ.get('NOREF') else (('straight', 30.0, 0.0), ('hover', 0.0, 0.0))):
    q, qv = constant_speed_trajectory(1200, speed, yaw_speed=yaw, init_pos=(0, 0, 1), body_rot_angle_y=-47.5, control_timestep=env.control_timestep())
    env.task._traj_generator.set_next_trajectory(q, qv)
    ts = env.reset()
    O, A, Q, V = [], [], [], []
    for k in range(600):
        o = concat(ts.observation)
        a = np.clip(policy(o), -1, 1)
        p = env.physics
        O.append(o); A.append(a); Q.append(p.data.qpos.copy()); V.append(p.data.qvel.copy())
        ts = env.step(lo + (a + 1) / 2 * (hi - lo))
    for n_, x in (('obs', O), ('action', A), ('qpos', Q), ('qvel', V)):
        np.array(x, dtype=np.float64).tofile(f'{ref}/{tag}_{n_}.bin')
    np.array(env.task._ref_qpos, dtype=np.float64).tofile(f'{ref}/{tag}_ref.bin')
    print(tag, 'end', Q[-1][:3].round(3), 'ref', env.task._ref_qpos[599, :3].round(3))
import mujoco
b2 = mujoco.MjModel.from_xml_path(f'{out}/fly.xml')
print(f'{len(asset.findall("mesh"))} meshes, {total} faces; nq {b2.nq} nu {b2.nu} na {b2.na} ngeom {b2.ngeom} dt {b2.opt.timestep}')
