"""Pack flybody's free-walking fly for the web: walker/fly.xml, walker/meshes/*.stl, walker/manifest.json,
walker/meta.json (names for walker.js), walker/policy.bin (the trained policy, from figshare 10.25378/janelia.25309105).

    python tools/pack_walker.py <walking policy dir> <reference dir from tools/walk_ref.py> walker
    (venv: mujoco 3.1.6, dm-control 1.0.20, flybody, tensorflow, fast-simplification)

Exported from flybody's own walk_imitation task, so the physics is the policy's, except: the mouth joints and
actuators stay (flybody removes them for walking; the brain drives them, the policy doesn't see them), and the
ghost fly, cameras, lights and path markers go. Visual meshes are decimated; contacts use flybody's primitives.
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
from dm_control import mjcf
from flybody.fly_envs import walk_imitation
from flybody.fruitfly import fruitfly

sys.path.insert(0, os.path.dirname(__file__))
KEEP = 0.12
MIN_FACES = 600
policy, ref, out = sys.argv[1:4]


def read_obj(path):
    verts, faces = [], []
    for line in open(path):
        if line.startswith('v '):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(p.split('/')[0]) - 1 for p in line.split()[1:]]
            faces += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    return np.array(verts), np.array(faces)


def weld(verts, faces):
    uniq, inv = np.unique(np.round(verts, 7), axis=0, return_inverse=True)
    faces = inv.reshape(-1)[faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return uniq, faces[keep]


def write_stl(path, verts, faces):
    rec = np.zeros(len(faces), np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['v'] = verts[faces].astype(np.float32)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80 + np.uint32(len(faces)).tobytes() + rec.tobytes())


build = fruitfly.FruitFly._build
def with_mouth(self, name='walker', use_mouth=False, **kw):
    return build(self, name=name, use_mouth=name == 'walker' or use_mouth, **kw)
fruitfly.FruitFly._build = with_mouth

env = walk_imitation(terminal_com_dist=float('inf'))
env.reset()
tmp = tempfile.mkdtemp()
mjcf.export_with_assets(env.task.root_entity.mjcf_model, tmp, out_file_name='scene.xml')
tree = ET.parse(f'{tmp}/scene.xml')
root = tree.getroot()
parent = {c: p for p in root.iter() for c in p}
ghost = lambda el: any((el.get(a) or '').startswith('ghost/') for a in ('name', 'class', 'body1', 'body2', 'childclass', 'joint', 'site', 'mesh', 'material', 'target', 'objname', 'joint1'))
for el in list(root.iter()):
    if el not in parent:
        continue
    if ghost(el) or el.tag in ('light', 'camera') or (el.tag == 'site' and (el.get('name') or '').startswith('traj')) or el.tag == 'key':
        if el in parent and any(el in list(p) for p in [parent[el]]):
            parent[el].remove(el)
size = root.find('size')
if size is not None:
    size.attrib.pop('nkey', None)
os.makedirs(f'{out}/meshes', exist_ok=True)
asset = root.find('asset')
total = 0
for mesh in asset.findall('mesh'):
    v, f = weld(*read_obj(f'{tmp}/{mesh.get("file")}'))
    target = max(MIN_FACES, int(len(f) * KEEP))
    if len(f) > target:
        v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), 1 - target / len(f))
    name = mesh.get('name').replace('walker/', '') + '.stl'
    write_stl(f'{out}/meshes/{name}', np.asarray(v), np.asarray(f))
    mesh.set('file', name)
    total += len(f)
comp = root.find('compiler')
comp.set('meshdir', 'meshes')
comp.attrib.pop('assetdir', None)
# the drop: a dome on the floor (half an ellipsoid), moved by the page (mocap); liquid: nothing collides with it
ET.SubElement(asset, 'material', name='drop', rgba='0.6 0.8 1 0.45')
wick = ET.SubElement(root.find('worldbody'), 'body', name='drop', mocap='true', pos='4 0 0')
ET.SubElement(wick, 'geom', name='drop', type='ellipsoid', size='0.2 0.2 0.16', material='drop', contype='0', conaffinity='0', group='1')
tree.write(f'{out}/fly.xml')
json.dump(sorted(os.listdir(f'{out}/meshes')), open(f'{out}/manifest.json', 'w'))
shutil.copy(f'{ref}/meta.json', f'{out}/meta.json')
shutil.copy(f'{ref}/policy.bin', f'{out}/policy.bin')
shutil.rmtree(tmp)

# the packed model must weigh the same as the one the policy was trained on
import mujoco
a = env.physics.model.ptr
b = mujoco.MjModel.from_xml_path(f'{out}/fly.xml')
for nm in ('thorax', 'head', 'coxa_T1_left', 'tibia_T2_right', 'rostrum'):
    ia, ib = a.body(f'walker/{nm}').id, b.body(f'walker/{nm}').id
    print(f'{nm}: mass {a.body_mass[ia]:.3e} vs {b.body_mass[ib]:.3e}, inertia max diff {np.abs(a.body_inertia[ia] / b.body_inertia[ib] - 1).max():.1e}')
print(f'{len(asset.findall("mesh"))} meshes, {total} faces; nq {b.nq} nu {b.nu} na {b.na} ngeom {b.ngeom} dt {b.opt.timestep}')
print('mouth actuators', [b.actuator(i).name for i in range(b.nu) if any(x in b.actuator(i).name for x in ('rostrum', 'haustellum', 'labrum'))])
