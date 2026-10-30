"""Pack flybody's fruit fly (Vaxenburg et al. 2025, github.com/TuragaLab/flybody, Apache-2.0) as a
tethered fly for the web: body/fly.xml, body/meshes/*.stl, body/manifest.json (the mesh files).

    python tools/pack_fly.py <flybody>/flybody/fruitfly/assets body
    (needs numpy, fast-simplification)

Like a fly in a proboscis extension assay, the thorax is held still (no free joint) and only the
proboscis moves: the rostrum, haustellum and labellum joints and their position actuators are
flybody's, unchanged. Every other joint, actuator, tendon and sensor is removed, so the legs,
wings and antennae stay in flybody's rest pose (the brain model has no nerve cord to move them).
Visual meshes are decimated; collision geoms are flybody's primitives. A wick (a capillary with a
drop at its tip, moved by the page) is added as a mocap body; the drop does not collide.
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

import fast_simplification
import numpy as np

KEEP_JOINTS = {'rostrum', 'haustellum', 'haustellum_abduct', 'labrum_left', 'labrum_right'}
KEEP = 0.12          # fraction of faces kept on visual meshes
MIN_FACES = 600


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
    tri = verts[faces].astype(np.float32)
    rec = np.zeros(len(faces), np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['v'] = tri
    with open(path, 'wb') as f:
        f.write(b'\0' * 80 + np.uint32(len(faces)).tobytes() + rec.tobytes())


src, out = sys.argv[1:3]
os.makedirs(os.path.join(out, 'meshes'), exist_ok=True)
tree = ET.parse(os.path.join(src, 'fruitfly.xml'))
root = tree.getroot()
parent = {c: p for p in root.iter() for c in p}

# keep only the proboscis joints; drop the free joint, lights, cameras
for el in list(root.iter()):
    if el.tag == 'freejoint' or (el.tag == 'joint' and el.get('name') not in KEEP_JOINTS and el in parent and parent[el].tag == 'body') \
            or el.tag in ('light', 'camera'):
        parent[el].remove(el)
for tag in ('tendon', 'sensor', 'keyframe'):
    for el in root.findall(tag):
        root.remove(el)
act = root.find('actuator')
for el in list(act):
    if el.tag != 'general' or el.get('joint') not in KEEP_JOINTS:
        act.remove(el)
# MuJoCo skips parent-child contacts only when the parent can move; with the head now welded to
# the world, head and rostrum would collide, which they never do in flybody
ET.SubElement(root.find('contact'), 'exclude', name='head_rostrum', body1='head', body2='rostrum')
size = root.find('size')
if size is not None:
    size.attrib.pop('nkey', None)

# meshes -> decimated binary STL
asset = root.find('asset')
total = 0
for mesh in asset.findall('mesh'):
    v, f = weld(*read_obj(os.path.join(src, mesh.get('file'))))
    target = max(MIN_FACES, int(len(f) * KEEP))
    if len(f) > target:
        v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), 1 - target / len(f))
    name = os.path.splitext(mesh.get('file'))[0] + '.stl'
    write_stl(os.path.join(out, 'meshes', name), np.asarray(v), np.asarray(f))
    mesh.set('file', name)
    total += len(f)
comp = root.find('compiler')
comp.set('meshdir', 'meshes')

# the wick: a glass capillary held by the page (mocap), with a drop at its tip
world = root.find('worldbody')
ET.SubElement(asset, 'material', name='glass', rgba='0.85 0.9 0.95 0.35')
ET.SubElement(asset, 'material', name='drop', rgba='0.6 0.8 1 0.4')
wick = ET.SubElement(world, 'body', name='wick', mocap='true', pos='0.3 0 -0.15')
ET.SubElement(wick, 'geom', name='capillary', type='capsule', size='0.012', fromto='0.025 0 0  0.5 0 0.15',
              material='glass', contype='1', conaffinity='1', condim='1', group='1')
ET.SubElement(wick, 'geom', name='drop', type='sphere', size='0.018', material='drop', contype='0', conaffinity='0', group='1')
tree.write(os.path.join(out, 'fly.xml'))
json.dump(sorted(os.listdir(os.path.join(out, 'meshes'))), open(os.path.join(out, 'manifest.json'), 'w'))
print(f'{len(asset.findall("mesh"))} meshes, {total} faces')
