"""Pack the Franka Panda scene from MuJoCo Playground (PandaPickCube) for the web.

The XMLs are copied as they are, so the physics matches training. Only the meshes
change, and none of them touch anything: in this scene the box only collides with
the floor and the two fingertip pads, which are boxes, and the hand capsule only
with the floor. So the collision meshes (contype 0) become convex hulls, and the
visual meshes are decimated. All become binary STL, ~35 MB of OBJ -> ~2 MB.

    python tools/pack_panda.py <dir with mjx_*.xml, sensor.xml, assets/> robot/
    (needs numpy, scipy, fast-simplification)

The source directory is Playground's franka_emika_panda/xmls plus Menagerie's
franka_emika_panda (mjx_panda.xml, assets/), which Playground ships in
mujoco_playground/external_deps/mujoco_menagerie.
"""
import json
import os
import re
import shutil
import struct
import sys

import fast_simplification
import numpy as np
from scipy.spatial import ConvexHull

VISUAL_KEEP = 0.2      # fraction of faces kept on visual meshes
MIN_FACES = 400        # small parts keep at least this many


def read_mesh(path):
    if path.endswith('.obj'):
        verts, faces = [], []
        for line in open(path):
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                idx = [int(p.split('/')[0]) - 1 for p in line.split()[1:]]
                faces += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
        return np.array(verts), np.array(faces)
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    if 84 + 50 * n == len(data):  # binary STL
        tri = np.frombuffer(data[84:], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n)
        verts = tri['v'].reshape(-1, 3).astype(np.float64)
    else:
        verts = np.array(re.findall(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)', data), dtype=np.float64)
    return verts, np.arange(len(verts)).reshape(-1, 3)


def weld(verts, faces):
    """Merge duplicate vertices (OBJ exports repeat them per face) so decimation can collapse edges."""
    uniq, inv = np.unique(np.round(verts, 7), axis=0, return_inverse=True)
    faces = inv.reshape(-1)[faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return uniq, faces[keep]


def write_stl(path, verts, faces):
    tri = verts[faces].astype(np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(faces), dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['n'], rec['v'] = normals, tri
    with open(path, 'wb') as f:
        f.write(b'\0' * 80 + struct.pack('<I', len(faces)) + rec.tobytes())


# link1.stl (collision) and link1.obj (visual) would collide, so hulls get their own names
def packed_name(f, hull):
    return f.rsplit('.', 1)[0] + ('_hull.stl' if hull else '.stl')


def main(src, out):
    shutil.rmtree(os.path.join(out, 'meshes'), ignore_errors=True)
    os.makedirs(os.path.join(out, 'meshes'), exist_ok=True)
    panda = open(os.path.join(src, 'mjx_panda.xml')).read()
    collision = {f for name, f in re.findall(r'<mesh name="([^"]+)" file="([^"]+)"/>', panda)}
    # finger_0 is drawn and also used as a (non-colliding) collision geom; treat it as visual
    totals = [0, 0]
    for f in sorted(set(re.findall(r'<mesh (?:name="[^"]+" )?file="([^"]+)"/>', panda))):
        verts, faces = read_mesh(os.path.join(src, 'assets', f))
        verts, faces = weld(verts, faces)
        totals[0] += len(faces)
        if f in collision:
            hull = ConvexHull(verts)
            verts, faces = hull.points[hull.vertices], None
            remap = {v: i for i, v in enumerate(hull.vertices)}
            faces = np.array([[remap[i] for i in s] for s in hull.simplices])
        elif len(faces) > MIN_FACES:
            keep = max(VISUAL_KEEP, MIN_FACES / len(faces))
            verts, faces = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int32), 1 - keep)
        totals[1] += len(faces)
        write_stl(os.path.join(out, 'meshes', packed_name(f, f in collision)), np.asarray(verts, np.float64), np.asarray(faces))
    panda = re.sub(r'<mesh (name="[^"]+" )?file="([^"]+)"/>',
                   lambda m: f'<mesh {m[1] or ""}file="{packed_name(m[2], bool(m[1]))}"/>', panda)
    panda = panda.replace('meshdir="assets"', 'meshdir="meshes"')
    open(os.path.join(out, 'mjx_panda.xml'), 'w').write(panda)
    for name in ['mjx_single_cube.xml', 'mjx_scene.xml', 'sensor.xml']:
        shutil.copy(os.path.join(src, name), os.path.join(out, name))
    shutil.copy(os.path.join(src, 'LICENSE'), os.path.join(out, 'LICENSE'))
    meshes = sorted('meshes/' + m for m in os.listdir(os.path.join(out, 'meshes')))
    manifest = {'scene': 'mjx_single_cube.xml',
                'files': ['mjx_single_cube.xml', 'mjx_scene.xml', 'mjx_panda.xml', 'sensor.xml'] + meshes}
    json.dump(manifest, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in manifest['files'])
    print(f'{len(meshes)} meshes, faces {totals[0]} -> {totals[1]}, {size / 1e6:.1f} MB')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
