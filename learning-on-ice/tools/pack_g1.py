"""Pack MuJoCo Menagerie's Unitree G1 for the web.

Collision geoms get their own convex-hull meshes: MuJoCo collides with a mesh's
convex hull anyway, so physics is unchanged while the files shrink to a few KB.
Visual geoms (no mass, no collisions) get meshes decimated to ~10% of their faces.

    python tools/pack_g1.py <menagerie>/unitree_g1 robot/   (needs numpy, scipy, fast-simplification)
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

VISUAL_KEEP = 0.10


def read_stl(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    if 84 + 50 * n == len(data):  # binary
        tri = np.frombuffer(data[84:], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n)
        verts = tri['v'].reshape(-1, 3).astype(np.float64)
    else:  # ASCII
        verts = np.array(re.findall(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)', data), dtype=np.float64)
    # Weld duplicate corners so faces share vertices
    uniq, inverse = np.unique(np.round(verts, 7), axis=0, return_inverse=True)
    return uniq, inverse.reshape(-1, 3)


def write_stl(path, verts, faces):
    tris = verts[faces].astype(np.float32)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(faces), dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['n'], rec['v'] = normals, tris
    with open(path, 'wb') as f:
        f.write(b'packed by tools/pack_g1.py'.ljust(80, b' '))
        f.write(struct.pack('<I', len(faces)))
        f.write(rec.tobytes())


def main(src, out):
    xml = open(os.path.join(src, 'g1.xml')).read()
    os.makedirs(os.path.join(out, 'meshes'), exist_ok=True)
    # <mesh file="x.STL"/> is named "x"; some declarations give an explicit name
    files = {}
    for attrs in re.findall(r'<mesh\s+([^>]*?)\s*/>', xml):
        file = re.search(r'file="([^"]+)"', attrs).group(1)
        name = re.search(r'name="([^"]+)"', attrs)
        files[name.group(1) if name else os.path.splitext(file)[0]] = file

    def geoms(cls):
        return set(re.findall(rf'<geom class="{cls}"[^>]*mesh="([^"]+)"', xml))

    # Everything that isn't a collision geom is drawn only (some, like the logo, have no class)
    collision = geoms('collision')
    visual = {m for attrs in re.findall(r'<geom\s([^>]*mesh="[^"]+"[^>]*)', xml)
              if 'class="collision"' not in attrs for m in re.findall(r'mesh="([^"]+)"', attrs)}
    total = 0
    for name in sorted(visual):
        v, f = read_stl(os.path.join(src, 'assets', files[name]))
        if len(f) > 400:
            v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1 - VISUAL_KEEP)
        write_stl(os.path.join(out, 'meshes', f'{name}.stl'), np.asarray(v), np.asarray(f))
        total += os.path.getsize(os.path.join(out, 'meshes', f'{name}.stl'))
    for name in sorted(collision):
        v, _ = read_stl(os.path.join(src, 'assets', files[name]))
        hull = ConvexHull(v)
        write_stl(os.path.join(out, 'meshes', f'{name}_hull.stl'), v, hull.simplices)
        total += os.path.getsize(os.path.join(out, 'meshes', f'{name}_hull.stl'))

    # Point visual geoms at the decimated meshes and collision geoms at the hulls
    xml = re.sub(r'(<geom class="collision"[^>]*mesh=")([^"]+)"', r'\1\2_hull"', xml)
    mesh_decls = ''.join(f'    <mesh name="{n}" file="{n}.stl"/>\n' for n in sorted(visual)) + \
        ''.join(f'    <mesh name="{n}_hull" file="{n}_hull.stl"/>\n' for n in sorted(collision))
    xml = re.sub(r'\s*<mesh\s+[^>]*?file="[^"]+"[^>]*/>', '', xml)
    xml = xml.replace('<asset>', '<asset>\n' + mesh_decls.rstrip('\n'), 1)
    xml = xml.replace('meshdir="assets"', 'meshdir="meshes"')
    header = '<!-- Unitree G1 from MuJoCo Menagerie (BSD-3-Clause, see LICENSE), meshes packed by tools/pack_g1.py -->\n'
    open(os.path.join(out, 'g1.xml'), 'w').write(header + xml)
    shutil.copy(os.path.join(src, 'scene.xml'), os.path.join(out, 'scene.xml'))
    shutil.copy(os.path.join(src, 'LICENSE'), os.path.join(out, 'LICENSE'))
    # File list for the page, which fetches everything into MuJoCo's virtual file system
    names = ['scene.xml', 'g1.xml'] + sorted(f'meshes/{f}' for f in os.listdir(os.path.join(out, 'meshes')))
    json.dump({'files': names}, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print(f'{len(visual)} visual meshes, {len(collision)} collision hulls, {total / 1e6:.2f} MB')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
