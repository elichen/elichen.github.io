"""Pack a Stanford PLY scan into the compact .mesh format the path tracer loads.

    python3 pack_mesh.py dragon_recon/dragon_vrip.ply ../model/dragon.mesh
    python3 pack_mesh.py bunny/reconstruction/bun_zipper.ply ../model/bunny.mesh --fill-holes
    python3 pack_mesh.py lucy.ply ../model/lucy.mesh --target 1000000 --z-up   # needs fast-simplification
    python3 pack_mesh.py teapot.ply ../model/teapot.mesh --weld

Reads ASCII or binary PLY, OBJ and glTF (.gltf with its .bin). Sources: the Stanford 3D Scanning
Repository, Keenan Crane's model repository, Poly Haven, and three.js's TeapotGeometry.

Layout (whole file gzipped):
  u32 magic 'MSH1', u32 nv, u32 nt, f32 bbox min[3], f32 bbox max[3], u32 index_bytes
  u16 positions[nv*3]   quantized over the bbox, delta + zigzag coded in vertex order
  u8  indices[index_bytes]  varint of (next_new - idx); 0 means "the next new vertex"
Triangles are Morton-sorted and vertices renumbered in first-use order, so both
streams are mostly tiny numbers and gzip well. The mesh is normalized to stand on
y = 0, centred in x and z, one unit tall.
"""
import gzip, struct, sys
from collections import defaultdict
import numpy as np


PLY_TYPES = {'char': 'i1', 'uchar': 'u1', 'short': 'i2', 'ushort': 'u2', 'int': 'i4', 'uint': 'u4',
             'float': 'f4', 'double': 'f8', 'int8': 'i1', 'uint8': 'u1', 'int16': 'i2', 'uint16': 'u2',
             'int32': 'i4', 'uint32': 'u4', 'float32': 'f4', 'float64': 'f8'}


def read_ply(path):
    with open(path, 'rb') as f:
        fmt, element, props = 'ascii', None, {'vertex': [], 'face': []}
        counts = {}
        while True:
            line = f.readline().decode().strip()
            if line.startswith('format'): fmt = line.split()[1]
            elif line.startswith('element'):
                _, element, n = line.split(); counts[element] = int(n); props.setdefault(element, [])
            elif line.startswith('property'): props[element].append(line.split()[1:])
            elif line == 'end_header': break
        nv, nf = counts['vertex'], counts['face']
        if fmt == 'ascii':
            body = f.read().decode().split('\n')
            v = np.array([l.split()[:3] for l in body[:nv]], dtype=np.float64)
            faces = np.array([l.split()[1:4] for l in body[nv:nv + nf]], dtype=np.int64)
            return v, faces
        end = '<' if fmt == 'binary_little_endian' else '>'
        vdt = np.dtype([(p[-1], end + PLY_TYPES[p[0]]) for p in props['vertex']])
        vert = np.frombuffer(f.read(vdt.itemsize * nv), vdt)
        v = np.stack([vert['x'], vert['y'], vert['z']], 1).astype(np.float64)
        fields = []                    # assumes triangles, so the list is a fixed 1 + 3 values
        for p in props['face']:
            if p[0] == 'list': fields += [('n', end + PLY_TYPES[p[1]]), ('i', end + PLY_TYPES[p[2]], 3)]
            else: fields.append((p[-1], end + PLY_TYPES[p[0]]))
        fdt = np.dtype(fields)
        face = np.frombuffer(f.read(fdt.itemsize * nf), fdt)
        assert (face['n'] == 3).all(), 'only triangles are supported'
        return v, face['i'].astype(np.int64)


def read_obj(path):
    v, faces = [], []
    for line in open(path):
        p = line.split()
        if not p: continue
        if p[0] == 'v': v.append([float(x) for x in p[1:4]])
        elif p[0] == 'f':
            idx = [int(x.split('/')[0]) for x in p[1:]]
            idx = [i - 1 if i > 0 else len(v) + i for i in idx]
            for k in range(1, len(idx) - 1): faces.append([idx[0], idx[k], idx[k + 1]])
    return np.array(v, np.float64), np.array(faces, np.int64)


def read_gltf(path):
    import json, os
    g = json.load(open(path))
    buffers = [open(os.path.join(os.path.dirname(path), b['uri']), 'rb').read() for b in g['buffers']]
    def accessor(i):
        a = g['accessors'][i]; bv = g['bufferViews'][a['bufferView']]
        dt = {5126: 'f4', 5125: 'u4', 5123: 'u2', 5121: 'u1'}[a['componentType']]
        n = {'SCALAR': 1, 'VEC3': 3, 'VEC4': 4, 'VEC2': 2}[a['type']]
        off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        stride = bv.get('byteStride', 0)
        item = np.dtype(dt).itemsize * n
        raw = buffers[bv['buffer']]
        if stride and stride != item:
            rows = [np.frombuffer(raw, dt, n, off + k * stride) for k in range(a['count'])]
            return np.array(rows)
        return np.frombuffer(raw, dt, a['count'] * n, off).reshape(-1, n) if n > 1 else np.frombuffer(raw, dt, a['count'], off)
    def matrix(node):
        if 'matrix' in node: return np.array(node['matrix'], np.float64).reshape(4, 4).T
        T = np.eye(4); T[:3, 3] = node.get('translation', [0, 0, 0])
        x, y, z, w = node.get('rotation', [0, 0, 0, 1])
        R = np.eye(4); R[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        S = np.diag(list(node.get('scale', [1, 1, 1])) + [1])
        return T @ R @ S
    verts, faces = [], []
    def visit(i, parent):
        node = g['nodes'][i]; M = parent @ matrix(node)
        if 'mesh' in node:
            for prim in g['meshes'][node['mesh']]['primitives']:
                p = accessor(prim['attributes']['POSITION']).astype(np.float64)
                p = (np.c_[p, np.ones(len(p))] @ M.T)[:, :3]
                idx = accessor(prim['indices']).astype(np.int64).reshape(-1, 3)
                faces.append(idx + sum(len(x) for x in verts)); verts.append(p)
        for c in node.get('children', []): visit(c, M)
    for i in g['scenes'][g.get('scene', 0)]['nodes']: visit(i, np.eye(4))
    return np.concatenate(verts), np.concatenate(faces)


def read_mesh(path):
    if path.endswith('.obj'): return read_obj(path)
    if path.endswith('.gltf'): return read_gltf(path)
    return read_ply(path)


def weld(v, f, tol=1e-5):
    """Merge vertices closer than tol, so patches and seams share normals and glass is closed."""
    key = np.round(v / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return v[first], inverse.reshape(-1)[f]


def fill_holes(v, f):
    """Close each boundary loop with a fan around its centroid, so glass has an inside."""
    count = defaultdict(int)
    for a, b, c in f:
        for e in ((a, b), (b, c), (c, a)):
            count[e] += 1
    nxt = {a: b for (a, b) in count if (b, a) not in count}   # boundary edges, oriented
    loops, seen = [], set()
    for start in nxt:
        if start in seen: continue
        loop, cur = [], start
        while cur not in seen and cur in nxt:
            seen.add(cur); loop.append(cur); cur = nxt[cur]
        if cur == start and len(loop) >= 3: loops.append(loop)
    v, new = list(v), []
    for loop in loops:
        c = len(v)
        v.append(np.mean([v[i] for i in loop], axis=0))
        for i in range(len(loop)):           # reverse of the boundary direction keeps winding consistent
            new.append((loop[(i + 1) % len(loop)], loop[i], c))
    print(f'filled {len(loops)} holes ({sum(map(len, loops))} boundary edges)')
    return np.array(v), np.concatenate([f, np.array(new, dtype=np.int64).reshape(-1, 3)])


def morton3(q):
    q = q.astype(np.uint64)
    def spread(x):
        x = (x | (x << np.uint64(16))) & np.uint64(0x030000FF)
        x = (x | (x << np.uint64(8))) & np.uint64(0x0300F00F)
        x = (x | (x << np.uint64(4))) & np.uint64(0x030C30C3)
        x = (x | (x << np.uint64(2))) & np.uint64(0x09249249)
        return x
    return spread(q[:, 0]) | (spread(q[:, 1]) << np.uint64(1)) | (spread(q[:, 2]) << np.uint64(2))


def varint(vals):
    out = bytearray()
    for x in vals.tolist():
        while x >= 0x80:
            out.append((x & 0x7F) | 0x80)
            x >>= 7
        out.append(x)
    return bytes(out)


def pack(src, dst, holes=False, welded=False, target=None, z_up=False):
    v, f = read_mesh(src)
    if z_up: v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)   # stand it up: z becomes y
    if welded: v, f = weld(v, f)
    if target and len(f) > target:
        import fast_simplification
        v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), 1 - target / len(f))
        v, f = v.astype(np.float64), f.astype(np.int64)
    # drop degenerate triangles
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    f = f[(area > 1e-12) & (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    if holes: v, f = fill_holes(v, f)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    volume = np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6
    if volume < 0: f = f[:, [0, 2, 1]]       # outward-facing winding
    # normalize: sit on y=0, centred in x/z, height 1
    used = np.unique(f)
    lo, hi = v[used].min(0), v[used].max(0)
    v = (v - [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]) / (hi[1] - lo[1])
    # Morton-sort triangles by centroid
    cen = v[f].mean(1)
    cl, ch = cen.min(0), cen.max(0)
    order = np.argsort(morton3(((cen - cl) / (ch - cl + 1e-12) * 1023).astype(np.int64)), kind='stable')
    f = f[order]
    # renumber vertices in first-use order (drops unreferenced ones)
    flat = f.reshape(-1)
    _, first = np.unique(flat, return_index=True)
    used = flat[np.sort(first)]
    remap = np.full(len(v), -1, np.int64)
    remap[used] = np.arange(len(used))
    f = remap[f]
    v = v[used]
    lo, hi = v.min(0), v.max(0)
    q = np.round((v - lo) / (hi - lo) * 65535).astype(np.int64)
    d = np.diff(q, axis=0, prepend=np.zeros((1, 3), np.int64))
    d = ((d + 32768) & 0xFFFF) - 32768       # wrap deltas to int16
    d = ((d << 1) ^ (d >> 15)) & 0xFFFF      # zigzag
    flat = f.reshape(-1)
    seen = np.maximum.accumulate(np.concatenate([[-1], flat[:-1]])) + 1   # next new vertex id
    codes = seen - flat
    assert codes.min() >= 0
    idx = varint(codes)
    head = struct.pack('<4sII6fI', b'MSH1', len(v), len(f), *lo.astype(np.float32), *hi.astype(np.float32), len(idx))
    raw = head + d.astype('<u2').tobytes() + idx
    packed = gzip.compress(raw, 9)
    with open(dst, 'wb') as out:
        out.write(packed)
    print(f'{dst}: {len(v)} verts, {len(f)} tris, signed volume {volume:+.3g}, gz {len(packed) / 1e6:.2f} MB')


if __name__ == '__main__':
    target = int(sys.argv[sys.argv.index('--target') + 1]) if '--target' in sys.argv else None
    pack(sys.argv[1], sys.argv[2], '--fill-holes' in sys.argv, '--weld' in sys.argv, target, '--z-up' in sys.argv)
