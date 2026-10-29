"""Build BAAIWorm's body the way Metaworm/sim does (Worm::Initialize, Muscle::Muscle,
Muscle::Initialize) and write body.bin for the Rust projective-dynamics solver.

usage: python export_body.py <BAAIWorm checkout> <model dir>/body.bin
(also writes body_muscles.bin: each segment's start and end, for drawing)

body.bin, little-endian float64:
  [nv, ntet, nmus, ntri, nsamp,
   nv*3 rest positions (scaled by worm.meta's 0.004),
   ntet * (4 indices, vol, 9 invDm row-major),
   nmus * (segment 0..95, 4 indices, vol, 9 invDm, 3 fiber, weight),
   ntri * 3 indices (outward),
   nsamp sampling indices, end effectors 2, local contour 3]
"""
import sys, numpy as np

D = sys.argv[1] + '/Metaworm/data/'
SCALE = 0.004

def read_mesh(path):
    tok = open(path).read().split()
    i = tok.index('Vertices') + 1
    nv = int(tok[i]); i += 1
    V = np.array(tok[i:i + 4 * nv], float).reshape(nv, 4)[:, :3]; i += 4 * nv
    i = tok.index('Tetrahedra') + 1
    nt = int(tok[i]); i += 1
    T = np.array(tok[i:i + 5 * nt], int).reshape(nt, 5)[:, :4] - 1
    return V * SCALE, T

V, T = read_mesh(D + 'worm_mesh_4.mesh')

def oriented(t):
    i0, i1, i2, i3 = t
    Dm = np.stack([V[i1] - V[i0], V[i2] - V[i0], V[i3] - V[i0]], 1)
    if np.linalg.det(Dm) < 0:
        i1, i2 = i2, i1
        Dm = np.stack([V[i1] - V[i0], V[i2] - V[i0], V[i3] - V[i0]], 1)
    return (i0, i1, i2, i3), Dm

tets = []
tris = []
for t in T:
    idx, Dm = oriented(t)
    tets.append((idx, np.linalg.det(Dm) / 6.0, np.linalg.inv(Dm)))
    s = sorted(idx)
    tris += [(s[0], s[1], s[2]), (s[0], s[1], s[3]), (s[0], s[2], s[3]), (s[1], s[2], s[3])]
from collections import Counter
cnt = Counter(tris)
contours = []
for k, tri in enumerate(tris):
    if cnt[tri] != 1:
        continue
    t = T[k // 4]  # original vertex order, as in Worm::Initialize
    center = V[t].mean(0)
    q0, q1, q2 = V[tri[0]], V[tri[1]], V[tri[2]]
    n = np.cross(q1 - q0, q2 - q1)
    tri = list(tri)
    if np.dot(center - q0, n) > 0.0:
        tri[0], tri[1] = tri[1], tri[0]
    contours.append(tri)

# muscles: polylines from the obj, each merged to 24 segments
def read_muscles(path):
    out, pts, new = [], [], False
    for line in open(path):
        w = line.split()
        if not w:
            continue
        if w[0] == 'o':
            new = True
        elif w[0] == 'v':
            pts.append(np.array([float(x) for x in w[1:4]]) * SCALE)
        elif w[0] == 'l' and new:
            out.append(pts); pts = []; new = False
    return out

def merge(points, n=24):
    segs = [[points[i], points[i + 1], np.linalg.norm(points[i] - points[i + 1])] for i in range(len(points) - 1)]
    while len(segs) > n:
        k = min(range(len(segs)), key=lambda j: segs[j][2])  # first minimum, like std::min_element
        m = segs[k]
        if k == 0:
            segs[1][0] = m[0]; segs[1][2] += m[2]
        elif k == len(segs) - 1:
            segs[k - 1][1] = m[1]; segs[k - 1][2] += m[2]
        elif segs[k - 1][2] < segs[k + 1][2]:
            segs[k - 1][1] = m[1]; segs[k - 1][2] += m[2]
        else:
            segs[k + 1][0] = m[0]; segs[k + 1][2] += m[2]
        del segs[k]
    return [(s[0], s[1]) for s in reversed(segs)]  # head to tail

def vol6(p0, p1, p2, p3):
    return np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0

def line_tri(p0, p1, p2, s, e):
    if vol6(s, p0, p1, p2) * vol6(e, p0, p1, p2) < 0.0:
        a, b, c = vol6(s, e, p0, p1), vol6(s, e, p1, p2), vol6(s, e, p2, p0)
        if a > 0 and b > 0 and c > 0:
            return True
        if a < 0 and b < 0 and c < 0:
            return True
    return False

def in_tet(p0, p1, p2, p3, s, e):
    return line_tri(p0, p1, p2, s, e) or line_tri(p0, p1, p3, s, e) or line_tri(p0, p2, p3, s, e) or line_tri(p1, p2, p3, s, e)

def det_pos(p0, p1, p2, p3):
    Dm = np.stack([p1 - p0, p2 - p0, p3 - p0], 1)
    d = np.linalg.det(Dm)
    if d < 0.0:  # GetDm swaps the first two columns
        Dm = np.stack([p2 - p0, p1 - p0, p3 - p0], 1)
        d = np.linalg.det(Dm)
    return d

muscles = read_muscles(D + 'worm_nerve_cord_highResCord.obj')
assert len(muscles) == 4, len(muscles)
mus = []
seg_ends = []
oriented_tets = [oriented(t) for t in T]
for mi, pts in enumerate(muscles):
    for si, (start, end) in enumerate(merge(pts)):
        seg = mi * 24 + si
        seg_ends.append((start, end))
        f = (end - start) / np.linalg.norm(end - start)
        L = np.linalg.norm(end - start)
        for idx, Dm in oriented_tets:
            p = [V[i] for i in idx]
            Dd = np.linalg.det(Dm)
            inv = np.linalg.inv(Dm)
            center = (p[0] + p[1] + p[2] + p[3]) / 4.0
            t = np.dot(center - start, f)
            q = start + t * f
            dist = np.linalg.norm(q - center) if max(np.linalg.norm(q - start), np.linalg.norm(q - end)) / L < 1.0 else 1e5
            dist = min(dist, np.linalg.norm(center - start), np.linalg.norm(center - end))
            w = np.exp(-abs(dist) / 0.005)
            add = lambda: mus.append((seg, idx, Dd / 6.0, inv, f, w))
            if in_tet(*p, start, end):
                add()
            elif w > 0.2:
                add()
            S = sum(det_pos(start, *[p[(k + j) % 4] for j in (1, 2, 3)]) for k in range(4)) / Dd
            E = sum(det_pos(end, *[p[(k + j) % 4] for j in (1, 2, 3)]) for k in range(4)) / Dd
            if abs(S - 1.0) < 1e-4:
                add()
            if abs(E - 1.0) < 1e-4:
                add()

samp = [int(x) for x in open(D + 'worm.sampling').read().split()]
meta = open(D + 'worm.meta').read().split('\n')
ends = [int(x) for x in meta[4].split()]
contour = [int(x) for x in meta[5].split()]
print('vertices', len(V), 'tets', len(tets), 'muscle constraints', len(mus), 'surface tris', len(contours),
      'sampling', len(samp), 'ends', ends, 'contour', contour)
print('head vertex', contour[0], V[contour[0]], 'z range', V[:, 2].min(), V[:, 2].max())
print('segment 0 start/end z', seg_ends[0][0][2], seg_ends[0][1][2], 'segment 23', seg_ends[23][0][2])
cps = Counter(m[0] for m in mus); print('constraints per segment min/max', min(cps.values()), max(cps.values()), len(cps))

out = [len(V), len(tets), len(mus), len(contours), len(samp)]
out += list(V.ravel())
for idx, vol, inv in tets:
    out += list(idx) + [vol] + list(inv.ravel())
for seg, idx, vol, inv, f, w in mus:
    out += [seg] + list(idx) + [vol] + list(inv.ravel()) + list(f) + [w]
for tri in contours:
    out += tri
out += samp + ends + contour
np.array(out, '<f8').tofile(sys.argv[2])
# muscle segment end points for drawing
np.array([np.r_[s, e] for s, e in seg_ends], '<f8').tofile(sys.argv[2].replace('.bin', '_muscles.bin'))
