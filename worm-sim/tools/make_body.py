"""A mirror-symmetric worm body for the solver, in body.bin's format (see export_body.py).

SoftCon's mesh (worm_mesh_4) is irregular and not a mirror image of itself, so even a
perfectly symmetric muscle wave makes it crawl in a curve. This mesh has the same length
and radius profile, but is built from one quadrant of a structured tube (cross-sections
along the body, each a centre vertex, a ring of 8 and a ring of 12) and mirrored across
the dorsal-ventral and the left-right midplanes, so it is exactly symmetric in both.
Muscles: 4 quadrants × 24 segments with BAAIWorm's segment ends along the body, at ±45°
and ±135° (x = dorsal, y = left), 0.78 of the radius out; fibers are picked and weighted
as in Metaworm's Muscle::Initialize (export_body.py), then scaled by --muscle so the gait
bends as much as on SoftCon's mesh.

usage: python make_body.py <model dir> <out body.bin> [--sections 48] [--muscle 1.0]
(reads <model dir>/body.bin for the radius profile and <model dir>/body_muscles.bin for
the segment ends; also writes <out>_muscles.bin)
"""
import argparse
from collections import Counter

import numpy as np
from scipy.ndimage import gaussian_filter1d

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--sections", type=int, default=48)
ap.add_argument("--muscle", type=float, default=1.0)
a = ap.parse_args()

src = np.fromfile(a.model + "/body.bin")
nv0 = int(src[0])
V0 = src[5:5 + nv0 * 3].reshape(nv0, 3)
z0, z1 = V0[:, 2].min(), V0[:, 2].max()
# radius profile of SoftCon's mesh, round and smoothed
edges = np.linspace(z0, z1, 81)
zc, rr = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    s = V0[(V0[:, 2] >= lo - 1e-9) & (V0[:, 2] <= hi + 1e-9)]
    zc.append((lo + hi) / 2)
    rr.append(((s[:, 0].max() - s[:, 0].min()) + (s[:, 1].max() - s[:, 1].min())) / 4)
zc, rr = np.array(zc), gaussian_filter1d(np.array(rr), 2.0, mode="nearest")
R0 = rr.max()


def outline(z):
    return max(np.interp(z, zc, rr), 0.22 * R0)


NS = a.sections
zs = np.linspace(z0, z1, NS)
# each section is as wide as the outline anywhere between it and its neighbours, plus 5%, so the
# mesh (linear between sections) contains the drawn skin (export_skin.py), round nose included
RS = [1.05 * max(outline(z) for z in np.linspace(zs[max(k - 1, 0)], zs[min(k + 1, NS - 1)], 41)) for k in range(NS)]


def radius(z):
    return np.interp(z, zs, RS)


# vertices: (section, ring, index); ring 0 = centre, 1 = inner (8, every 45°), 2 = outer (12, every 30°)
RING = {0: 1, 1: 8, 2: 12}
RFRAC = {0: 0.0, 1: 0.5, 2: 1.0 / np.cos(np.pi / 12)}  # the outer 12-gon's flat sides at the radius
vid, V = {}, []
for k, z in enumerate(zs):
    r = radius(z)
    for ring, n in RING.items():
        for j in range(n):
            th = 2 * np.pi * j / n
            vid[(k, ring, j)] = len(V)
            V.append((RFRAC[ring] * r * np.cos(th), RFRAC[ring] * r * np.sin(th), z))
V = np.array(V)

# one quadrant (0..90°) of each layer, triangulated in 2D: (ring, j) corners
Q1 = [((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (1, 2)),
      ((1, 0), (2, 0), (2, 1)), ((1, 0), (2, 1), (1, 1)), ((1, 1), (2, 1), (2, 2)), ((1, 1), (2, 2), (2, 3)), ((1, 1), (2, 3), (1, 2))]


def mirror(rj, mx, my):
    """(ring, j) under x -> -x (dorsal <-> ventral) and/or y -> -y (left <-> right)"""
    ring, j = rj
    n = RING[ring]
    if ring == 0:
        return rj
    if mx:
        j = (n // 2 - j) % n
    if my:
        j = (-j) % n
    return (ring, j)


def prism_tets(b, t):
    """3 tets of the prism with bottom b = (A, B, C), top t = (D, E, F), split by Dompierre et
    al.'s rule on these (quadrant-local) ids, so neighbouring prisms share their diagonals"""
    v = list(b) + list(t)
    m = int(np.argmin(v))
    # rotate so the smallest vertex is A (bottom, position 0)
    rot = {0: [0, 1, 2, 3, 4, 5], 1: [1, 2, 0, 4, 5, 3], 2: [2, 0, 1, 5, 3, 4],
           3: [3, 5, 4, 0, 2, 1], 4: [4, 3, 5, 1, 0, 2], 5: [5, 4, 3, 2, 1, 0]}[m]
    A, B, C, D, E, F = [v[i] for i in rot]
    if min(B, F) < min(C, E):
        return [(A, B, C, F), (A, B, F, E), (A, E, F, D)]
    return [(A, B, C, E), (A, E, C, F), (A, E, F, D)]


tets = []
for k in range(NS - 1):
    for tri in Q1:
        for mx in (False, True):
            for my in (False, True):
                # split the first quadrant's prism, then mirror its tets (not re-split), so the mesh is an exact mirror image
                b = [vid[(k, *rj)] for rj in tri]
                t = [vid[(k + 1, *rj)] for rj in tri]
                local = prism_tets(b, t)
                inv = {}
                for rj in tri:
                    for kk in (k, k + 1):
                        inv[vid[(kk, *rj)]] = vid[(kk, *mirror(rj, mx, my))]
                tets += [tuple(inv[i] for i in tt) for tt in local]
tets = [tuple(t) for t in {tuple(sorted(t)): t for t in tets}.values()]  # on-plane prisms map to themselves


def oriented(t):
    i0, i1, i2, i3 = t
    Dm = np.stack([V[i1] - V[i0], V[i2] - V[i0], V[i3] - V[i0]], 1)
    if np.linalg.det(Dm) < 0:
        i1, i2 = i2, i1
        Dm = np.stack([V[i1] - V[i0], V[i2] - V[i0], V[i3] - V[i0]], 1)
    return (i0, i1, i2, i3), Dm


ot = [oriented(t) for t in tets]
assert all(np.linalg.det(Dm) > 1e-15 for _, Dm in ot), "degenerate tet"

# surface triangles, outward
faces = []
for idx, _ in ot:
    s = sorted(idx)
    faces += [((s[0], s[1], s[2]), s[3]), ((s[0], s[1], s[3]), s[2]), ((s[0], s[2], s[3]), s[1]), ((s[1], s[2], s[3]), s[0])]
cnt = Counter(f for f, _ in faces)
contours = []
for f, other in faces:
    if cnt[f] != 1:
        continue
    q0, q1, q2 = V[list(f)]
    n = np.cross(q1 - q0, q2 - q0)
    contours.append(list(f) if np.dot(n, q0 - V[other]) > 0 else [f[1], f[0], f[2]])

# muscles: BAAIWorm's segment ends along the body (mean over the four quadrants), at ±45°/±135°
ends = np.fromfile(a.model + "/body_muscles.bin").reshape(96, 6)
zs0 = ends[:, 2].reshape(4, 24).mean(0)
zs1 = ends[:, 5].reshape(4, 24).mean(0)
ANG = {0: -45, 1: -135, 2: 45, 3: 135}  # DR, VR, DL, VL (x dorsal, y left)


def mpoint(z, q):
    r = 0.78 * outline(z)
    th = np.radians(ANG[q])
    return np.array([r * np.cos(th), r * np.sin(th), z])


def vol6(p0, p1, p2, p3):
    return np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0


def line_tri(p0, p1, p2, s, e):
    if vol6(s, p0, p1, p2) * vol6(e, p0, p1, p2) < 0.0:
        x, y, w = vol6(s, e, p0, p1), vol6(s, e, p1, p2), vol6(s, e, p2, p0)
        return (x > 0 and y > 0 and w > 0) or (x < 0 and y < 0 and w < 0)
    return False


def in_tet(p, s, e):
    return line_tri(p[0], p[1], p[2], s, e) or line_tri(p[0], p[1], p[3], s, e) or line_tri(p[0], p[2], p[3], s, e) or line_tri(p[1], p[2], p[3], s, e)


mus, seg_ends = [], []
for q in range(4):
    for si in range(24):
        start, end = mpoint(zs0[si], q), mpoint(zs1[si], q)
        seg_ends.append((start, end))
        L = np.linalg.norm(end - start)
        f = (end - start) / L
        for idx, Dm in ot:
            p = V[list(idx)]
            centre = p.mean(0)
            t = np.dot(centre - start, f)
            qq = start + t * f
            dist = np.linalg.norm(qq - centre) if max(np.linalg.norm(qq - start), np.linalg.norm(qq - end)) / L < 1.0 else 1e5
            dist = min(dist, np.linalg.norm(centre - start), np.linalg.norm(centre - end))
            w = np.exp(-abs(dist) / 0.005)
            if in_tet(p, start, end) or w > 0.2:
                mus.append((q * 24 + si, idx, np.linalg.det(Dm) / 6.0, np.linalg.inv(Dm), f, w * a.muscle))

# the solver's other inputs: 17 sampled points along the axis, the two ends, the nose
axis = [vid[(k, 0, 0)] for k in range(NS)]
samp = [axis[int(round(i * (NS - 1) / 16))] for i in range(17)]
endv = [axis[0], axis[-1]]
contour = [axis[0], vid[(0, 2, 0)], vid[(0, 2, 3)]]
cps = Counter(m[0] for m in mus)
print(f"vertices {len(V)}, tets {len(ot)}, surface tris {len(contours)}, muscle constraints {len(mus)} "
      f"(per segment {min(cps.values())}-{max(cps.values())}), volume {sum(np.linalg.det(Dm) / 6 for _, Dm in ot):.6f}")

out = [len(V), len(ot), len(mus), len(contours), len(samp)]
out += list(V.ravel())
for idx, Dm in ot:
    out += list(idx) + [np.linalg.det(Dm) / 6.0] + list(np.linalg.inv(Dm).ravel())
for seg, idx, vol, inv, f, w in mus:
    out += [seg] + list(idx) + [vol] + list(inv.ravel()) + list(f) + [w]
for tri in contours:
    out += tri
out += samp + endv + contour
np.array(out, "<f8").tofile(a.out)
np.array([np.r_[s, e] for s, e in seg_ends], "<f8").tofile(a.out.replace(".bin", "_muscles.bin"))
