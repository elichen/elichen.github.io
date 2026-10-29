"""Place every compartment of the circuit inside the FEM body so the page can draw
the nervous system moving with the worm.

The NeuroML morphology is in OpenWorm's frame (µm; x = left, y = head→tail,
z = dorsal). Each node's position along y maps linearly onto the mesh's long
axis; its offset from the body's midline (estimated per slice from all
compartments' extent) maps onto the mesh cross-section (mesh: x = dorsal,
y = left). Each point is then fixed in the rest-mesh tetrahedron that holds it
(barycentric weights), so it deforms with the body.

usage: python export_neurons.py <geom.json> <model dir>/body.bin <model dir>/neurons.bin

neurons.bin, little-endian float32, per node (brain.bin order):
  4 vertex indices, 4 weights, parent node (-1 = none), cell index
"""
import json, sys, numpy as np
from scipy.spatial import cKDTree

g = json.load(open(sys.argv[1]))
body = np.fromfile(sys.argv[2])
nv, nt = int(body[0]), int(body[1])
V = body[5:5 + nv * 3].reshape(nv, 3)
T = body[5 + nv * 3:5 + nv * 3 + nt * 14].reshape(nt, 14)[:, :4].astype(int)

P, parent, cell = [], [], []
for ci, c in enumerate(g['cells']):
    P += c['pos']
    parent += c['parent']
    cell += [ci] * c['n']
P = np.array(P)
y0, y1 = P[:, 1].min(), P[:, 1].max()
zr0, zr1 = V[:, 2].min(), V[:, 2].max()

# body midline and half-widths in the anatomical frame, per slice of y
bins = np.linspace(y0, y1, 41)
mid_x, mid_z, half_x, half_z = [], [], [], []
for a, b in zip(bins[:-1], bins[1:]):
    s = P[(P[:, 1] >= a) & (P[:, 1] <= b)]
    mid_x.append((s[:, 0].min() + s[:, 0].max()) / 2); half_x.append((s[:, 0].max() - s[:, 0].min()) / 2)
    mid_z.append((s[:, 2].min() + s[:, 2].max()) / 2); half_z.append((s[:, 2].max() - s[:, 2].min()) / 2)
yc = (bins[:-1] + bins[1:]) / 2
sm = lambda a: np.convolve(np.pad(a, 2, mode='edge'), np.ones(5) / 5, mode='valid')
mid_x, mid_z = sm(np.array(mid_x)), sm(np.array(mid_z))
half = sm(np.maximum(np.array(half_x), np.array(half_z)))

# mesh half-widths per slice of z
mz = np.linspace(zr0, zr1, 61)
mhx, mhy, mcx, mcy = [], [], [], []
for a in mz:
    s = V[np.abs(V[:, 2] - a) < (zr1 - zr0) / 60 + 1e-9]
    mcx.append((s[:, 0].min() + s[:, 0].max()) / 2); mhx.append((s[:, 0].max() - s[:, 0].min()) / 2)
    mcy.append((s[:, 1].min() + s[:, 1].max()) / 2); mhy.append((s[:, 1].max() - s[:, 1].min()) / 2)

Q = np.zeros_like(P)
t = (P[:, 1] - y0) / (y1 - y0)
Q[:, 2] = zr0 + t * (zr1 - zr0) * 0.98 + (zr1 - zr0) * 0.01
u_l = (P[:, 0] - np.interp(P[:, 1], yc, mid_x)) / np.interp(P[:, 1], yc, half)   # left
u_d = (P[:, 2] - np.interp(P[:, 1], yc, mid_z)) / np.interp(P[:, 1], yc, half)   # dorsal
inside = 0.8
Q[:, 0] = np.interp(Q[:, 2], mz, mcx) + inside * u_d * np.interp(Q[:, 2], mz, mhx)
Q[:, 1] = np.interp(Q[:, 2], mz, mcy) + inside * u_l * np.interp(Q[:, 2], mz, mhy)

# barycentric embedding in the rest mesh
cent = V[T].mean(1)
tree = cKDTree(cent)
_, near = tree.query(Q, k=24)
out = np.zeros((len(Q), 10), np.float32)
outside = 0
for i, q in enumerate(Q):
    best, bw = None, -1e9
    for j in near[i]:
        p = V[T[j]]
        M = np.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], 1)
        l = np.linalg.solve(M, q - p[0])
        w = np.array([1 - l.sum(), *l])
        if w.min() > bw:
            bw, best = w.min(), (j, w)
    j, w = best
    if bw < -1e-6:
        outside += 1
        w = np.clip(w, 0, None); w /= w.sum()
    out[i, :4] = T[j]; out[i, 4:8] = w; out[i, 8] = parent[i]; out[i, 9] = cell[i]
print('nodes', len(Q), 'outside the mesh (clamped):', outside)
out.astype('<f4').tofile(sys.argv[3])
