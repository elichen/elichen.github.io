"""A drawn skin for the worm: a smooth, fine surface that follows the FEM mesh's
outline, and the organs that show through a real worm's cuticle (pharynx with its two
bulbs and grinder, intestine, eggs around the vulva). Every vertex is fixed in a
rest-mesh tetrahedron (barycentric weights, extrapolated a little for points just
outside), so the page moves it with the body as it does the neurons. Physics never
sees it.

Anatomy in fractions of the body length L from the nose (adult hermaphrodite, WormAtlas):
pharynx 0-0.13 L (metacorpus bulb at 0.065, terminal bulb at 0.115), intestine 0.135-0.86 L,
vulva at 0.5 L ventral, anus at 0.88 L ventral.

usage: python export_skin.py <model dir>/body.bin <model dir>/skin.bin

skin.bin, little-endian float32: nv, ntri, then per vertex
  4 vertex indices, 4 weights, s (0 nose .. 1 tail), angle around the body (rad, 0 = dorsal), part
then ntri × 3 vertex indices. part: 0 skin, 1 pharynx, 2 grinder, 3 intestine, 4 egg
"""
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

body = np.fromfile(sys.argv[1])
nv, nt = int(body[0]), int(body[1])
V = body[5:5 + nv * 3].reshape(nv, 3)
T = body[5 + nv * 3:5 + nv * 3 + nt * 14].reshape(nt, 14)[:, :4].astype(int)
z0, z1 = V[:, 2].min(), V[:, 2].max()
L = z1 - z0

# the mesh's outline per slice: centre and half-widths (x = dorsal, y = left)
NS = 80
edges = np.linspace(z0, z1, NS + 1)
zc, cx, cy, hx, hy = [], [], [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    s = V[(V[:, 2] >= a - 1e-9) & (V[:, 2] <= b + 1e-9)]
    zc.append((a + b) / 2)
    cx.append((s[:, 0].max() + s[:, 0].min()) / 2); hx.append((s[:, 0].max() - s[:, 0].min()) / 2)
    cy.append((s[:, 1].max() + s[:, 1].min()) / 2); hy.append((s[:, 1].max() - s[:, 1].min()) / 2)
zc, cx, cy, hx, hy = map(np.array, (zc, cx, cy, hx, hy))
# a worm's cross-section is round: one radius per slice, smoothed; mirror-symmetric centre line
rad = gaussian_filter1d((hx + hy) / 2, 2.0, mode="nearest")
cx = gaussian_filter1d(cx, 3.0, mode="nearest")
cy = gaussian_filter1d(cy, 3.0, mode="nearest")
R0 = rad.max()


def radius(s):
    """body radius at s (0 nose, 1 tail): the mesh's, with a rounded nose and a tail that
    narrows to a fine tip as a real worm's does"""
    r = np.interp(z0 + s * L, zc, rad)
    nose = np.sqrt(np.clip(s / 0.03, 0, 1)) * 0.75 + 0.25 * np.clip(s / 0.03, 0, 1)
    tail = np.clip((1 - s) / 0.14, 0, 1) ** 0.85
    return np.minimum(r, R0 * np.minimum(nose, 1) * np.where(s > 0.86, tail, 1)) * 0.97 + 1e-4 * (s > 0) * (s < 1)


def centre(s):
    z = z0 + s * L
    return np.interp(z, zc, cx), np.interp(z, zc, cy), z


verts, tris, attr = [], [], []


def tube(s0, s1, rfun, part, nring=None, nseg=36, off=(0.0, 0.0), cap=True):
    """a closed tube along the body from s0 to s1 with radius rfun(s) (fraction of nothing:
    absolute units), offset (dorsal, left) in body radii"""
    nring = nring or max(8, int((s1 - s0) * 300))
    ss = np.linspace(s0, s1, nring)
    base = len(verts)
    for s in ss:
        x, y, z = centre(s)
        r = rfun(s)
        for k in range(nseg):
            th = 2 * np.pi * k / nseg
            verts.append((x + off[0] * radius(s) + r * np.cos(th), y + off[1] * radius(s) + r * np.sin(th), z))
            attr.append((s, th, part))
    for i in range(nring - 1):
        for k in range(nseg):
            a, b = base + i * nseg + k, base + i * nseg + (k + 1) % nseg
            c, d = a + nseg, b + nseg
            tris.extend([(a, c, b), (b, c, d)])
    if cap:
        for end, s in ((0, ss[0]), (nring - 1, ss[-1])):
            x, y, z = centre(s)
            ci = len(verts)
            verts.append((x + off[0] * radius(s), y + off[1] * radius(s), z))
            attr.append((s, 0.0, part))
            for k in range(nseg):
                a, b = base + end * nseg + k, base + end * nseg + (k + 1) % nseg
                tris.append((ci, b, a) if end == 0 else (ci, a, b))


def blob(s, dors, left, rl, rw, part, n=14):
    """an ellipsoid (half-length rl along the body, half-width rw) at s, offset in body radii"""
    x, y, z = centre(s)
    x += dors * radius(s)
    y += left * radius(s)
    base = len(verts)
    for i in range(n + 1):
        ph = np.pi * i / n
        for k in range(n):
            th = 2 * np.pi * k / n
            verts.append((x + rw * np.sin(ph) * np.cos(th), y + rw * np.sin(ph) * np.sin(th), z - rl * np.cos(ph)))
            attr.append((s, th, part))
    for i in range(n):
        for k in range(n):
            a, b = base + i * n + k, base + i * n + (k + 1) % n
            tris.extend([(a, a + n, b), (b, a + n, b + n)])


# skin: dense rings so the bends stay smooth
tube(0.0, 1.0, radius, 0, nring=320, nseg=40, cap=True)


def pharynx(s):
    r = radius(s)
    bulb = lambda c, w, h: h * np.exp(-((s - c) / w) ** 2)
    return r * (0.26 + bulb(0.065, 0.012, 0.22) + bulb(0.115, 0.013, 0.32)) * np.clip(s / 0.012, 0.3, 1)


tube(0.004, 0.128, pharynx, 1, nring=90, nseg=24)
blob(0.115, 0.0, 0.0, 0.006 * L, 0.18 * R0, 2, n=10)  # grinder in the terminal bulb


def gut(s):
    taper = np.clip((s - 0.135) / 0.02, 0, 1) * np.clip((0.86 - s) / 0.05, 0, 1)
    return radius(s) * (0.18 + 0.32 * taper)


tube(0.13, 0.875, gut, 3, nring=220, nseg=24, off=(0.1, 0.0))

# eggs in the uterus on both sides of the vulva, beside the gut
rng = np.random.default_rng(7)
for k, s in enumerate(np.linspace(0.40, 0.60, 9)):
    if abs(s - 0.5) < 0.015:
        continue
    side = 1 if k % 2 else -1
    blob(s + rng.uniform(-0.008, 0.008), -0.35 + rng.uniform(-0.1, 0.1), side * 0.32, 0.026 * L, 0.33 * R0, 4, n=12)

verts = np.array(verts)
attr = np.array(attr)
tris = np.array(tris)

# barycentric embedding in the rest mesh
cent = V[T].mean(1)
tree = cKDTree(cent)
_, near = tree.query(verts, k=24)
emb = np.zeros((len(verts), 8))
far = 0
for i, q in enumerate(verts):
    best, bw = None, -1e9
    for j in near[i]:
        p = V[T[j]]
        M = np.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], 1)
        l = np.linalg.solve(M, q - p[0])
        w = np.array([1 - l.sum(), *l])
        if w.min() > bw:
            bw, best = w.min(), (j, w)
    j, w = best
    if bw < -0.6:  # well outside: clamp rather than extrapolate far
        far += 1
        w = np.clip(w, 0, None); w /= w.sum()
    emb[i, :4] = T[j]
    emb[i, 4:] = w
print(f"skin: {len(verts)} vertices, {len(tris)} triangles; clamped {far}; radius max {R0:.4f}, length {L:.4f}")
out = np.concatenate([[len(verts), len(tris)], np.c_[emb, attr].ravel(), tris.ravel()]).astype("<f4")
out.tofile(sys.argv[2])
