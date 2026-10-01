"""Vibration modes of a drum: -Δu = λu inside the outline, u = 0 on the rim.

Linear (P1) finite elements on a triangle mesh. The general mesher lays a
jittered triangular lattice inside the outline, adds points every h along the
rim, and keeps the Delaunay triangles whose centroids fall inside. Drums are
scaled to unit area first, so h means the same thing for every shape, and the
network sees ratios λ_k / λ_1, which don't depend on size or tension.

The browser runs a port of this (drum/fem.js); the two must agree.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import Delaunay

H = 0.025          # mesh spacing for a unit-area drum
K_MODES = 40


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def normalise(p):
    """Counter-clockwise, centroid at the origin, unit area."""
    p = np.asarray(p, float)
    a = polygon_area(p)
    if a < 0:
        p, a = p[::-1], -a
    x, y = p[:, 0], p[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = np.sum((x + np.roll(x, -1)) * cr) / (6 * a)
    cy = np.sum((y + np.roll(y, -1)) * cr) / (6 * a)
    return (p - [cx, cy]) / np.sqrt(a)


def resample_rim(p, h):
    """Points along the outline at most h apart, keeping every vertex."""
    out = []
    n = len(p)
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        L = np.hypot(*(b - a))
        k = max(1, int(np.ceil(L / h - 1e-9)))
        t = np.arange(k) / k
        out.append(a + t[:, None] * (b - a))
    return np.concatenate(out)


def inside(pts, p):
    """Even-odd point-in-polygon test, vectorised over pts."""
    x, y = pts[:, 0:1], pts[:, 1:2]
    ax, ay = p[:, 0][None], p[:, 1][None]
    bx, by = np.roll(p[:, 0], -1)[None], np.roll(p[:, 1], -1)[None]
    cond = (ay > y) != (by > y)
    with np.errstate(divide='ignore', invalid='ignore'):
        xint = (bx - ax) * (y - ay) / (by - ay) + ax
    return (np.sum(cond & (x < xint), axis=1) % 2) == 1


def dist_to_rim(pts, p):
    a = p[None]
    b = np.roll(p, -1, 0)[None]
    q = pts[:, None]
    ab = b - a
    t = np.clip(np.sum((q - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-18), 0, 1)
    d = q - (a + t[..., None] * ab)
    return np.sqrt(np.min(np.sum(d * d, -1), axis=1))


def lattice(p, h, jit=0.12):
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    dy = h * np.sqrt(3) / 2
    j0, j1 = int(np.floor(y0 / dy)) - 1, int(np.ceil(y1 / dy)) + 1
    i0, i1 = int(np.floor(x0 / h)) - 1, int(np.ceil(x1 / h)) + 1
    I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing='ij')
    I, J = I.ravel(), J.ravel()
    s = (I * 73856093) ^ (J * 19349663)
    s = s.astype(np.int64) & 0xFFFFFFFF
    s = ((s ^ (s >> 13)) * 1274126177) & 0xFFFFFFFF
    u = ((s & 0xFFFF) / 65536.0) * 2 - 1
    v = (((s >> 16) & 0xFFFF) / 65536.0) * 2 - 1
    x = I * h + (J & 1) * h / 2 + jit * h * u
    y = J * dy + jit * h * v
    pts = np.stack([x, y], 1)
    pts = pts[inside(pts, p)]
    return pts[dist_to_rim(pts, p) > 0.55 * h]


def mesh(p, h=H):
    """Nodes (rim first), triangles, and the number of rim nodes."""
    rim = resample_rim(p, h)
    inner = lattice(p, h)
    nodes = np.concatenate([rim, inner])
    tri = Delaunay(nodes).simplices
    c = nodes[tri].mean(1)
    tri = tri[inside(c, p)]
    # orient counter-clockwise
    a = nodes[tri[:, 1]] - nodes[tri[:, 0]]
    b = nodes[tri[:, 2]] - nodes[tri[:, 0]]
    flip = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return nodes, tri, len(rim)


def crisscross_mesh(p, n):
    """Structured mesh for polygons whose edges run along the unit grid or its
    diagonals (the isospectral drums): each grid square of side 1/n is split by
    both diagonals. It is symmetric under every reflection of the grid, so the
    solver's spectra of the two drums agree exactly, as the theory says."""
    x0, y0 = np.floor(p.min(0)).astype(int)
    x1, y1 = np.ceil(p.max(0)).astype(int)
    xs = np.arange(x0 * n, x1 * n + 1) / n
    ys = np.arange(y0 * n, y1 * n + 1) / n
    key = {}
    nodes = []

    def node(x, y):
        k = (round(x * 2 * n), round(y * 2 * n))
        if k not in key:
            key[k] = len(nodes)
            nodes.append((x, y))
        return key[k]

    tris = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            xa, xb, ya, yb = xs[i], xs[i + 1], ys[j], ys[j + 1]
            xm, ym = (xa + xb) / 2, (ya + yb) / 2
            corners = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
            for k in range(4):
                (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
                cx, cy = (ax + bx + xm) / 3, (ay + by + ym) / 3
                if inside(np.array([[cx, cy]]), p)[0]:
                    tris.append((node(ax, ay), node(bx, by), node(xm, ym)))
    nodes = np.array(nodes)
    tri = np.array(tris)
    on_rim = dist_to_rim(nodes, p) < 1e-9
    order = np.concatenate([np.flatnonzero(on_rim), np.flatnonzero(~on_rim)])
    remap = np.empty(len(nodes), int)
    remap[order] = np.arange(len(nodes))
    return nodes[order], remap[tri], int(on_rim.sum())


def assemble(nodes, tri):
    """P1 stiffness and consistent mass matrices."""
    P = nodes[tri]                                   # (T, 3, 2)
    x, y = P[..., 0], P[..., 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
    A = 0.5 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    Ke = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4 * A[:, None, None])
    Me = A[:, None, None] / 12 * (np.ones((3, 3)) + np.eye(3))[None]
    rows = np.repeat(tri, 3, axis=1).ravel()
    cols = np.tile(tri, (1, 3)).ravel()
    n = len(nodes)
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    M = sp.coo_matrix((Me.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    return K, M


def modes(nodes, tri, n_rim, k=K_MODES, vectors=False):
    """Lowest k eigenvalues (and optionally mode shapes on all nodes)."""
    K, M = assemble(nodes, tri)
    K, M = K[n_rim:, n_rim:], M[n_rim:, n_rim:]
    vals, vecs = sla.eigsh(K.tocsc(), k=k, M=M.tocsc(), sigma=0, which='LM')
    o = np.argsort(vals)
    if not vectors:
        return vals[o]
    full = np.zeros((len(nodes), k))
    full[n_rim:] = vecs[:, o]
    return vals[o], full


def drum_spectrum(outline, k=K_MODES, h=H):
    p = normalise(outline)
    nodes, tri, nr = mesh(p, h)
    return modes(nodes, tri, nr, k)


# The Gordon–Webb–Wolpert pair, as in Driscoll (1997) and Moler (2012).
GWW1 = np.array([[0, 0], [0, 1], [2, 3], [2, 2], [3, 2], [2, 1], [1, 1], [1, 0]], float)
GWW2 = np.array([[1, 0], [0, 1], [0, 2], [2, 2], [2, 3], [3, 2], [2, 1], [1, 1]], float)
