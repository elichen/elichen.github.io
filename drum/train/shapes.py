"""Drum outlines to learn from: simple (non-self-intersecting) closed polygons
of N_OUT points, from several families. Each generator returns a (n, 2) array
or None when it produced something unusable."""

import numpy as np

N_OUT = 96


def simple(p):
    """True if the closed polygon p has no self-intersections."""
    a = p
    b = np.roll(p, -1, 0)
    n = len(p)
    d1 = b - a
    # segment i vs segment j
    ax, ay = a[:, None, 0], a[:, None, 1]
    dx, dy = d1[:, None, 0], d1[:, None, 1]
    cx, cy = a[None, :, 0], a[None, :, 1]
    ex, ey = d1[None, :, 0], d1[None, :, 1]
    den = dx * ey - dy * ex
    with np.errstate(divide='ignore', invalid='ignore'):
        t = ((cx - ax) * ey - (cy - ay) * ex) / den
        u = ((cx - ax) * dy - (cy - ay) * dx) / den
    hit = (t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)
    i, j = np.triu_indices(n, 2)
    keep = ~((i == 0) & (j == n - 1))
    return not np.any(hit[i[keep], j[keep]])


def resample(p, n=N_OUT):
    closed = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])], 1)


def area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def perimeter(p):
    return np.sum(np.linalg.norm(np.roll(p, -1, 0) - p, axis=1))


def usable(p):
    """Simple, and not so thin that a membrane of it is mostly rim."""
    if p is None or not np.all(np.isfinite(p)):
        return False
    a = abs(area(p))
    if a <= 0:
        return False
    iso = 4 * np.pi * a / perimeter(p) ** 2
    return iso > 0.18 and simple(p)


def smooth(p, passes=2):
    for _ in range(passes):
        p = (np.roll(p, 1, 0) + 2 * p + np.roll(p, -1, 0)) / 4
    return p


def fourier_blob(rng, kmax=8, decay=1.6, amp=0.35):
    """Smooth closed curve from a random complex Fourier series (not always
    star-shaped)."""
    t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    z = np.exp(1j * t)
    for k in range(2, kmax + 1):
        for s in (1, -1):
            c = (rng.normal() + 1j * rng.normal()) * amp / k ** decay
            z = z + c * np.exp(1j * s * k * t)
    # a random stretch makes elongated drums common
    a = np.exp(rng.normal(0, 0.35))
    z = z.real * a + 1j * z.imag / a
    return resample(np.stack([z.real, z.imag], 1))


def star_shaped(rng, kmax=7):
    t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    r = np.ones_like(t)
    for k in range(1, kmax + 1):
        r += rng.normal(0, 0.28 / k ** 1.2) * np.cos(k * t + rng.uniform(0, 2 * np.pi))
    if r.min() < 0.15:
        return None
    return resample(np.stack([r * np.cos(t), r * np.sin(t)], 1))


def polygon(rng):
    kind = rng.integers(0, 4)
    if kind == 0:     # regular polygon
        n = rng.integers(3, 9)
        a = np.arange(n) * 2 * np.pi / n
        p = np.stack([np.cos(a), np.sin(a)], 1)
    elif kind == 1:   # star
        n = rng.integers(4, 8)
        inner = rng.uniform(0.35, 0.75)
        a = np.arange(2 * n) * np.pi / n
        r = np.where(np.arange(2 * n) % 2, inner, 1.0)
        p = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    elif kind == 2:   # random polygon around the origin
        n = rng.integers(3, 11)
        a = np.sort(rng.uniform(0, 2 * np.pi, n))
        r = rng.uniform(0.4, 1.0, n)
        p = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    else:             # rectangle / triangle
        if rng.random() < 0.5:
            w = np.exp(rng.normal(0, 0.5))
            p = np.array([[0, 0], [w, 0], [w, 1], [0, 1]], float)
        else:
            p = np.array([[0, 0], [1, 0], [rng.uniform(-0.5, 1.5), rng.uniform(0.3, 1.5)]])
    th = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return resample(p @ R.T)


def ellipse(rng):
    t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    a = np.exp(rng.uniform(0, 1.2))
    return resample(np.stack([a * np.cos(t), np.sin(t) / a], 1))


def doodle(raw):
    """A closed Quick, Draw! stroke, lightly smoothed."""
    p = resample(smooth(resample(raw, 192), 3))
    return p
