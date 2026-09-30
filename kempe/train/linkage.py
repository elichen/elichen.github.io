"""Planar linkages built from dyads, and curve matching up to similarity.

A mechanism is a list of joints in construction order:
  grounds      fixed pivots; joint 0 carries the motor
  cranks       a crank tip turning about a ground pivot; the first is the
               motor's, the rest are geared to it at an integer speed ratio
  dyads        a joint tied by two bars to two earlier joints (a, b)

Each dyad adds two degrees of freedom and two length constraints, so every
mechanism built this way has exactly one degree of freedom (the motor angle)
and every pose is found by circle-circle intersection in order, keeping the
side of line a-b that the joint starts on. A dyad whose triangle degenerates
at some crank angle (the linkage would lock) is rejected, as is one whose
transmission angle gets too shallow to drive.
"""

import numpy as np
np.seterr(all="ignore")

T_SIM = 200        # crank angles per revolution when simulating
N_CURVE = 64       # arc-length samples per normalised curve
MIN_SIN = 0.2      # minimum sine of the transmission angle at every dyad


def circle_meet(pa, pb, la, lb, side):
    """Intersection of circles (pa, la) and (pb, lb) on the given side of a->b.

    pa, pb: (..., T, 2); la, lb, side: (...,). Returns point (..., T, 2) and
    the sine of the angle at the new joint, (..., T); NaN where no meeting.
    """
    d_vec = pb - pa
    d = np.linalg.norm(d_vec, axis=-1)
    la_ = la[..., None]
    lb_ = lb[..., None]
    a = (la_ ** 2 - lb_ ** 2 + d ** 2) / (2 * d)
    h2 = la_ ** 2 - a ** 2
    h = np.sqrt(np.where(h2 > 0, h2, np.nan))
    u = d_vec / d[..., None]
    perp = np.stack([-u[..., 1], u[..., 0]], -1)
    p = pa + a[..., None] * u + (side[..., None, None] * h[..., None]) * perp
    sin_g = d * h / (la_ * lb_)
    return p, sin_g


RATIOS = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
RATIO_W = np.array([1, 2, 3, 3, 2, 3, 2, 1], float)


def generate(rng, batch, n_ground, n_crank, n_dyad, tries=24, T=T_SIM):
    """Random valid mechanisms with the same joint counts.

    Joint order: grounds 0..G-1, cranks G..G+C-1 (crank c turns about ground
    c; crank 0 is the motor, the others are geared to it at an integer speed
    ratio), then dyads. Returns dict with 'pos' (M, J, 2) initial positions,
    'parents' (M, J, 2) (-1 unused; a crank's first parent is its pivot),
    'ratio' (M, J) int8 speed ratio of cranks (0 otherwise), 'kind' (J,)
    0 ground 1 crank 2 dyad, and 'traj' (M, J, T, 2).
    """
    G, C = n_ground, n_crank
    assert G >= C >= 1
    J = G + C + n_dyad
    theta = np.linspace(0, 2 * np.pi, T, endpoint=False)
    kind = np.zeros(J, np.int8)
    kind[G:G + C] = 1
    kind[G + C:] = 2
    pos = np.zeros((batch, J, 2))
    parents = -np.ones((batch, J, 2), np.int16)
    ratio = np.zeros((batch, J), np.int8)
    traj = np.zeros((batch, J, T, 2))

    pos[:, 0] = rng.uniform(-0.5, 0.5, (batch, 2))
    for g in range(1, G):
        pos[:, g] = rng.uniform(-1, 1, (batch, 2))
    traj[:, :G] = pos[:, :G, None, :]
    for c in range(C):
        k = G + c
        r = rng.uniform(0.12, 0.45, batch) if c == 0 else rng.uniform(0.08, 0.5, batch)
        phi = rng.uniform(0, 2 * np.pi, batch)
        rho = np.ones(batch, int) if c == 0 else rng.choice(RATIOS, batch, p=RATIO_W / RATIO_W.sum())
        ratio[:, k] = rho
        parents[:, k, 0] = c
        pos[:, k] = pos[:, c] + r[:, None] * np.stack([np.cos(phi), np.sin(phi)], -1)
        ang = rho[:, None] * theta[None] + phi[:, None]
        traj[:, k] = pos[:, c, None, :] + r[:, None, None] * np.stack([np.cos(ang), np.sin(ang)], -1)

    alive = np.ones(batch, bool)
    moving = list(range(G, G + C))
    for k in range(G + C, J):
        done = np.zeros(batch, bool)
        for attempt in range(tries):
            todo = np.flatnonzero(alive & ~done)
            if todo.size == 0:
                break
            n = todo.size
            # parent a: a moving joint, favouring recent ones; parent b: any
            # earlier joint other than a. A geared machine usually starts by
            # tying its two crank tips together (the geared five-bar).
            w = np.arange(1, len(moving) + 1, dtype=float) ** 1.5
            a = np.array(moving)[rng.choice(len(moving), n, p=w / w.sum())]
            b = rng.integers(0, k - 1, n)
            b = np.where(b >= a, b + 1, b)
            if C >= 2 and k == G + C:
                five = rng.random(n) < 0.7
                a = np.where(five, G, a)
                b = np.where(five, G + 1, b)
            pa0 = pos[todo, a]
            pb0 = pos[todo, b]
            mid = 0.5 * (pa0 + pb0)
            span = np.linalg.norm(pb0 - pa0, axis=-1)
            pk = mid + rng.normal(0, 1, (n, 2)) * (0.25 + 0.6 * span)[:, None]
            pk = np.clip(pk, -1.6, 1.6)
            la = np.linalg.norm(pk - pa0, axis=-1)
            lb = np.linalg.norm(pk - pb0, axis=-1)
            cross = (pb0[:, 0] - pa0[:, 0]) * (pk[:, 1] - pa0[:, 1]) - \
                (pb0[:, 1] - pa0[:, 1]) * (pk[:, 0] - pa0[:, 0])
            side = np.sign(cross)
            ok = (la > 0.08) & (lb > 0.08) & (side != 0)
            p, sin_g = circle_meet(traj[todo, a], traj[todo, b], la, lb, side)
            ok &= np.all(np.isfinite(sin_g), -1)
            ok &= np.nan_to_num(sin_g, nan=0).min(-1) > MIN_SIN
            # the joint must actually move, and not ride rigidly on a crank
            # (a point on a crank body only traces a circle)
            ok &= np.nan_to_num(np.ptp(p[..., 0], -1) + np.ptp(p[..., 1], -1)) > 0.05
            for c in range(C):
                r0 = np.linalg.norm(p - traj[todo, c], axis=-1)
                r1 = np.linalg.norm(p - traj[todo, G + c], axis=-1)
                ok &= ~((np.nan_to_num(np.ptp(r0, -1), nan=1) < 1e-6) &
                        (np.nan_to_num(np.ptp(r1, -1), nan=1) < 1e-6))
            idx = todo[ok]
            pos[idx, k] = pk[ok]
            parents[idx, k, 0] = a[ok]
            parents[idx, k, 1] = b[ok]
            traj[idx, k] = p[ok]
            done[idx] = True
        alive &= done
        moving.append(k)
    return dict(pos=pos[alive], parents=parents[alive], ratio=ratio[alive], kind=kind,
                traj=traj[alive])


def ancestors(parents, kind, j):
    """Joints the motion of joint j depends on (including j), sorted."""
    need, stack = {j}, [j]
    while stack:
        k = stack.pop()
        if kind[k] == 0:
            continue
        for p in parents[k]:
            if p >= 0 and p not in need:
                need.add(int(p))
                stack.append(int(p))
    return sorted(need)


def prune(pos, parents, kind, gear, tracer):
    """The sub-mechanism the tracer joint depends on, re-indexed in order.
    Returns pos, parents, kind, gear, tracer index (in the new numbering)."""
    keep = ancestors(parents, kind, tracer)
    remap = -np.ones(len(kind), int)
    remap[keep] = np.arange(len(keep))
    par = np.where(parents[keep] >= 0, remap[np.maximum(parents[keep], 0)], -1)
    return pos[keep], par, kind[keep], gear[keep], int(remap[tracer])


def resample(curves, n=N_CURVE):
    """Uniform arc-length resampling of closed curves (M, T, 2) -> (M, n, 2)."""
    M, T, _ = curves.shape
    closed = np.concatenate([curves, curves[:, :1]], 1)
    seg = np.linalg.norm(np.diff(closed, axis=1), axis=-1)       # (M, T)
    cum = np.concatenate([np.zeros((M, 1)), np.cumsum(seg, 1)], 1)  # (M, T+1)
    total = cum[:, -1:]
    s = (np.arange(n) / n)[None] * total                          # (M, n)
    # batched searchsorted via row offsets
    off = (np.arange(M) * 4.0)[:, None]
    cumn = cum / np.maximum(total, 1e-12)
    idx = np.searchsorted((cumn + off).ravel(), ((s / np.maximum(total, 1e-12)) + off).ravel(), 'right')
    idx = idx.reshape(M, n) - np.arange(M)[:, None] * (T + 1) - 1
    idx = np.clip(idx, 0, T - 1)
    rows = np.arange(M)[:, None]
    c0 = cum[rows, idx]
    c1 = cum[rows, idx + 1]
    w = ((s - c0) / np.maximum(c1 - c0, 1e-12))[..., None]
    return closed[rows, idx] * (1 - w) + closed[rows, idx + 1] * w


def normalise(z):
    """Complex curves (M, n): centre and scale to unit RMS; returns z, scale."""
    z = z - z.mean(-1, keepdims=True)
    s = np.sqrt((np.abs(z) ** 2).mean(-1, keepdims=True))
    return z / np.maximum(s, 1e-12), s


def to_complex(xy):
    return xy[..., 0] + 1j * xy[..., 1]


def variants(z):
    """The four traversals a mechanism can realise of curve z (M, n):
    as is, reversed crank, mirrored, mirrored and reversed."""
    rev = np.roll(z[..., ::-1], 1, -1)
    return [z, rev, np.conj(z), np.conj(rev)]


def align_dist(t, C):
    """Best similarity-aligned residual between target t (n,) and curves C
    (M, n), both normalised. Returns (dist (M,), variant (M,), shift (M,)).
    dist = 1 - max |<c_shift, t>|^2 / n^2, in [0, 1]."""
    n = t.shape[-1]
    Ft = np.fft.fft(t)
    best = np.full(C.shape[0], -1.0)
    var = np.zeros(C.shape[0], np.int8)
    shift = np.zeros(C.shape[0], np.int16)
    for v, Cv in enumerate(variants(C)):
        corr = np.fft.ifft(Ft[None] * np.conj(np.fft.fft(Cv, axis=-1)), axis=-1)
        mag = np.abs(corr) / n
        s = mag.argmax(-1)
        m = mag[np.arange(len(C)), s]
        better = m > best
        best = np.where(better, m, best)
        var = np.where(better, v, var)
        shift = np.where(better, s, shift)
    return 1 - best ** 2, var, shift
