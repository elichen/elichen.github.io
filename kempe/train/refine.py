"""Design parameters of a dyad linkage and Levenberg-Marquardt refinement of
its tracer curve towards a target shape (up to similarity).

Design vector for a mechanism with ground set G (joint 0 first) and dyads D:
  [x, y of each ground joint except joint 0] + [la, lb of each dyad]
Joint 0 and the crank radius and phase are held fixed: together they pin the
translation and scale the similarity alignment makes irrelevant anyway.
"""

import numpy as np
from linkage import circle_meet, T_SIM, N_CURVE, variants

np.seterr(all="ignore")


class Topo:
    """One mechanism's fixed structure; its continuous design is a vector.

    Design vector: [x, y of each ground except joint 0] + [r, phase of each
    geared crank] + [la, lb of each dyad]. Joint 0 and the motor crank's
    radius and phase stay fixed: they pin the translation, scale and start
    point that the similarity alignment makes irrelevant anyway.
    """

    def __init__(self, pos, parents, kind, ratio, tracer):
        self.kind = np.asarray(kind)
        self.parents = np.asarray(parents)
        self.ratio = np.asarray(ratio)
        self.tracer = int(tracer)
        self.J = len(kind)
        self.grounds = [j for j in range(self.J) if self.kind[j] == 0]
        self.cranks = [j for j in range(self.J) if self.kind[j] == 1]
        self.dyads = [j for j in range(self.J) if self.kind[j] == 2]
        pos = np.asarray(pos, float)
        self.p0 = pos[0].copy()
        m = self.cranks[0]
        d = pos[m] - pos[self.parents[m, 0]]
        self.r = np.hypot(*d)
        self.phi = np.arctan2(d[1], d[0])
        self.side = np.zeros(self.J)
        for j in self.dyads:
            a, b = self.parents[j]
            pa, pb, pk = pos[a], pos[b], pos[j]
            self.side[j] = np.sign((pb[0] - pa[0]) * (pk[1] - pa[1]) - (pb[1] - pa[1]) * (pk[0] - pa[0]))

    def params_from_pos(self, pos):
        pos = np.asarray(pos, float)
        out = [pos[g] for g in self.grounds[1:]]
        for c in self.cranks[1:]:
            d = pos[c] - pos[self.parents[c, 0]]
            out.append(np.array([np.hypot(*d), np.arctan2(d[1], d[0])]))
        for j in self.dyads:
            a, b = self.parents[j]
            out.append(np.array([np.hypot(*(pos[j] - pos[a])), np.hypot(*(pos[j] - pos[b]))]))
        return np.concatenate(out) if out else np.zeros(0)

    def bar_lengths(self, P):
        """Crank radii and dyad bar lengths, (B, n_bars)."""
        P = np.atleast_2d(P)
        cols = [np.full(len(P), self.r)]
        i = 2 * (len(self.grounds) - 1)
        for _ in self.cranks[1:]:
            cols.append(np.abs(P[:, i]))
            i += 2
        for _ in self.dyads:
            cols += [P[:, i], P[:, i + 1]]
            i += 2
        return np.stack(cols, 1)

    def simulate(self, P, T=T_SIM):
        """P (B, D) design vectors -> traj (B, J, T, 2), min transmission sine (B,)."""
        P = np.atleast_2d(P)
        B = len(P)
        theta = np.linspace(0, 2 * np.pi, T, endpoint=False)
        traj = np.zeros((B, self.J, T, 2))
        traj[:, 0] = self.p0
        i = 0
        for g in self.grounds[1:]:
            traj[:, g] = P[:, None, i:i + 2]
            i += 2
        for n, c in enumerate(self.cranks):
            piv = traj[:, self.parents[c, 0]]
            if n == 0:
                r, phi = np.full(B, self.r), np.full(B, self.phi)
            else:
                r, phi = P[:, i], P[:, i + 1]
                i += 2
            ang = self.ratio[c] * theta[None] + phi[:, None]
            traj[:, c] = piv + r[:, None, None] * np.stack([np.cos(ang), np.sin(ang)], -1)
        min_sin = np.ones(B)
        for j in self.dyads:
            a, b = self.parents[j]
            la, lb = P[:, i], P[:, i + 1]
            i += 2
            p, s = circle_meet(traj[:, a], traj[:, b], la, lb, np.full(B, self.side[j]))
            traj[:, j] = p
            min_sin = np.minimum(min_sin, np.nan_to_num(s, nan=-1).min(-1))
        return traj, min_sin


def resample_offset(curves, delta, n=N_CURVE):
    """Arc-length resample closed curves (B, T, 2) at s_i = (i + delta_b)/n of
    the perimeter, starting from sample 0. Returns complex (B, n)."""
    B, T, _ = curves.shape
    closed = np.concatenate([curves, curves[:, :1]], 1)
    seg = np.linalg.norm(np.diff(closed, axis=1), axis=-1)
    cum = np.concatenate([np.zeros((B, 1)), np.cumsum(seg, 1)], 1)
    total = cum[:, -1:]
    u = ((np.arange(n)[None] + np.asarray(delta).reshape(-1, 1)) / n) % 1.0
    cumn = cum / np.maximum(total, 1e-12)
    out = np.empty((B, n), complex)
    z = closed[..., 0] + 1j * closed[..., 1]
    for b in range(B):
        idx = np.clip(np.searchsorted(cumn[b], u[b], 'right') - 1, 0, T - 1)
        c0, c1 = cumn[b, idx], cumn[b, idx + 1]
        w = (u[b] - c0) / np.maximum(c1 - c0, 1e-12)
        out[b] = z[b, idx] * (1 - w) + z[b, idx + 1] * w
    return out


def normalise(z):
    z = z - z.mean(-1, keepdims=True)
    return z / np.maximum(np.sqrt((np.abs(z) ** 2).mean(-1, keepdims=True)), 1e-12)


def target_frame(t, variant, shift):
    """Re-express target t (n,) so that it lines up index-for-index with the
    candidate's own arc-length samples (see align_dist)."""
    f = variants(t[None])[variant][0]
    s = shift if variant in (0, 2) else -shift
    return np.roll(f, -s)


def best_alignment(t, c):
    """Variant, integer shift and dist of candidate c (n,) against target t."""
    n = len(t)
    Ft = np.fft.fft(t)
    best = (-1, 0, 0)
    for v, cv in enumerate(variants(c[None])):
        corr = np.fft.ifft(Ft * np.conj(np.fft.fft(cv[0])))
        s = int(np.abs(corr).argmax())
        m = np.abs(corr[s]) / n
        if m > best[0]:
            best = (m, v, s)
    return 1 - best[0] ** 2, best[1], best[2]


EXTENT_MAX = 3.0     # machine reach, in curve RMS radii from the curve centre
BAR_MIN = 0.25       # shortest bar, in curve RMS radii


def residuals(topo, P, delta, tt, min_sin_floor=0.25, w_sin=2.0, w_ext=0.3, w_bar=0.5):
    """Residual vectors (B, 2n+3) for designs P (B, D) against aligned target
    tt (n,), with each curve resampled from start offset delta (B,). The last
    three entries keep the machine drivable (transmission angle), compact
    around the drawing, and free of stubby bars."""
    traj, min_sin = topo.simulate(P)
    raw = resample_offset(traj[:, topo.tracer], delta)
    mu = raw.mean(-1)
    sc = np.sqrt((np.abs(raw - mu[:, None]) ** 2).mean(-1))
    c = (raw - mu[:, None]) / sc[:, None]
    n = c.shape[1]
    alpha = (np.conj(c) * tt[None]).sum(-1) / n
    r = (tt[None] - alpha[:, None] * c) / np.sqrt(n)
    pen_sin = w_sin * np.maximum(0, min_sin_floor - min_sin)
    tz = traj[..., 0] + 1j * traj[..., 1]
    extent = np.abs(tz - mu[:, None, None]).max((-1, -2)) / sc
    pen_ext = w_ext * np.maximum(0, extent - EXTENT_MAX)
    bars = topo.bar_lengths(P) / sc[:, None]
    pen_bar = w_bar * np.sqrt((np.maximum(0, BAR_MIN - bars) ** 2).sum(-1))
    R = np.concatenate([r.real, r.imag, pen_sin[:, None], pen_ext[:, None], pen_bar[:, None]], 1)
    R[~np.isfinite(R).all(1)] = np.nan
    return R, alpha


def fit(topo, p, target, iters=60, lam=1e-3, verbose=False):
    """Levenberg-Marquardt on the design (and the start offset) of one mechanism.
    target: normalised complex (n,). Returns p, rms error, history."""
    n = len(target)
    traj, _ = topo.simulate(p[None])
    c = normalise(resample_offset(traj[:, topo.tracer], [0.0]))[0]
    dist, v, s = best_alignment(target, c)
    tt = target_frame(target, v, s)
    x = np.concatenate([p, [0.0]])      # last entry: start offset delta
    R, _ = residuals(topo, x[None, :-1], x[None, -1:], tt)
    cost = np.nansum(R[0] ** 2) if np.isfinite(R).all() else np.inf
    hist = [np.sqrt(cost)]
    for it in range(iters):
        if it % 10 == 9:   # re-pick the discrete alignment as the curve moves
            traj, _ = topo.simulate(x[None, :-1])
            c = normalise(resample_offset(traj[:, topo.tracer], [x[-1]]))[0]
            _, v2, s2 = best_alignment(target, c)
            tt_new = target_frame(target, v2, s2)
            Rn, _ = residuals(topo, x[None, :-1], x[None, -1:], tt_new)
            cn = np.sum(Rn[0] ** 2)
            if cn < cost:
                tt, cost = tt_new, cn
        h = 1e-5 * np.maximum(1, np.abs(x))
        X = x[None] + np.diag(h)
        X = np.vstack([x[None], X])
        R, _ = residuals(topo, X[:, :-1], X[:, -1], tt)
        if not np.isfinite(R[0]).all():
            break
        Jm = (R[1:] - R[0][None]) / h[:, None]
        Jm = np.nan_to_num(Jm).T
        g = Jm.T @ R[0]
        H = Jm.T @ Jm
        improved = False
        for _ in range(8):
            step = -np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-9 * np.eye(len(x))), g)
            xn = x + step
            Rn, _ = residuals(topo, xn[None, :-1], xn[None, -1:], tt)
            cn = np.sum(Rn[0] ** 2) if np.isfinite(Rn).all() else np.inf
            if cn < cost:
                x, cost = xn, cn
                lam = max(lam / 3, 1e-7)
                improved = True
                break
            lam *= 4
        hist.append(np.sqrt(cost))
        if not improved and lam > 1e6:
            break
    R, _ = residuals(topo, x[None, :-1], x[None, -1:], tt)
    shape_err = np.sqrt(np.sum(R[0, :2 * n] ** 2))
    return x[:-1], shape_err, hist, (tt, x[-1])
