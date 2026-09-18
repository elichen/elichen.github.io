"""JAX port of sim.py for GPU training.

Same solver, same constraint order. Every quantity is a per-env vector and
bodies are kept as Python lists of those vectors, so the whole step is
elementwise and XLA fuses it into a handful of kernels. `jparity.py` checks it
against the numba simulator.
"""
import jax
import jax.numpy as jnp
from jax import lax
import consts as C

NB, NJ, NC = C.NB, C.NJ, C.NC
IM = [float(v) for v in C.INV_M]
II = [float(v) for v in C.INV_I]
JA = [int(v) for v in C.J_A]
JB = [int(v) for v in C.J_B]
CBODY = [int(v) for v in C.C_BODY]
BH = C.BETA / C.H


def unpack(S, ACC):
    cols = [[S[:, i, k] for i in range(NB)] for k in range(6)]
    return cols, [ACC[:, k] for k in range(C.NACC)]


def pack(cols, acc):
    S = jnp.stack([jnp.stack(cols[k], axis=1) for k in range(6)], axis=2)
    return S, jnp.stack(acc, axis=1)


def _targets(action):
    hip, knee = action // 3, action % 3
    hs = jnp.where(hip == 1, C.HIP_SPEED, jnp.where(hip == 2, -C.HIP_SPEED, 0.0))
    ar = jnp.where(hip == 1, -C.ARM_SPEED, jnp.where(hip == 2, C.ARM_SPEED, 0.0))
    kr = jnp.where(knee == 1, C.KNEE_SPEED, jnp.where(knee == 2, -C.KNEE_SPEED, 0.0))
    return [hs, -hs, kr, -kr, None, None, ar, -ar]


def _substep(state, tgt):
    x, y, a, vx, vy, w, acc = state
    vy = [v - C.GRAVITY * C.H for v in vy]
    cs = [jnp.cos(t) for t in a]
    sn = [jnp.sin(t) for t in a]

    J = []
    tg = list(tgt)
    for j in range(NJ):
        ia, ib = JA[j], JB[j]
        rax = cs[ia] * C.J_AX[j] - sn[ia] * C.J_AY[j]
        ray = sn[ia] * C.J_AX[j] + cs[ia] * C.J_AY[j]
        rbx = cs[ib] * C.J_BX[j] - sn[ib] * C.J_BY[j]
        rby = sn[ib] * C.J_BX[j] + cs[ib] * C.J_BY[j]
        ms = IM[ia] + IM[ib]
        k11 = ms + II[ia] * ray * ray + II[ib] * rby * rby
        k12 = -II[ia] * rax * ray - II[ib] * rbx * rby
        k22 = ms + II[ia] * rax * rax + II[ib] * rbx * rbx
        det = k11 * k22 - k12 * k12
        ang = a[ib] - a[ia]
        c_lo = ang - C.J_LO[j]
        c_hi = C.J_HI[j] - ang
        J.append(dict(rax=rax, ray=ray, rbx=rbx, rby=rby, i11=k22 / det, i12=-k12 / det, i22=k11 / det,
                      bx=BH * (x[ib] + rbx - x[ia] - rax), by=BH * (y[ib] + rby - y[ia] - ray),
                      ka=1.0 / (II[ia] + II[ib]),
                      lo=jnp.where(c_lo > 0.0, c_lo / C.H, BH * c_lo),
                      hi=jnp.where(c_hi > 0.0, c_hi / C.H, BH * c_hi)))
        if j in (C.ANKLE_R, C.ANKLE_L):
            tg[j] = jnp.clip(-C.ANKLE_GAIN * ang, -C.ANKLE_SPEED, C.ANKLE_SPEED)

    K = []
    for c in range(NC):
        i = CBODY[c]
        rx = cs[i] * C.C_X[c] - sn[i] * C.C_Y[c]
        ry = sn[i] * C.C_X[c] + cs[i] * C.C_Y[c] - C.C_R[c]
        sep = y[i] + ry
        bias = jnp.where(sep > 0.0, sep / C.H, jnp.clip(BH * (sep + C.SLOP), -C.MAX_CORR, 0.0))
        K.append(dict(rx=rx, ry=ry, kn=1.0 / (IM[i] + II[i] * rx * rx), kt=1.0 / (IM[i] + II[i] * ry * ry), b=bias))

    vx, vy, w, acc = list(vx), list(vy), list(w), list(acc)
    # warm start
    for j in range(NJ):
        ia, ib, o, q = JA[j], JB[j], j * 5, J[j]
        px, py = acc[o], acc[o + 1]
        aimp = acc[o + 2] + acc[o + 3] - acc[o + 4]
        vx[ia] = vx[ia] - IM[ia] * px
        vy[ia] = vy[ia] - IM[ia] * py
        w[ia] = w[ia] - II[ia] * (q['rax'] * py - q['ray'] * px + aimp)
        vx[ib] = vx[ib] + IM[ib] * px
        vy[ib] = vy[ib] + IM[ib] * py
        w[ib] = w[ib] + II[ib] * (q['rbx'] * py - q['rby'] * px + aimp)
    for c in range(NC):
        i, o, k = CBODY[c], NJ * 5 + c * 2, K[c]
        pn, pt = acc[o], acc[o + 1]
        vx[i] = vx[i] + IM[i] * pt
        vy[i] = vy[i] + IM[i] * pn
        w[i] = w[i] + II[i] * (k['rx'] * pn - k['ry'] * pt)

    def iteration(carry, _):
        vx, vy, w, acc = [list(v) for v in carry]
        for j in range(NJ):
            ia, ib, o, q = JA[j], JB[j], j * 5, J[j]
            mx = float(C.J_TORQUE[j]) * C.H
            new = jnp.clip(acc[o + 2] + (tg[j] - (w[ib] - w[ia])) * q['ka'], -mx, mx)       # motor
            lam, acc[o + 2] = new - acc[o + 2], new
            w[ia] = w[ia] - II[ia] * lam
            w[ib] = w[ib] + II[ib] * lam
            new = jnp.maximum(acc[o + 3] - ((w[ib] - w[ia]) + q['lo']) * q['ka'], 0.0)      # lower limit
            lam, acc[o + 3] = new - acc[o + 3], new
            w[ia] = w[ia] - II[ia] * lam
            w[ib] = w[ib] + II[ib] * lam
            new = jnp.maximum(acc[o + 4] - ((w[ia] - w[ib]) + q['hi']) * q['ka'], 0.0)      # upper limit
            lam, acc[o + 4] = new - acc[o + 4], new
            w[ia] = w[ia] + II[ia] * lam
            w[ib] = w[ib] - II[ib] * lam
            cdx = vx[ib] - w[ib] * q['rby'] - vx[ia] + w[ia] * q['ray'] + q['bx']           # point
            cdy = vy[ib] + w[ib] * q['rbx'] - vy[ia] - w[ia] * q['rax'] + q['by']
            px = -(q['i11'] * cdx + q['i12'] * cdy)
            py = -(q['i12'] * cdx + q['i22'] * cdy)
            acc[o] = acc[o] + px
            acc[o + 1] = acc[o + 1] + py
            vx[ia] = vx[ia] - IM[ia] * px
            vy[ia] = vy[ia] - IM[ia] * py
            w[ia] = w[ia] - II[ia] * (q['rax'] * py - q['ray'] * px)
            vx[ib] = vx[ib] + IM[ib] * px
            vy[ib] = vy[ib] + IM[ib] * py
            w[ib] = w[ib] + II[ib] * (q['rbx'] * py - q['rby'] * px)
        for c in range(NC):
            i, o, k = CBODY[c], NJ * 5 + c * 2, K[c]
            mx = C.FRICTION * acc[o]
            new = jnp.clip(acc[o + 1] - (vx[i] - w[i] * k['ry']) * k['kt'], -mx, mx)        # friction
            lam, acc[o + 1] = new - acc[o + 1], new
            vx[i] = vx[i] + IM[i] * lam
            w[i] = w[i] - II[i] * k['ry'] * lam
            new = jnp.maximum(acc[o] - ((vy[i] + w[i] * k['rx']) + k['b']) * k['kn'], 0.0)  # normal
            lam, acc[o] = new - acc[o], new
            vy[i] = vy[i] + IM[i] * lam
            w[i] = w[i] + II[i] * k['rx'] * lam
        return (vx, vy, w, acc), None

    (vx, vy, w, acc), _ = lax.scan(iteration, (vx, vy, w, acc), None, length=C.ITERS)
    vx = [jnp.clip(v, -C.MAX_V, C.MAX_V) for v in vx]
    vy = [jnp.clip(v, -C.MAX_V, C.MAX_V) for v in vy]
    w = [jnp.clip(v, -C.MAX_W, C.MAX_W) for v in w]
    x = [p + C.H * v for p, v in zip(x, vx)]
    y = [p + C.H * v for p, v in zip(y, vy)]
    a = [p + C.H * v for p, v in zip(a, w)]
    return (x, y, a, vx, vy, w, acc)


def step(S, ACC, action):
    """One control step for a batch. Returns S, ACC, fell (bool), dx."""
    cols, acc = unpack(S, ACC)
    tgt = _targets(action)
    zero = jnp.zeros_like(S[:, 0, 0])
    tgt = [zero if t is None else t + zero for t in tgt]
    state = (*cols, acc)
    state, _ = lax.scan(lambda st, _: (_substep(st, tgt), None), state, None, length=C.NSUB)
    x, y, a = state[0], state[1], state[2]
    fell = jnp.zeros_like(zero, dtype=bool)
    for c in range(C.C_FATAL0, NC):
        i = CBODY[c]
        fell = fell | (y[i] + jnp.sin(a[i]) * C.C_X[c] + jnp.cos(a[i]) * C.C_Y[c] - C.C_R[c] < C.FATAL_Y)
    S2, ACC2 = pack(list(state[:6]), state[6])
    return S2, ACC2, fell, S2[:, 0, 0] - S[:, 0, 0]


def observe(S, ACC, prev):
    t = S[:, 0]
    rest = S[:, 1:]
    head = jnp.stack([(t[:, 1] - 1.2) * 2.0, jnp.sin(t[:, 2]), jnp.cos(t[:, 2]),
                      t[:, 3] * 0.2, t[:, 4] * 0.2, t[:, 5] * 0.2], axis=1)
    body = jnp.stack([rest[:, :, 0] - t[:, None, 0], rest[:, :, 1] - t[:, None, 1],
                      jnp.sin(rest[:, :, 2]), jnp.cos(rest[:, :, 2]),
                      (rest[:, :, 3] - t[:, None, 3]) * 0.2, (rest[:, :, 4] - t[:, None, 4]) * 0.2,
                      rest[:, :, 5] * 0.1], axis=2).reshape(S.shape[0], -1)
    contact = (ACC[:, NJ * 5:NJ * 5 + 12:2] > 0.0).astype(S.dtype)
    return jnp.concatenate([head, body, contact, jax.nn.one_hot(prev, C.NACT, dtype=S.dtype)], axis=1)
