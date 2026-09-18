"""Batched 2D ragdoll-runner physics (QWOP-like), numba-compiled.

This file is mirrored line-for-line by ../physics.js. Any change to the
constants or the solver here must be made there too; `parity.py` checks that
the two implementations produce the same trajectories.

Conventions: x forward, y up, ground at y=0, angles CCW. Limbs hang along
local -y, so a positive hip angle swings the leg forward.
"""
import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------- model ----
NB = 9
TORSO, THIGH_R, THIGH_L, CALF_R, CALF_L, FOOT_R, FOOT_L, ARM_R, ARM_L = range(NB)
MASS = np.array([40.0, 7.0, 7.0, 4.0, 4.0, 2.0, 2.0, 3.5, 3.5])
INERTIA = np.array([2.2, 0.13, 0.13, 0.072, 0.072, 0.03, 0.03, 0.09, 0.09])
INV_M = 1.0 / MASS
INV_I = 1.0 / INERTIA

NJ = 8
HIP_R, HIP_L, KNEE_R, KNEE_L, ANKLE_R, ANKLE_L, SHO_R, SHO_L = range(NJ)
J_A = np.array([TORSO, TORSO, THIGH_R, THIGH_L, CALF_R, CALF_L, TORSO, TORSO])
J_B = np.array([THIGH_R, THIGH_L, CALF_R, CALF_L, FOOT_R, FOOT_L, ARM_R, ARM_L])
J_AX = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
J_AY = np.array([-0.30, -0.30, -0.225, -0.225, -0.225, -0.225, 0.27, 0.27])
J_BX = np.array([0.0, 0.0, 0.0, 0.0, -0.06, -0.06, 0.0, 0.0])
J_BY = np.array([0.225, 0.225, 0.225, 0.225, 0.03, 0.03, 0.275, 0.275])
J_LO = np.array([-0.8, -0.8, -2.4, -2.4, -0.6, -0.6, -1.6, -1.6])
J_HI = np.array([1.5, 1.5, 0.0, 0.0, 0.6, 0.6, 1.6, 1.6])
J_TORQUE = np.array([450.0, 450.0, 350.0, 350.0, 40.0, 40.0, 60.0, 60.0])
HIP_SPEED = 4.5
KNEE_SPEED = 5.5
ARM_SPEED = 3.0
ANKLE_GAIN = 8.0
ANKLE_SPEED = 4.0

NC = 11
C_BODY = np.array([FOOT_R, FOOT_R, FOOT_L, FOOT_L, CALF_R, CALF_L,
                   TORSO, TORSO, TORSO, ARM_R, ARM_L])
C_X = np.array([-0.13, 0.13, -0.13, 0.13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
C_Y = np.array([0.0, 0.0, 0.0, 0.0, 0.225, 0.225, -0.30, 0.30, 0.45, -0.275, -0.275])
C_R = np.array([0.04, 0.04, 0.04, 0.04, 0.055, 0.055, 0.10, 0.10, 0.12, 0.045, 0.045])
C_FATAL0 = 6  # contacts with index >= this end the run when they touch

GRAVITY = 9.81
FRICTION = 0.9
NSUB = 4
H = 1.0 / 120.0
DT = NSUB * H  # one control step = 1/30 s
ITERS = 4
BETA = 0.2
SLOP = 0.005
MAX_CORR = 3.0
MAX_V = 50.0
MAX_W = 40.0
FATAL_Y = 0.01  # fatal contact circles closer than this to the ground end the run

NACC = NJ * 5 + NC * 2
NACT = 9
NOBS = 6 + 7 * (NB - 1) + 6 + NACT
GOAL_X = 100.0

# default standing pose: joint angles in joint order
POSE0 = np.array([0.4, -0.26, -0.19, -0.01, -0.18, 0.29, 0.0, 0.0])
POSE0_TORSO = -0.03


@njit(cache=True)
def set_pose(s, x, torso_a, q):
    """Forward kinematics: place bodies so joints are satisfied, feet on ground."""
    for i in range(NB):
        for k in range(6):
            s[i, k] = 0.0
    s[TORSO, 0] = x
    s[TORSO, 1] = 0.0
    s[TORSO, 2] = torso_a
    for j in range(NJ):
        a = J_A[j]
        b = J_B[j]
        ca = np.cos(s[a, 2])
        sa = np.sin(s[a, 2])
        px = s[a, 0] + ca * J_AX[j] - sa * J_AY[j]
        py = s[a, 1] + sa * J_AX[j] + ca * J_AY[j]
        ang = s[a, 2] + q[j]
        cb = np.cos(ang)
        sb = np.sin(ang)
        s[b, 2] = ang
        s[b, 0] = px - (cb * J_BX[j] - sb * J_BY[j])
        s[b, 1] = py - (sb * J_BX[j] + cb * J_BY[j])
    low = 1e9
    for c in range(NC):
        i = C_BODY[c]
        ci = np.cos(s[i, 2])
        si = np.sin(s[i, 2])
        y = s[i, 1] + si * C_X[c] + ci * C_Y[c] - C_R[c]
        if y < low:
            low = y
    for i in range(NB):
        s[i, 1] -= low


@njit(cache=True)
def step_env(s, acc, action):
    """Advance one env by one control step. Returns 1 if a fatal contact touches."""
    hip = action // 3
    knee = action % 3
    tgt = np.zeros(NJ)
    if hip == 1:  # Q
        tgt[HIP_R] = HIP_SPEED
        tgt[HIP_L] = -HIP_SPEED
        tgt[SHO_R] = -ARM_SPEED
        tgt[SHO_L] = ARM_SPEED
    elif hip == 2:  # W
        tgt[HIP_R] = -HIP_SPEED
        tgt[HIP_L] = HIP_SPEED
        tgt[SHO_R] = ARM_SPEED
        tgt[SHO_L] = -ARM_SPEED
    if knee == 1:  # O
        tgt[KNEE_L] = -KNEE_SPEED
        tgt[KNEE_R] = KNEE_SPEED
    elif knee == 2:  # P
        tgt[KNEE_L] = KNEE_SPEED
        tgt[KNEE_R] = -KNEE_SPEED

    cs = np.empty(NB)
    sn = np.empty(NB)
    jrax = np.empty(NJ)
    jray = np.empty(NJ)
    jrbx = np.empty(NJ)
    jrby = np.empty(NJ)
    jk11 = np.empty(NJ)
    jk12 = np.empty(NJ)
    jk22 = np.empty(NJ)
    jbx = np.empty(NJ)
    jby = np.empty(NJ)
    jka = np.empty(NJ)
    jlo = np.empty(NJ)
    jhi = np.empty(NJ)
    crx = np.empty(NC)
    cry = np.empty(NC)
    ckn = np.empty(NC)
    ckt = np.empty(NC)
    cb = np.empty(NC)

    for sub in range(NSUB):
        for i in range(NB):
            s[i, 4] -= GRAVITY * H
            cs[i] = np.cos(s[i, 2])
            sn[i] = np.sin(s[i, 2])

        # ---- joint precompute
        for j in range(NJ):
            a = J_A[j]
            b = J_B[j]
            rax = cs[a] * J_AX[j] - sn[a] * J_AY[j]
            ray = sn[a] * J_AX[j] + cs[a] * J_AY[j]
            rbx = cs[b] * J_BX[j] - sn[b] * J_BY[j]
            rby = sn[b] * J_BX[j] + cs[b] * J_BY[j]
            jrax[j] = rax
            jray[j] = ray
            jrbx[j] = rbx
            jrby[j] = rby
            ms = INV_M[a] + INV_M[b]
            k11 = ms + INV_I[a] * ray * ray + INV_I[b] * rby * rby
            k12 = -INV_I[a] * rax * ray - INV_I[b] * rbx * rby
            k22 = ms + INV_I[a] * rax * rax + INV_I[b] * rbx * rbx
            det = k11 * k22 - k12 * k12
            jk11[j] = k22 / det  # inverse of K
            jk12[j] = -k12 / det
            jk22[j] = k11 / det
            jbx[j] = (BETA / H) * (s[b, 0] + rbx - s[a, 0] - rax)
            jby[j] = (BETA / H) * (s[b, 1] + rby - s[a, 1] - ray)
            jka[j] = 1.0 / (INV_I[a] + INV_I[b])
            ang = s[b, 2] - s[a, 2]
            c_lo = ang - J_LO[j]
            c_hi = J_HI[j] - ang
            jlo[j] = (c_lo / H) if c_lo > 0.0 else (BETA / H) * c_lo
            jhi[j] = (c_hi / H) if c_hi > 0.0 else (BETA / H) * c_hi
            if j == ANKLE_R or j == ANKLE_L:
                t = -ANKLE_GAIN * ang
                if t > ANKLE_SPEED:
                    t = ANKLE_SPEED
                elif t < -ANKLE_SPEED:
                    t = -ANKLE_SPEED
                tgt[j] = t

        # ---- contact precompute
        for c in range(NC):
            i = C_BODY[c]
            rx = cs[i] * C_X[c] - sn[i] * C_Y[c]
            ry = sn[i] * C_X[c] + cs[i] * C_Y[c] - C_R[c]
            crx[c] = rx
            cry[c] = ry
            ckn[c] = 1.0 / (INV_M[i] + INV_I[i] * rx * rx)
            ckt[c] = 1.0 / (INV_M[i] + INV_I[i] * ry * ry)
            sep = s[i, 1] + ry
            if sep > 0.0:
                cb[c] = sep / H
            else:
                bias = (BETA / H) * (sep + SLOP)
                if bias > 0.0:
                    bias = 0.0
                if bias < -MAX_CORR:
                    bias = -MAX_CORR
                cb[c] = bias

        # ---- warm start
        for j in range(NJ):
            a = J_A[j]
            b = J_B[j]
            o = j * 5
            px = acc[o]
            py = acc[o + 1]
            aimp = acc[o + 2] + acc[o + 3] - acc[o + 4]
            s[a, 3] -= INV_M[a] * px
            s[a, 4] -= INV_M[a] * py
            s[a, 5] -= INV_I[a] * (jrax[j] * py - jray[j] * px + aimp)
            s[b, 3] += INV_M[b] * px
            s[b, 4] += INV_M[b] * py
            s[b, 5] += INV_I[b] * (jrbx[j] * py - jrby[j] * px + aimp)
        for c in range(NC):
            i = C_BODY[c]
            o = NJ * 5 + c * 2
            pn = acc[o]
            pt = acc[o + 1]
            s[i, 3] += INV_M[i] * pt
            s[i, 4] += INV_M[i] * pn
            s[i, 5] += INV_I[i] * (crx[c] * pn - cry[c] * pt)

        # ---- velocity iterations
        for it in range(ITERS):
            for j in range(NJ):
                a = J_A[j]
                b = J_B[j]
                o = j * 5
                # motor
                lam = (tgt[j] - (s[b, 5] - s[a, 5])) * jka[j]
                mx = J_TORQUE[j] * H
                old = acc[o + 2]
                new = old + lam
                if new > mx:
                    new = mx
                elif new < -mx:
                    new = -mx
                acc[o + 2] = new
                lam = new - old
                s[a, 5] -= INV_I[a] * lam
                s[b, 5] += INV_I[b] * lam
                # lower limit
                lam = -((s[b, 5] - s[a, 5]) + jlo[j]) * jka[j]
                old = acc[o + 3]
                new = old + lam
                if new < 0.0:
                    new = 0.0
                acc[o + 3] = new
                lam = new - old
                s[a, 5] -= INV_I[a] * lam
                s[b, 5] += INV_I[b] * lam
                # upper limit
                lam = -((s[a, 5] - s[b, 5]) + jhi[j]) * jka[j]
                old = acc[o + 4]
                new = old + lam
                if new < 0.0:
                    new = 0.0
                acc[o + 4] = new
                lam = new - old
                s[a, 5] += INV_I[a] * lam
                s[b, 5] -= INV_I[b] * lam
                # point
                cdx = s[b, 3] - s[b, 5] * jrby[j] - s[a, 3] + s[a, 5] * jray[j] + jbx[j]
                cdy = s[b, 4] + s[b, 5] * jrbx[j] - s[a, 4] - s[a, 5] * jrax[j] + jby[j]
                px = -(jk11[j] * cdx + jk12[j] * cdy)
                py = -(jk12[j] * cdx + jk22[j] * cdy)
                acc[o] += px
                acc[o + 1] += py
                s[a, 3] -= INV_M[a] * px
                s[a, 4] -= INV_M[a] * py
                s[a, 5] -= INV_I[a] * (jrax[j] * py - jray[j] * px)
                s[b, 3] += INV_M[b] * px
                s[b, 4] += INV_M[b] * py
                s[b, 5] += INV_I[b] * (jrbx[j] * py - jrby[j] * px)
            for c in range(NC):
                i = C_BODY[c]
                o = NJ * 5 + c * 2
                # friction
                vt = s[i, 3] - s[i, 5] * cry[c]
                lam = -vt * ckt[c]
                mx = FRICTION * acc[o]
                old = acc[o + 1]
                new = old + lam
                if new > mx:
                    new = mx
                elif new < -mx:
                    new = -mx
                acc[o + 1] = new
                lam = new - old
                s[i, 3] += INV_M[i] * lam
                s[i, 5] -= INV_I[i] * cry[c] * lam
                # normal
                vn = s[i, 4] + s[i, 5] * crx[c]
                lam = -(vn + cb[c]) * ckn[c]
                old = acc[o]
                new = old + lam
                if new < 0.0:
                    new = 0.0
                acc[o] = new
                lam = new - old
                s[i, 4] += INV_M[i] * lam
                s[i, 5] += INV_I[i] * crx[c] * lam

        # ---- integrate
        for i in range(NB):
            for k in range(3, 5):
                if s[i, k] > MAX_V:
                    s[i, k] = MAX_V
                elif s[i, k] < -MAX_V:
                    s[i, k] = -MAX_V
            if s[i, 5] > MAX_W:
                s[i, 5] = MAX_W
            elif s[i, 5] < -MAX_W:
                s[i, 5] = -MAX_W
            s[i, 0] += H * s[i, 3]
            s[i, 1] += H * s[i, 4]
            s[i, 2] += H * s[i, 5]

    fell = 0
    for c in range(C_FATAL0, NC):
        i = C_BODY[c]
        y = s[i, 1] + np.sin(s[i, 2]) * C_X[c] + np.cos(s[i, 2]) * C_Y[c] - C_R[c]
        if y < FATAL_Y:
            fell = 1
    return fell


@njit(cache=True)
def obs_env(s, acc, prev_action, out):
    t = s[TORSO]
    out[0] = (t[1] - 1.2) * 2.0
    out[1] = np.sin(t[2])
    out[2] = np.cos(t[2])
    out[3] = t[3] * 0.2
    out[4] = t[4] * 0.2
    out[5] = t[5] * 0.2
    k = 6
    for i in range(1, NB):
        out[k] = s[i, 0] - t[0]
        out[k + 1] = s[i, 1] - t[1]
        out[k + 2] = np.sin(s[i, 2])
        out[k + 3] = np.cos(s[i, 2])
        out[k + 4] = (s[i, 3] - t[3]) * 0.2
        out[k + 5] = (s[i, 4] - t[4]) * 0.2
        out[k + 6] = s[i, 5] * 0.1
        k += 7
    for c in range(6):
        out[k] = 1.0 if acc[NJ * 5 + c * 2] > 0.0 else 0.0
        k += 1
    for a in range(NACT):
        out[k] = 1.0 if prev_action == a else 0.0
        k += 1


@njit(cache=True, parallel=True)
def step_batch(S, ACC, actions, fell, dx):
    for e in prange(S.shape[0]):
        x0 = S[e, TORSO, 0]
        fell[e] = step_env(S[e], ACC[e], actions[e])
        dx[e] = S[e, TORSO, 0] - x0


@njit(cache=True, parallel=True)
def obs_batch(S, ACC, prev_actions, OUT):
    for e in prange(S.shape[0]):
        obs_env(S[e], ACC[e], prev_actions[e], OUT[e])


@njit(cache=True)
def reset_batch(S, ACC, idx, noise):
    q = np.empty(NJ)
    for n in range(idx.shape[0]):
        e = idx[n]
        for j in range(NJ):
            q[j] = POSE0[j] + noise * np.random.uniform(-1.0, 1.0)
            if q[j] < J_LO[j] + 0.02:
                q[j] = J_LO[j] + 0.02
            if q[j] > J_HI[j] - 0.02:
                q[j] = J_HI[j] - 0.02
        set_pose(S[e], 0.0, POSE0_TORSO + noise * 0.5 * np.random.uniform(-1.0, 1.0), q)
        for i in range(NB):
            S[e, i, 3] = noise * 2.0 * np.random.uniform(-1.0, 1.0)
            S[e, i, 4] = 0.0
        for k in range(NACC):
            ACC[e, k] = 0.0


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


class QwopVec:
    """Vectorised env. Reset is manual so callers can see terminal obs."""

    def __init__(self, n, noise=0.1, max_steps=1000, seed=0):
        self.n = n
        self.noise = noise
        self.max_steps = max_steps
        self.S = np.zeros((n, NB, 6))
        self.ACC = np.zeros((n, NACC))
        self.prev = np.zeros(n, dtype=np.int64)
        self.t = np.zeros(n, dtype=np.int64)
        self.fell = np.zeros(n, dtype=np.int64)
        self.dx = np.zeros(n)
        self._obs = np.zeros((n, NOBS))
        self.reset(np.arange(n))

    def reset(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size:
            reset_batch(self.S, self.ACC, idx, self.noise)
            self.prev[idx] = 0
            self.t[idx] = 0

    def obs(self):
        obs_batch(self.S, self.ACC, self.prev, self._obs)
        return self._obs.copy()

    def step(self, actions):
        actions = np.ascontiguousarray(actions, dtype=np.int64)
        step_batch(self.S, self.ACC, actions, self.fell, self.dx)
        self.prev[:] = actions
        self.t += 1
        fell = self.fell.astype(bool)
        goal = self.S[:, TORSO, 0] >= GOAL_X
        timeout = self.t >= self.max_steps
        return self.dx.copy(), fell, goal, timeout


def joint_angles(S):
    return S[:, J_B, 2] - S[:, J_A, 2]
