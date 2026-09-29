"""The teacher's physics as a loss: the incremental potential of one implicit-Euler frame step
(cloth.py's substep energy with h = FRAME_DT), batched in PyTorch.

The teacher only labels states it produced itself, so a network trained on its frames alone never
learns what to do from the crumpled or stretched states its own rollouts drift into. Minimising this
energy from any state is what the solver does, so it can supervise those states too (HOOD, Grigorev
et al. 2023). Differences from cloth.py: one step per frame instead of SUBSTEPS, barriers continue
quadratically below B_MIN so penetrating guesses get finite energy, and the lagged normal force is
capped for states the solver would never reach.
"""
import numpy as np
import torch

import mgn_torch as M

# Constants mirror cloth.py (kept here so this module does not import JAX).
GRAVITY = (0.0, -9.81, 0.0)
DENSITY = 0.25
YOUNG = 1500.0
POISSON = 0.3
DHAT = 0.003
SELF_THICK = 0.012
KAPPA = 1.0e4
FRICTION_EPS_V = 1.0e-3
AIR_DENSITY = 1.2
DRAG_COEF = 1.2
MU_S = YOUNG / (2 * (1 + POISSON))
LAM_S = YOUNG * POISSON / (1 - POISSON ** 2)
Q_STVK = ((MU_S + LAM_S / 2, LAM_S / 2, 0.0), (LAM_S / 2, MU_S + LAM_S / 2, 0.0), (0.0, 0.0, 2 * MU_S))

B_MIN = 5e-4               # barrier distance below which it continues as its quadratic Taylor expansion
LAM_MAX = 5 * 9.81         # cap on the lagged normal force per unit mass
T_MAX = 2 * 30 * 30        # triangles of the largest (31 x 31) cloth
H_MAX = 3 * 30 * 30 - 60   # its interior edges (hinges)
TERMS = ("inertia", "membrane", "bending", "obstacle", "self", "friction")


# ---------------------------------------------------------------------------- mesh tables (numpy)

def dihedral(xh):
    """cloth.dihedral, batched: xh [..., 4, 3] -> signed angle [...]."""
    a, b, c, d = xh.unbind(-2)
    e = b - a
    n1 = torch.cross(b - a, c - a, dim=-1)
    n2 = torch.cross(a - b, d - b, dim=-1)
    en = e / torch.sqrt((e * e).sum(-1, keepdim=True) + 1e-20)
    return torch.atan2((torch.cross(n1, n2, dim=-1) * en).sum(-1), (n1 * n2).sum(-1) + 1e-30)


def energy_tables(nx, ny):
    """cloth.build_mesh's element data for an nx x ny grid, padded to T_MAX / H_MAX.

    Padded elements point at vertex N_MAX (the zero row M.pad appends) and have zero weight.
    """
    n = nx * ny
    tris = M.grid_tris(nx, ny)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    rest = np.stack([ii.ravel(), jj.ravel()], -1).astype(np.float64) * M.SPACING
    p = rest[tris]
    Dm = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], -1)
    area = 0.5 * np.abs(np.linalg.det(Dm))
    mass = np.zeros(n)
    np.add.at(mass, tris.ravel(), np.repeat(area / 3, 3))
    mass *= DENSITY

    edge_tris = {}
    for t, tri in enumerate(tris):
        for k in range(3):
            a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            edge_tris.setdefault((min(a, b), max(a, b)), []).append((t, a, b, c))
    hinges, hinge_w = [], []
    for lst in edge_tris.values():
        if len(lst) == 2:
            (t1, a1, b1, c1), (t2, _, _, c2) = lst
            hinges.append((a1, b1, c1, c2))
            hinge_w.append(3 * np.sum((rest[a1] - rest[b1]) ** 2) / (area[t1] + area[t2]))
    hinges = np.array(hinges, np.int64).reshape(-1, 4)
    rest3 = torch.tensor(np.stack([rest[:, 0], np.zeros(n), rest[:, 1]], 1)[hinges], requires_grad=True)
    dihedral(rest3).sum().backward()
    g = rest3.grad[..., 1].numpy()

    T, H = len(tris), len(hinges)
    out = {
        "e_tris": np.full((T_MAX, 3), M.N_MAX, np.int64), "e_dm_inv": np.zeros((T_MAX, 2, 2), np.float32),
        "e_area": np.zeros(T_MAX, np.float32), "e_mass": np.zeros(M.N_MAX, np.float32),
        "e_hinges": np.full((H_MAX, 4), M.N_MAX, np.int64), "e_hinge_g": np.zeros((H_MAX, 4), np.float32),
        "e_hinge_w": np.zeros(H_MAX, np.float32),
    }
    out["e_tris"][:T] = tris
    out["e_dm_inv"][:T] = np.linalg.inv(Dm)
    out["e_area"][:T] = area
    out["e_mass"][:n] = mass
    out["e_hinges"][:H] = hinges
    out["e_hinge_g"][:H] = g
    out["e_hinge_w"][:H] = hinge_w
    return out


# ---------------------------------------------------------------------------- energy

def barrier(d):
    """cloth.barrier for d >= B_MIN, its quadratic continuation below; 0 beyond DHAT (d must be finite)."""
    dc = d.clamp(B_MIN, DHAT)
    val = -((dc - DHAT) ** 2) * torch.log(dc / DHAT)
    m, dh = B_MIN, DHAT
    b0 = -((m - dh) ** 2) * np.log(m / dh)
    b1 = -2 * (m - dh) * np.log(m / dh) - (m - dh) ** 2 / m
    b2 = -2 * np.log(m / dh) - 4 * (m - dh) / m + (m - dh) ** 2 / m ** 2
    s = d - m
    return torch.where(d < m, b0 + b1 * s + 0.5 * b2 * s * s, val)


def barrier_d1(d):
    dc = d.clamp(B_MIN, DHAT)
    return torch.where(d < DHAT, -2 * (dc - DHAT) * torch.log(dc / DHAT) - (dc - DHAT) ** 2 / dc,
                       torch.zeros_like(d))


def f0(y, eps):
    return torch.where(y < eps, -y ** 3 / (3 * eps ** 2) + y ** 2 / eps + eps / 3, y)


def _finite(d):
    return torch.where(torch.isfinite(d), d, torch.full_like(d, 1.0))


def frame_energy(y, x1, x0, free, obs0, obs1, kb, mu, wind, et, rest, valid):
    """Energy terms [B] of the step x0 -> y (x1: the frame before x0).

    y [B, N, 3]: candidate next positions, held vertices already at their targets.
    free [B, N]: vertices the step moves (valid and not held).
    obs0 / obs1: obstacles at the start / end of the step (as train_torch.obs_at).
    et: batched energy_tables; rest/valid: the graph's rest positions and valid mask.
    """
    h = M.FRAME_DT
    B = y.shape[0]
    mass = et["e_mass"]
    fm = mass * free
    v = (x0 - x1) / h

    # explicit forces: gravity + aerodynamic drag on triangles (from the current state)
    tris = et["e_tris"]
    xt = M.bgather(M.pad(x0), tris)
    vt = M.bgather(M.pad(v), tris)
    nrm = torch.cross(xt[:, :, 1] - xt[:, :, 0], xt[:, :, 2] - xt[:, :, 0], dim=-1)
    a2 = torch.sqrt((nrm * nrm).sum(-1) + 1e-20)
    nrm = nrm / a2[..., None]
    vn = ((vt.mean(2) - wind[:, None]) * nrm).sum(-1)
    f_tri = -0.5 * AIR_DENSITY * DRAG_COEF * (0.5 * a2) * vn * vn.abs() * (et["e_area"] > 0)
    f_air = torch.zeros(B, M.N_MAX + 1, 3, device=y.device)
    f_air.scatter_add_(1, tris.reshape(B, -1, 1).expand(-1, -1, 3),
                       (f_tri[..., None, None] * nrm[:, :, None] / 3).expand(-1, -1, 3, -1).reshape(B, -1, 3))
    acc = torch.tensor(GRAVITY, device=y.device) + f_air[:, :-1] / mass.clamp(min=1e-12)[..., None]
    x_tilde = x0 + h * v + h * h * acc

    e_in = 0.5 * (fm * ((y - x_tilde) ** 2).sum(-1)).sum(1) / (h * h)

    yt = M.bgather(M.pad(y), tris)
    Ds = torch.stack([yt[:, :, 1] - yt[:, :, 0], yt[:, :, 2] - yt[:, :, 0]], -1)  # [B,T,3,2]
    F = Ds @ et["e_dm_inv"]
    C = F.transpose(-1, -2) @ F
    e = torch.stack([0.5 * (C[..., 0, 0] - 1), 0.5 * (C[..., 1, 1] - 1), 0.5 * C[..., 0, 1]], -1)
    Q = torch.tensor(Q_STVK, device=y.device)
    e_mem = (et["e_area"] * torch.einsum("bti,ij,btj->bt", e, Q, e)).sum(1)

    kx = (et["e_hinge_g"][..., None] * M.bgather(M.pad(y), et["e_hinges"])).sum(2)
    e_bend = kb * (et["e_hinge_w"] * (kx * kx).sum(-1)).sum(1)

    d, _, _ = M.obstacle_sdf(y, obs1)
    e_obs = (KAPPA * fm[..., None] * torch.where(torch.isfinite(d), barrier(_finite(d)), torch.zeros_like(d))).sum((1, 2))

    idx, ok = M.world_neighbors(x0, rest, valid)
    dp = torch.sqrt(((y[:, :, None] - M.bgather(y, idx)) ** 2).sum(-1) + 1e-20) - SELF_THICK
    pm = 0.5 * (mass[:, :, None] + M.bgather(mass, idx))
    pair_free = free[:, :, None] | M.bgather(free, idx)
    e_self = 0.5 * torch.where(ok & pair_free, KAPPA * pm * barrier(dp), torch.zeros_like(dp)).sum((1, 2))

    d_c, n_c, v_c = M.obstacle_sdf(x0, obs0)
    near = torch.isfinite(d_c) & (d_c < DHAT)
    lam = (KAPPA * barrier_d1(_finite(d_c)).abs()).clamp(max=LAM_MAX) * mass[..., None]
    lam = torch.where(near, lam, torch.zeros_like(lam))
    u = (y - x0)[:, :, None, :] - h * v_c
    ut = u - (u * n_c).sum(-1, keepdim=True) * n_c
    e_fric = (mu[:, None, None] * lam * free[..., None] *
              f0(torch.sqrt((ut * ut).sum(-1) + 1e-20), FRICTION_EPS_V * h)).sum((1, 2))

    return {"inertia": e_in, "membrane": e_mem, "bending": e_bend, "obstacle": e_obs, "self": e_self,
            "friction": e_fric}


def energy_scale(free, et, acc_std):
    """Divisor that puts energies in the units of the supervised loss: a displacement of one acc_std
    at every free vertex costs 0.5 in inertia."""
    h = M.FRAME_DT
    return (et["e_mass"] * free).sum(1) / (h * h) * acc_std ** 2
