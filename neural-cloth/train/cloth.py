"""Accurate cloth solver: the teacher the graph network learns from.

Physics (SI units, y up):
  * membrane: St. Venant-Kirchhoff FEM on triangles (Green strain, 2D Young's modulus)
  * bending: isometric quadratic bending (Bergou et al. 2006), flat rest state; it matches
    discrete shells for small bends and has a constant, exact Hessian
  * time integration: implicit Euler written as an optimization (Gast et al. 2015),
    solved with Newton + preconditioned CG + backtracking line search
  * contact: IPC-style log barriers (Li et al. 2020) against analytic obstacles
    (table plane, spheres, capsules) and between cloth vertices (self-contact),
    with lagged, smoothed Coulomb friction against obstacles
  * air drag on triangles (optional wind); grasped vertices follow scripted paths

The Newton Hessian is the Gauss-Newton part of each energy (always positive
semi-definite), applied matrix-free. One frame = SUBSTEPS implicit steps.
Run it with jax_default_matmul_precision="highest": TF32 tensor-core matmuls
leave Newton stuck above its tolerance.
"""
import numpy as np
import jax
import jax.numpy as jnp

GRAVITY = np.array([0.0, -9.81, 0.0], np.float32)
FRAME_DT = 1.0 / 60.0
SUBSTEPS = 10
SPACING = 0.02            # rest edge length (m)
DENSITY = 0.25            # kg / m^2
YOUNG = 1500.0            # 2D Young's modulus (N/m)
POISSON = 0.3
DHAT = 0.003              # barrier activation distance (m)
OBS_OFFSET = 0.003        # cloth half-thickness against obstacles (m)
SELF_THICK = 0.012        # vertex-vertex separation for self-contact (m)
SELF_EXCLUDE = 2.1 * SPACING  # no self-contact between vertices this close at rest
KAPPA = 1.0e4             # barrier stiffness per unit vertex mass (1/s^2)
FRICTION_EPS_V = 1.0e-3   # static-friction velocity threshold (m/s)
AIR_DENSITY = 1.2
DRAG_COEF = 1.2
MAX_SPHERES = 2
MAX_CAPSULES = 2
MAX_HANDLES = 2
SELF_K = 8                # self-contact candidates kept per vertex
NEWTON_ITERS = 16
NEWTON_TOL = 5e-6         # max |dx| (m) to stop Newton
CG_ITERS = 200
CG_TOL = 1e-4             # relative residual
DEBUG = False

MU_S = YOUNG / (2 * (1 + POISSON))
LAM_S = YOUNG * POISSON / (1 - POISSON ** 2)
Q_STVK = np.array([[MU_S + LAM_S / 2, LAM_S / 2, 0], [LAM_S / 2, MU_S + LAM_S / 2, 0], [0, 0, 2 * MU_S]], np.float32)


# ---------------------------------------------------------------------------- mesh

def grid_mesh(nx, ny, spacing=SPACING):
    """Rectangular cloth on the rest plane, cells split along alternating diagonals."""
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    rest = np.stack([ii.ravel(), jj.ravel()], -1).astype(np.float64) * spacing
    vid = lambda i, j: i * ny + j
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            if (i + j) % 2 == 0:
                tris += [(a, b, c), (a, c, d)]
            else:
                tris += [(a, b, d), (b, c, d)]
    tris = np.array(tris, np.int32)
    return build_mesh(rest, tris)


def build_mesh(rest, tris):
    """Precompute everything the solver and the network need from a rest mesh."""
    n = len(rest)
    p = rest[tris]  # [T,3,2]
    Dm = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], -1)  # [T,2,2] columns
    area = 0.5 * np.abs(np.linalg.det(Dm))
    Dm_inv = np.linalg.inv(Dm)
    mass = np.zeros(n)
    np.add.at(mass, tris.ravel(), np.repeat(area / 3, 3))
    mass *= DENSITY

    edge_tris = {}
    for t, tri in enumerate(tris):
        for k in range(3):
            a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            edge_tris.setdefault((min(a, b), max(a, b)), []).append((t, a, b, c))
    hinges, hinge_w, edges = [], [], []
    for (a, b), lst in edge_tris.items():
        edges.append((a, b))
        if len(lst) == 2:
            (t1, a1, b1, c1), (t2, a2, b2, c2) = lst
            # hinge (a1, b1, c1, d): tri1 = (a1, b1, c1) in winding order, d opposite in tri2
            hinges.append((a1, b1, c1, c2))
            e2 = np.sum((rest[a1] - rest[b1]) ** 2)
            hinge_w.append(3 * e2 / (area[t1] + area[t2]))
    hinges = np.array(hinges, np.int32).reshape(-1, 4)
    # Bending: E = kb * w * |sum_i g_i x_i|^2 with g_i = d(theta)/d(normal offset of vertex i) at rest.
    rest3 = np.stack([rest[:, 0], np.zeros(n), rest[:, 1]], 1)
    g = np.asarray(jax.vmap(jax.grad(dihedral))(jnp.asarray(rest3[hinges], jnp.float32)))[..., 1]
    def incidence(elems, width):
        """Per-vertex list of flat (element * k + slot) indices, padded with -1."""
        lst = [[] for _ in range(n)]
        for e, el in enumerate(elems):
            for slot, vtx in enumerate(el):
                lst[vtx].append(e * elems.shape[1] + slot)
        out = -np.ones((n, width), np.int32)
        for vtx, l in enumerate(lst):
            assert len(l) <= width, (len(l), width)
            out[vtx, :len(l)] = l
        return out

    return dict(
        tri_inc=incidence(tris, 8), hinge_inc=incidence(hinges, 16),
        hinge_g=g.astype(np.float32),
        rest=rest.astype(np.float32), tris=tris, Dm_inv=Dm_inv.astype(np.float32),
        area=area.astype(np.float32), mass=mass.astype(np.float32),
        hinges=hinges, hinge_w=np.array(hinge_w, np.float32),
        edges=np.array(edges, np.int32),
    )


# ---------------------------------------------------------------------------- energies

def green_strain(xt, dm_inv):
    """xt [3,3] triangle vertices -> (E11, E22, E12)."""
    Ds = jnp.stack([xt[1] - xt[0], xt[2] - xt[0]], -1)  # [3,2]
    F = Ds @ dm_inv
    C = F.T @ F
    return jnp.stack([0.5 * (C[0, 0] - 1), 0.5 * (C[1, 1] - 1), 0.5 * C[0, 1]])


def dihedral(xh):
    """Signed dihedral angle of hinge (a, b, c, d); 0 when flat."""
    a, b, c, d = xh
    e = b - a
    n1 = jnp.cross(b - a, c - a)
    n2 = jnp.cross(a - b, d - b)
    en = e / jnp.sqrt(jnp.sum(e * e) + 1e-20)
    return jnp.arctan2(jnp.dot(jnp.cross(n1, n2), en), jnp.dot(n1, n2) + 1e-30)


def barrier(d):
    """IPC barrier -(d - dhat)^2 log(d / dhat) for 0 < d < dhat, else 0 (inf if d <= 0)."""
    dc = jnp.clip(d, 1e-12, DHAT)
    val = -((dc - DHAT) ** 2) * jnp.log(dc / DHAT)
    return jnp.where(d <= 0, jnp.inf, jnp.where(d < DHAT, val, 0.0))


def barrier_d1(d):
    dc = jnp.clip(d, 1e-12, DHAT)
    g = -2 * (dc - DHAT) * jnp.log(dc / DHAT) - (dc - DHAT) ** 2 / dc
    return jnp.where(d < DHAT, g, 0.0)


def barrier_d2(d):
    dc = jnp.clip(d, 1e-12, DHAT)
    h = -2 * jnp.log(dc / DHAT) - 4 * (dc - DHAT) / dc + (dc - DHAT) ** 2 / dc ** 2
    return jnp.where(d < DHAT, h, 0.0)


def f0(y, eps):
    return jnp.where(y < eps, -y ** 3 / (3 * eps ** 2) + y ** 2 / eps + eps / 3, y)


def f1_over_y(y, eps):
    return jnp.where(y < eps, 2 / eps - y / eps ** 2, 1 / jnp.maximum(y, 1e-12))


def psd2(S):
    """Closed-form projection of symmetric 2x2 matrices [..., 2, 2] onto the PSD cone."""
    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    mid = 0.5 * (a + c)
    rad = jnp.sqrt(0.25 * (a - c) ** 2 + b * b)
    l1, l2 = mid + rad, mid - rad  # l1 >= l2
    # eigenvector of l1: (b, l1 - a) or (l1 - c, b), whichever is better conditioned
    ux, uy = b, l1 - a
    wx, wy = l1 - c, b
    use_w = (wx * wx + wy * wy) > (ux * ux + uy * uy)
    vx, vy = jnp.where(use_w, wx, ux), jnp.where(use_w, wy, uy)
    nrm = jnp.sqrt(vx * vx + vy * vy)
    ok = nrm > 1e-30
    vx, vy = jnp.where(ok, vx / jnp.where(ok, nrm, 1.0), 1.0), jnp.where(ok, vy / jnp.where(ok, nrm, 1.0), 0.0)
    p1, p2 = jnp.maximum(l1, 0.0), jnp.maximum(l2, 0.0)
    # S+ = p1 v v^T + p2 w w^T with w = (-vy, vx)
    s00 = p1 * vx * vx + p2 * vy * vy
    s01 = (p1 - p2) * vx * vy
    s11 = p1 * vy * vy + p2 * vx * vx
    return jnp.stack([jnp.stack([s00, s01], -1), jnp.stack([s01, s11], -1)], -2)


# ---------------------------------------------------------------------------- obstacles

def obstacle_sdf(x, obs):
    """Distances [N,O], outward normals [N,O,3] and surface velocities [N,O,3].

    Obstacle 0 is the table plane y = 0, then spheres, then capsules. Inactive
    obstacles get distance +inf.
    """
    n = x.shape[0]
    d_table = x[:, 1:2]
    n_table = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), (n, 1, 3))
    v_table = jnp.zeros((n, 1, 3))

    sc, sr, son, sv = obs["sph_c"], obs["sph_r"], obs["sph_on"], obs["sph_v"]
    ds = x[:, None, :] - sc[None]
    dist = jnp.sqrt(jnp.sum(ds * ds, -1) + 1e-20)
    d_sph = jnp.where(son[None] > 0, dist - sr[None], jnp.inf)
    n_sph = ds / dist[..., None]
    v_sph = jnp.broadcast_to(sv[None], ds.shape)

    ca, cb, cr, con = obs["cap_a"], obs["cap_b"], obs["cap_r"], obs["cap_on"]
    va, vb = obs["cap_va"], obs["cap_vb"]
    ab = cb - ca
    t = jnp.clip(jnp.sum((x[:, None, :] - ca[None]) * ab[None], -1) / (jnp.sum(ab * ab, -1)[None] + 1e-12), 0, 1)
    q = ca[None] + t[..., None] * ab[None]
    dq = x[:, None, :] - q
    dist = jnp.sqrt(jnp.sum(dq * dq, -1) + 1e-20)
    d_cap = jnp.where(con[None] > 0, dist - cr[None], jnp.inf)
    n_cap = dq / dist[..., None]
    v_cap = va[None] + t[..., None] * (vb - va)[None]

    d = jnp.concatenate([d_table, d_sph, d_cap], 1) - OBS_OFFSET
    nrm = jnp.concatenate([n_table, n_sph, n_cap], 1)
    vel = jnp.concatenate([v_table, v_sph, v_cap], 1)
    return d, nrm, vel


# ---------------------------------------------------------------------------- self-contact pairs

def self_pairs(x, rest, active, radius, k=SELF_K):
    """Up to k close, non-neighbouring vertices per vertex (brute force; N is small)."""
    n = x.shape[0]
    d2 = jnp.sum((x[:, None] - x[None]) ** 2, -1)
    r2 = jnp.sum((rest[:, None] - rest[None]) ** 2, -1)
    ok = (d2 < radius * radius) & (r2 > SELF_EXCLUDE ** 2) & active[:, None] & active[None]
    score = jnp.where(ok, -d2, -jnp.inf)
    val, idx = jax.lax.top_k(score, k)
    return idx, jnp.isfinite(val)


# ---------------------------------------------------------------------------- one implicit step

def make_step(mesh):
    tris = jnp.asarray(mesh["tris"])
    dm_inv = jnp.asarray(mesh["Dm_inv"])
    area = jnp.asarray(mesh["area"])
    hinges = jnp.asarray(mesh["hinges"])
    hinge_w = jnp.asarray(mesh["hinge_w"])
    hinge_g = jnp.asarray(mesh["hinge_g"])  # [H,4]
    mass = jnp.asarray(mesh["mass"])
    rest = jnp.asarray(np.concatenate([mesh["rest"], np.zeros((len(mesh["rest"]), 1), np.float32)], 1))
    Q = jnp.asarray(Q_STVK)
    n = mass.shape[0]
    dmi = mesh["Dm_inv"]
    Bm = jnp.asarray(np.stack([-(dmi[:, 0] + dmi[:, 1]), dmi[:, 0], dmi[:, 1]], 1))  # [T,3(v),2(a)]
    strain_jac = jax.vmap(jax.jacfwd(green_strain))
    strains = jax.vmap(green_strain)
    h = FRAME_DT / SUBSTEPS
    tri_area_rest = area

    tri_inc = jnp.asarray(mesh["tri_inc"])
    hinge_inc = jnp.asarray(mesh["hinge_inc"])

    def scatter(vals, idx):
        return jnp.zeros((n, 3)).at[idx.reshape(-1)].add(vals.reshape(-1, 3))

    def pull(vals, inc):
        """Sum per-element vertex contributions [E, k, 3] into vertices by gathering (no atomics)."""
        flat = jnp.concatenate([vals.reshape(-1, 3), jnp.zeros((1, 3))], 0)
        return jnp.sum(flat[jnp.where(inc >= 0, inc, flat.shape[0] - 1)], 1)

    def substep(x, v, free, x_pin, obs0, obs1, kb, mu, wind):
        """obs0/obs1: obstacle state at the start/end of the substep."""
        # --- explicit forces: gravity + aerodynamic drag on triangles
        xt, vt = x[tris], v[tris]
        nrm = jnp.cross(xt[:, 1] - xt[:, 0], xt[:, 2] - xt[:, 0])
        a2 = jnp.sqrt(jnp.sum(nrm * nrm, -1) + 1e-20)
        nrm = nrm / a2[:, None]
        vrel = jnp.mean(vt, 1) - wind
        vn = jnp.sum(vrel * nrm, -1)
        f_tri = -0.5 * AIR_DENSITY * DRAG_COEF * (0.5 * a2) * vn * jnp.abs(vn)
        f_air = scatter(jnp.repeat((f_tri[:, None] * nrm / 3)[:, None], 3, 1), tris)
        x_tilde = x + h * v + h * h * (jnp.asarray(GRAVITY) + f_air / mass[:, None])

        # --- start point: current positions, pinned vertices at their targets,
        # anything the moved obstacles now overlap pushed back out.
        x0 = jnp.where(free[:, None], x, x_pin)
        d_new, n_new, _ = obstacle_sdf(x0, obs1)
        push = jnp.sum(jnp.where(d_new < 0.3 * DHAT, (0.3 * DHAT - d_new), 0.0)[..., None] * n_new, 1)
        x0 = x0 + push

        # --- lagged contact data (friction, self pairs)
        d_c, n_c, v_c = obstacle_sdf(x, obs0)
        lam = KAPPA * mass[:, None] * jnp.abs(barrier_d1(d_c))  # normal force magnitude
        lam = jnp.where(jnp.isfinite(d_c) & (d_c < DHAT), lam, 0.0)
        obs_disp = h * v_c
        eps_x = FRICTION_EPS_V * h
        pair_idx, pair_ok = self_pairs(x0, rest, jnp.ones(n, bool), SELF_THICK + DHAT + 0.01)
        dpair = jnp.sqrt(jnp.sum((x0[:, None] - x0[pair_idx]) ** 2, -1) + 1e-20) - SELF_THICK
        pair_ok = pair_ok & (dpair > 0)
        pm = 0.5 * (mass[:, None] + mass[pair_idx])

        def energy(y):
            e_in = 0.5 * jnp.sum(mass[:, None] * (y - x_tilde) ** 2) / (h * h)
            e = strains(y[tris], dm_inv)
            e_mem = jnp.sum(area * jnp.einsum("ti,ij,tj->t", e, Q, e))
            kx = jnp.einsum("hi,hid->hd", hinge_g, y[hinges])
            e_bend = kb * jnp.sum(hinge_w * jnp.sum(kx * kx, -1))
            d, _, _ = obstacle_sdf(y, obs1)
            e_obs = jnp.sum(KAPPA * mass[:, None] * barrier(d))
            dp = jnp.sqrt(jnp.sum((y[:, None] - y[pair_idx]) ** 2, -1) + 1e-20) - SELF_THICK
            e_self = 0.5 * jnp.sum(jnp.where(pair_ok, KAPPA * pm * barrier(dp), 0.0))
            u = (y - x)[:, None, :] - obs_disp
            ut = u - jnp.sum(u * n_c, -1, keepdims=True) * n_c
            e_fric = jnp.sum(mu * lam * f0(jnp.sqrt(jnp.sum(ut * ut, -1) + 1e-20), eps_x))
            return e_in + e_mem + e_bend + e_obs + e_self + e_fric

        grad_e = jax.grad(energy)

        def hess_parts(y):
            J = strain_jac(y[tris], dm_inv).reshape(-1, 3, 9)  # [T,3,9]
            # Geometric stiffness from the membrane stress, with the 2x2 stress clamped to PSD.
            e = strains(y[tris], dm_inv)
            tr = e[:, 0] + e[:, 1]
            S = area[:, None, None] * jnp.stack([
                jnp.stack([2 * MU_S * e[:, 0] + LAM_S * tr, 2 * MU_S * e[:, 2]], -1),
                jnp.stack([2 * MU_S * e[:, 2], 2 * MU_S * e[:, 1] + LAM_S * tr], -1)], -2)
            S = psd2(S)
            Gm = jnp.einsum("tva,tab,twb->tvw", Bm, S, Bm)  # [T,3,3]
            d, nn, _ = obstacle_sdf(y, obs1)
            c_obs = jnp.where(jnp.isfinite(d), KAPPA * mass[:, None] * barrier_d2(d), 0.0)
            diff = y[:, None] - y[pair_idx]
            dist = jnp.sqrt(jnp.sum(diff * diff, -1) + 1e-20)
            pn = diff / dist[..., None]
            c_self = jnp.where(pair_ok, 0.5 * KAPPA * pm * barrier_d2(dist - SELF_THICK), 0.0)
            u = (y - x)[:, None, :] - obs_disp
            ut = u - jnp.sum(u * n_c, -1, keepdims=True) * n_c
            c_fric = mu * lam * f1_over_y(jnp.sqrt(jnp.sum(ut * ut, -1) + 1e-20), eps_x)
            return J, c_obs, nn, c_self, pn, c_fric, Gm

        def hvp(parts, p):
            J, c_obs, nn, c_self, pn, c_fric, Gm = parts
            p = jnp.where(free[:, None], p, 0.0)
            out = mass[:, None] * p / (h * h)
            pt = p[tris].reshape(-1, 9)
            jp = jnp.einsum("tkd,td->tk", J, pt)
            tri_out = (2 * area[:, None] * jnp.einsum("tkd,kl,tl->td", J, Q, jp)).reshape(-1, 3, 3)
            tri_out = tri_out + jnp.einsum("tvw,twd->tvd", Gm, p[tris])
            out += pull(tri_out, tri_inc)
            kp = jnp.einsum("hi,hid->hd", hinge_g, p[hinges])
            out += pull((2 * kb * hinge_w)[:, None, None] * hinge_g[..., None] * kp[:, None, :], hinge_inc)
            out += jnp.sum(c_obs[..., None] * jnp.sum(nn * p[:, None, :], -1, keepdims=True) * nn, 1)
            # Pairs are listed from both ends, so each vertex only pulls its own list (x2 for both halves).
            dp = p[:, None, :] - p[pair_idx]
            out += 2 * jnp.sum(c_self[..., None] * jnp.sum(pn * dp, -1, keepdims=True) * pn, 1)
            pt_ = p[:, None, :] - jnp.sum(p[:, None, :] * n_c, -1, keepdims=True) * n_c
            out += jnp.sum(c_fric[..., None] * pt_, 1)
            return jnp.where(free[:, None], out, 0.0)

        def diag(parts):
            J, c_obs, nn, c_self, pn, c_fric, Gm = parts
            dg = jnp.broadcast_to(mass[:, None] / (h * h), (n, 3))
            dt = 2 * area[:, None] * jnp.einsum("tkd,kl,tld->td", J, Q, J)
            dg = dg + pull(dt.reshape(-1, 3, 3) + jnp.diagonal(Gm, axis1=1, axis2=2)[..., None], tri_inc)
            dg = dg + pull(jnp.broadcast_to(((2 * kb * hinge_w)[:, None] * hinge_g ** 2)[..., None], hinge_g.shape + (3,)), hinge_inc)
            dg = dg + jnp.sum(c_obs[..., None] * nn * nn, 1)
            dg = dg + 2 * jnp.sum(c_self[..., None] * pn * pn, 1)
            dg = dg + jnp.sum(c_fric[..., None] * (1 - n_c * n_c), 1)
            return jnp.where(free[:, None], dg, 1.0)

        def feasible(y):
            d, _, _ = obstacle_sdf(y, obs1)
            dp = jnp.sqrt(jnp.sum((y[:, None] - y[pair_idx]) ** 2, -1) + 1e-20) - SELF_THICK
            return jnp.all(d > 0) & jnp.all(jnp.where(pair_ok, dp > 0, True))

        def cg(parts, b):
            dinv = 1.0 / diag(parts)
            xk = jnp.zeros_like(b)
            r = b
            z = dinv * r
            p = z
            rz = jnp.sum(r * z)
            b_norm = jnp.sqrt(jnp.sum(b * b)) + 1e-30

            def cond(s):
                k, _, r, _, _ = s
                return (k < CG_ITERS) & (jnp.sqrt(jnp.sum(r * r)) > CG_TOL * b_norm)

            def body(s):
                k, xk, r, p, rz = s
                Ap = hvp(parts, p)
                alpha = rz / (jnp.sum(p * Ap) + 1e-30)
                xk = xk + alpha * p
                r = r - alpha * Ap
                z = dinv * r
                rz_new = jnp.sum(r * z)
                p = z + (rz_new / (rz + 1e-30)) * p
                return k + 1, xk, r, p, rz_new

            k, xk, _, _, _ = jax.lax.while_loop(cond, body, (0, xk, r, p, rz))
            return xk, k

        def newton_cond(s):
            it, _, dx, _ = s
            return (it < NEWTON_ITERS) & (dx > NEWTON_TOL)

        def newton_body(s):
            it, y, _, cg_total = s
            g = jnp.where(free[:, None], grad_e(y), 0.0)
            parts = hess_parts(y)
            p, k = cg(parts, -g)
            e0 = energy(y)
            slope = jnp.sum(g * p)

            # Step-size filter (first-order CCD): never close more than 80% of any contact gap.
            d, nn, _ = obstacle_sdf(y, obs1)
            closing = -jnp.sum(nn * p[:, None, :], -1)
            a_obs = jnp.min(jnp.where((closing > 0) & jnp.isfinite(d), 0.8 * d / jnp.maximum(closing, 1e-20), 1.0))
            diff = y[:, None] - y[pair_idx]
            dist = jnp.sqrt(jnp.sum(diff * diff, -1) + 1e-20)
            closing = -jnp.sum(diff / dist[..., None] * (p[:, None] - p[pair_idx]), -1)
            a_self = jnp.min(jnp.where(pair_ok & (closing > 0), 0.8 * (dist - SELF_THICK) / jnp.maximum(closing, 1e-20), 1.0))
            a_max = jnp.minimum(1.0, jnp.minimum(a_obs, a_self))

            def ls_cond(a):
                y1 = y + a * p
                bad = ~feasible(y1) | (energy(y1) > e0 + 1e-4 * a * slope + 1e-7 * jnp.abs(e0))
                return bad & (a > 1e-4 * a_max)

            alpha = jax.lax.while_loop(ls_cond, lambda a: 0.5 * a, a_max)
            alpha = jnp.where(feasible(y + alpha * p), alpha, 0.0)
            y = y + alpha * p
            if DEBUG:
                jax.debug.print("newton it={it} |p|={pn:.2e} alpha={a:.3f} cg={k} |g|={g:.2e}", it=it, pn=jnp.max(jnp.abs(p)), a=alpha, k=k, g=jnp.max(jnp.abs(g)))
            return it + 1, y, jnp.max(jnp.abs(alpha * p)), cg_total + k

        # Start from the velocity predictor, backing off towards x0 until it is feasible.
        pred = jnp.where(free[:, None], h * v, 0.0)
        frac = jax.lax.while_loop(lambda a: (~feasible(x0 + a * pred)) & (a > 0.01), lambda a: 0.5 * a, 1.0)
        start = x0 + jnp.where(feasible(x0 + frac * pred), frac, 0.0) * pred
        it, y, _, cg_total = jax.lax.while_loop(newton_cond, newton_body, (0, start, jnp.inf, 0))
        v_new = (y - x) / h
        return y, v_new, it, cg_total

    return substep


# ---------------------------------------------------------------------------- trajectories

def obstacle_state(sc, t_frac):
    """Interpolate per-frame obstacle tracks at fractional frame t_frac; velocities per frame."""
    f0_, f1_ = jnp.floor(t_frac).astype(jnp.int32), jnp.floor(t_frac).astype(jnp.int32) + 1
    w = t_frac - f0_
    lerp = lambda a: a[f0_] * (1 - w) + a[f1_] * w
    vel = lambda a: (a[f1_] - a[f0_]) / FRAME_DT
    return {
        "sph_c": lerp(sc["sph_c"]), "sph_v": vel(sc["sph_c"]), "sph_r": sc["sph_r"], "sph_on": sc["sph_on"],
        "cap_a": lerp(sc["cap_a"]), "cap_b": lerp(sc["cap_b"]), "cap_va": vel(sc["cap_a"]),
        "cap_vb": vel(sc["cap_b"]), "cap_r": sc["cap_r"], "cap_on": sc["cap_on"],
    }


def make_rollout(mesh, frames):
    """Returns f(scene) -> (positions [frames+1, N, 3], newton iters, cg iters)."""
    substep = make_step(mesh)
    n = len(mesh["mass"])

    def rollout(sc):
        hv = sc["h_vid"]  # [H] vertex index (-1 unused)
        valid_h = hv >= 0

        def frame(carry, f):
            x, v, anchor = carry
            # grasps start at the beginning of their start frame; anchor = where the vertex is then
            starting = valid_h & (sc["h_t0"] == f)
            anchor = jnp.where(starting[:, None], x[jnp.maximum(hv, 0)], anchor)

            def sub(s, st):
                x, v, its, cgs = st
                t0 = f + s / SUBSTEPS
                t1 = f + (s + 1.0) / SUBSTEPS
                held = valid_h & (sc["h_t0"] <= f) & (f < sc["h_t1"])
                w = (s + 1.0) / SUBSTEPS
                disp = sc["h_disp"][f] * (1 - w) + sc["h_disp"][f + 1] * w
                target = anchor + disp
                idx = jnp.where(held, hv, n)  # out of range -> dropped
                free = ~jnp.zeros(n, bool).at[idx].set(True, mode="drop")
                x_pin = x.at[idx].set(target, mode="drop")
                wind = sc["wind"][f]
                y, vn, it, cg = substep(x, v, free, x_pin, obstacle_state(sc, t0), obstacle_state(sc, t1),
                                        sc["kb"], sc["mu"], wind)
                return y, vn, its + it, cgs + cg

            x, v, its, cgs = jax.lax.fori_loop(0, SUBSTEPS, sub, (x, v, 0, 0))
            return (x, v, anchor), (x, its, cgs)

        init = (sc["x0"], jnp.zeros((n, 3)), jnp.zeros((MAX_HANDLES, 3)))
        _, (xs, its, cgs) = jax.lax.scan(frame, init, jnp.arange(frames))
        return jnp.concatenate([sc["x0"][None], xs], 0), its, cgs

    return rollout
