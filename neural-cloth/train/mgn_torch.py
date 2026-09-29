"""PyTorch port of mgn.py (the same network, op for op), for training on GPUs where XLA's
generated backward kernels fault. Parameters convert to and from mgn.py's dict layout,
so export.py, the golden tests and the web engine are unchanged.
"""
import numpy as np
import torch
import torch.nn.functional as F

# Constants mirror mgn.py / cloth.py (kept here so this module does not import JAX).
SPACING = 0.02
OBS_OFFSET = 0.003
FRAME_DT = 1.0 / 60.0
SELF_EXCLUDE = 2.1 * SPACING
D_CLIP = 0.05
WORLD_R = 0.03
WORLD_K = 8
NBR_K = 8
STRIDES = (1, 2, 4)
N_MAX = 31 * 31
LEVEL_MAX = (N_MAX, 16 * 16, 9 * 9)
KB_LOG_MID, KB_LOG_HALF = -5.25, 1.0
WIND_SCALE = 4.0
GRIP_K = 2
GRIP_SCALE = 0.5
CONTACT_D = 0.006
NODE_IN = 21 + 10 * GRIP_K


# ---------------------------------------------------------------------------- graph tables (numpy)

def grid_edges(nx, ny):
    """Edges of cloth.grid_mesh (cells split along alternating diagonals)."""
    vid = lambda i, j: i * ny + j
    edges = set()
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris = [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
            for t in tris:
                for k in range(3):
                    p, q = t[k], t[(k + 1) % 3]
                    edges.add((min(p, q), max(p, q)))
    return sorted(edges)


def _coarse_axis(n, s):
    idx = list(range(0, n, s))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def graph_tables(nx, ny):
    """Same tables as mgn.graph_tables."""
    n = nx * ny
    out = {}
    nb = [[] for _ in range(n)]
    for a, b in grid_edges(nx, ny):
        nb[a].append(b)
        nb[b].append(a)
    nodes = np.full(LEVEL_MAX[0], N_MAX, np.int64)
    nodes[:n] = np.arange(n)
    nbr = np.zeros((LEVEL_MAX[0], NBR_K), np.int64)
    mask = np.zeros((LEVEL_MAX[0], NBR_K), bool)
    for v, lst in enumerate(nb):
        nbr[v, :len(lst)] = lst
        mask[v, :len(lst)] = True
    out.update(nodes0=nodes, nbr0=nbr, mask0=mask)
    for lvl, s in enumerate(STRIDES[1:], start=1):
        I, J = _coarse_axis(nx, s), _coarse_axis(ny, s)
        pos = {(a, b): k for k, (a, b) in enumerate((a, b) for a in range(len(I)) for b in range(len(J)))}
        nodes = np.full(LEVEL_MAX[lvl], N_MAX, np.int64)
        nbr = np.zeros((LEVEL_MAX[lvl], NBR_K), np.int64)
        mask = np.zeros((LEVEL_MAX[lvl], NBR_K), bool)
        for (a, b), k in pos.items():
            nodes[k] = I[a] * ny + J[b]
            slot = 0
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if (da, db) != (0, 0) and (a + da, b + db) in pos:
                        nbr[k, slot] = I[a + da] * ny + J[b + db]
                        mask[k, slot] = True
                        slot += 1
        out.update({f"nodes{lvl}": nodes, f"nbr{lvl}": nbr, f"mask{lvl}": mask})
    rest = np.zeros((N_MAX, 2), np.float32)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    rest[:n] = np.stack([ii.ravel(), jj.ravel()], -1) * SPACING
    valid = np.zeros(N_MAX, bool)
    valid[:n] = True
    out.update(rest=rest, valid=valid)
    return out


# ---------------------------------------------------------------------------- parameters

class Params:
    """Flat list of tensors with the same tree structure as mgn.init_params."""

    def __init__(self, jax_params, device):
        self.tree = _map_tree(jax_params, lambda a: torch.tensor(np.asarray(a), dtype=torch.float32,
                                                                   device=device, requires_grad=True))

    def tensors(self):
        return _leaves(self.tree)

    def to_numpy(self):
        return _map_tree(self.tree, lambda t: t.detach().cpu().numpy())


def _map_tree(t, f):
    if isinstance(t, dict):
        return {k: _map_tree(v, f) for k, v in t.items()}
    if isinstance(t, (list, tuple)):
        return [_map_tree(v, f) for v in t]
    return f(t)


def _leaves(t):
    if isinstance(t, dict):
        return [x for k in sorted(t) for x in _leaves(t[k])]
    if isinstance(t, (list, tuple)):
        return [x for v in t for x in _leaves(v)]
    return [t]


def _mlp(p, x):
    layers = p["layers"]
    for i, lyr in enumerate(layers):
        x = x @ lyr["w"] + lyr["b"]
        if i < len(layers) - 1:
            x = F.relu(x)
    if "ln" in p:
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        x = (x - mu) * torch.rsqrt(var + 1e-5) * p["ln"]["g"] + p["ln"]["b"]
    return x


def _edge_mlp(p, e, v_send, v_recv):
    L = e.shape[-1]
    first = p["layers"][0]
    w = first["w"]
    h = e @ w[:L] + v_send @ w[L:2 * L] + (v_recv @ w[2 * L:]).unsqueeze(-2) + first["b"]
    return _mlp({"layers": p["layers"][1:], "ln": p["ln"]}, F.relu(h))


# ---------------------------------------------------------------------------- batched helpers

def bgather(x, idx):
    """x [B, N, ...], idx [B, ...] -> x[b, idx[b]]."""
    B = x.shape[0]
    bi = torch.arange(B, device=x.device).view((B,) + (1,) * (idx.dim() - 1))
    return x[bi, idx]


def pad(x):
    return torch.cat([x, torch.zeros_like(x[:, :1])], 1)


def obstacle_sdf(x, obs):
    """Batched cloth.obstacle_sdf. x [B,N,3]; obs tensors with a leading B dim."""
    B, N, _ = x.shape
    d_table = x[..., 1:2]
    n_table = torch.tensor([0.0, 1.0, 0.0], device=x.device).expand(B, N, 1, 3)
    v_table = torch.zeros(B, N, 1, 3, device=x.device)
    ds = x[:, :, None, :] - obs["sph_c"][:, None]
    dist = torch.sqrt((ds * ds).sum(-1) + 1e-20)
    d_sph = torch.where(obs["sph_on"][:, None] > 0, dist - obs["sph_r"][:, None], torch.full_like(dist, float("inf")))
    n_sph = ds / dist[..., None]
    v_sph = obs["sph_v"][:, None].expand_as(ds)
    ca, cb = obs["cap_a"], obs["cap_b"]
    ab = cb - ca
    t = ((x[:, :, None, :] - ca[:, None]) * ab[:, None]).sum(-1) / ((ab * ab).sum(-1)[:, None] + 1e-12)
    t = t.clamp(0, 1)
    q = ca[:, None] + t[..., None] * ab[:, None]
    dq = x[:, :, None, :] - q
    dist = torch.sqrt((dq * dq).sum(-1) + 1e-20)
    d_cap = torch.where(obs["cap_on"][:, None] > 0, dist - obs["cap_r"][:, None], torch.full_like(dist, float("inf")))
    n_cap = dq / dist[..., None]
    v_cap = obs["cap_va"][:, None] + t[..., None] * (obs["cap_vb"] - obs["cap_va"])[:, None]
    d = torch.cat([d_table, d_sph, d_cap], 2) - OBS_OFFSET
    return d, torch.cat([n_table, n_sph, n_cap], 2), torch.cat([v_table, v_sph, v_cap], 2)


def world_neighbors(x, rest, valid):
    d2 = ((x[:, :, None] - x[:, None]) ** 2).sum(-1)
    r2 = ((rest[:, :, None] - rest[:, None]) ** 2).sum(-1)
    ok = (d2 < WORLD_R ** 2) & (r2 > SELF_EXCLUDE ** 2) & valid[:, :, None] & valid[:, None]
    val, idx = torch.topk(torch.where(ok, -d2, torch.full_like(d2, -float("inf"))), WORLD_K, dim=-1)
    return idx, torch.isfinite(val)


def grip_features(x0, rest, held):
    """mgn.grip_features, batched."""
    B, N, _ = x0.shape
    cells = torch.round(rest / SPACING).long()
    d = cells[:, None] - cells[:, :, None]
    key = (d[..., 0] ** 2 + d[..., 1] ** 2) * 1024 + torch.arange(N, device=x0.device)[None, None]
    big = torch.iinfo(torch.int64).max
    val, idx = torch.topk(torch.where(held[:, None], key, torch.full_like(key, big)), GRIP_K, dim=-1, largest=False)
    on = (val < big)[..., None]
    off = torch.where(on, (bgather(x0, idx) - x0[:, :, None]) / GRIP_SCALE, torch.zeros(1, device=x0.device))
    dist = torch.where(on, torch.sqrt((val // 1024).clamp(max=10 ** 6).to(x0.dtype))[..., None] * SPACING / GRIP_SCALE,
                       torch.zeros(1, device=x0.device))
    return torch.cat([on.to(x0.dtype), off, dist], -1).reshape(B, N, 5 * GRIP_K)


def node_features(stats, x1, x0, kin_next, held, obs, kb, mu, wind, rest, valid):
    vs = stats["vel_std"]
    v = (x0 - x1 - stats["vel_mean"]) / vs
    typ = torch.stack([~held, held], -1).to(x0.dtype)
    kin = torch.where(held[..., None], (kin_next - x0) / vs, torch.zeros_like(x0))
    table = ((x0[..., 1:2] - OBS_OFFSET) / D_CLIP).clamp(0, 1)
    d, nrm, vel = obstacle_sdf(x0, obs)
    d, nrm, vel = d[..., 1:], nrm[..., 1:, :], vel[..., 1:, :]
    k = torch.argmin(d, -1)
    dn = torch.gather(d, 2, k[..., None])
    near = dn < D_CLIP
    nn_ = torch.where(near, torch.gather(nrm, 2, k[..., None, None].expand(-1, -1, 1, 3))[:, :, 0], torch.zeros_like(x0))
    nv = torch.gather(vel, 2, k[..., None, None].expand(-1, -1, 1, 3))[:, :, 0] * FRAME_DT / vs
    nv = torch.where(near, nv, torch.zeros_like(x0))
    dfeat = (dn / D_CLIP).clamp(-1, 1)
    B, N, _ = x0.shape
    mat = torch.stack([(torch.log10(kb) - KB_LOG_MID) / KB_LOG_HALF, mu], -1)[:, None].expand(B, N, 2)
    w = (wind / WIND_SCALE)[:, None].expand(B, N, 3)
    touching = (dn[..., 0] < CONTACT_D) & ~held & valid
    return torch.cat([v, typ, kin, table, dfeat, nn_, nv, mat, w, grip_features(x0, rest, held),
                      grip_features(x0, rest, touching)], -1)


def edge_features(xp, restp, nodes, nbr, scale):
    u = (bgather(restp, nbr) - bgather(restp, nodes)[:, :, None]) / scale
    dx = (bgather(xp, nbr) - bgather(xp, nodes)[:, :, None]) / scale
    lu = torch.sqrt((u * u).sum(-1, keepdim=True) + 1e-12)
    lx = torch.sqrt((dx * dx).sum(-1, keepdim=True) + 1e-12)
    return torch.cat([u, lu, dx, lx, (lx / lu - 1.0) * 10.0], -1)


def world_edge_features(x, nbr):
    dx = (bgather(x, nbr) - x[:, :, None]) / WORLD_R
    return torch.cat([dx, torch.sqrt((dx * dx).sum(-1, keepdim=True) + 1e-12)], -1)


def forward(p, schedule, node_in, x, g, world_nbr, world_mask):
    v = _mlp(p["enc_node"], node_in)
    xp, restp = pad(x), pad(g["rest"])
    e = [_mlp(p["enc_edge"][l], edge_features(xp, restp, g[f"nodes{l}"], g[f"nbr{l}"], s * SPACING))
         for l, s in enumerate(STRIDES)]
    ew = _mlp(p["enc_world"], world_edge_features(x, world_nbr))
    wm = world_mask[..., None].to(v.dtype)
    for lvl, blk in zip(schedule, p["proc"]):
        nodes, nbr = g[f"nodes{lvl}"], g[f"nbr{lvl}"]
        mask = g[f"mask{lvl}"][..., None].to(v.dtype)
        vp = pad(v)
        v_recv = bgather(vp, nodes)
        e_new = _edge_mlp(blk["edge"], e[lvl], bgather(vp, nbr), v_recv) * mask
        if lvl == 0:
            ew_new = _edge_mlp(blk["world"], ew, bgather(v, world_nbr), v) * wm
            v = v + _mlp(blk["node"], torch.cat([v, e_new.sum(2), ew_new.sum(2)], -1))
            ew = ew + ew_new
        else:
            upd = _mlp(blk["node"], torch.cat([v_recv, e_new.sum(2)], -1))
            # v[nodes] += upd, padding rows (index N_MAX) land in a dropped extra row
            vp = pad(v)
            bi = torch.arange(v.shape[0], device=v.device)[:, None].expand_as(nodes)
            vp = vp.index_put((bi, nodes), upd, accumulate=True)
            v = vp[:, :-1]
        e[lvl] = e[lvl] + e_new
    return _mlp(p["dec"], v)


def predict(p, schedule, stats, x1, x0, kin_next, held, obs, kb, mu, wind, g):
    node_in = node_features(stats, x1, x0, kin_next, held, obs, kb, mu, wind, g["rest"], g["valid"])
    world_nbr, world_mask = world_neighbors(x0, g["rest"], g["valid"])
    return forward(p, schedule, node_in, x0, g, world_nbr, world_mask)


def project(x, obs):
    d, nrm, _ = obstacle_sdf(x, obs)
    push = (torch.where(torch.isfinite(d) & (d < 0), -d, torch.zeros_like(d))[..., None] * nrm).sum(2)
    return x + push


def integrate(stats, a_norm, x1, x0, kin_next, held, obs):
    x_next = 2 * x0 - x1 + a_norm * stats["acc_std"] + stats["acc_mean"]
    x_next = project(x_next, obs)
    return torch.where(held[..., None], kin_next, x_next)


def grid_tris(nx, ny):
    """Triangles of cloth.grid_mesh (for rendering)."""
    vid = lambda i, j: i * ny + j
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    return np.array(tris, np.int32)
