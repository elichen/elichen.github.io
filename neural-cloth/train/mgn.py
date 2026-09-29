"""Hierarchical MeshGraphNets-style cloth model, in plain JAX.

Based on MeshGraphNets (Pfaff et al. 2021) with a HOOD-style hierarchy (Grigorev
et al. 2023): stiff, nearly inextensible cloth has to feel a pull at one corner
across the whole sheet within a frame, which fine-mesh message passing alone
cannot reach.

Graph:
  * nodes: cloth vertices; latents are shared by every level
  * level 0: the cloth's own mesh edges, plus "world" edges between vertices that
    are close in space but far apart on the cloth (self-contact)
  * levels 1, 2: grid-native coarse levels that keep every 2nd / 4th row and
    column (plus the last), linked to their 8 neighbours at that stride
All edge sets are padded per-receiver slots: receiver r, slot k, sender nbr[r, k].
A message-passing step works on one level: edge MLP for that level's edges,
sum at receivers, node MLP updates only that level's nodes (residual, LayerNorm).

Node features: current velocity, whether a gripper holds the vertex and where it
moves it next, height above the table, the nearest other obstacle (distance,
normal, surface velocity), bending stiffness, friction and wind.
Edge features: rest offset, current offset, lengths and strain.

The web engine reimplements this forward pass; keep them in sync.
"""
import numpy as np
import jax
import jax.numpy as jnp

import cloth as C

D_CLIP = 0.05          # obstacle distance features saturate here (m)
WORLD_R = 0.03         # world-edge radius (m)
WORLD_K = 8            # world neighbours kept per vertex
NBR_K = 8              # neighbour slots per receiver on every level
STRIDES = (1, 2, 4)
SCHEDULE = (0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0)
N_MAX = 31 * 31
LEVEL_MAX = (N_MAX, 16 * 16, 9 * 9)
GRIP_K = 2             # held vertices, and vertices touching a ball or rod, each vertex is told about
GRIP_SCALE = 0.5       # their offsets and rest distances are divided by this (m)
CONTACT_D = 0.006      # a vertex this close to a ball or rod (not the table) counts as touching it (m)
NODE_IN = 21 + 10 * GRIP_K
EDGE_IN = 8
WORLD_IN = 4
KB_LOG_MID, KB_LOG_HALF = -5.25, 1.0   # log10(kb) -> roughly [-1, 1]
WIND_SCALE = 4.0


def default_config():
    return dict(latent=128, hidden=128, mlp_layers=2, schedule=list(SCHEDULE), noise_std=5e-4, noise_gamma=0.1)


# ---------------------------------------------------------------------------- graph tables

def _coarse_axis(n, s):
    idx = list(range(0, n, s))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def graph_tables(nx, ny, mesh):
    """Padded tables for one cloth size (vertex id = i * ny + j).

    Returns dict with, per level l: nodes_l [LEVEL_MAX[l]] (vertex ids, N_MAX = padding),
    nbr_l [LEVEL_MAX[l], NBR_K] (vertex ids), mask_l, plus rest [N_MAX, 2] and valid [N_MAX].
    """
    n = nx * ny
    out = {}
    # level 0: mesh edges
    nb = [[] for _ in range(n)]
    for a, b in mesh["edges"]:
        nb[a].append(b)
        nb[b].append(a)
    nodes = np.full(LEVEL_MAX[0], N_MAX, np.int32)
    nodes[:n] = np.arange(n)
    nbr = np.zeros((LEVEL_MAX[0], NBR_K), np.int32)
    mask = np.zeros((LEVEL_MAX[0], NBR_K), bool)
    for v, lst in enumerate(nb):
        nbr[v, :len(lst)] = lst
        mask[v, :len(lst)] = True
    out.update(nodes0=nodes, nbr0=nbr, mask0=mask)
    # coarse levels: 8-neighbourhood on the strided grid
    for lvl, s in enumerate(STRIDES[1:], start=1):
        I, J = _coarse_axis(nx, s), _coarse_axis(ny, s)
        pos = {(a, b): k for k, (a, b) in enumerate((a, b) for a in range(len(I)) for b in range(len(J)))}
        m = len(pos)
        assert m <= LEVEL_MAX[lvl]
        nodes = np.full(LEVEL_MAX[lvl], N_MAX, np.int32)
        nbr = np.zeros((LEVEL_MAX[lvl], NBR_K), np.int32)
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
    rest[:n] = mesh["rest"]
    valid = np.zeros(N_MAX, bool)
    valid[:n] = True
    out.update(rest=rest, valid=valid)
    return out


# ---------------------------------------------------------------------------- params

def _init_linear(key, din, dout):
    return {"w": jax.random.truncated_normal(key, -2.0, 2.0, (din, dout)) / np.sqrt(din), "b": jnp.zeros((dout,))}


def _init_mlp(key, din, hidden, layers, dout, layer_norm=True):
    sizes = [din] + [hidden] * layers + [dout]
    keys = jax.random.split(key, len(sizes) - 1)
    p = {"layers": [_init_linear(k, a, b) for k, a, b in zip(keys, sizes[:-1], sizes[1:])]}
    if layer_norm:
        p["ln"] = {"g": jnp.ones((dout,)), "b": jnp.zeros((dout,))}
    return p


def init_params(key, cfg):
    L, H, n = cfg["latent"], cfg["hidden"], cfg["mlp_layers"]
    sched = cfg["schedule"]
    keys = jax.random.split(key, 8 + 3 * len(sched))
    proc = []
    for i, lvl in enumerate(sched):
        blk = {"edge": _init_mlp(keys[8 + 3 * i], 3 * L, H, n, L),
               "node": _init_mlp(keys[9 + 3 * i], (3 if lvl == 0 else 2) * L, H, n, L)}
        if lvl == 0:
            blk["world"] = _init_mlp(keys[10 + 3 * i], 3 * L, H, n, L)
        proc.append(blk)
    return {
        "enc_node": _init_mlp(keys[0], NODE_IN, H, n, L),
        "enc_edge": [_init_mlp(keys[1 + l], EDGE_IN, H, n, L) for l in range(len(STRIDES))],
        "enc_world": _init_mlp(keys[4], WORLD_IN, H, n, L),
        "proc": proc,
        "dec": _init_mlp(keys[5], L, H, n, 3, layer_norm=False),
    }


def _mlp(p, x):
    layers = p["layers"]
    for i, lyr in enumerate(layers):
        x = x @ lyr["w"] + lyr["b"]
        if i < len(layers) - 1:
            x = jax.nn.relu(x)
    if "ln" in p:
        mu = jnp.mean(x, -1, keepdims=True)
        var = jnp.mean((x - mu) ** 2, -1, keepdims=True)
        x = (x - mu) * jax.lax.rsqrt(var + 1e-5) * p["ln"]["g"] + p["ln"]["b"]
    return x


def _edge_mlp(p, e, v_send, v_recv):
    """First layer split into e @ We + (v @ Ws)[sender] + (v @ Wr)[receiver]; v_send is [R,K,L]-gathered."""
    L = e.shape[-1]
    first = p["layers"][0]
    w = first["w"]
    h = e @ w[:L] + v_send @ w[L:2 * L] + (v_recv @ w[2 * L:])[:, None, :] + first["b"]
    return _mlp({"layers": p["layers"][1:], "ln": p["ln"]}, jax.nn.relu(h))


# ---------------------------------------------------------------------------- features

def world_neighbors(x, rest, valid):
    """Up to WORLD_K vertices within WORLD_R that are far apart on the cloth."""
    d2 = jnp.sum((x[:, None] - x[None]) ** 2, -1)
    r2 = jnp.sum((rest[:, None] - rest[None]) ** 2, -1)
    ok = (d2 < WORLD_R ** 2) & (r2 > C.SELF_EXCLUDE ** 2) & valid[:, None] & valid[None]
    val, idx = jax.lax.top_k(jnp.where(ok, -d2, -jnp.inf), WORLD_K)
    return idx, jnp.isfinite(val)


def grip_features(x0, rest, held):
    """For the GRIP_K vertices in `held` nearest in the rest shape (ties to the lower index): a flag, the
    world offset to it and the rest distance. Whether and where the cloth is held up (by a gripper, or
    draped over a ball or rod) decides if a vertex half a metre away should fall or hang, and message
    passing alone barely reaches that far."""
    n = x0.shape[0]
    big = jnp.iinfo(jnp.int32).max
    cells = jnp.round(rest / C.SPACING).astype(jnp.int32)
    d = cells[None] - cells[:, None]
    key = (d[..., 0] ** 2 + d[..., 1] ** 2) * 1024 + jnp.arange(n, dtype=jnp.int32)[None]
    neg, idx = jax.lax.top_k(-jnp.where(held[None], key, big), GRIP_K)
    on = (-neg < big)[..., None]
    off = jnp.where(on, (x0[idx] - x0[:, None]) / GRIP_SCALE, 0.0)
    dist = jnp.where(on, jnp.sqrt(((-neg) // 1024).astype(x0.dtype))[..., None] * C.SPACING / GRIP_SCALE, 0.0)
    return jnp.concatenate([on.astype(x0.dtype), off, dist], -1).reshape(n, 5 * GRIP_K)


def node_features(stats, x1, x0, kin_next, held, obs, kb, mu, wind, rest, valid):
    """x1, x0: positions at t-1, t. kin_next: gripper targets for t+1 (used where held)."""
    vs = stats["vel_std"]
    v = (x0 - x1 - stats["vel_mean"]) / vs
    typ = jnp.stack([~held, held], -1).astype(x0.dtype)
    kin = jnp.where(held[:, None], (kin_next - x0) / vs, 0.0)
    table = jnp.clip((x0[:, 1:2] - C.OBS_OFFSET) / D_CLIP, 0.0, 1.0)
    d, nrm, vel = C.obstacle_sdf(x0, obs)
    d, nrm, vel = d[:, 1:], nrm[:, 1:], vel[:, 1:]  # everything but the table
    k = jnp.argmin(d, 1)
    dn = jnp.take_along_axis(d, k[:, None], 1)
    near = dn < D_CLIP
    nn = jnp.where(near, jnp.take_along_axis(nrm, k[:, None, None], 1)[:, 0], 0.0)
    nv = jnp.where(near, jnp.take_along_axis(vel, k[:, None, None], 1)[:, 0] * C.FRAME_DT / vs, 0.0)
    dfeat = jnp.clip(dn / D_CLIP, -1.0, 1.0)
    n = x0.shape[0]
    mat = jnp.broadcast_to(jnp.stack([(jnp.log10(kb) - KB_LOG_MID) / KB_LOG_HALF, mu]), (n, 2))
    w = jnp.broadcast_to(wind / WIND_SCALE, (n, 3))
    touching = (dn[:, 0] < CONTACT_D) & ~held & valid
    return jnp.concatenate([v, typ, kin, table, dfeat, nn, nv, mat, w, grip_features(x0, rest, held),
                            grip_features(x0, rest, touching)], -1)


def edge_features(x, rest, nodes, nbr, scale):
    """Receiver = nodes[r], sender = nbr[r, k]; offsets normalised by the level's rest spacing."""
    u = (rest[nbr] - rest[nodes][:, None]) / scale
    dx = (x[nbr] - x[nodes][:, None]) / scale
    lu = jnp.sqrt(jnp.sum(u * u, -1, keepdims=True) + 1e-12)
    lx = jnp.sqrt(jnp.sum(dx * dx, -1, keepdims=True) + 1e-12)
    return jnp.concatenate([u, lu, dx, lx, (lx / lu - 1.0) * 10.0], -1)


def world_edge_features(x, nbr):
    dx = (x[nbr] - x[:, None]) / WORLD_R
    return jnp.concatenate([dx, jnp.sqrt(jnp.sum(dx * dx, -1, keepdims=True) + 1e-12)], -1)


def _pad(x):
    return jnp.concatenate([x, jnp.zeros((1,) + x.shape[1:], x.dtype)], 0)


def forward(params, cfg, node_in, x, g, world_nbr, world_mask):
    """g: graph tables (jnp) for one cloth. Returns normalized accelerations [N_MAX, 3]."""
    v = _mlp(params["enc_node"], node_in)
    xp, restp = _pad(x), _pad(g["rest"])
    e = []
    for l, s in enumerate(STRIDES):
        feats = edge_features(xp, restp, g[f"nodes{l}"], g[f"nbr{l}"], s * C.SPACING)
        e.append(_mlp(params["enc_edge"][l], feats))
    ew = _mlp(params["enc_world"], world_edge_features(x, world_nbr))
    wm = world_mask[..., None].astype(v.dtype)
    for lvl, blk in zip(cfg["schedule"], params["proc"]):
        nodes, nbr = g[f"nodes{lvl}"], g[f"nbr{lvl}"]
        mask = g[f"mask{lvl}"][..., None].astype(v.dtype)
        vp = _pad(v)
        v_recv = vp[nodes]
        e_new = _edge_mlp(blk["edge"], e[lvl], vp[nbr], v_recv) * mask
        if lvl == 0:
            ew_new = _edge_mlp(blk["world"], ew, v[world_nbr], v) * wm
            v = v + _mlp(blk["node"], jnp.concatenate([v, jnp.sum(e_new, 1), jnp.sum(ew_new, 1)], -1))
            ew = ew + ew_new
        else:
            upd = _mlp(blk["node"], jnp.concatenate([v_recv, jnp.sum(e_new, 1)], -1))
            v = v.at[nodes].add(upd, mode="drop")
        e[lvl] = e[lvl] + e_new
    return _mlp(params["dec"], v)


def predict(params, cfg, stats, x1, x0, kin_next, held, obs, kb, mu, wind, g):
    node_in = node_features(stats, x1, x0, kin_next, held, obs, kb, mu, wind, g["rest"], g["valid"])
    world_nbr, world_mask = world_neighbors(x0, g["rest"], g["valid"])
    return forward(params, cfg, node_in, x0, g, world_nbr, world_mask)


def project(x, obs):
    """Hard constraint: push vertices out of the table and obstacles (distance >= 0)."""
    d, nrm, _ = C.obstacle_sdf(x, obs)
    push = jnp.sum(jnp.where(jnp.isfinite(d) & (d < 0), -d, 0.0)[..., None] * nrm, 1)
    return x + push


def integrate(stats, a_norm, x1, x0, kin_next, held, obs):
    x_next = 2 * x0 - x1 + a_norm * stats["acc_std"] + stats["acc_mean"]
    x_next = project(x_next, obs)
    return jnp.where(held[:, None], kin_next, x_next)


def step(params, cfg, stats, x1, x0, kin_next, held, obs, obs_next, kb, mu, wind, g):
    """One frame: predict, integrate, project against the obstacles' next poses, apply grippers."""
    a = predict(params, cfg, stats, x1, x0, kin_next, held, obs, kb, mu, wind, g)
    return integrate(stats, a, x1, x0, kin_next, held, obs_next)
