"""Train the cloth graph network on teacher trajectories. Resumable from <run>/latest_full.pkl.

python train.py --data data/train --valid data/valid --run runs/c1 --steps 1000000
"""
import argparse
import glob
import os
import pickle
import queue
import random
import threading
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

import cloth as C
import mgn

N_MAX = mgn.N_MAX
HUBER = 3.0      # normalized-acceleration error where the loss turns linear
ACC_CLIP = 50.0  # normalized acceleration clip inside training unrolls
GRAPH_KEYS = ["nodes0", "nbr0", "mask0", "nodes1", "nbr1", "mask1", "nodes2", "nbr2", "mask2", "rest", "valid"]
SCENE_KEYS = ["h_vid", "h_t0", "h_t1", "sph_c", "sph_r", "sph_on", "cap_a", "cap_b", "cap_r", "cap_on", "wind", "kb", "mu"]
_TABLES = {}


def tables(size):
    if size not in _TABLES:
        _TABLES[size] = mgn.graph_tables(*size, C.grid_mesh(*size))
    return _TABLES[size]


def graph_of(sc):
    return {k: sc[k] for k in GRAPH_KEYS}


def load_chunk(path):
    """Chunk padded to N_MAX vertices, as a dict of numpy arrays with a leading trajectory axis."""
    with np.load(path) as z:
        size = tuple(int(v) for v in z["size"])
        pos = z["pos"]
        B, T1, n, _ = pos.shape
        out = {"pos": np.zeros((B, T1, N_MAX, 3), np.float32)}
        out["pos"][:, :, :n] = pos
        for k in SCENE_KEYS:
            out[k] = z[k]
    for k, v in tables(size).items():
        out[k] = np.broadcast_to(v, (B,) + v.shape).copy()
    # Drop clips where the teacher let a vertex sink below the table surface.
    ok = pos[..., 1].min((1, 2)) > 0.0
    if not ok.all():
        print(f"{os.path.basename(path)}: dropping {int((~ok).sum())} clip(s) with table penetration", flush=True)
        out = {k: v[ok] for k, v in out.items()}
    return out


def compute_stats(chunk):
    pos, valid = chunk["pos"], chunk["valid"]
    vel = (pos[:, 1:] - pos[:, :-1])[np.broadcast_to(valid[:, None], pos[:, 1:].shape[:3])]
    acc = (pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2])[np.broadcast_to(valid[:, None], pos[:, 2:].shape[:3])]
    return {"vel_mean": vel.mean(0), "vel_std": vel.std(0), "acc_mean": acc.mean(0), "acc_std": acc.std(0)}


def held_at(sc, t):
    """Boolean [N_MAX]: vertices a gripper holds during step t -> t+1."""
    active = (sc["h_vid"] >= 0) & (sc["h_t0"] <= t) & (t < sc["h_t1"])
    idx = jnp.where(active, sc["h_vid"], N_MAX)
    return jnp.zeros(N_MAX, bool).at[idx].set(True, mode="drop")


def obs_at(sc, t):
    nxt = jnp.minimum(t + 1, sc["sph_c"].shape[0] - 1)
    return {"sph_c": sc["sph_c"][t], "sph_v": (sc["sph_c"][nxt] - sc["sph_c"][t]) / C.FRAME_DT,
            "sph_r": sc["sph_r"], "sph_on": sc["sph_on"],
            "cap_a": sc["cap_a"][t], "cap_b": sc["cap_b"][t],
            "cap_va": (sc["cap_a"][nxt] - sc["cap_a"][t]) / C.FRAME_DT,
            "cap_vb": (sc["cap_b"][nxt] - sc["cap_b"][t]) / C.FRAME_DT,
            "cap_r": sc["cap_r"], "cap_on": sc["cap_on"]}


class ChunkFeeder(threading.Thread):
    def __init__(self, pattern):
        super().__init__(daemon=True)
        self.pattern = pattern
        self.q = queue.Queue(maxsize=2)
        self.seen = set()

    def run(self):
        while True:
            files = sorted(glob.glob(self.pattern))
            fresh = [f for f in files if f not in self.seen]
            path = fresh[0] if fresh else random.choice(files)
            self.seen.add(path)
            try:
                self.q.put(load_chunk(path))
            except Exception as exc:
                print("feeder:", path, exc, flush=True)
                time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--swap-every", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--lr-decay-steps", type=float, default=700_000)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--noise", type=float, default=5e-4)
    ap.add_argument("--strain-weight", type=float, default=0.05)
    ap.add_argument("--unroll-start", type=int, default=100_000, help="step at which unrolls start growing")
    ap.add_argument("--unroll-every", type=int, default=60_000, help="steps per extra unroll step")
    ap.add_argument("--unroll-max", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--save-every", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default=None, help="jax_default_matmul_precision, e.g. highest (no TF32)")
    ap.add_argument("--duty-sleep", type=float, default=0.0, help="idle seconds after each step (keeps a laptop GPU cooler)")
    args = ap.parse_args()
    if args.precision:
        jax.config.update("jax_default_matmul_precision", args.precision)
    os.makedirs(args.run, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data, "chunk_*.npz")))
    assert files, "no training chunks yet"
    first = load_chunk(files[0])
    T1 = first["pos"].shape[1]
    T = T1 - 1

    full_path = os.path.join(args.run, "latest_full.pkl")
    if os.path.exists(full_path):
        with open(full_path, "rb") as f:
            ck = pickle.load(f)
        cfg, stats_np, params, opt_state, start = ck["cfg"], ck["stats"], ck["params"], ck["opt_state"], ck["step"]
        print(f"resumed at step {start}", flush=True)
    else:
        cfg = mgn.default_config()
        cfg.update(latent=args.latent, hidden=args.latent, noise_std=args.noise)
        raw = compute_stats(first)
        print("raw stats", raw, flush=True)
        g = cfg["noise_gamma"]
        stats_np = {"vel_mean": raw["vel_mean"], "vel_std": np.sqrt(raw["vel_std"] ** 2 + args.noise ** 2),
                    "acc_mean": raw["acc_mean"], "acc_std": np.sqrt(raw["acc_std"] ** 2 + ((1 + g) * args.noise) ** 2)}
        stats_np = {k: np.asarray(v, np.float32) for k, v in stats_np.items()}
        params = mgn.init_params(jax.random.PRNGKey(args.seed), cfg)
        opt_state, start = None, 0
    stats = {k: jnp.asarray(v) for k, v in stats_np.items()}
    sched = lambda s: args.lr_min + (args.lr - args.lr_min) * 0.1 ** (s / args.lr_decay_steps)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    if opt_state is None:
        opt_state = opt.init(params)

    # ---- trajectory pool on the GPU
    P = args.pool
    pool = {k: jnp.zeros((P,) + v.shape[1:], v.dtype) for k, v in first.items()}
    set_slots = jax.jit(lambda arr, idx, new: arr.at[idx].set(new), donate_argnums=0)
    filled = 0
    feeder = ChunkFeeder(os.path.join(args.data, "chunk_*.npz"))
    feeder.start()

    def swap_in(block=False):
        nonlocal filled
        try:
            chunk = feeder.q.get(block=block)
        except queue.Empty:
            return
        jax.block_until_ready(pool)  # don't donate pool buffers an in-flight step may still read
        k = chunk["pos"].shape[0]
        if filled < P:
            idx = np.arange(filled, min(P, filled + k))
            filled = min(P, filled + k)
        else:
            idx = np.random.choice(P, k, replace=False)
        for key in pool:
            pool[key] = set_slots(pool[key], jnp.asarray(idx), jnp.asarray(chunk[key][:len(idx)]))

    n_files = len(files)
    while filled < min(P, n_files * first["pos"].shape[0]):
        swap_in(block=True)
    print(f"pool filled with {filled} trajectories", flush=True)

    B = args.batch
    noise_std, gamma = cfg["noise_std"], cfg["noise_gamma"]

    def strain_penalty(x, g):
        xp, rp = mgn._pad(x), mgn._pad(g["rest"])
        dx = xp[g["nbr0"]] - xp[g["nodes0"]][:, None]
        du = rp[g["nbr0"]] - rp[g["nodes0"]][:, None]
        strain = jnp.sqrt(jnp.sum(dx * dx, -1) + 1e-12) / jnp.sqrt(jnp.sum(du * du, -1) + 1e-12) - 1.0
        m = g["mask0"]
        return jnp.sum(jnp.where(m, strain ** 2, 0.0)) / jnp.maximum(jnp.sum(m), 1) / 0.01 ** 2

    def one_step_loss(params, sc, t, x1, x0, key):
        """Loss for one network step from state (x1, x0) at frame t, scored against the acceleration that
        lands exactly on the solver's frame t+1. (x1, x0) is the solver's state or the network's own."""
        pos, g = sc["pos"], graph_of(sc)
        held = held_at(sc, t)
        free = g["valid"] & ~held
        noise = jax.random.normal(key, x0.shape) * noise_std * free[:, None]
        x0 = x0 + noise
        a = mgn.predict(params, cfg, stats, x1, x0, pos[t + 1], held, obs_at(sc, t), sc["kb"], sc["mu"],
                        sc["wind"][jnp.minimum(t, T - 1)], g)
        a_t = ((pos[t + 1] + (1 - gamma) * noise) - 2 * x0 + x1 - stats["acc_mean"]) / stats["acc_std"]
        err = jnp.abs(a - a_t)
        hub = jnp.where(err < HUBER, 0.5 * err ** 2, HUBER * (err - 0.5 * HUBER)) * 2
        loss = jnp.sum(jnp.sum(hub, -1) * free) / jnp.maximum(jnp.sum(free), 1)
        x_next = mgn.integrate(stats, jnp.clip(a, -ACC_CLIP, ACC_CLIP), x1, x0, pos[t + 1], held, obs_at(sc, t + 1))
        return loss + args.strain_weight * strain_penalty(x_next, g), loss

    def loss_fn(params, batch, ts, x1, x0, keys):
        l, a = jax.vmap(one_step_loss, in_axes=(None, 0, 0, 0, 0, 0))(params, batch, ts, x1, x0, keys)
        return jnp.mean(l), jnp.mean(a)

    @jax.jit
    def train_step(params, opt_state, batch, ts, x1, x0, key):
        (loss, acc_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, ts, x1, x0, jax.random.split(key, B))
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, acc_loss

    @jax.jit
    def sample(key, pool, n_filled, push):
        """B random (clip, frame) pairs with room for `push` pushforward steps; returns the solver state."""
        k1, k2 = jax.random.split(key)
        p = jax.random.randint(k1, (B,), 0, n_filled)
        ts = jax.random.randint(k2, (B,), 1, T - push)
        batch = {k: v[p] for k, v in pool.items()}
        idx = jnp.arange(B)
        return batch, ts, batch["pos"][idx, ts - 1], batch["pos"][idx, ts]

    @jax.jit
    def push_step(params, batch, ts, x1, x0):
        """One inference step of the network for every batch element (no gradients)."""
        def one(sc, t, x1, x0):
            nxt = mgn.step(params, cfg, stats, x1, x0, sc["pos"][t + 1], held_at(sc, t), obs_at(sc, t),
                           obs_at(sc, t + 1), sc["kb"], sc["mu"], sc["wind"][jnp.minimum(t, T - 1)], graph_of(sc))
            return jnp.clip(jnp.nan_to_num(nxt), -3.0, 3.0)
        return jax.vmap(one)(batch, ts, x1, x0)

    # ---- evaluation: full rollouts on validation trajectories
    vfiles = sorted(glob.glob(os.path.join(args.valid, "chunk_*.npz")))
    vchunks = [load_chunk(f) for f in vfiles]
    vtraj = [{k: v[i] for k, v in c.items()} for c in vchunks for i in range(min(2, c["pos"].shape[0]))]

    @jax.jit
    def rollout(params, sc):
        pos, g = sc["pos"], graph_of(sc)
        Tv = pos.shape[0] - 1

        def body(carry, t):
            x1, x0 = carry
            nxt = mgn.step(params, cfg, stats, x1, x0, pos[t + 1], held_at(sc, t), obs_at(sc, t), obs_at(sc, t + 1),
                           sc["kb"], sc["mu"], sc["wind"][jnp.minimum(t, Tv - 1)], g)
            nxt = jnp.where(g["valid"][:, None], nxt, 0.0)
            return (x0, nxt), nxt

        _, xs = jax.lax.scan(body, (pos[0], pos[1]), jnp.arange(1, Tv))
        return jnp.concatenate([pos[:2], xs], 0)

    def evaluate(step_i):
        errs = []
        for i, sc in enumerate(vtraj):
            sc_j = {k: jnp.asarray(v) for k, v in sc.items()}
            pred = np.asarray(rollout(params, sc_j))
            valid = sc["valid"]
            e = np.sqrt(np.mean(np.sum((pred[:, valid] - sc["pos"][:, valid]) ** 2, -1), 1))
            errs.append(e)
            if i < 2:
                from render3d import render_pair
                n = int(valid.sum())
                size = next(s for s in _TABLES if s[0] * s[1] == n)
                render_pair(os.path.join(args.run, f"rollout_{step_i:08d}_{i}.gif"), sc["pos"][:, :n], pred[:, :n],
                            C.grid_mesh(*size)["tris"], {k: sc[k] for k in SCENE_KEYS})
        errs = np.stack(errs)
        print(f"eval step {step_i}: rmse@30 {errs[:, 30].mean():.4f} @100 {errs[:, 100].mean():.4f} "
              f"@end {errs[:, -1].mean():.4f} m", flush=True)

    def save_full(step_i):
        tmp = full_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"cfg": cfg, "stats": stats_np, "params": jax.device_get(params),
                         "opt_state": jax.device_get(opt_state), "step": step_i}, f)
        os.replace(tmp, full_path)

    key = jax.random.PRNGKey(args.seed + start)
    t_log = time.time()
    losses = []
    for step_i in range(start, args.steps):
        if step_i % args.swap_every == 0 and step_i > start:
            swap_in()
        key, sk, pk, tk = jax.random.split(key, 4)
        K = int(min(args.unroll_max, 1 + max(0, step_i - args.unroll_start) // args.unroll_every))
        # pushforward: 0..K-1 network steps away from the solver's trajectory, then train one step from there
        push = int(np.random.randint(K))
        batch, ts, x1, x0 = sample(sk, pool, filled, push)
        for _ in range(push):
            x1, x0 = x0, push_step(params, batch, ts, x1, x0)
            ts = ts + 1
        params, opt_state, loss = train_step(params, opt_state, batch, ts, x1, x0, tk)
        if args.duty_sleep > 0:
            jax.block_until_ready(loss)
            time.sleep(args.duty_sleep)
        if (step_i + 1) % max(1, args.log_every // 20) == 0:
            losses.append(float(loss))
        if (step_i + 1) % args.log_every == 0:
            dt = time.time() - t_log
            t_log = time.time()
            print(f"step {step_i + 1} loss {np.mean(losses):.4f} unroll {K} lr {sched(step_i):.2e} "
                  f"{args.log_every / dt:.1f} it/s", flush=True)
            losses = []
        if (step_i + 1) % args.save_every == 0:
            save_full(step_i + 1)
        if (step_i + 1) % args.eval_every == 0:
            with open(os.path.join(args.run, f"params_{step_i + 1:08d}.pkl"), "wb") as f:
                pickle.dump({"cfg": cfg, "stats": stats_np, "params": jax.device_get(params)}, f)
            evaluate(step_i + 1)
    save_full(args.steps)


if __name__ == "__main__":
    main()
