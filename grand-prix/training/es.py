"""Evolution strategies on lap time, from the DAgger network.

Each generation perturbs all of the actor's weights with antithetic Gaussian
noise and drives every candidate around all 22 circuits, from the rolling start
and from a perturbed start (off the line, a little faster or slower), so the
improvement holds beyond one exact lap. A lap scores the reference lap time
over its own (> 1 when faster); a crash scores half the share of the lap it
completed. Scores are turned into centered ranks, and Adam steps along the
rank-weighted noise. The unperturbed network is checked every generation and
saved when it finishes all 44 laps faster than the best so far.

  python es.py <start params> <out dir> [--generations 300]
"""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.io import model
from jax.flatten_util import ravel_pytree

from qualify import TRACKS, make_env

parser = argparse.ArgumentParser()
parser.add_argument("start")
parser.add_argument("out")
parser.add_argument("--generations", type=int, default=300)
parser.add_argument("--pairs", type=int, default=64)
parser.add_argument("--sigma", type=float, default=0.002)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--steps", type=int, default=6000)  # 120 s; the slowest lap is ~113 s
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

env = make_env()
T = len(TRACKS)
normalizer, policy, value = model.load_params(args.start)
flat0, unravel = ravel_pytree(policy["params"])
print(f"{flat0.size} weights", flush=True)

# (track, lateral offset m, speed factor): the rolling start, and one perturbed start per track
STARTS = [(t, 0.0, 1.0) for t in range(T)] + [(t, 0.4 * (1 if t % 2 else -1), 1.03 if t % 3 else 0.97) for t in range(T)]
S_TRACK = jnp.asarray([s[0] for s in STARTS])
S_LAT = jnp.asarray([s[1] for s in STARTS], jnp.float32)
S_SPEED = jnp.asarray([s[2] for s in STARTS], jnp.float32)


def forward(p, x):
    h = jax.nn.swish(x @ p["hidden_0"]["kernel"] + p["hidden_0"]["bias"])
    h = jax.nn.swish(h @ p["hidden_1"]["kernel"] + p["hidden_1"]["bias"])
    return jnp.tanh((h @ p["hidden_2"]["kernel"] + p["hidden_2"]["bias"])[..., :2])


def start(track_id, lateral, speed):
    state = env.reset(jax.random.PRNGKey(0))
    index = jnp.asarray(0, dtype=jnp.int32)
    ones = jnp.ones((4,), dtype=jnp.float32)
    prev = jnp.zeros((2,), dtype=jnp.float32)
    pipeline = jnp.asarray([
        env.x[track_id, index] - env.tangent_y[track_id, index] * lateral,
        env.y[track_id, index] + env.tangent_x[track_id, index] * lateral,
        env.heading[track_id, index],
        0.76 * env.target_speed[track_id, index] * 0.985 * speed, 0, 0, 0, 0.75, 0], jnp.float32)
    ts, _, _, _ = env._track_state(pipeline[0], pipeline[1], track_id, index)
    obs = env._observation(pipeline, track_id, index, ones, prev, pipeline)
    return state.replace(pipeline_state=pipeline, obs=obs, info={
        **state.info, "track_id": track_id, "index": index, "track_s": ts,
        "progress": jnp.asarray(0.0), "elapsed": jnp.asarray(0.0), "params": ones, "previous_action": prev})


def laps(flat):
    """All 44 laps for one weight vector: (lap time or 0, share of the lap done)."""
    p = unravel(flat)
    states = jax.vmap(start)(S_TRACK, S_LAT, S_SPEED)

    def body(carry, _):
        s, stopped, lap = carry
        nxt = jax.vmap(env.step)(s, forward(p, s.obs["state"]))
        keep = lambda a, b: jnp.where(stopped.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
        nxt = jax.tree.map(keep, s, nxt)
        now = (nxt.done > 0) & ~stopped
        lap = jnp.where(now & (nxt.metrics["lap_complete"] > 0), nxt.info["elapsed"], lap)
        return (nxt, stopped | now, lap), None

    init = (states, jnp.zeros(len(STARTS), bool), jnp.zeros(len(STARTS)))
    (s, _, lap), _ = jax.lax.scan(body, init, None, args.steps)
    return lap, jnp.minimum(s.info["progress"] / env.lengths[S_TRACK], 1.0)


laps_one = jax.jit(laps)
laps_many = jax.jit(jax.vmap(laps))

ref_lap, ref_done = jax.device_get(laps_one(flat0))
assert (ref_lap > 0).all(), f"the start network must finish all {len(STARTS)} laps: {np.asarray(TRACKS * 2)[ref_lap == 0]}"
REF = jnp.asarray(ref_lap)
print(f"start: {ref_lap[:T].sum():.2f} s from the rolling start, {ref_lap[T:].sum():.2f} s from the perturbed starts", flush=True)


def score(lap, done):
    return jnp.mean(jnp.where(lap > 0, REF / jnp.maximum(lap, 1.0), 0.5 * done), axis=-1)


def centered_ranks(x):
    ranks = jnp.argsort(jnp.argsort(x))
    return ranks / (x.size - 1) - 0.5


opt = optax.adam(args.lr)
flat = flat0
ostate = opt.init(flat)
best = float(ref_lap.sum())
key = jax.random.PRNGKey(0)
log = []
for gen in range(args.generations):
    t0 = time.time()
    key, k = jax.random.split(key)
    eps = jax.random.normal(k, (args.pairs, flat.size))
    cands = jnp.concatenate([flat + args.sigma * eps, flat - args.sigma * eps])
    lap, done = laps_many(cands)
    f = centered_ranks(score(lap, done))
    grad = -(f[:args.pairs] - f[args.pairs:]) @ eps / (2 * args.pairs * args.sigma)  # ascent -> minimize
    updates, ostate = opt.update(grad, ostate, flat)
    flat = optax.apply_updates(flat, updates)
    c_lap, _ = jax.device_get(laps_one(flat))
    finished = int((c_lap > 0).sum())
    total = float(c_lap.sum()) if finished == len(STARTS) else None
    row = {"gen": gen, "s": round(time.time() - t0, 1), "finished": finished,
           "rolling": float(c_lap[:T].sum()) if (c_lap[:T] > 0).all() else None,
           "total": total, "pop_finished": float((np.asarray(lap) > 0).mean())}
    if total is not None and total < best:
        best = total
        model.save_params(str(out / "es_best"), (normalizer, {**policy, "params": unravel(flat)}, value))
        row["saved"] = True
    log.append(row)
    (out / "es_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(row), flush=True)
    if gen % 25 == 24:
        model.save_params(str(out / f"es_gen_{gen + 1:04d}"), (normalizer, {**policy, "params": unravel(flat)}, value))
