"""DAgger: teach the network to drive like the path-following teacher.

The teacher (teacher.py) follows the racing line on a minimum-time speed
profile for this car (speed_profile.py). It needs the map and its own profile;
the network sees only its 73 inputs. Each round the network drives (with
probability BETA the teacher's action is executed instead) from anywhere on any
circuit and from the fixed start; every state it visits is labelled with the
teacher's action; it is refit on all labels so far; then it is qualified on all
22 circuits. Driving its own laps is what teaches it to recover from its own
mistakes, which copying the teacher's laps alone does not.

  python dagger.py <start params> <profiles_0.9.npz> <out dir> [--rounds 16]
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

from qualify import TRACKS, make_env, make_qualifier, qualify
from teacher import DEFAULT_GAINS, fixed_start, make_teacher, make_qualifier as make_teacher_qualifier, summary

parser = argparse.ArgumentParser()
parser.add_argument("start")
parser.add_argument("profile")
parser.add_argument("out")
parser.add_argument("--rounds", type=int, default=16)
parser.add_argument("--envs", type=int, default=1024)
parser.add_argument("--horizon", type=int, default=300)
parser.add_argument("--chunks", type=int, default=4)
parser.add_argument("--updates", type=int, default=40000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--beta0", type=float, default=0.8)
parser.add_argument("--buffer", type=int, default=6_000_000)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

env = make_env()
T = len(TRACKS)
profile_speed = jnp.asarray(np.load(args.profile)["speed"])
teacher = make_teacher(env, profile_speed, DEFAULT_GAINS)
head, rows, _, _ = summary(*make_teacher_qualifier(env, profile_speed)(DEFAULT_GAINS))
print(f"teacher: {head}\n   {rows}", flush=True)

normalizer, policy, value = model.load_params(args.start)


def forward(p, x):
    h = jax.nn.swish(x @ p["hidden_0"]["kernel"] + p["hidden_0"]["bias"])
    h = jax.nn.swish(h @ p["hidden_1"]["kernel"] + p["hidden_1"]["bias"])
    return jnp.tanh((h @ p["hidden_2"]["kernel"] + p["hidden_2"]["bias"])[..., :2])


def place(pipeline, track_id, index, prev):
    state = env.reset(jax.random.PRNGKey(0))
    ones = jnp.ones((4,), dtype=jnp.float32)
    ts, _, _, _ = env._track_state(pipeline[0], pipeline[1], track_id, index)
    obs = env._observation(pipeline, track_id, index, ones, prev, pipeline)
    return state.replace(pipeline_state=pipeline, obs=obs, done=jnp.asarray(0.0), info={
        **state.info, "track_id": track_id, "index": index, "track_s": ts,
        "progress": jnp.asarray(0.0), "elapsed": jnp.asarray(0.0), "params": ones, "previous_action": prev})


def random_start(key):
    """Anywhere on any circuit, up to 1 m off the line, at 85-105% of the
    teacher's speed there (or, 1 in 8, the fixed start)."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    track_id = jax.random.randint(k1, (), 0, T)
    index = jax.random.randint(k2, (), 0, env.counts[track_id])
    n = jax.random.uniform(k3, (3,), minval=-1.0, maxval=1.0)
    pipeline = jnp.asarray([
        env.x[track_id, index] - env.tangent_y[track_id, index] * n[0],
        env.y[track_id, index] + env.tangent_x[track_id, index] * n[0],
        env.heading[track_id, index] + 0.02 * n[1],
        profile_speed[track_id, index] * (0.95 + 0.1 * n[2]), 0.0, 0.0, 0.0, 0.3, 0.0])
    state = place(pipeline, track_id, index, jnp.asarray([0.0, 0.3]))
    use_fixed = jax.random.uniform(k4) < 0.125
    return jax.tree.map(lambda a, b: jnp.where(use_fixed, a, b), fixed_start(env, track_id), state)


def teacher_action(state):
    return teacher(state.pipeline_state, state.info["track_id"], state.info["index"])


@jax.jit
def collect(p, key, beta):
    """Run the cars for `horizon` ticks; return (observations, teacher actions)."""
    states = jax.vmap(random_start)(jax.random.split(key, args.envs))

    def body(states, step_key):
        obs = states.obs["state"]
        label = jax.vmap(teacher_action)(states)
        use_teacher = jax.random.uniform(step_key, (args.envs,)) < beta
        nxt = jax.vmap(env.step)(states, jnp.where(use_teacher[:, None], label, forward(p, obs)))
        # cars that crashed (or finished) start again somewhere else
        fresh = jax.vmap(random_start)(jax.random.split(step_key, args.envs))
        restart = nxt.done > 0
        nxt = jax.tree.map(lambda f, n: jnp.where(restart.reshape((-1,) + (1,) * (n.ndim - 1)), f, n), fresh, nxt)
        return nxt, (obs, label)

    _, (obs, labels) = jax.lax.scan(body, states, jax.random.split(key, args.horizon))
    return obs.reshape(-1, obs.shape[-1]), labels.reshape(-1, 2)


opt = optax.adam(args.lr)


@jax.jit
def update(p, s, x, y):
    def loss(p):
        # steering errors weigh more: a small wheel angle is a lot at speed
        return jnp.mean(jnp.sum((forward(p, x) - y) ** 2 * jnp.asarray([4.0, 1.0]), -1))
    l, g = jax.value_and_grad(loss)(p)
    u, s = opt.update(g, s, p)
    return optax.apply_updates(p, u), s, l


params = policy["params"]
ostate = opt.init(params)
run = make_qualifier(env)
X = np.zeros((0, 73), np.float32)
Y = np.zeros((0, 2), np.float32)
key = jax.random.PRNGKey(1)
report = {}
for rnd in range(args.rounds):
    started = time.time()
    beta = max(0.0, args.beta0 - 0.15 * rnd)
    for _ in range(args.chunks):
        key, k = jax.random.split(key)
        x, y = collect(params, k, beta)
        X = np.concatenate([X, np.asarray(x)])[-args.buffer:]
        Y = np.concatenate([Y, np.asarray(y)])[-args.buffer:]
    # the labels stay in host memory; only minibatches go to the GPU
    rng = np.random.default_rng(rnd)
    losses = []
    for step in range(args.updates):
        idx = rng.integers(0, len(X), 4096)
        params, ostate, l = update(params, ostate, jnp.asarray(X[idx]), jnp.asarray(Y[idx]))
        if step % 100 == 0:
            losses.append(float(l))
    full = (normalizer, {**policy, "params": params}, value)
    path = out / f"dagger_round_{rnd:02d}"
    model.save_params(str(path), full)
    q = qualify(run, full)
    report[str(path)] = {**q, "beta": beta, "rows": len(X), "loss": float(np.mean(losses))}
    (out / "dagger_report.json").write_text(json.dumps(report, indent=1))
    print(f"round {rnd} ({time.time() - started:.0f} s) beta {beta:.2f} rows {len(X)} loss {np.mean(losses):.4f}: "
          f"{q['completed']}/22 total {q['total_time']:.2f} s, straight pedal {q['straight_pedal']:.3f}", flush=True)
    print("   " + " ".join(f"{r['track'][:6]}={r['lap_time']:.2f}" if r["lap_time"] else f"{r['track'][:6]}=DNF@{r['progress']:.0f}"
                          for r in q["tracks"]), flush=True)
