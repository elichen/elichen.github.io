"""The teacher: a path-following controller on the racing line plus a pedal
tracking a minimum-time speed profile (speed_profile.py) for the load-sensitive car.

Steering (radians, then normalized by the 0.38 rad lock):
  curvature feedforward  atan(L * k(s + preview))
  heading                K_HEADING * (path heading - course angle)   course = yaw + slip
  lateral (Stanley)      -atan(K_LATERAL * offset / max(u, 8))
  yaw damping            K_YAW * (u * k(s) - yaw rate)
Pedal: (profile speed 0.25 s ahead - u) / 1.5, throttle capped at what the rear
tires can take (plant.throttle_limit).

  python teacher.py --profile profiles_0.85.npz --search    # grid-search the steering gains
"""
import argparse
import itertools

import jax
import jax.numpy as jnp
import numpy as np

import racecar.env as legacy
from racecar import vehicle_params as V
from qualify import TRACKS, STEPS, make_env
import plant

# from the grid search: 22/22 on both the 0.8 and the 0.85 profile
DEFAULT_GAINS = dict(heading=6.0, lateral=2.0, yaw=0.6, preview=0.4)


def make_teacher(env, profile_speed, gains):
    def teacher(pipeline, track_id, index):
        x, y, yaw, u, v, r = pipeline[:6]
        spacing, count = env.spacing[track_id], env.counts[track_id]
        ahead = jnp.mod(index + jnp.asarray((3.0 + gains["preview"] * u) / spacing, jnp.int32), count)
        _, lateral, _, _ = env._track_state(x, y, track_id, index)
        course = yaw + jnp.arctan2(v, jnp.maximum(u, 4.0))
        steer = (jnp.arctan(V.WHEELBASE_M * env.curvature[track_id, ahead])
                 + gains["heading"] * legacy.wrap_angle(env.heading[track_id, index] - course)
                 - jnp.arctan(gains["lateral"] * lateral / jnp.maximum(u, 8.0))
                 + gains["yaw"] * (u * env.curvature[track_id, index] - r))
        pedal_at = jnp.mod(index + jnp.asarray(u * 0.25 / spacing, jnp.int32), count)
        pedal = jnp.clip((profile_speed[track_id, pedal_at] - u) / 1.5, -1.0, 1.0)
        pedal = jnp.minimum(pedal, plant.throttle_limit(pipeline))  # no power oversteer out of slow corners
        return jnp.stack([jnp.clip(steer / legacy.MAX_STEER_RAD, -1.0, 1.0), pedal])
    return teacher


def fixed_start(env, track_id):
    state = env.reset(jax.random.PRNGKey(0))
    index = jnp.asarray(0, dtype=jnp.int32)
    ones = jnp.ones((4,), dtype=jnp.float32)
    prev = jnp.zeros((2,), dtype=jnp.float32)
    pipeline = jnp.asarray([env.x[track_id, index], env.y[track_id, index], env.heading[track_id, index],
                            0.76 * env.target_speed[track_id, index] * 0.985, 0, 0, 0, 0.75, 0], jnp.float32)
    ts, _, _, _ = env._track_state(pipeline[0], pipeline[1], track_id, index)
    obs = env._observation(pipeline, track_id, index, ones, prev, pipeline)
    return state.replace(pipeline_state=pipeline, obs=obs, info={
        **state.info, "track_id": track_id, "index": index, "track_s": ts,
        "progress": jnp.asarray(0.0), "elapsed": jnp.asarray(0.0), "params": ones, "previous_action": prev})


def make_qualifier(env, profile_speed):
    """jitted (gains vector [heading, lateral, yaw, preview]) -> (lap times, progress)"""
    def run(g):
        gains = dict(heading=g[0], lateral=g[1], yaw=g[2], preview=g[3])
        return jax.vmap(lambda t: _lap(env, make_teacher(env, profile_speed, gains), t))(jnp.arange(len(TRACKS)))
    run = jax.jit(run)
    return lambda gains: jax.device_get(run(jnp.asarray([gains["heading"], gains["lateral"], gains["yaw"], gains["preview"]], jnp.float32)))


def _lap(env, teacher, track_id):
    if True:
        def body(carry, _):
            state, stopped, lap_time = carry
            a = teacher(state.pipeline_state, state.info["track_id"], state.info["index"])
            nxt = env.step(state, a)
            nxt = jax.tree.map(lambda p, q: jnp.where(stopped, p, q), state, nxt)
            now = (nxt.done > 0) & ~stopped
            lap_time = jnp.where(now & (nxt.metrics["lap_complete"] > 0), nxt.info["elapsed"], lap_time)
            return (nxt, stopped | now, lap_time), None
        (state, _, lap_time), _ = jax.lax.scan(body, (fixed_start(env, track_id), jnp.asarray(False), jnp.asarray(0.0)), None, STEPS)
        return lap_time, state.info["progress"]


def summary(times, progress):
    done = times > 0
    head = f"{done.sum()}/22, total {times[done].sum():.2f} s"
    rows = " ".join(f"{n[:6]}={t:.2f}" if t > 0 else f"{n[:6]}=DNF@{p:.0f}" for n, t, p in zip(TRACKS, times, progress))
    return head, rows, done.sum(), times[done].sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", nargs="+", required=True)
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    env = make_env()
    for path in args.profile:
        speed = jnp.asarray(np.load(path)["speed"])
        qualify_gains = make_qualifier(env, speed)
        grid = [DEFAULT_GAINS]
        if args.search:
            grid = [dict(heading=h, lateral=l, yaw=y, preview=p) for h, l, y, p in
                    itertools.product([3.0, 4.5, 6.0], [0.5, 1.0, 2.0], [0.3, 0.6], [0.25, 0.4])]
        best = None
        for g in grid:
            head, rows, n, total = summary(*qualify_gains(g))
            print(f"{path} {g}: {head}", flush=True)
            if best is None or (n, -total) > (best[0], -best[1]):
                best = (n, total, g, rows)
        print(f"BEST {path}: {best[0]}/22 {best[1]:.2f} s {best[2]}\n   {best[3]}", flush=True)
