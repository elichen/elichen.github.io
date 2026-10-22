"""Deterministic fixed-start laps of every circuit for one or more checkpoints.

  python qualify.py <checkpoint> [<checkpoint> ...] [--out report.json]

The start is the shipped qualifier's (racecar/standalone/qualify_multitrack_policy.py):
index 0 on the racing line, 0.985 x the 0.76-scaled profile speed. Besides lap
times it reports how the car uses the pedal where full throttle is safe: on
straights (|curvature| < 2.5e-3 for the next 150 m) with no braking needed
within 650 m.
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks

from racecar.standalone.multitrack_env import MultiTrackRaceCar
import plant

plant.use()  # every script that imports this module runs on the load-sensitive tires

TRACKS = (
    "Austin", "Budapest", "Catalunya", "Hockenheim", "IMS", "MexicoCity",
    "Montreal", "Monza", "MoscowRaceway", "Norisring", "Nuerburgring",
    "Oschersleben", "Sakhir", "SaoPaulo", "Sepang", "Shanghai", "Silverstone",
    "Sochi", "Spa", "Spielberg", "Suzuka", "YasMarina",
)
STEPS = 8000


def make_env():
    return MultiTrackRaceCar(
        TRACKS, qualification=True, profile_speed_scale=0.76,
        reset_profile_speed_scale=0.76, randomize_dynamics=False,
        actuator_history=True, speed_profile_observation=True,
        extended_preview_observation=True, braking_preview_observation=True,
    )


def make_qualifier(env):
    nets = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        policy_hidden_layer_sizes=(512, 512), value_hidden_layer_sizes=(512, 512),
        policy_obs_key="state", value_obs_key="privileged_state",
        distribution_type="tanh_normal",
    )
    make_policy = ppo_networks.make_inference_fn(nets)

    def start(track_id):
        state = env.reset(jax.random.PRNGKey(0))
        index = jnp.asarray(0, dtype=jnp.int32)
        ones = jnp.ones((4,), dtype=jnp.float32)
        previous_action = jnp.zeros((2,), dtype=jnp.float32)
        pipeline = jnp.asarray([
            env.x[track_id, index], env.y[track_id, index],
            env.heading[track_id, index],
            0.76 * env.target_speed[track_id, index] * 0.985,
            0.0, 0.0, 0.0, 0.75, 0.0,
        ], dtype=jnp.float32)
        track_s, _, _, _ = env._track_state(pipeline[0], pipeline[1], track_id, index)
        obs = env._observation(pipeline, track_id, index, ones, previous_action, pipeline)
        return state.replace(pipeline_state=pipeline, obs=obs, info={
            **state.info, "track_id": track_id, "index": index,
            "track_s": track_s, "progress": jnp.asarray(0.0),
            "elapsed": jnp.asarray(0.0), "params": ones,
            "previous_action": previous_action,
        })

    def safe_straight(state):
        track_id, index = state.info["track_id"], state.info["index"]
        spacing, count = env.spacing[track_id], env.counts[track_id]
        near = jnp.mod(index + jnp.asarray(jnp.arange(0.0, 150.1, 3.0) / spacing, jnp.int32), count)
        distances = jnp.arange(3.0, 650.1, 3.0)
        far = jnp.mod(index + jnp.asarray(distances / spacing, jnp.int32), count)
        u = state.pipeline_state[3]
        need = jnp.max(jnp.maximum((u * u - env.target_speed[track_id, far] ** 2) / (2.0 * distances), 0.0))
        return (jnp.max(jnp.abs(env.curvature[track_id, near])) < 2.5e-3) & (need < 0.10)

    def lap(params, track_id):
        policy = make_policy(params, deterministic=True)

        def body(carry, _):
            state, stopped, lap_time, failed, top, straight, pedal, full = carry
            action = policy(state.obs, jax.random.PRNGKey(0))[0]
            on_straight = safe_straight(state) & ~stopped
            nxt = env.step(state, action)
            nxt = jax.tree.map(lambda a, b: jnp.where(stopped, a, b), state, nxt)
            done_now = (nxt.done > 0) & ~stopped
            lap_time = jnp.where(done_now & (nxt.metrics["lap_complete"] > 0), nxt.info["elapsed"], lap_time)
            failed = failed | (done_now & (nxt.metrics["lap_complete"] == 0))
            top = jnp.maximum(top, jnp.where(stopped, 0.0, nxt.pipeline_state[3]))
            return (nxt, stopped | done_now, lap_time, failed, top,
                    straight + on_straight, pedal + on_straight * action[1],
                    full + on_straight * (action[1] > 0.95)), None

        zero = jnp.asarray(0.0)
        init = (start(track_id), jnp.asarray(False), zero, jnp.asarray(False), zero, zero, zero, zero)
        (state, _, lap_time, failed, top, straight, pedal, full), _ = jax.lax.scan(body, init, None, STEPS)
        return {
            "lap_time": lap_time, "failed": failed, "progress": state.info["progress"],
            "top_kmh": 3.6 * top, "straight_ticks": straight,
            "straight_pedal": pedal / jnp.maximum(straight, 1.0),
            "straight_full": full / jnp.maximum(straight, 1.0),
        }

    return jax.jit(jax.vmap(lap, in_axes=(None, 0)))


def qualify(run, params):
    r = jax.device_get(run(params, jnp.arange(len(TRACKS))))
    rows = []
    for i, name in enumerate(TRACKS):
        rows.append({
            "track": name,
            "lap_time": float(r["lap_time"][i]) if float(r["lap_time"][i]) > 0 else None,
            "progress": float(r["progress"][i]), "top_kmh": float(r["top_kmh"][i]),
            "straight_pedal": float(r["straight_pedal"][i]),
            "straight_full": float(r["straight_full"][i]),
        })
    done = [row for row in rows if row["lap_time"]]
    return {
        "completed": len(done),
        "total_time": sum(row["lap_time"] for row in done),
        "straight_pedal": float(np.mean([row["straight_pedal"] for row in rows])),
        "straight_full": float(np.mean([row["straight_full"] for row in rows])),
        "tracks": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    run = make_qualifier(make_env())
    report = {}
    for path in args.checkpoints:
        q = qualify(run, model.load_params(path))
        report[path] = q
        print(f"{path}: {q['completed']}/22 total {q['total_time']:.2f} s, "
              f"straight pedal {q['straight_pedal']:.3f}, full {q['straight_full']:.3f}", flush=True)
        print("   " + " ".join(
            f"{row['track'][:6]}={row['lap_time']:.2f}" if row["lap_time"] else f"{row['track'][:6]}=DNF"
            for row in q["tracks"]), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
