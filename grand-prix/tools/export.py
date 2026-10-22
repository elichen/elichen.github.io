"""Export the racing policy and the 22 circuits for the page, plus reference
rollouts from the JAX environment itself so the JS port can be checked.

  python export.py <mjx-rl-experiments checkout> <params checkpoint> <out dir>

Writes policy.json + policy.bin (the actor MLP), tracks.json (plant constants,
track list) + tracks/<name>.bin (each circuit's arrays exactly as the
environment uses them, float32) and test.json (closed-loop laps recorded tick by
tick from racecar.standalone.multitrack_env.MultiTrackRaceCar in the nominal
qualification setup the checkpoint was selected with).
"""
import hashlib
import json
import os
import sys

repo, checkpoint, out = sys.argv[1:4]
sys.path.insert(0, repo)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
os.makedirs(os.path.join(out, "tracks"), exist_ok=True)

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks

import racecar.env as legacy
from racecar import vehicle_params as vehicle
from racecar import track as track_module
from racecar.standalone.multitrack_env import MultiTrackRaceCar, MULTITRACK_EPISODE_STEPS
import plant

plant.use()  # the load-sensitive tires (training/plant.py)

TRACKS = (
    "Austin", "Budapest", "Catalunya", "Hockenheim", "IMS", "MexicoCity",
    "Montreal", "Monza", "MoscowRaceway", "Norisring", "Nuerburgring",
    "Oschersleben", "Sakhir", "SaoPaulo", "Sepang", "Shanghai", "Silverstone",
    "Sochi", "Spa", "Spielberg", "Suzuka", "YasMarina",
)
SCALE = 0.76
TEACHER_TICKS = 400  # ticks per track stored in full for teacher-forced checks
FIELDS = (
    "x", "y", "center_x", "center_y", "center_tangent_x", "center_tangent_y",
    "width_right", "width_left", "tangent_x", "tangent_y", "heading",
    "curvature", "legacy_target_speed", "target_speed", "target_accel", "s",
)

# ---- policy ----------------------------------------------------------------
params = model.load_params(checkpoint)
actor = params[1]["params"]
layers, blobs = [], []
for i in range(3):
    kernel = np.asarray(actor[f"hidden_{i}"]["kernel"], np.float32)  # (in, out)
    bias = np.asarray(actor[f"hidden_{i}"]["bias"], np.float32)
    layers.append({"in": kernel.shape[0], "out": kernel.shape[1]})
    blobs += [kernel.T.reshape(-1), bias]  # row-major (out, in)
with open(os.path.join(out, "policy.bin"), "wb") as f:
    f.write(np.concatenate(blobs).astype("<f4").tobytes())
with open(checkpoint, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
with open(os.path.join(out, "policy.json"), "w") as f:
    json.dump({
        "obs": layers[0]["in"], "act": 2, "activation": "swish",
        # tanh_normal: the first two outputs are the location; the deterministic
        # action is tanh(location). The last two are the exploration scale.
        "output": "tanh(loc)", "layers": layers,
        "checkpoint": os.path.basename(checkpoint), "sha256": sha,
    }, f, indent=1)

# ---- tracks ----------------------------------------------------------------
env = MultiTrackRaceCar(
    TRACKS, qualification=True, profile_speed_scale=SCALE,
    reset_profile_speed_scale=SCALE, randomize_dynamics=False,
    actuator_history=True, speed_profile_observation=True,
    extended_preview_observation=True, braking_preview_observation=True,
)
meta_tracks = []
for t, name in enumerate(TRACKS):
    count = int(env.counts[t])
    arrays = [np.asarray(getattr(env, field)[t, :count], np.float32) for field in FIELDS]
    with open(os.path.join(out, "tracks", f"{name}.bin"), "wb") as f:
        f.write(np.stack(arrays).astype("<f4").tobytes())
    # For drawing only: the TUM centerline with its widths, every ~2 m.
    raw = track_module._read_csv(f"tracks/{name}.csv")[:, :4]
    _, center_length = track_module._closed_arclength(raw[:, :2])
    dense = track_module._resample_periodic(raw, int(round(center_length / 2.0)))
    with open(os.path.join(out, "tracks", f"{name}.center.bin"), "wb") as f:
        f.write(dense.T.astype("<f4").tobytes())
    meta_tracks.append({
        "center_count": len(dense),
        "name": name, "count": count,
        "length": float(env.lengths[t]), "spacing": float(env.spacing[t]),
    })
    print(name, count, float(env.lengths[t]))

consts = {
    "CONTROL_DT": legacy.CONTROL_DT, "MAX_STEER_RAD": plant.MAX_STEER_RAD,
    "MAX_STEER_RATE_RAD_S": legacy.MAX_STEER_RATE_RAD_S,
    "TRACK_LIMIT_CENTER_ALLOWANCE_M": legacy.TRACK_LIMIT_CENTER_ALLOWANCE_M,
    "TRACK_LIMIT_MAP_TOLERANCE_M": legacy.TRACK_LIMIT_MAP_TOLERANCE_M,
    **{k: getattr(vehicle, k) for k in dir(vehicle) if k.isupper()},
}
with open(os.path.join(out, "tracks.json"), "w") as f:
    json.dump({"fields": FIELDS, "profile_speed_scale": SCALE, "plant": consts, "tires": {k: getattr(plant, k) for k in ("FZ0", "LOAD_SENSITIVITY", "STIFFNESS_PEAK_LOAD", "SLIP_STIFFNESS", "BRAKE_BIAS_FRONT")},
               "tracks": meta_tracks}, f, indent=1)

# ---- reference laps -----------------------------------------------------------
nets = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    policy_hidden_layer_sizes=(512, 512), value_hidden_layer_sizes=(512, 512),
    policy_obs_key="state", value_obs_key="privileged_state",
    distribution_type="tanh_normal",
)
policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
step = jax.jit(env.step)
act = jax.jit(lambda obs: policy(obs, jax.random.PRNGKey(0))[0])


def fixed_start(track_id):
    # As racecar/standalone/qualify_multitrack_policy.py: index 0 on the racing
    # line at 0.985 x the 0.76-scaled profile speed, throttle 0.75.
    state = env.reset(jax.random.PRNGKey(0))
    index = jnp.asarray(0, dtype=jnp.int32)
    ones = jnp.ones((4,), dtype=jnp.float32)
    previous_action = jnp.zeros((2,), dtype=jnp.float32)
    pipeline = jnp.asarray([
        env.x[track_id, index], env.y[track_id, index], env.heading[track_id, index],
        SCALE * env.target_speed[track_id, index] * 0.985, 0.0, 0.0, 0.0, 0.75, 0.0,
    ], dtype=jnp.float32)
    track_s, _, _, _ = env._track_state(pipeline[0], pipeline[1], track_id, index)
    obs = env._observation(pipeline, track_id, index, ones, previous_action, pipeline)
    return state.replace(pipeline_state=pipeline, obs=obs, info={
        **state.info, "track_id": jnp.asarray(track_id, dtype=jnp.int32),
        "index": index, "track_s": track_s, "progress": jnp.asarray(0.0),
        "elapsed": jnp.asarray(0.0), "params": ones, "previous_action": previous_action,
    })


episodes = []
for t, name in enumerate(TRACKS):
    state = fixed_start(t)
    ticks, lap_time, result = [], None, "timeout"
    obs0 = np.asarray(state.obs["state"], np.float64).tolist()
    for k in range(MULTITRACK_EPISODE_STEPS):
        action = act(state.obs)
        nxt = step(state, action)
        if k < TEACHER_TICKS:
            ticks.append({
                "state": np.asarray(state.pipeline_state, np.float64).tolist(),
                "index": int(state.info["index"]),
                "action": np.asarray(action, np.float64).tolist(),
                "obs": np.asarray(state.obs["state"], np.float64).tolist() if k % 50 == 0 else None,
            })
        state = nxt
        if float(state.done):
            if float(state.metrics["lap_complete"]):
                lap_time, result = float(state.info["elapsed"]), "lap"
            else:
                result = "off" if float(state.metrics["off_track"]) else "spin"
            break
    print(name, result, lap_time, k + 1)
    episodes.append({"track": name, "result": result, "lap_time": lap_time,
                     "steps": k + 1, "obs0": obs0, "ticks": ticks})

with open(os.path.join(out, "test.json"), "w") as f:
    json.dump({"episodes": episodes}, f)

# the AI's lap from the rolling start, shown in the circuit picker
with open(os.path.join(out, "tracks.json")) as f:
    meta = json.load(f)
meta["ai_laps"] = {e["track"]: e["lap_time"] for e in episodes if e["lap_time"]}
with open(os.path.join(out, "tracks.json"), "w") as f:
    json.dump(meta, f, indent=1)
