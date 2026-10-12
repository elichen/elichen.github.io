"""Export the racing policy, courses and plant for the page, plus reference
rollouts from the MJX environment itself so the JS port can be checked.

  python export.py <mjx-rl-experiments checkout> <params checkpoint> <out dir>

Writes policy.json + policy.bin (MLP and observation normalizer), track.json
(plant constants and the six UZH courses) and test.json (closed-loop episodes
recorded tick by tick from drone.racing.env.RingRace in qualification mode).
"""
import json
import os
import sys
from dataclasses import fields, replace

repo, checkpoint, out = sys.argv[1:4]
sys.path.insert(0, repo)
os.makedirs(out, exist_ok=True)

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from drone.racing.challenge import (
    apply_challenge_config,
    fixed_course_for_challenge,
    open_course_for_challenge,
)
from drone.racing.checkpoint_contract import load_checkpoint_contract
from drone.racing.config import RaceConfig, config_for_stage
from drone.racing.env import RingRace

COURSES = (
    ("uzh_swift_race_v1", "Swift"),
    ("uzh_big_s_v2", "Big-S"),
    ("uzh_kidney_v1", "Kidney"),
    ("uzh_figure_eight_v1", "Figure-8"),
    # Split-S (uzh_split_s_v1) is in the training mixture too, but its geometry is
    # Multistory's, so it is not exported separately.
    ("uzh_multistory_slalom_v1", "Multistory"),
)
REFERENCE_TICKS = 220

contract = load_checkpoint_contract(checkpoint)
task = contract["task_config"]
names = {item.name for item in fields(RaceConfig)}
base = replace(
    config_for_stage(int(task["stage"]), str(task["hardware_profile"])),
    **{k: (tuple(v) if isinstance(v, list) else v) for k, v in task.items() if k in names},
)


def course_config(profile):
    config = apply_challenge_config(base, profile)
    if not open_course_for_challenge(profile):
        config = replace(config, required_laps=2)
    config.validate()
    return config


# ---- policy ----------------------------------------------------------------
normalizer, policy, _ = model.load_params(checkpoint)
p = policy["params"]
hidden = [p["MLP_0"][f"hidden_{i}"] for i in range(3)] + [p["Dense_0"]]
obs_size = int(contract["observation"]["state_size"])
meta = {
    "obs": obs_size,
    "act": 4,
    "activation": "silu",
    "output": "mean",  # distribution_type normal: the deterministic action is the mean
    "mean": np.asarray(normalizer.mean["state"], np.float64).tolist(),
    "std": np.asarray(normalizer.std["state"], np.float64).tolist(),
    "layers": [],
    "checkpoint": os.path.basename(checkpoint),
    "model_sha256": contract["model_sha256"],
}
blobs = []
for layer in hidden:
    w = np.asarray(layer["kernel"], np.float32)
    b = np.asarray(layer["bias"], np.float32)
    meta["layers"].append({"in": w.shape[0], "out": w.shape[1]})
    blobs += [w.T.ravel(), b]  # row-major [out][in]
np.concatenate(blobs).astype("<f4").tofile(os.path.join(out, "policy.bin"))
json.dump(meta, open(os.path.join(out, "policy.json"), "w"))

networks = ppo_networks.make_ppo_networks(
    {"state": obs_size, "privileged_state": 80},
    4,
    policy_hidden_layer_sizes=tuple(contract["network_hidden"]),
    value_hidden_layer_sizes=tuple(contract["network_hidden"]),
    policy_obs_key="state",
    value_obs_key="privileged_state",
    preprocess_observations_fn=running_statistics.normalize,
    distribution_type="normal",
    noise_std_type="log",
)
inference = ppo_networks.make_inference_fn(networks)(
    (normalizer, policy), deterministic=True
)


@jax.jit
def actor(state):
    action, _ = inference(
        {"state": state, "privileged_state": jnp.zeros(80)}, jax.random.PRNGKey(0)
    )
    return action


# ---- plant + courses -----------------------------------------------------------
probe = RingRace(config=course_config("uzh_swift_race_v1"),
                 fixed_course=fixed_course_for_challenge("uzh_swift_race_v1"),
                 qualification=True)
m = probe._model
control = probe._control
plant = {
    "timestep": float(m.opt.timestep),
    "substeps": int(probe._nsub),
    "mass": float(m.body_mass[probe._body_id]),
    "inertia": np.asarray(m.body_inertia[probe._body_id]).tolist(),
    "allocation": np.asarray(control.allocation).tolist(),
    "allocation_inverse": np.asarray(control.allocation_inverse).tolist(),
    "hover_per_rotor": float(control.hover_per_rotor),
    "actuator_max": float(control.actuator_max),
    "rotor_sites": np.asarray(m.site_pos[:4]).tolist(),
    "yaw_moment": np.asarray(m.actuator_gear[:, 5]).tolist(),
    "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in base.manifest().items()}
    if hasattr(base, "manifest") else task,
}
courses = []
for profile, label in COURSES:
    config = course_config(profile)
    course = fixed_course_for_challenge(profile)
    entry = {"id": profile, "name": label, "open": open_course_for_challenge(profile),
             "laps": int(config.required_laps),
             "flight_radius": float(config.flight_radius_m),
             "flight_ceiling": float(config.flight_ceiling_m)}
    for field in ("centers", "previous_centers", "normals", "up", "right",
                  "inner_radius", "outer_radius", "mask", "leg_lengths"):
        entry[field] = np.asarray(getattr(course, field)).tolist()
    courses.append(entry)
json.dump({"plant": plant, "courses": courses}, open(os.path.join(out, "track.json"), "w"))

# ---- reference episodes from the MJX environment -------------------------------------
reference = []
summary = []
for profile, label in COURSES:
    config = course_config(profile)
    env = RingRace(
        config=config,
        frame_stack=int(contract["observation"]["stack"]),
        qualification=True,
        fixed_course=fixed_course_for_challenge(profile),
        open_course=open_course_for_challenge(profile),
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    @jax.jit
    def detector_draws(rng):
        keys = jax.random.split(rng, 5)
        bad, active, successor = jax.random.split(keys[3], 3)
        return jnp.stack([jax.random.uniform(bad), jax.random.uniform(active),
                          jax.random.uniform(successor)])

    for seed in (0, 1):
        state = reset(jax.random.PRNGKey(1000 + seed))
        info = state.info
        episode = {
            "course": profile,
            "seed": seed,
            "qpos": np.asarray(state.pipeline_state.qpos).tolist(),
            "qvel": np.asarray(state.pipeline_state.qvel).tolist(),
            "obs0": np.asarray(state.obs["state"]).tolist(),
            "ticks": [],
        }
        result = None
        for tick in range(config.episode_steps):
            draws = detector_draws(state.info["rng"])
            action = actor(state.obs["state"])
            before = state
            state = step(state, action)
            info = state.info
            if seed == 0 and tick < REFERENCE_TICKS:
                episode["ticks"].append({
                    "action": np.asarray(action).tolist(),
                    "draws": np.asarray(draws).tolist(),
                    "qpos": np.asarray(state.pipeline_state.qpos).tolist(),
                    "qvel": np.asarray(state.pipeline_state.qvel).tolist(),
                    "motor": np.asarray(info["motor_state"]).tolist(),
                    "rate_filter": np.asarray(info["rate_filter_state"]).tolist(),
                    "frame": np.asarray(state.obs["state"][-51:]).tolist(),
                    "active": int(info["active_index"]),
                    "lap": int(info["lap"]),
                    "terminal": int(info["terminal_reason"]),
                })
            if float(state.done) or int(info["ep_step"]) >= config.episode_steps:
                result = {"terminal": int(info["terminal_reason"]),
                          "ticks": int(info["ep_step"]),
                          "passes": int(info["ordered_passes"])}
                break
        summary.append({"course": profile, "seed": seed, **(result or {})})
        print(label, seed, result, flush=True)
        if seed == 0:
            reference.append(episode)


def float32(value):
    """Rounds floats to float32's 7-8 significant digits to keep test.json small."""
    if isinstance(value, float):
        return float(f"{value:.8g}")
    if isinstance(value, list):
        return [float32(v) for v in value]
    if isinstance(value, dict):
        return {k: float32(v) for k, v in value.items()}
    return value


json.dump(float32({"episodes": reference, "summary": summary}),
          open(os.path.join(out, "test.json"), "w"), separators=(",", ":"))
print("wrote", out)
