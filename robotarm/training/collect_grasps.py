"""Run the pick policy and save the states where it has just lifted the box: the
throw policy starts from these. The handover rule matches the page: the box within
5 cm of the gripper and at least 10 cm up, for 10 control steps (0.2 s) in a row.
From each episode, keep the handover state and up to 15 steps after it.

  python collect_grasps.py <pick params.pkl> <out.npz> [batches=8] [hidden=256,256,128]
"""
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

import panda_live

HELD, LIFTED, SUSTAIN, STEPS, ENVS = 0.05, 0.10, 10, 300, 1024
path, out = sys.argv[1], sys.argv[2]
batches = int(sys.argv[3]) if len(sys.argv) > 3 else 8
hidden = tuple(int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "256,256,128").split(","))

cfg = panda_live.live_config()
cfg.target_jump_prob = 0.0
cfg.box_respawn_prob = 0.0
env = panda_live.PandaPickLive(cfg, config_overrides={"impl": os.environ.get("IMPL", "warp")})
normalizer, policy, _ = model.load_params(path)
nets = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize, policy_hidden_layer_sizes=hidden)
infer = ppo_networks.make_inference_fn(nets)((normalizer, policy), deterministic=True)


def rollout(rng):
  state = env.reset(rng)

  def body(state, _):
    act, _ = infer(state.obs, rng)
    state = env.step(state, act)
    box = state.data.xpos[env._obj_body]
    held = (jp.linalg.norm(box - state.data.site_xpos[env._gripper_site]) < HELD) & (box[2] > LIFTED)
    return state, (state.data.qpos, state.data.qvel, state.data.ctrl, held)

  return jax.lax.scan(body, state, None, length=STEPS)[1]


run = jax.jit(jax.vmap(rollout))
rng = np.random.default_rng(0)
keep = {"qpos": [], "qvel": [], "ctrl": []}
episodes = handed = 0
for b in range(batches):
  qpos, qvel, ctrl, held = (np.asarray(x) for x in run(jax.random.split(jax.random.PRNGKey(100 + b), ENVS)))
  for e in range(ENVS):
    episodes += 1
    h = held[e].astype(int)
    run_len = np.convolve(h, np.ones(SUSTAIN, int), "valid")  # run_len[t] = held steps in t..t+9
    hits = np.nonzero(run_len == SUSTAIN)[0]
    if not len(hits):
      continue
    handed += 1
    t0 = hits[0] + SUSTAIN - 1
    for t in sorted(set([t0] + list(rng.integers(t0, min(STEPS, t0 + 16), 3)))):
      if held[e, t]:
        for k, v in (("qpos", qpos), ("qvel", qvel), ("ctrl", ctrl)):
          keep[k].append(v[e, t])
  print(f"batch {b}: {handed}/{episodes} episodes reached the handover, {len(keep['qpos'])} states", flush=True)
np.savez(out, **{k: np.array(v, np.float32) for k, v in keep.items()})
print("saved", out)
