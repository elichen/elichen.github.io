"""PPO for the Panda pick task on Nitro (Brax + MJX), following
mjx-rl-experiments/NITRO_MJX_TRAINING_PLAYBOOK.md.

  TASK=live|throw|stock IMPL=warp|jax OUT=/mnt/c/w/robotarm RUN_ID=<unique> python train.py
  (throw needs GRASPS=<grasps.npz from collect_grasps.py>)

Writes <OUT>/runs/<RUN_ID>/: config.json, log.json, full Brax checkpoints at every
evaluation (checkpoints/<step>), and params_<step>.pkl / params_final.pkl
(normalizer, policy, value) for export.py.
"""

import functools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp

_clip = jnp.clip
def _clip_compat(a, min=None, max=None, *, a_min=None, a_max=None):  # older Playground code passes a_min/a_max
  return _clip(a, a_min if a_min is not None else min, a_max if a_max is not None else max)
jnp.clip = _clip_compat

from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params

import panda_live
import panda_throw

TASK = os.environ.get("TASK", "live")
IMPL = os.environ.get("IMPL", "warp")
OUT = Path(os.environ.get("OUT", "/mnt/c/w/robotarm"))
RUN_ID = os.environ.get("RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = OUT / "runs" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=False)


def make_env():
  if TASK == "stock":
    return registry.load("PandaPickCube", config_overrides={"impl": IMPL})
  if TASK == "throw":
    cfg = panda_throw.throw_config()
    cfg.reward_config.scales.lateral = float(os.environ.get("LATERAL", "0"))
    return panda_throw.PandaThrow(cfg, config_overrides={"impl": IMPL}, grasps=os.environ["GRASPS"])
  cfg = panda_live.live_config()
  cfg.reward_config.scales.box_fine = float(os.environ.get("BOX_FINE", "0"))
  return panda_live.PandaPickLive(cfg, config_overrides={"impl": IMPL})


env, eval_env = make_env(), make_env()
pp = dict(manipulation_params.brax_ppo_config("PandaPickCube"))
pp.pop("network_factory")
policy_layers = tuple(int(x) for x in os.environ.get("POLICY", "32,32,32,32").split(","))
value_layers = tuple(int(x) for x in os.environ.get("VALUE", "256,256,256,256,256").split(","))
network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=policy_layers, value_hidden_layer_sizes=value_layers)
pp["episode_length"] = env._config.episode_length
for key, cast in [("num_envs", int), ("num_timesteps", int), ("num_evals", int),
                  ("learning_rate", float), ("entropy_cost", float), ("discounting", float),
                  ("batch_size", int), ("num_minibatches", int), ("unroll_length", int),
                  ("num_updates_per_batch", int)]:
  if key.upper() in os.environ:
    pp[key] = cast(os.environ[key.upper()])
seed = int(os.environ.get("SEED", "1"))
pp["save_checkpoint_path"] = RUN_DIR / "checkpoints"
if os.environ.get("RESTORE"):
  pp["restore_checkpoint_path"] = os.environ["RESTORE"]


def write_json(path, value):
  tmp = Path(f"{path}.tmp")
  tmp.write_text(json.dumps(value, indent=2, default=str))
  os.replace(tmp, path)


config = {"task": TASK, "impl": IMPL, "run_id": RUN_ID, "seed": seed,
          "restore": os.environ.get("RESTORE"),
          "policy_layers": policy_layers, "value_layers": value_layers,
          "env_config": env._config.to_dict(), "ppo": pp}
write_json(RUN_DIR / "config.json", config)
print(json.dumps(config, default=str), flush=True)

t0 = time.time()
log = []


def progress(step, m):
  ep_len = float(m.get("eval/avg_episode_length", 1.0))
  rec = {"step": int(step), "t": round(time.time() - t0, 1),
         "reward": float(m.get("eval/episode_reward", float("nan"))), "ep_len": ep_len}
  if TASK == "throw":
    # throw is the landing distance on the landing step (0 otherwise), landed 1 then
    landed = float(m.get("eval/episode_landed", float("nan")))
    rec["landed"] = landed
    rec["throw"] = float(m.get("eval/episode_throw", float("nan"))) / max(landed, 1e-6)
    extra = f"landed={landed:.3f} throw={rec['throw']:.3f} m"
  else:
    # at_target is summed over the episode; divide for the fraction of time at the target
    rec["at_target"] = float(m.get("eval/episode_at_target", float("nan"))) / ep_len
    extra = f"at_target={rec['at_target']:.3f}"
  log.append(rec)
  write_json(RUN_DIR / "log.json", log)
  print(f"[{rec['t']:7.1f}s] step={rec['step']:>11} reward={rec['reward']:8.2f} {extra} ep_len={ep_len:.0f}", flush=True)


def save_params(step, make_policy, params):
  del make_policy
  model.save_params(RUN_DIR / f"params_{int(step)}.pkl", params)


train_fn = functools.partial(
    ppo.train, **pp, network_factory=network_factory, seed=seed,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    progress_fn=progress, policy_params_fn=save_params)
print("starting PPO", flush=True)
_, params, _ = train_fn(environment=env, eval_env=eval_env)
model.save_params(RUN_DIR / "params_final.pkl", params)
print(f"DONE {time.time() - t0:.1f}s -> {RUN_DIR}", flush=True)
