"""Score a throw policy in MJX, to compare with tools/eval.mjs on MuJoCo's wasm build:
throws from the saved handover states in random directions; reports how often the box
lands and the landing distance along the aim direction.

  IMPL=warp|jax python eval_mjx.py <throw params.pkl> <grasps.npz> [episodes=2048] [hidden=256,256,128]
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

import panda_throw

path, grasps = sys.argv[1], sys.argv[2]
episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
hidden = tuple(int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "256,256,128").split(","))
impl = os.environ.get("IMPL", "warp")

env = panda_throw.PandaThrow(config_overrides={"impl": impl}, grasps=grasps)
normalizer, policy, _ = model.load_params(path)
nets = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize, policy_hidden_layer_sizes=hidden)
infer = ppo_networks.make_inference_fn(nets)((normalizer, policy), deterministic=True)


def rollout(rng):
  state = env.reset(rng)

  def body(carry, _):
    state, done, dist, point = carry
    act, _ = infer(state.obs, rng)
    nxt = env.step(state, act)
    land = (done == 0) & (nxt.metrics["landed"] > 0)
    box = nxt.data.xpos[env._obj_body][:2]
    dist = jp.where(land, nxt.metrics["throw"], dist)
    point = jp.where(land, box, point)
    done = jp.maximum(done, nxt.done)
    return (nxt, done, dist, point), None

  (_, _, dist, point), _ = jax.lax.scan(body, (state, 0.0, jp.nan, jp.zeros(2)), None, length=env._config.episode_length)
  return dist, point, state.info["heading"]


run = jax.jit(jax.vmap(rollout))
keys = jax.random.split(jax.random.PRNGKey(7), episodes)
out = [run(keys[i:i + 512]) for i in range(0, episodes, 512)]
dist = np.concatenate([np.asarray(o[0]) for o in out])
point = np.concatenate([np.asarray(o[1]) for o in out])
heading = np.concatenate([np.asarray(o[2]) for o in out])
ok = ~np.isnan(dist)
d = np.sort(dist[ok])
off = np.abs((np.arctan2(point[ok, 1], point[ok, 0]) - heading[ok] + 3 * np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi
print(f"MJX {impl}: {ok.sum()}/{episodes} landed; distance mean {d.mean():.2f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.2f} m, "
      f"10th pct {np.percentile(d, 10):.2f}, median {np.median(d):.2f}, 90th pct {np.percentile(d, 90):.2f}, best {d.max():.2f}; "
      f"off the arrow median {np.median(off):.0f}°, 90th pct {np.percentile(off, 90):.0f}°", flush=True)
