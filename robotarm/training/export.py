"""Export a Brax PPO policy for the page: policy.json (layer shapes, observation
normalizer) + policy.bin (float32 weights), and test.json (observations with
Brax's own deterministic actions) so the JS forward pass can be checked.

  python export.py <params.pkl> <out dir> [hidden sizes, default 256,256,128]
"""
import json
import os
import sys

import jax
import jax.numpy as jp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

ACT = 8

path, out = sys.argv[1], sys.argv[2]
hidden = tuple(int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "256,256,128").split(","))
os.makedirs(out, exist_ok=True)
normalizer, policy, _ = model.load_params(path)
OBS = int(np.asarray(normalizer.mean).shape[-1])  # 66 for picking, 59 for throwing

layers = policy["params"]
names = sorted(layers, key=lambda n: int(n.split("_")[1]))
meta = {"obs": OBS, "act": ACT, "activation": "swish", "layers": [],
        "mean": np.asarray(normalizer.mean, np.float64).tolist(),
        "std": np.asarray(normalizer.std, np.float64).tolist()}
blobs = []
for n in names:
  w, b = np.asarray(layers[n]["kernel"], np.float32), np.asarray(layers[n]["bias"], np.float32)
  meta["layers"].append({"in": w.shape[0], "out": w.shape[1]})
  blobs += [w.T.ravel(), b]  # row-major [out][in]
np.concatenate(blobs).astype("<f4").tofile(os.path.join(out, "policy.bin"))
json.dump(meta, open(os.path.join(out, "policy.json"), "w"))

# Reference actions from Brax itself
nets = ppo_networks.make_ppo_networks(
    OBS, ACT, preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=hidden)
infer = jax.jit(ppo_networks.make_inference_fn(nets)((normalizer, policy), deterministic=True))
rng = np.random.default_rng(0)
mean, std = np.asarray(normalizer.mean), np.asarray(normalizer.std)
obs = (mean + std * rng.normal(size=(16, OBS))).astype(np.float32)
acts = np.stack([np.asarray(infer(jp.asarray(o), jax.random.PRNGKey(0))[0]) for o in obs])
json.dump({"obs": obs.tolist(), "act": acts.tolist()}, open(os.path.join(out, "test.json"), "w"))
print("layers", [(l["in"], l["out"]) for l in meta["layers"]], "->", out)
