"""Load the live env on the GPU, reset and step it, and print shapes and timings."""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jp
import panda_live

impl = os.environ.get("IMPL", "warp")
env = panda_live.PandaPickLive(config_overrides={"impl": impl})
print("obs", env.observation_size, "act", env.action_size, "nq", env._mj_model.nq, "nv", env._mj_model.nv,
      "ngeom", env._mj_model.ngeom, "substeps", env.n_substeps)
n = int(os.environ.get("N", "2048"))
reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))
s = reset(jax.random.split(jax.random.PRNGKey(0), n))
a = jp.zeros((n, env.action_size))
s = step(s, a); jax.block_until_ready(s.obs)
t = time.time()
for i in range(50):
  s = step(s, jax.random.uniform(jax.random.PRNGKey(i), (n, env.action_size), minval=-1, maxval=1))
jax.block_until_ready(s.obs)
dt = time.time() - t
print(f"{n} envs x 50 steps: {dt:.2f}s = {n * 50 / dt:,.0f} env steps/s")
print("box z range", float(s.data.xpos[:, env._obj_body, 2].min()), float(s.data.xpos[:, env._obj_body, 2].max()))
print("nan", bool(jp.isnan(s.obs).any()), "done", float(s.done.mean()), "metrics", {k: float(v.mean()) for k, v in s.metrics.items()})
