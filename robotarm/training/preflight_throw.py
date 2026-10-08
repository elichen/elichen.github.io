"""Check the throw env: shapes, holding with zero action, and landing when the gripper opens."""
import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jp
import panda_throw

env = panda_throw.PandaThrow(config_overrides={"impl": os.environ.get("IMPL", "warp")}, grasps=sys.argv[1])
n = 512
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
s = reset(jax.random.split(jax.random.PRNGKey(0), n))
print("obs", s.obs.shape, "size", env.observation_size, "box z", float(s.data.xpos[:, env._obj_body, 2].mean()))
for name, a in [("hold", jp.zeros((n, 8))), ("open", jp.zeros((n, 8)).at[:, 7].set(1.0))]:
  t = s
  done_at = jp.full(n, -1)
  for i in range(60):
    t = step(t, a)
    done_at = jp.where((done_at < 0) & (t.done > 0), i, done_at)
  print(name, "landed", float((done_at >= 0).mean()), "mean step", float(jp.where(done_at >= 0, done_at, 0).sum() / jp.maximum((done_at >= 0).sum(), 1)),
        "box z", float(t.data.xpos[:, env._obj_body, 2].mean()), "nan", bool(jp.isnan(t.obs).any()))
