"""One-step parity of hockey.step against browser transitions from parity.mjs (float64 on CPU)."""
import json, sys
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np
from hockey import St, step, serve, H

rows = json.load(open(sys.argv[1]))
f = lambda key: St(*(jnp.array(np.array([r[key][k] for r in rows]), dtype=jnp.int32 if k == 't' else jnp.float64) for k in St._fields))
S, N = f('s'), f('n')
A = jnp.array(np.array([r['a'] for r in rows]), dtype=jnp.float64)
g, to = np.array([r['goal'] for r in rows]), np.array([r['tout'] for r in rows])
out, goal, tout, (h0, h1) = jax.jit(jax.vmap(step))(S, A)
print(f'{len(rows)} frames: {(g != 0).sum()} goals, {to.sum()} timeouts, {int(np.array(h0 | h1).sum())} paddle hits')
assert (np.array(goal) == g).all(), np.nonzero(np.array(goal) != g)
assert (np.array(tout) == to).all()
exp = jax.vmap(serve)(jnp.array(np.where(g == 1, H / 4, np.where(g == -1, 3 * H / 4, H / 2))))
ok = jnp.array((g == 0) & ~to)
pred = jax.tree.map(lambda a, b: jnp.where(ok.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), out, exp)
worst = max(np.abs(np.array(getattr(pred, k)) - np.array(getattr(N, k))).max() for k in St._fields)
for k in St._fields: print(f'{k:3s} max err {np.abs(np.array(getattr(pred, k)) - np.array(getattr(N, k))).max():.3g}')
print('PASS' if worst < 1e-9 else 'FAIL')
