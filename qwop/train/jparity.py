"""Check jsim.py (JAX) against sim.py (numba): same actions -> same states and observations."""
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64', '--f32' not in sys.argv)
import jax.numpy as jnp
import sim, jsim

N, T = 64, 150
dt = np.float32 if '--f32' in sys.argv else np.float64
rng = np.random.default_rng(0)
sim.seed_numba(0)
env = sim.QwopVec(N, noise=0.05)
# copy: with x64 on, jnp.asarray can alias numba's buffer, which env.step mutates in place
S, ACC = jnp.array(env.S.astype(dt)), jnp.array(env.ACC.astype(dt))
jstep = jax.jit(jsim.step)
jobs = jax.jit(jsim.observe)
act = rng.integers(0, sim.NACT, N)
for t in range(T):
    if t % 5 == 0:
        act = rng.integers(0, sim.NACT, N)
    dx, fell, _, _ = env.step(act)
    S, ACC, jfell, jdx = jstep(S, ACC, jnp.asarray(act))
    if t in (0, 9, 29, 59, 149):
        ds = np.abs(np.asarray(S) - env.S).max()
        do = np.abs(np.asarray(jobs(S, ACC, jnp.asarray(act))) - env.obs()).max()
        print(f"t={t:3d} max|dS|={ds:.2e} max|dobs|={do:.2e} fell agree={(np.asarray(jfell) == fell).mean():.3f} "
              f"max|ddx|={np.abs(np.asarray(jdx) - dx).max():.2e}")
