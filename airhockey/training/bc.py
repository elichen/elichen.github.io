"""Clone the scripted expert into the policy net with DAgger, as a starting point for league PPO."""
import argparse, pickle
import numpy as np, jax, jax.numpy as jnp, optax
from jax import lax
from hockey import *

p = argparse.ArgumentParser()
p.add_argument('--out', default='bc.pkl'); p.add_argument('--envs', type=int, default=2048); p.add_argument('--frames', type=int, default=1200)
p.add_argument('--iters', type=int, default=6); p.add_argument('--epochs', type=int, default=6); p.add_argument('--hid', type=int, default=256)
A = p.parse_args()
G = A.envs
teach = lambda n: Bot(jnp.zeros(n, int), jnp.zeros(n, int), jnp.zeros(n), jnp.full(n, 70.), jnp.full(n, 70.), jnp.zeros(n, bool))


def opp_bots(key, n):
    k1, k2 = jax.random.split(key)
    b = jax.vmap(lambda k: sample_bot(k, .6, .2))(jax.random.split(k1, n))
    return b._replace(kind=jnp.where(jax.random.uniform(k2, (n,)) < .1, 3, b.kind))     # kind 3: never moves


@jax.jit
def rollout(actor, beta, key):
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    s = jax.vmap(start_state)(jax.random.split(k0, G)); side = jax.random.randint(k1, (G,), 0, 2)
    ob, drive = opp_bots(k2, G), jax.random.uniform(k3, (G,)) < beta
    hist, ptr = jax.vmap(fill_hist)(s); ptr = ptr[0]
    def f(c, k):
        s, side, ob, drive, hist, ptr = c
        k = jax.random.split(k, 6)
        o = jax.vmap(obs)(s, side)
        lab = jax.vmap(bot_action, (0, 0, 0, None, 0, 0))(teach(G), s, hist, ptr, side, jax.random.split(k[0], G))
        al = jnp.where(drive[:, None], lab + .15 * jax.random.normal(k[1], (G, 2)), pi(actor, o))
        ao = jax.vmap(bot_action, (0, 0, 0, None, 0, 0))(ob, s, hist, ptr, 1 - side, jax.random.split(k[2], G))
        ao = jnp.where((ob.kind == 3)[:, None], 0., ao)
        a = jnp.where((side == 0)[:, None, None], jnp.stack([al, ao], 1), jnp.stack([ao, al], 1))
        s2, goal, tout, _ = jax.vmap(step)(s, a)
        done = (goal != 0) | tout
        w = lambda x, y: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), y, x)
        s2 = jax.tree.map(w, s2, jax.vmap(start_state)(jax.random.split(k[3], G)))
        side, ob = w(side, jax.random.randint(k[4], (G,), 0, 2)), jax.tree.map(w, ob, opp_bots(k[5], G))
        hist, ptr = jax.vmap(push_hist, (0, None, 0))(hist, ptr, s2)
        hist = w(hist, jax.vmap(fill_hist)(s2)[0])
        return (s2, side, ob, drive, hist, ptr[0]), (o, lab)
    return lax.scan(f, (s, side, ob, drive, hist, ptr), jax.random.split(k4, A.frames))[1]


opt = optax.adam(1e-3)
@jax.jit
def epoch(net, ost, X, Y, key):
    perm = jax.random.permutation(key, X.shape[0])[: X.shape[0] // 65536 * 65536].reshape(-1, 65536)
    def mb(c, ix):
        net, ost = c
        l, g = jax.value_and_grad(lambda n: ((mlp(n, X[ix] * 2 - 1) - Y[ix]) ** 2).mean())(net)
        up, ost = opt.update(g, ost, net)
        return (optax.apply_updates(net, up), ost), l
    (net, ost), l = lax.scan(mb, (net, ost), perm)
    return net, ost, l.mean()


key = jax.random.PRNGKey(0)
actor = {'net': init_mlp(key, [12, A.hid, A.hid, A.hid, 2], .01), 'logstd': jnp.full(2, -1.)}
ost = opt.init(actor['net'])
X, Y = [], []
EV = [('expert0', bot_player(0, 0.)), ('human12', bot_player(12, 6.)), ('goalie', bot_player(0, 0., 2))]
ev = jax.jit(lambda ac, f, k: play(net_player(ac), f, 3072, k), static_argnums=1)
for it in range(A.iters):
    o, lab = rollout(actor, 1. if it == 0 else .5 / it, jax.random.PRNGKey(100 + it))
    X.append(np.asarray(o).reshape(-1, 12)); Y.append(np.asarray(lab).reshape(-1, 2))
    Xa, Ya = jnp.asarray(np.concatenate(X[-4:])), jnp.asarray(np.concatenate(Y[-4:]))
    for e in range(A.epochs):
        actor['net'], ost, l = epoch(actor['net'], ost, Xa, Ya, jax.random.PRNGKey(it * 100 + e))
    r = {n: summary(*ev(actor, f, jax.random.PRNGKey(it))) for n, f in EV}
    print(f'iter {it} data {Xa.shape[0] / 1e6:.1f}M mse {float(l):.4f} {r}', flush=True)
    pickle.dump(jax.tree.map(np.asarray, actor), open(A.out, 'wb'))
