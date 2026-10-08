"""Continuous browser-style matches (conceder serves, 20 s timeouts): goals per minute for a policy against a set of opponents."""
import argparse, pickle, json
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from hockey import *

p = argparse.ArgumentParser()
p.add_argument('policy'); p.add_argument('--vs', nargs='*', default=[]); p.add_argument('--minutes', type=float, default=5)
p.add_argument('--matches', type=int, default=512)
A = p.parse_args()
load = lambda f: jax.tree.map(jnp.asarray, (lambda c: c['actor'] if 'actor' in c else c)(pickle.load(open(f, 'rb'))))


def match(pa, pb, key):
    G, F = A.matches, int(A.minutes * 3600)
    side = jnp.arange(G) % 2
    s = jax.vmap(serve)(jnp.full(G, H / 2))
    hist, ptr = jax.vmap(fill_hist)(s); ptr = ptr[0]
    def f(c, k):
        s, hist, ptr = c
        ka, kb = jax.random.split(k)
        aa, ab = pa(s, hist, ptr, side, ka), pb(s, hist, ptr, 1 - side, kb)
        a = jnp.where((side == 0)[:, None, None], jnp.stack([aa, ab], 1), jnp.stack([ab, aa], 1))
        s2, goal, tout, _ = jax.vmap(step)(s, a)
        s2 = jax.tree.map(lambda x, y: jnp.where(((goal != 0) | tout).reshape((-1,) + (1,) * (x.ndim - 1)), y, x), s2,
                          jax.vmap(serve)(jnp.where(goal == 1, H / 4, jnp.where(goal == -1, 3 * H / 4, H / 2))))
        hist, ptr = jax.vmap(push_hist, (0, None, 0))(hist, ptr, s2)
        return (s2, hist, ptr[0]), (goal * jnp.where(side == 0, 1, -1), tout)
    _, (g, to) = lax.scan(f, (s, hist, ptr), jax.random.split(key, F))
    return (g == 1).sum(0), (g == -1).sum(0), to.sum(0)


def player(spec):
    """'still', 'bot:delay:noise[:kind]', 'late:delay:motor:path' or a policy path (noise-free)."""
    if spec == 'still':
        return lambda s, h, p, side, k: jnp.zeros((side.shape[0], 2))
    if spec.startswith('bot:'):
        d, n, *k = spec.split(':')[1:]
        return bot_player(int(d), float(n), int(k[0]) if k else 0)
    if spec.startswith('late:'):
        d, m, f = spec.split(':', 3)[1:]
        return late_player(load(f), int(d), float(m))
    return net_player(load(spec))


me = net_player(load(A.policy))
out = {}
for spec in A.vs:
    gf, ga, to = map(np.asarray, jax.jit(lambda k, s=spec: match(me, player(s), k))(jax.random.PRNGKey(0)))
    out[spec] = dict(goals_for_per_min=round(float(gf.mean() / A.minutes), 2), goals_against_per_min=round(float(ga.mean() / A.minutes), 2),
                     timeouts_per_min=round(float(to.mean() / A.minutes), 2), matches_won=round(float((gf > ga).mean()), 3),
                     matches_lost=round(float((gf < ga).mean()), 3))
    print(spec, out[spec], flush=True)
json.dump(out, open(A.policy + '.eval.json', 'w'), indent=1)
