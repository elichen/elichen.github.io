"""Browser-exact air hockey physics in JAX (environment.js + game.js moveAgentPaddle), plus scripted opponents."""
from typing import NamedTuple

import numpy as np, jax, jax.numpy as jnp
from jax import lax

W, H, RP, RK, GW, GP, FR, VMAX, SPD, TMAX = 600., 800., 20., 15., 200., 20., .997, 30., 10., 1200
EW, EP, SUB = .9, .8, 4                                        # wall / paddle restitution, substeps per frame
GL, GR, CD = (W - GW) / 2, (W + GW) / 2, RP + RK
HOME = jnp.array([[W / 2, H - 50.], [W / 2, 50.]])            # [bottom, top]
LO = jnp.array([[RP, H / 2 + RP], [RP, RP]])
HI = jnp.array([[W - RP, H - RP], [W - RP, H / 2 - RP]])
ASIGN = jnp.array([[1., 1.], [1., -1.]])                       # top player's y action is flipped
D = 16                                                         # history ring for delayed bots


class St(NamedTuple):
    pp: jax.Array; pv: jax.Array; kp: jax.Array; kv: jax.Array; t: jax.Array


def serve(y):
    return St(HOME, jnp.zeros((2, 2)), jnp.array([W / 2, y]), jnp.zeros(2), jnp.int32(0))


def in_goal(k):
    inx = (k[0] > GL) & (k[0] < GR)
    return inx & (k[1] - RK < GP), inx & (k[1] + RK > H - GP)


def walls(k, v, goal_aware=True):
    l, r = k[0] - RK < 0, k[0] + RK > W
    k0 = jnp.where(l, RK, jnp.where(r, W - RK, k[0])); v0 = jnp.where(l, jnp.abs(v[0]) * EW, jnp.where(r, -jnp.abs(v[0]) * EW, v[0]))
    gt, gb = in_goal(jnp.array([k0, k[1]]))
    open_ = ~(gt | gb) if goal_aware else ~((k0 > GL) & (k0 < GR))
    t, b = open_ & (k[1] - RK < 0), open_ & (k[1] + RK > H)
    k1 = jnp.where(t, RK, jnp.where(b, H - RK, k[1])); v1 = jnp.where(t, jnp.abs(v[1]) * EW, jnp.where(b, -jnp.abs(v[1]) * EW, v[1]))
    return jnp.array([k0, k1]), jnp.array([v0, v1])


def collide(k, v, p, pv):
    d = k - p; dist = jnp.sqrt(d @ d)
    hit = (dist < CD) & (dist > 0)
    n = d / jnp.where(hit, dist, 1.)
    vn = (v - pv) @ n
    nv = jnp.where(vn < 0, v - (1 + EP) * vn * n, v)
    s = jnp.sqrt(nv @ nv)
    nv = jnp.where(s > VMAX, nv * (VMAX / jnp.where(s > 0, s, 1.)), nv)
    return jnp.where(hit, p + n * CD, k), jnp.where(hit, nv, v), hit


def step(s, a):
    """a: (2,2) actions in each player's frame [bottom, top]. Returns state, goal (+1 bottom scored, -1 top scored), timeout, hits."""
    req = jnp.clip(a, -1, 1) * SPD * ASIGN
    pp = jnp.clip(s.pp + s.pv * .6 + req * .4, LO, HI); pv = pp - s.pp
    k, v, goal, h0, h1 = s.kp, s.kv, jnp.int32(0), False, False
    for j in range(1, SUB + 1):
        f = j / SUB
        k2, v2, c0 = collide(k + v / SUB, v, pp[0] - pv[0] * (1 - f), pv[0])
        k2, v2, c1 = collide(k2, v2, pp[1] - pv[1] * (1 - f), pv[1])
        k2, v2 = walls(k2, v2)
        gt, gb = in_goal(k2)
        live = goal == 0
        k, v = jnp.where(live, k2, k), jnp.where(live, v2, v)
        h0, h1 = h0 | (live & c0), h1 | (live & c1)
        goal = jnp.where(live, jnp.where(gt, 1, jnp.where(gb, -1, 0)), goal)
    t = s.t + 1
    return St(pp, pv, k, v * FR, t), goal, (goal == 0) & (t >= TMAX), (h0, h1)


def obs(s, i):
    """12 features, identical to ppo_agent.js getState for player i (0 bottom, 1 top)."""
    sy = jnp.where(i == 0, -1., 1.); oy = jnp.where(i == 0, H, 0.)
    P = lambda p: jnp.array([p[0] / W, (oy + sy * p[1]) / H])
    V = lambda v: jnp.clip(jnp.array([v[0], sy * v[1]]) / VMAX, -1, 1) * .5 + .5
    return jnp.concatenate([P(s.pp[i]), P(s.kp), V(s.pv[i]), V(s.kv), P(s.pp[1 - i]), V(s.pv[1 - i])])


def random_state(key):
    k = jax.random.split(key, 7)
    kp = jax.random.uniform(k[0], (2,), minval=jnp.array([RK, 80.]), maxval=jnp.array([W - RK, H - 80]))
    sp = jax.random.uniform(k[1], (), maxval=VMAX) * (jax.random.uniform(k[2]) < .8)
    ang = jax.random.uniform(k[3], (), maxval=2 * jnp.pi)
    pp = jax.random.uniform(k[4], (2, 2), minval=LO, maxval=HI)
    pv = jax.random.uniform(k[5], (2, 2), minval=-8., maxval=8.) * (jax.random.uniform(k[6], (2, 1)) < .6)
    # keep the puck from starting inside a paddle
    d = kp - pp; dist = jnp.sqrt((d ** 2).sum(1, keepdims=True))
    kp = jnp.where((dist < CD + 1).any(), jnp.array([W / 2, H / 2]), kp)
    return St(pp, pv, kp, sp * jnp.array([jnp.cos(ang), jnp.sin(ang)]), jnp.int32(0))


def start_state(key, p_random=.4):
    k1, k2, k3 = jax.random.split(key, 3)
    sv = serve(jnp.array([H / 4, H / 2, 3 * H / 4])[jax.random.randint(k2, (), 0, 3)])
    rs = random_state(k3)
    return jax.tree.map(lambda a, b: jnp.where(jax.random.uniform(k1) < p_random, a, b), rs, sv)


# ---------------- scripted opponents (mouse-target controllers, like a human in game.js) ----------------

class Bot(NamedTuple):
    kind: jax.Array   # 0 expert, 1 random actions, 2 goalie
    delay: jax.Array  # reaction delay in frames (< D)
    noise: jax.Array  # mouse jitter (px)
    aimx: jax.Array   # aim offset from goal centre (px)
    defy: jax.Array   # defence line (px from own goal)
    bank: jax.Array   # prefers bank shots


def sample_bot(key, p_expert=.8, p_random=.1):
    k = jax.random.split(key, 7)
    u = jax.random.uniform(k[0])
    return Bot(jnp.where(u < p_expert, 0, jnp.where(u < p_expert + p_random, 1, 2)),
               jax.random.randint(k[1], (), 0, D), jax.random.uniform(k[2], (), maxval=12.),
               jax.random.uniform(k[3], (), minval=40., maxval=95.), jax.random.uniform(k[4], (), minval=40., maxval=110.),
               jax.random.uniform(k[5]) < .3)


def own(i, p): return jnp.array([p[0], jnp.where(i == 0, H - p[1], p[1])])     # own frame: y = distance from own goal line


def path(k, v, n):
    def f(c, _):
        k, v = walls(c[0] + c[1], c[1] * FR, goal_aware=False)
        return (k, v), (k, v)
    return lax.scan(f, (k, v), None, length=n)[1]


def expert_target(b, me, k, kv, opp, key):
    """Mouse target in own frame. k, kv, opp are (possibly delayed) observations in own frame."""
    ks, vs = (lax.dynamic_slice(x, (b.delay, 0), (48, 2)) for x in path(k, kv, D + 48))   # extrapolate through the reaction delay
    n = jnp.arange(1, 49.)
    d = jnp.sqrt(((ks - me) ** 2).sum(1))
    ok = (d <= jnp.maximum(9. * n - 8., 0.) + CD - 4) & (ks[:, 1] < H / 2 + CD - 5) & (ks[:, 1] > RK)
    reach, j = ok.any(), jnp.argmax(ok)
    ip, iv = jnp.where(reach, ks[j], k), jnp.where(reach, vs[j], kv)
    side = jnp.where(opp[0] < W / 2, 1., -1.)
    ax = W / 2 + side * b.aimx + jax.random.normal(key) * b.noise
    ax = jnp.where(b.bank, jnp.where(side > 0, 2 * W - ax - 2 * RK, -ax + 2 * RK), ax)   # mirror the aim point over a side wall
    sd = jnp.array([ax, H + 10.]) - ip; sd = sd / jnp.sqrt(sd @ sd)
    nrm = 20. * sd - iv; nrm = nrm / jnp.sqrt(nrm @ nrm)                                 # contact normal that turns iv into a shot along sd
    rel = ip - me; along = rel @ nrm; lat = rel[0] * nrm[1] - rel[1] * nrm[0]
    perp = jnp.array([nrm[1], -nrm[0]]) * jnp.where(lat > 0, -1., 1.)
    back = ip - nrm * (CD + 25)
    cramped = (jnp.abs(jnp.clip(back, jnp.array([RP, RP]), jnp.array([W - RP, H / 2 - RP])) - back) > 3).any()
    atk = jnp.where(along < 0, ip + perp * (CD + 20) - nrm * 20,                                  # wrong side: go around
                    jnp.where((jnp.abs(lat) < 10) | cramped | (along < CD + 30), ip + nrm * 80, back))                # aligned: strike, else wind up behind
    coming = kv[1] < -.5
    below = ks[:, 1] <= b.defy + CD
    cx = jnp.where(below.any(), ks[jnp.argmax(below), 0], k[0])
    guard = jnp.array([jnp.clip(cx, GL - 15, GR + 15), b.defy])
    home = jnp.array([W / 2 + (k[0] - W / 2) * .3, b.defy])
    attack = reach & (~coming | (ip[1] > b.defy + 40))
    return jnp.where(attack, atk, jnp.where(coming, guard, home))


def bot_action(b, s, hist, ptr, i, key):
    """Action for player i in its own frame, using observations delayed by b.delay frames."""
    k1, k2, k3 = jax.random.split(key, 3)
    h = hist[(ptr - b.delay) % D]                        # [kp, kv, pp0, pp1] absolute
    sy = jnp.where(i == 0, -1., 1.)
    me = own(i, s.pp[i])
    tgt = expert_target(b, me, own(i, h[0:2]), jnp.array([h[2], sy * h[3]]), own(i, jnp.where(i == 0, h[6:8], h[4:6])), k1)
    goalie = jnp.array([jnp.clip(own(i, h[0:2])[0], GL, GR), 40.])
    tgt = jnp.where(b.kind == 2, goalie, tgt) + jax.random.normal(k2, (2,)) * b.noise * .5
    dv = tgt - me
    req = dv * jnp.minimum(1., SPD / jnp.maximum(jnp.abs(dv).max(), 1e-6)) / SPD   # mouse placed along the intended direction
    a = jnp.array([req[0], -req[1]])                     # positive action y moves toward own goal
    return jnp.where(b.kind == 1, jax.random.uniform(k3, (2,), minval=-1., maxval=1.), a)


def push_hist(hist, ptr, s):
    return hist.at[(ptr + 1) % D].set(jnp.concatenate([s.kp, s.kv, s.pp.ravel(), s.pv.ravel()])), ptr + 1


def fill_hist(s):
    return jnp.tile(jnp.concatenate([s.kp, s.kv, s.pp.ravel(), s.pv.ravel()]), (D, 1)), jnp.int32(0)


def obs_late(s, hist, ptr, i, delay):
    """Observation with the puck and opponent seen `delay` frames late; own paddle is current (the hand knows where it is)."""
    h = hist[(ptr - delay) % D]
    pp, pv = h[4:8].reshape(2, 2).at[i].set(s.pp[i]), h[8:12].reshape(2, 2).at[i].set(s.pv[i])
    return obs(St(pp, pv, h[0:2], h[2:4], s.t), i)


# ---------------- policies and matches ----------------

def init_mlp(key, sizes, last):
    ks = jax.random.split(key, len(sizes))
    return [(jax.nn.initializers.orthogonal(last if i == len(sizes) - 2 else np.sqrt(2))(ks[i], (a, b)), jnp.zeros(b))
            for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]


def mlp(ps, x):
    for w, b in ps[:-1]: x = jnp.tanh(x @ w + b)
    return x @ ps[-1][0] + ps[-1][1]


pi = lambda ac, o: mlp(ac['net'], o * 2 - 1)
logp = lambda a, mu, ls: (-.5 * ((a - mu) / jnp.exp(ls)) ** 2 - ls - .5 * np.log(2 * np.pi)).sum(-1)


def play(pa, pb, games, key, start=None):
    """pa, pb: (fn(state, hist, ptr, side, key) -> action). Returns per-game outcome for pa (+1/-1/0) and length."""
    k1, k2 = jax.random.split(key)
    side = jnp.arange(games) % 2
    s = jax.vmap(serve)(jnp.array([H / 4, H / 2, 3 * H / 4])[(jnp.arange(games) // 2) % 3]) if start is None else start
    hist, ptr = jax.vmap(fill_hist)(s); ptr = ptr[0]
    def f(c, k):
        s, hist, ptr, res, ln = c
        ka, kb = jax.random.split(k)
        aa, ab = pa(s, hist, ptr, side, ka), pb(s, hist, ptr, 1 - side, kb)
        a = jnp.where((side == 0)[:, None, None], jnp.stack([aa, ab], 1), jnp.stack([ab, aa], 1))
        s2, goal, tout, _ = jax.vmap(step)(s, a)
        live = res == 2
        res = jnp.where(live & ((goal != 0) | tout), goal * jnp.where(side == 0, 1, -1), res)
        s2 = jax.tree.map(lambda x, y: jnp.where(live.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), s2, s)
        hist, ptr = jax.vmap(push_hist, (0, None, 0))(hist, ptr, s2)
        return (s2, hist, ptr[0], res, ln + live), None
    (_, _, _, res, ln), _ = lax.scan(f, (s, hist, ptr, jnp.full(games, 2), jnp.zeros(games)), jax.random.split(k2, TMAX))
    return res, ln


net_player = lambda ac: lambda s, h, p, side, k: pi(ac, jax.vmap(obs)(s, side))
def late_player(ac, delay, motor):
    return lambda s, h, p, side, k: pi(ac, jax.vmap(obs_late, (0, 0, None, 0, None))(s, h, p, side, delay)) + motor * jax.random.normal(k, (side.shape[0], 2))
def bot_player(delay, noise, kind=0):
    def f(s, h, p, side, k):
        n = side.shape[0]
        b = jax.vmap(sample_bot)(jax.random.split(jax.random.PRNGKey(7), n))
        b = b._replace(kind=jnp.full(n, kind), delay=jnp.full(n, delay), noise=jnp.full(n, noise * 1.))
        return jax.vmap(bot_action, (0, 0, 0, None, 0, 0))(b, s, h, p, side, jax.random.split(k, n))
    return f


def summary(res, ln):
    res, ln = np.array(res), np.array(ln)
    w, l, d = (res == 1).mean(), (res == -1).mean(), (res == 0).mean()
    return dict(w=round(float(w), 3), l=round(float(l), 3), d=round(float(d), 3), score=round(float(w + d / 2), 3), len=int(ln.mean()))


def export(actor, path):
    """float32 [n, sizes..., then per layer W (out x in, row-major), b]; read by ppo_agent.js."""
    net = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in actor['net']]
    sizes = [net[0][0].shape[0]] + [w.shape[1] for w, _ in net]
    np.concatenate([np.array([len(sizes)] + sizes, np.float32)] + [x for w, b in net for x in (w.T.ravel(), b)]).tofile(path)
