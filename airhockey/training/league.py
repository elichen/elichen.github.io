"""League PPO on the GPU: learner vs latest self, a PFSP-weighted snapshot pool, and delayed scripted experts."""
import argparse, json, os, pickle, time
from typing import NamedTuple
import numpy as np, jax, jax.numpy as jnp, optax
from jax import lax
from hockey import *

p = argparse.ArgumentParser()
p.add_argument('--name', default='run'); p.add_argument('--out', default='/mnt/c/w/airhockey/runs')
p.add_argument('--envs', type=int, default=8192); p.add_argument('--T', type=int, default=64)
p.add_argument('--updates', type=int, default=4000); p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--hid', type=int, default=256); p.add_argument('--gamma', type=float, default=.993)
p.add_argument('--lam', type=float, default=.95); p.add_argument('--epochs', type=int, default=4)
p.add_argument('--mb', type=int, default=8); p.add_argument('--ent', type=float, default=0.)
p.add_argument('--bots', type=float, default=.25); p.add_argument('--self_blocks', type=int, default=3)
p.add_argument('--blocks', type=int, default=12); p.add_argument('--draw', type=float, default=0.)
p.add_argument('--shape', type=float, default=0.); p.add_argument('--shape_end', type=int, default=300)
p.add_argument('--snap', type=int, default=10); p.add_argument('--eval', type=int, default=25)
p.add_argument('--pool', type=int, default=64); p.add_argument('--seed', type=int, default=0)
p.add_argument('--resume', default=''); p.add_argument('--init', default='')
p.add_argument('--logstd', type=float, default=-.5); p.add_argument('--warmup', type=int, default=0)
p.add_argument('--p_expert', type=float, default=.6); p.add_argument('--p_random', type=float, default=.1)
p.add_argument('--delay', type=int, default=0); p.add_argument('--motor', type=float, default=0.)
p.add_argument('--vs', default=''); p.add_argument('--vs_blocks', type=int, default=0)
p.add_argument('--min_logstd', type=float, default=-9.)
A = p.parse_args()
N, T, K = A.envs, A.T, A.blocks
NB = int(N * A.bots) // K * K if A.bots < 1 else N
NN = N - NB; NN -= NN % K; NB = N - NN
OUT = f'{A.out}/{A.name}'; os.makedirs(OUT, exist_ok=True)


class E(NamedTuple):
    s: St; side: jax.Array; bot: Bot; hist: jax.Array; ptr: jax.Array


def critic_in(e, o):
    return jnp.concatenate([o * 2 - 1, (e.s.t / TMAX * 2 - 1)[:, None], (jnp.arange(N) >= NN)[:, None] * 1.,
                            (e.bot.delay / D)[:, None] * (jnp.arange(N) >= NN)[:, None]], 1)


def fresh(key, n):
    k1, k2, k3 = jax.random.split(key, 3)
    s = jax.vmap(start_state)(jax.random.split(k1, n))
    return s, jax.random.randint(k2, (n,), 0, 2), jax.vmap(lambda k: sample_bot(k, A.p_expert, A.p_random))(jax.random.split(k3, n))


def env_step(e, actor, opp, shape, key):
    k = jax.random.split(key, 6)
    oc, oo = jax.vmap(obs)(e.s, e.side), jax.vmap(obs)(e.s, 1 - e.side)
    ol = jax.vmap(obs_late, (0, 0, None, 0, None))(e.s, e.hist, e.ptr, e.side, A.delay) if A.delay else oc
    mu = pi(actor, ol); al = mu + jnp.exp(actor['logstd']) * jax.random.normal(k[0], mu.shape)
    mo = jax.vmap(pi)(opp, oo[:NN].reshape(K, NN // K, 12))
    ao = [(mo + jnp.exp(opp['logstd'])[:, None] * jax.random.normal(k[1], mo.shape)).reshape(NN, 2)]
    if NB:
        sl = lambda x: x[NN:]
        ao.append(jax.vmap(bot_action, (0, 0, 0, None, 0, 0))(jax.tree.map(sl, e.bot), jax.tree.map(sl, e.s), e.hist[NN:], e.ptr,
                                                              1 - e.side[NN:], jax.random.split(k[2], NB)))
    ao = jnp.concatenate(ao)
    ae = al + A.motor * jax.random.normal(k[4], al.shape)
    a = jnp.where((e.side == 0)[:, None, None], jnp.stack([ae, ao], 1), jnp.stack([ao, ae], 1))
    s2, goal, tout, _ = jax.vmap(step)(e.s, a)
    win = goal * jnp.where(e.side == 0, 1, -1)
    done = (goal != 0) | tout
    phi = lambda s: shape * (jax.vmap(own)(e.side, s.kp)[:, 1] / H - .5)
    r = win + tout * A.draw + jnp.where(done, 0., A.gamma * phi(s2)) - phi(e.s)
    ns, nside, nbot = fresh(k[3], N)
    w = lambda x, y: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), y, x)
    s3 = jax.tree.map(w, s2, ns)
    hist, ptr = jax.vmap(push_hist, (0, None, 0))(e.hist, e.ptr, s3)
    hist = w(hist, jax.vmap(fill_hist)(s3)[0])
    e2 = E(s3, w(e.side, nside), jax.tree.map(w, e.bot, nbot), hist, ptr[0])
    return e2, (ol, critic_in(e, oc), al, logp(al, mu, actor['logstd']), r, done, win, tout)


def make_update(opt):
    def update(carry, pool, idx, shape, act_on):
        actor, critic, ost, e, key = carry
        pool = jax.tree.map(lambda P, x: P.at[0].set(x), pool, actor)
        opp = jax.tree.map(lambda P: P[idx], pool)
        def roll(c, k):
            e = c
            e2, tr = env_step(e, actor, opp, shape, k)
            return e2, tr
        key, kr, kp = jax.random.split(key, 3)
        e, (o, ci, a, lp, r, d, win, tout) = lax.scan(roll, e, jax.random.split(kr, T))
        v = jax.vmap(lambda x: mlp(critic, x)[:, 0])(ci)
        lv = mlp(critic, critic_in(e, jax.vmap(obs)(e.s, e.side)))[:, 0]
        def g(c, x):
            adv, nv = c; v_, r_, d_ = x
            adv = r_ + A.gamma * nv * (1 - d_) - v_ + A.gamma * A.lam * (1 - d_) * adv
            return (adv, v_), adv
        _, adv = lax.scan(g, (jnp.zeros(N), lv), (v, r, d * 1.), reverse=True)
        ret = adv + v
        B = jax.tree.map(lambda x: x.reshape((T * N,) + x.shape[2:]), (o, ci, a, lp, adv, ret))
        def loss(params, b):
            ac, cr = params; o, ci, a, lp, adv, ret = b
            mu = pi(ac, o); nlp = logp(a, mu, ac['logstd']); ratio = jnp.exp(nlp - lp)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pl = -jnp.minimum(ratio * adv, jnp.clip(ratio, .8, 1.2) * adv).mean()
            vl = ((mlp(cr, ci)[:, 0] - ret) ** 2).mean()
            ent = (ac['logstd'] + .5 * np.log(2 * np.pi * np.e)).sum()
            return pl + .5 * vl - A.ent * ent, (pl, vl, ((ratio - 1) - (nlp - lp)).mean())
        def epoch(c, k):
            params, ost = c
            perm = jax.random.permutation(k, T * N).reshape(A.mb, -1)
            def mb(c, ix):
                params, ost = c
                (l, aux), gr = jax.value_and_grad(loss, has_aux=True)(params, jax.tree.map(lambda x: x[ix], B))
                up, ost = opt.update(gr, ost, params)
                up = (jax.tree.map(lambda x: x * act_on, up[0]), up[1])
                ac, cr = optax.apply_updates(params, up)
                return (({**ac, 'logstd': jnp.maximum(ac['logstd'], A.min_logstd)}, cr), ost), aux
            return lax.scan(mb, (params, ost), perm)
        ((actor, critic), ost), aux = lax.scan(epoch, ((actor, critic), ost), jax.random.split(kp, A.epochs))
        blk = lambda x: x[:, :NN].reshape(T, K, NN // K).sum((0, 2))
        stats = dict(w=blk(win == 1), l=blk(win == -1), dr=blk(tout), bw=(win[:, NN:] == 1).sum(), bl=(win[:, NN:] == -1).sum(),
                     bd=tout[:, NN:].sum(), pl=aux[0].mean(), vl=aux[1].mean(), kl=aux[2].mean(), r=r.mean())
        return (actor, critic, ost, e, key), pool, stats
    return update


EVALS = {'expert0': bot_player(0, 0.), 'expert6': bot_player(6, 4.), 'human12': bot_player(12, 6.), 'random': bot_player(0, 0., 1), 'goalie': bot_player(0, 0., 2)}
me = lambda ac: late_player(ac, A.delay, A.motor) if A.delay or A.motor else net_player(ac)
rstart = lambda key: jax.vmap(lambda k: start_state(k, 1.))(jax.random.split(key, 3072))
_ev = {name: jax.jit(lambda ac, key, f=f: play(me(ac), f, 3072, key)) for name, f in EVALS.items()}
_ev_net = jax.jit(lambda ac, op, key: play(me(ac), net_player(op), 3072, key, rstart(key)))


key = jax.random.PRNGKey(A.seed)
k1, k2, k3, key = jax.random.split(key, 4)
actor = {'net': init_mlp(k1, [12, A.hid, A.hid, A.hid, 2], .01), 'logstd': jnp.full(2, A.logstd)}
if A.init: ck = pickle.load(open(A.init, 'rb')); actor['net'] = jax.tree.map(jnp.asarray, (ck['actor'] if 'actor' in ck else ck)['net'])
critic = init_mlp(k2, [15, A.hid, A.hid, A.hid, 1], 1.)
opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(A.lr, eps=1e-5))
ost = opt.init((actor, critic))
s, side, bot = fresh(k3, N)
hist, ptr = jax.vmap(fill_hist)(s)
e = E(s, side, bot, hist, ptr[0])
pool = jax.tree.map(lambda x: jnp.zeros((A.pool,) + x.shape, x.dtype) + x, actor)
score = np.full(A.pool, .5); filled, start_u, hist_log = 1, 0, []
if A.resume:
    ck = pickle.load(open(A.resume, 'rb'))
    actor, critic, ost, pool = jax.tree.map(jnp.asarray, (ck['actor'], ck['critic'], ck['ost'], ck['pool']))
    score, filled, start_u = ck['score'], ck['filled'], ck['u'] + 1
VS = [jax.tree.map(jnp.asarray, pickle.load(open(f, 'rb'))['actor']) for f in A.vs.split(',')] if A.vs else []
NV = len(VS)
update = jax.jit(make_update(opt))
for j, v in enumerate(VS): pool = jax.tree.map(lambda P, x: P.at[A.pool - 1 - j].set(x), pool, {**v, 'logstd': v.get('logstd', jnp.full(2, -3.))})
push = jax.jit(lambda pool, ac, i: jax.tree.map(lambda P, x: P.at[i].set(x), pool, ac))
pool = push(pool, actor, 1); filled = max(filled, 2); init_actor = actor
carry = (actor, critic, ost, e, key)
rng = np.random.default_rng(A.seed)
log = open(f'{OUT}/log.txt', 'a')
say = lambda *x: (print(*x, flush=True), print(*x, file=log, flush=True))
say(f'# {vars(A)}  net envs {NN} in {K} blocks, bot envs {NB}')
t0, steps = time.time(), 0
for u in range(start_u, A.updates):
    wts = (1 - score[1:filled]) ** 2 + .02
    idx = np.concatenate([np.zeros(A.self_blocks, int), A.pool - 1 - np.arange(A.vs_blocks) % max(NV, 1),
                          1 + rng.choice(filled - 1, K - A.self_blocks - A.vs_blocks, p=wts / wts.sum())])
    shape = A.shape * max(0., 1 - u / A.shape_end)
    carry, pool, st = update(carry, pool, jnp.array(idx), shape, float(u >= A.warmup))
    st = jax.tree.map(np.asarray, st); steps += N * T
    for j, i in enumerate(idx):
        n = st['w'][j] + st['l'][j] + st['dr'][j]
        if i and n: score[i] = .9 * score[i] + .1 * (st['w'][j] + .5 * st['dr'][j]) / n
    if u % A.snap == 0 and u:
        cap = A.pool - max(NV, 1)
        slot = 1 + (u // A.snap - 1) % (cap - 1) if filled >= cap else filled
        pool = push(pool, carry[0], slot); score[slot] = .5; filled = min(filled + 1, cap)
    if u % 5 == 0:
        sb = slice(0, A.self_blocks); pb = slice(A.self_blocks, K)
        f = lambda s_: f"{st['w'][s_].sum() / max(1, st['w'][s_].sum() + st['l'][s_].sum() + st['dr'][s_].sum()):.2f}/{st['l'][s_].sum() / max(1, st['w'][s_].sum() + st['l'][s_].sum() + st['dr'][s_].sum()):.2f}"
        nb = st['bw'] + st['bl'] + st['bd']
        say(f"u {u} steps {steps / 1e6:.0f}M sps {steps / (time.time() - t0) / 1e3:.0f}k | self w/l {f(sb)} pool w/l {f(pb)} "
            f"bots w/l {st['bw'] / max(nb, 1):.2f}/{st['bl'] / max(nb, 1):.2f} d {st['bd'] / max(nb, 1):.2f} | std {np.exp(np.asarray(carry[0]['logstd'])).round(3)} "
            f"vl {st['vl']:.4f} kl {st['kl']:.4f} pool {filled} minscore {score[1:filled].min():.2f}")
    if u % A.eval == 0:
        ev = {n: summary(*f(carry[0], jax.random.PRNGKey(u))) for n, f in _ev.items()}
        old = jax.tree.map(lambda P: P[max(1, filled - 11)], pool)
        ev['snap-10'] = summary(*_ev_net(carry[0], old, jax.random.PRNGKey(u)))
        ev['init'] = summary(*_ev_net(carry[0], init_actor, jax.random.PRNGKey(u)))
        for j, v in enumerate(VS): ev[f'vs{j}'] = summary(*_ev_net(carry[0], v, jax.random.PRNGKey(u)))
        say(f"EVAL u {u} " + json.dumps(ev))
        export(carry[0], f'{OUT}/policy_{u:05d}.bin'); export(carry[0], f'{OUT}/policy.bin')
        pickle.dump(dict(actor=carry[0], critic=carry[1], ost=carry[2], pool=pool, score=score, filled=filled, u=u, args=vars(A)),
                    open(f'{OUT}/ckpt.pkl', 'wb'))
