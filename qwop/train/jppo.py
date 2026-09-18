"""End-to-end GPU PPO for the QWOP runner (JAX). Mirrors train.py's PPO.

Trains in float32 on the GPU simulator (jsim.py). Float32 physics drifts from
the float64 reference, so checkpoints written here are only candidates:
score them with `train.py finaleval` in the numba simulator before use.

  python jppo.py --init init.npz --out /mnt/c/w/qwop/run1 --steps 1000000000
"""
import argparse, json, os, pickle, time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax
import consts as C
import jsim

NOBS, NACT = C.NOBS, C.NACT


def forward(p, x):
    for i, (W, b) in enumerate(p):
        x = x @ W.T + b
        if i < len(p) - 1:
            x = jnp.tanh(x)
    return x


def init_mlp(key, n_in, n_out, hidden=256):
    sizes = [n_in, hidden, hidden, n_out]
    ks = jax.random.split(key, 3)
    return [(jax.random.uniform(k, (o, i), minval=-1, maxval=1) / np.sqrt(i), jnp.zeros(o))
            for k, i, o in zip(ks, sizes[:-1], sizes[1:])]


def make_train(args, pool_S):
    N, T = args.envs, args.horizon
    gamma, lam = args.gamma, 0.95
    n_pool = pool_S.shape[0]
    opt_a = optax.chain(optax.clip_by_global_norm(0.5), optax.inject_hyperparams(optax.adam)(learning_rate=args.lr, eps=1e-5))
    opt_c = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3, eps=1e-5))

    def reset_where(done, S, ACC, prev, t, key):
        idx = jax.random.randint(key, (S.shape[0],), 0, n_pool)
        S = jnp.where(done[:, None, None], pool_S[idx], S)
        ACC = jnp.where(done[:, None], 0.0, ACC)
        return S, ACC, jnp.where(done, 0, prev), jnp.where(done, 0, t)

    def rollout(actor, critic, env, key):
        def one(carry, k):
            S, ACC, prev, t = carry
            k1, k2 = jax.random.split(k)
            obs = jsim.observe(S, ACC, prev)
            logits = forward(actor, obs)
            a = jax.random.categorical(k1, logits)
            logp = jax.nn.log_softmax(logits)[jnp.arange(N), a]
            v = forward(critic, obs)[:, 0]
            S2, ACC2, fell, dx = jsim.step(S, ACC, a)
            t2 = t + 1
            trunc = ((S2[:, 0, 0] >= C.GOAL_X) | (t2 >= args.max_steps)) & ~fell
            v2 = forward(critic, jsim.observe(S2, ACC2, a))[:, 0]
            knees = (ACC2[:, C.NJ * 5 + 8] > 0) | (ACC2[:, C.NJ * 5 + 10] > 0)
            rew = dx * (0.1 / C.DT) - 2.0 * fell - args.knee_penalty * knees + gamma * v2 * trunc
            done = fell | trunc
            S3, ACC3, prev3, t3 = reset_where(done, S2, ACC2, a, t2, k2)
            return (S3, ACC3, prev3, t3), (obs, a, logp, v, rew, done.astype(jnp.float32), dx, fell, knees)
        env, traj = lax.scan(one, env, jax.random.split(key, T))
        S, ACC, prev, t = env
        last_v = forward(critic, jsim.observe(S, ACC, prev))[:, 0]
        return env, traj, last_v

    def gae(v, rew, done, last_v):
        def back(carry, x):
            nv, last = carry
            v_t, r_t, d_t = x
            delta = r_t + gamma * nv * (1 - d_t) - v_t
            last = delta + gamma * lam * (1 - d_t) * last
            return (v_t, last), last
        _, adv = lax.scan(back, (last_v, jnp.zeros_like(last_v)), (v, rew, done), reverse=True)
        return adv

    def update(state, batch, key, ent_coef, train_actor):
        obs, a, logp0, adv, ret = batch
        n = obs.shape[0]
        mb = n // args.minibatches

        def loss_a(p, o, a, lp0, ad):
            logits = forward(p, o)
            ls = jax.nn.log_softmax(logits)
            lp = ls[jnp.arange(o.shape[0]), a]
            ratio = jnp.exp(lp - lp0)
            pl = -jnp.minimum(ratio * ad, jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * ad).mean()
            ent = -(jnp.exp(ls) * ls).sum(1).mean()
            return pl - ent_coef * ent, (lp0 - lp).mean()

        def loss_c(p, o, r):
            return ((forward(p, o)[:, 0] - r) ** 2).mean()

        def minibatch(st, idx):
            actor, critic, sa, sc = st
            vl, gc = jax.value_and_grad(loss_c)(critic, obs[idx], ret[idx])
            uc, sc = opt_c.update(gc, sc)
            critic = optax.apply_updates(critic, uc)
            (_, kl), ga = jax.value_and_grad(loss_a, has_aux=True)(actor, obs[idx], a[idx], logp0[idx], adv[idx])
            ua, sa2 = opt_a.update(ga, sa)
            actor2 = optax.apply_updates(actor, ua)
            actor, sa = jax.tree.map(lambda new, old: jnp.where(train_actor, new, old), (actor2, sa2), (actor, sa))
            return (actor, critic, sa, sc), (vl, kl)

        def epoch(st, k):
            perm = jax.random.permutation(k, n)[:mb * args.minibatches].reshape(args.minibatches, mb)
            return lax.scan(minibatch, st, perm)

        state, (vl, kl) = lax.scan(epoch, state, jax.random.split(key, args.epochs))
        return state, vl[-1, -1], kl[-1, -1]

    @jax.jit
    def iteration(state, env, key, lr, ent_coef, train_actor):
        actor, critic, sa, sc = state
        sa[1].hyperparams['learning_rate'] = lr
        k1, k2 = jax.random.split(key)
        env, (obs, a, logp, v, rew, done, dx, fell, knees), last_v = rollout(actor, critic, env, k1)
        adv = gae(v, rew, done, last_v)
        ret = adv + v
        advf = adv.reshape(-1)
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        batch = (obs.reshape(-1, NOBS), a.reshape(-1), logp.reshape(-1), advf, ret.reshape(-1))
        state, vl, kl = update((actor, critic, sa, sc), batch, k2, ent_coef, train_actor)
        stats = dict(vel=dx.mean() / C.DT, falls_per_1k=1000.0 * fell.mean(), knee=knees.mean(), vloss=vl, kl=kl)
        return state, env, stats

    @jax.jit
    def evaluate(actor, key):
        """Greedy float32 GPU eval, 1024 runners, 40 s cap: a progress signal only."""
        n = 1024
        S = pool_S[jax.random.randint(key, (n,), 0, n_pool)]

        def one(carry, _):
            S, ACC, prev, active, fin, steps = carry
            a = jnp.argmax(forward(actor, jsim.observe(S, ACC, prev)), axis=1)
            S, ACC, fell, _ = jsim.step(S, ACC, a)
            steps = steps + active
            goal = S[:, 0, 0] >= C.GOAL_X
            fin = fin | (active & goal)
            active = active & ~(fell | goal)
            return (S, ACC, a, active, fin, steps), None
        init = (S, jnp.zeros((n, C.NACC)), jnp.zeros(n, jnp.int32), jnp.ones(n, bool), jnp.zeros(n, bool), jnp.zeros(n))
        (S, _, _, active, fin, steps), _ = lax.scan(one, init, None, length=1200)
        t = steps * C.DT
        return dict(finish=fin.mean(), time=(t * fin).sum() / jnp.maximum(fin.sum(), 1), best=jnp.where(fin, t, 1e9).min())

    return opt_a, opt_c, iteration, evaluate, reset_where


def to_np(tree):
    return jax.tree.map(lambda x: np.asarray(x), tree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default='')          # npz with actor (and optionally critic) weights
    ap.add_argument('--resume', default='')        # full checkpoint .pkl
    ap.add_argument('--pool', default='reset_pool.npy')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=1_000_000_000)
    ap.add_argument('--envs', type=int, default=8192)
    ap.add_argument('--horizon', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--minibatches', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lr-final', type=float, default=0.1)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--max-steps', type=int, default=1000)
    ap.add_argument('--knee-penalty', type=float, default=0.0)
    ap.add_argument('--critic-warmup', type=int, default=0)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, 'args.json'), 'w'), indent=1)

    pool_S = jnp.asarray(np.load(args.pool), jnp.float32)
    opt_a, opt_c, iteration, evaluate, reset_where = make_train(args, pool_S)
    key = jax.random.PRNGKey(args.seed)
    key, ka, kc, ke = jax.random.split(key, 4)
    actor, critic = init_mlp(ka, NOBS, NACT), init_mlp(kc, NOBS, 1)
    if args.init:
        z = np.load(args.init)
        actor = [(jnp.asarray(z[f'actor_W{i}']), jnp.asarray(z[f'actor_b{i}'])) for i in range(3)]
        if 'critic_W0' in z:
            critic = [(jnp.asarray(z[f'critic_W{i}']), jnp.asarray(z[f'critic_b{i}'])) for i in range(3)]
    state = (actor, critic, opt_a.init(actor), opt_c.init(critic))
    start = 0
    if args.resume:
        ck = pickle.load(open(args.resume, 'rb'))
        state = jax.tree.map(jnp.asarray, ck['state'])
        start = ck['iter'] + 1
    N = args.envs
    env = (pool_S[jax.random.randint(ke, (N,), 0, pool_S.shape[0])], jnp.zeros((N, C.NACC), jnp.float32),
           jnp.zeros(N, jnp.int32), jnp.zeros(N, jnp.int32))
    iters = args.steps // (N * args.horizon)
    log = open(os.path.join(args.out, 'history.jsonl'), 'a')
    best = 1e9
    t0 = time.time()
    for it in range(start, iters):
        frac = it / iters
        key, k = jax.random.split(key)
        lr = args.lr * (1.0 - (1.0 - args.lr_final) * frac)
        state, env, stats = iteration(state, env, k, lr, args.ent * (1.0 - frac), it >= args.critic_warmup)
        if it % args.eval_every == 0 or it == iters - 1:
            key, k = jax.random.split(key)
            ev = {k2: float(v) for k2, v in evaluate(state[0], k).items()}
            rec = dict(iter=it, steps=(it + 1) * N * args.horizon, wall=time.time() - t0,
                       **{k2: float(v) for k2, v in stats.items()}, **{f'eval_{k2}': v for k2, v in ev.items()})
            log.write(json.dumps(rec) + '\n')
            log.flush()
            # full trainer state first (resume point), then flat inference weights
            pickle.dump(dict(state=to_np(state), iter=it), open(os.path.join(args.out, 'latest_full.pkl.tmp'), 'wb'))
            os.replace(os.path.join(args.out, 'latest_full.pkl.tmp'), os.path.join(args.out, 'latest_full.pkl'))
            flat = {f'actor_W{i}': np.asarray(W) for i, (W, b) in enumerate(state[0])}
            flat.update({f'actor_b{i}': np.asarray(b) for i, (W, b) in enumerate(state[0])})
            np.savez(os.path.join(args.out, f'actor_{it:06d}.npz'), **flat)
            score = ev['time'] if ev['finish'] > 0.97 else 1e9
            if score < best:
                best = score
                np.savez(os.path.join(args.out, 'best.npz'), **flat)
            print(f"it {it:5d} {rec['steps'] / 1e6:8.1f}M  vel {rec['vel']:5.2f}  falls/1k {rec['falls_per_1k']:5.2f}  "
                  f"knee {rec['knee'] * 100:4.1f}%  kl {rec['kl']:.4f}  {rec['wall']:6.0f}s | "
                  f"gpu-eval finish {ev['finish'] * 100:5.1f}%  100m {ev['time']:.2f}s  best {ev['best']:.2f}s", flush=True)


if __name__ == '__main__':
    main()
