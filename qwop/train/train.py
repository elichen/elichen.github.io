"""Three-stage training for the QWOP runner.

  python train.py imitate   # behaviour cloning from the scripted teacher, then DAgger
  python train.py ppo       # PPO fine-tuning starting from the DAgger policy
  python train.py export    # write ../models.json for the browser

Everything runs on a laptop: the numba simulator does ~1M env-steps/s on CPU.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sim
from teacher import Teacher

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
HIDDEN = 256


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.Tanh(),
                         nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                         nn.Linear(HIDDEN, n_out))


class CpuActor:
    """CPU copy of the actor for per-step rollout inference (avoids MPS sync latency)."""

    def __init__(self, actor):
        self.net = mlp(sim.NOBS, sim.NACT)
        self.sync(actor)

    def sync(self, actor):
        self.net.load_state_dict({k: v.detach().cpu() for k, v in actor.state_dict().items()})

    @torch.no_grad()
    def logits(self, obs):
        return self.net(torch.from_numpy(obs).float())


def evaluate(policy_fn, n=1024, max_steps=3000, noise=0.05, seed=123, on_reset=None):
    """Run each env until it falls, reaches 100 m, or times out (100 s).

    Returns task-unit metrics: finish rate, mean 100 m time of finishers,
    mean distance, mean speed while alive."""
    np.random.seed(seed)
    sim.seed_numba(seed)
    env = sim.QwopVec(n, noise=noise, max_steps=max_steps)
    if on_reset:
        on_reset(np.arange(n))
    active = np.ones(n, bool)
    finished = np.zeros(n, bool)
    steps = np.zeros(n)
    knee_steps = np.zeros(n)
    for _ in range(max_steps):
        a = policy_fn(env)
        dx, fell, goal, timeout = env.step(a)
        steps += active
        knee_steps += active & ((env.ACC[:, sim.NJ * 5 + 8] > 0) | (env.ACC[:, sim.NJ * 5 + 10] > 0))
        finished |= active & goal
        active &= ~(fell | goal)
        if not active.any():
            break
    dist = np.minimum(env.S[:, sim.TORSO, 0], sim.GOAL_X)
    t = steps * sim.DT
    fin_t = t[finished]
    ci = 1.96 * fin_t.std() / np.sqrt(max(len(fin_t), 1)) if len(fin_t) else float('nan')
    return dict(finish_rate=float(finished.mean()),
                time_100m=float(fin_t.mean()) if len(fin_t) else None,
                time_100m_ci95=float(ci) if len(fin_t) else None,
                best_time=float(fin_t.min()) if len(fin_t) else None,
                distance=float(dist.mean()),
                knee_frac=float((knee_steps / np.maximum(steps, 1)).mean()),
                speed=float((dist / np.maximum(t, sim.DT)).mean()),
                fall_rate=float(1.0 - finished.mean() - active.mean()))


def greedy(cpu_actor):
    return lambda env: cpu_actor.logits(env.obs()).argmax(1).numpy()


def fmt(m):
    t = f"{m['time_100m']:.1f}s" if m['time_100m'] is not None else "--"
    return (f"finish {m['finish_rate'] * 100:5.1f}%  100m {t}  dist {m['distance']:6.2f} m  "
            f"speed {m['speed']:.2f} m/s  falls {m['fall_rate'] * 100:5.1f}%  knee-down {m['knee_frac'] * 100:4.1f}%")


# ------------------------------------------------------------ imitation ----
def collect(n, steps, teacher_path, cpu_actor, beta, rng, eps=0.03):
    """Roll out a teacher/student mixture; label every visited state with the teacher.

    Each env is driven by the teacher with probability beta (chosen per episode),
    otherwise by the student. The teacher's FSM tracks the physical state either way."""
    env = sim.QwopVec(n, noise=0.05, max_steps=steps + 1)
    teacher = Teacher.from_json(teacher_path, n)
    use_teacher = rng.random(n) < beta
    X, Y = [], []
    for _ in range(steps):
        obs = env.obs()
        label = teacher.act(env.S)
        if cpu_actor is None:
            a = label
        else:
            a = np.where(use_teacher, label, cpu_actor.logits(obs).argmax(1).numpy())
        a = np.where(rng.random(n) < eps, rng.integers(0, sim.NACT, n), a)
        X.append(obs.astype(np.float32))
        Y.append(label.copy())
        _, fell, goal, _ = env.step(a)
        done = np.nonzero(fell | goal)[0]
        if done.size:
            env.reset(done)
            teacher.reset(done)
            use_teacher[done] = rng.random(done.size) < beta
    return np.concatenate(X), np.concatenate(Y)


def fit(actor, X, Y, epochs, lr=1e-3, bs=8192):
    X = torch.from_numpy(X).to(DEV)
    Y = torch.from_numpy(Y).to(DEV)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(actor(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
    with torch.no_grad():
        accs = [(actor(X[i:i + 65536]).argmax(1) == Y[i:i + 65536]).float().sum().item()
                for i in range(0, n, 65536)]
    return tot / n, sum(accs) / n


def imitate(args):
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    tpath = os.path.join(HERE, 'teacher.json')
    log = {}

    t_eval = Teacher.from_json(tpath, 1024)
    log['teacher'] = evaluate(lambda env: t_eval.act(env.S), on_reset=t_eval.reset)
    print("teacher        ", fmt(log['teacher']), flush=True)

    actor = mlp(sim.NOBS, sim.NACT).to(DEV)
    cpu = CpuActor(actor)
    X, Y = collect(2048, 300, tpath, None, 1.0, rng, eps=0.0)
    loss, acc = fit(actor, X, Y, 30)
    cpu.sync(actor)
    log['bc'] = dict(evaluate(greedy(cpu)), samples=int(len(X)), label_acc=acc)
    print(f"bc  ({len(X) / 1e6:.2f}M, acc {acc:.3f})", fmt(log['bc']), flush=True)
    torch.save(actor.state_dict(), os.path.join(HERE, 'bc.pt'))

    log['dagger'] = []
    best = (-1.0, None)
    for it in range(args.dagger_iters):
        beta = 0.5 ** (it + 1)
        Xn, Yn = collect(2048, 300, tpath, cpu, beta, rng)
        X = np.concatenate([X, Xn])
        Y = np.concatenate([Y, Yn])
        loss, acc = fit(actor, X, Y, 8, lr=5e-4)
        cpu.sync(actor)
        m = dict(evaluate(greedy(cpu)), samples=int(len(X)), label_acc=acc, beta=beta)
        log['dagger'].append(m)
        print(f"dagger {it:2d} ({len(X) / 1e6:.2f}M, acc {acc:.3f})", fmt(m), flush=True)
        if m['distance'] > best[0]:
            best = (m['distance'], it)
            torch.save(actor.state_dict(), os.path.join(HERE, 'dagger.pt'))
    log['dagger_best_iter'] = best[1]
    json.dump(log, open(os.path.join(HERE, 'imitation_log.json'), 'w'), indent=1)


# ------------------------------------------------------------------ PPO ----
def ppo(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    N, T = args.envs, args.horizon
    gamma, lam = 0.99, 0.95
    actor = mlp(sim.NOBS, sim.NACT).to(DEV)
    if args.init:
        actor.load_state_dict(torch.load(os.path.join(HERE, args.init), map_location=DEV))
    critic = mlp(sim.NOBS, 1).to(DEV)
    cpu = CpuActor(actor)
    cpu_critic = mlp(sim.NOBS, 1)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr, eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3, eps=1e-5)
    env = sim.QwopVec(N, noise=0.05, max_steps=1000, seed=args.seed)
    iters = args.steps // (N * T)
    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    history = []
    best = -1e9
    ep_ret = np.zeros(N)
    ep_len = np.zeros(N)
    t0 = time.time()
    obs = env.obs()
    for it in range(iters):
        frac = it / iters
        warm = it < args.critic_warmup
        for g in opt_a.param_groups:
            g['lr'] = args.lr * (1.0 - 0.9 * frac)
        ent_coef = args.ent * (1.0 - frac)
        cpu_critic.load_state_dict({k: v.detach().cpu() for k, v in critic.state_dict().items()})
        B_obs = np.zeros((T, N, sim.NOBS), np.float32)
        B_act = np.zeros((T, N), np.int64)
        B_logp = np.zeros((T, N), np.float32)
        B_rew = np.zeros((T, N), np.float32)
        B_done = np.zeros((T, N), np.float32)
        B_val = np.zeros((T + 1, N), np.float32)
        falls, dones_len, dist_done = 0, [], []
        vel = 0.0
        for t in range(T):
            ot = torch.from_numpy(obs).float()
            with torch.no_grad():
                logits = cpu.net(ot)
                B_val[t] = cpu_critic(ot).squeeze(1).numpy()
            d = torch.distributions.Categorical(logits=logits)
            a = d.sample()
            B_obs[t], B_act[t], B_logp[t] = obs, a.numpy(), d.log_prob(a).numpy()
            dx, fell, goal, timeout = env.step(a.numpy())
            rew = dx * (0.1 / sim.DT) - 2.0 * fell
            if args.knee_penalty:  # shaping only: knees on the ground are legal, just discouraged
                knees = (env.ACC[:, sim.NJ * 5 + 8] > 0) | (env.ACC[:, sim.NJ * 5 + 10] > 0)
                rew -= args.knee_penalty * knees
            trunc = (goal | timeout) & ~fell
            if trunc.any():  # bootstrap through time-limit / finish-line resets
                with torch.no_grad():
                    vf = cpu_critic(torch.from_numpy(env.obs()[trunc]).float()).squeeze(1).numpy()
                rew[trunc] += gamma * vf
            done = fell | trunc
            B_rew[t], B_done[t] = rew, done
            vel += dx.mean() / sim.DT
            ep_len += 1
            idx = np.nonzero(done)[0]
            if idx.size:
                falls += int(fell.sum())
                dones_len += ep_len[idx].tolist()
                dist_done += env.S[idx, sim.TORSO, 0].tolist()
                ep_len[idx] = 0
                env.reset(idx)
            obs = env.obs()
        with torch.no_grad():
            B_val[T] = cpu_critic(torch.from_numpy(obs).float()).squeeze(1).numpy()
        adv = np.zeros((T, N), np.float32)
        last = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nd = 1.0 - B_done[t]
            delta = B_rew[t] + gamma * B_val[t + 1] * nd - B_val[t]
            last = delta + gamma * lam * nd * last
            adv[t] = last
        ret = adv + B_val[:T]

        o = torch.from_numpy(B_obs.reshape(-1, sim.NOBS)).to(DEV)
        a = torch.from_numpy(B_act.reshape(-1)).to(DEV)
        lp0 = torch.from_numpy(B_logp.reshape(-1)).to(DEV)
        ad = torch.from_numpy(adv.reshape(-1)).to(DEV)
        rt = torch.from_numpy(ret.reshape(-1)).to(DEV)
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        n = o.shape[0]
        mb = n // args.minibatches
        kl = 0.0
        for ep in range(args.epochs):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                vloss = F.mse_loss(critic(o[idx]).squeeze(1), rt[idx])
                opt_c.zero_grad()
                vloss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_c.step()
                if warm:
                    continue
                d = torch.distributions.Categorical(logits=actor(o[idx]))
                lp = d.log_prob(a[idx])
                ratio = torch.exp(lp - lp0[idx])
                pl = -torch.min(ratio * ad[idx], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * ad[idx]).mean()
                loss = pl - ent_coef * d.entropy().mean()
                opt_a.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                opt_a.step()
                kl = (lp0[idx] - lp).mean().item()
        cpu.sync(actor)
        rec = dict(iter=it, steps=(it + 1) * N * T, vel=vel / T, vloss=vloss.item(), kl=kl,
                   falls_per_1k=1000.0 * falls / (N * T),
                   ep_len=float(np.mean(dones_len)) if dones_len else None,
                   wall=time.time() - t0)
        if it % args.eval_every == 0 or it == iters - 1:
            m = evaluate(greedy(cpu), n=512)
            rec['eval'] = m
            score = m['distance'] + (100.0 * (30.0 / m['time_100m']) if m['time_100m'] else 0.0) * m['finish_rate']
            torch.save(actor.state_dict(), os.path.join(out, f'actor_{it:05d}.pt'))
            if score > best:
                best = score
                torch.save(actor.state_dict(), os.path.join(out, 'best.pt'))
            print(f"it {it:4d} {rec['steps'] / 1e6:7.1f}M  vel {rec['vel']:5.2f}  vloss {rec['vloss']:.3f} "
                  f"kl {kl:.4f}  {rec['wall']:5.0f}s | eval: {fmt(m)}", flush=True)
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict()), os.path.join(out, 'latest_full.pt'))
        history.append(rec)
        json.dump(history, open(os.path.join(out, 'history.json'), 'w'))


# ----------------------------------------------------------- final eval ----
def load_actor(path):
    net = mlp(sim.NOBS, sim.NACT)
    net.load_state_dict(torch.load(os.path.join(HERE, path), map_location='cpu'))
    return CpuActor(net)


def final_eval(args):
    """Score every stage on the same 2,048 episodes (4 seeds x 512)."""
    stats = {}
    paths = [('bc', 'bc.pt'), ('dagger', 'dagger.pt')] + [(f'rl:{p}', p) for p in args.rl.split(',')]
    for name, path in [('teacher', None)] + paths:
        runs = []
        for seed in (11, 22, 33, 44):
            if path is None:
                t = Teacher.from_json(os.path.join(HERE, 'teacher.json'), 512)
                runs.append(evaluate(lambda env: t.act(env.S), n=512, seed=seed, on_reset=t.reset))
            else:
                runs.append(evaluate(greedy(load_actor(path)), n=512, seed=seed))
        m = {k: (float(np.mean([r[k] for r in runs if r[k] is not None]))
                 if any(r[k] is not None for r in runs) else None) for k in runs[0]}
        m['best_time'] = min([r['best_time'] for r in runs if r['best_time'] is not None], default=None)
        m['episodes'] = 2048
        stats[name] = m
        print(f"{name:28s}", fmt(m), f"ci95 +-{m['time_100m_ci95']}" if m['time_100m'] else "", flush=True)
    json.dump(stats, open(os.path.join(HERE, 'final_stats.json'), 'w'), indent=1)


# --------------------------------------------------------------- export ----
def export(args):
    def layers(path):
        sd = torch.load(os.path.join(HERE, path), map_location='cpu')
        ks = sorted({k.split('.')[0] for k in sd}, key=int)
        return [dict(W=[[float(f"{v:.5g}") for v in row] for row in sd[f'{k}.weight'].tolist()],
                     b=[float(f"{v:.5g}") for v in sd[f'{k}.bias'].tolist()]) for k in ks]
    models = dict(teacher=json.load(open(os.path.join(HERE, 'teacher.json'))))
    for name, path in [('bc', 'bc.pt'), ('dagger', 'dagger.pt'), ('rl', args.rl)]:
        models[name] = layers(path)
    sp = os.path.join(HERE, 'final_stats.json')
    if os.path.exists(sp):
        st = json.load(open(sp))
        models['stats'] = {k.split(':')[0]: v for k, v in st.items() if ':' not in k or k == f'rl:{args.rl}'}
    dst = os.path.join(HERE, '..', 'models.json')
    json.dump(models, open(dst, 'w'), separators=(',', ':'))
    print("wrote", dst, os.path.getsize(dst) // 1024, "KB")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['imitate', 'ppo', 'finaleval', 'export'])
    ap.add_argument('--dagger-iters', type=int, default=10)
    ap.add_argument('--steps', type=int, default=300_000_000)
    ap.add_argument('--envs', type=int, default=8192)
    ap.add_argument('--horizon', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--minibatches', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--critic-warmup', type=int, default=10)
    ap.add_argument('--eval-every', type=int, default=20)
    ap.add_argument('--knee-penalty', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', default='dagger.pt')
    ap.add_argument('--out', default='runs/ppo')
    ap.add_argument('--rl', default='rl.pt')
    args = ap.parse_args()
    dict(imitate=imitate, ppo=ppo, finaleval=final_eval, export=export)[args.cmd](args)
