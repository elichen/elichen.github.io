"""SOTA GPU-accelerated PPO for Shadow Hand Rubik's Cube (Reorient & Twist) on MuJoCo-Warp.

Features:
  - Zero-copy Warp -> PyTorch CUDA rollouts (no CPU-GPU sync in training loop)
  - Running observation normalization (mean/std, clipped to [-10, 10])
  - Adaptive KL learning rate schedule (rsl_rl / rl_games style)
  - Value loss clipping, GAE(lambda=0.95, gamma=0.99), grad norm clipping
  - Automatic handover state collection (saves real successful terminal states for partner policy!)
  - Periodic checkpointing + live browser-ready export (policy.json + policy.bin + test.json)

Usage:
  python train_ppo.py --task reorient --xml /content/rubik/robot/shadow_rubik.xml --out /content/runs/reorient1 --steps 150000000
  python train_ppo.py --task twist --xml /content/rubik/robot/shadow_rubik.xml --handover /content/runs/reorient1/handovers.npz --out /content/runs/twist1 --steps 200000000
"""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn

from rubik_warp_env import RubikWarpEnv
from export import export_checkpoint


class RunningNorm:
    def __init__(self, dim: int, device: torch.device, clip: float = 10.0):
        self.mean = torch.zeros(dim, device=device, dtype=torch.float64)
        self.var = torch.ones(dim, device=device, dtype=torch.float64)
        self.count = 1e-4
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x64 = x.to(torch.float64)
        bm = x64.mean(dim=0)
        bv = x64.var(dim=0, unbiased=False)
        bc = x64.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * (bc / tot)
        m_a = self.var * self.count
        m_b = bv * bc
        M2 = m_a + m_b + (delta ** 2) * (self.count * bc / tot)
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.var + 1e-8).to(torch.float32)
        mean = self.mean.to(torch.float32)
        return torch.clamp((x - mean) / std, -self.clip, self.clip)

    def std_np(self):
        return np.sqrt(self.var.cpu().numpy() + 1e-8)

    def mean_np(self):
        return self.mean.cpu().numpy()

    def load_np(self, mean: np.ndarray, std: np.ndarray, count: float = 1e6):
        self.mean.copy_(torch.tensor(mean, device=self.mean.device, dtype=torch.float64))
        self.var.copy_(torch.tensor(np.maximum(std ** 2 - 1e-8, 1e-6), device=self.var.device, dtype=torch.float64))
        self.count = count


def build_mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        layers.append(nn.ELU())
        d = h
    last = nn.Linear(d, out_dim)
    nn.init.orthogonal_(last.weight, gain=0.01 if out_dim > 1 else 1.0)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, actor_hidden=(512, 256, 128), critic_hidden=(512, 256, 128), init_log_std=-0.3):
        super().__init__()
        self.actor = build_mlp(obs_dim, actor_hidden, act_dim)
        self.critic = build_mlp(obs_dim, critic_hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def act(self, obs_n: torch.Tensor, deterministic: bool = False):
        raw_mean = self.actor(obs_n)
        mean = torch.tanh(raw_mean)
        std = torch.exp(torch.clamp(self.log_std, -2.5, 0.0))
        dist = torch.distributions.Normal(mean, std)
        action = mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)
        val = self.critic(obs_n).squeeze(-1)
        return action, logp, val, mean, std

    def evaluate(self, obs_n: torch.Tensor, actions: torch.Tensor):
        mean = torch.tanh(self.actor(obs_n))
        std = torch.exp(torch.clamp(self.log_std, -2.5, 0.0))
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(actions).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        val = self.critic(obs_n).squeeze(-1)
        return logp, ent, val, mean, std


def save_ckpt(path, model, norm, obs_dim, act_dim, step, metrics):
    ckpt = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "obs_mean": norm.mean_np(),
        "obs_std": norm.std_np(),
        "obs_clip": norm.clip,
        "activation": "elu",
        "tanh_out": True,
        "step": step,
        "metrics": metrics,
        "actor_state_dict": model.actor.state_dict(),
        "critic_state_dict": model.critic.state_dict(),
        "log_std": model.log_std.detach().cpu(),
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def collect_handovers(env: RubikWarpEnv, model: ActorCritic, norm: RunningNorm, n_target: int = 4096, out_path: str = "handovers.npz"):
    """Run deterministic policy rollouts and collect real post-success states (qpos, qvel, ctrl)."""
    env.reset_all()
    obs = env.get_obs()
    qpos_buf, qvel_buf, ctrl_buf = [], [], []
    steps = 0
    max_steps = 800
    with torch.no_grad():
        while sum(len(x) for x in qpos_buf) < n_target and steps < max_steps:
            obs_n = norm.normalize(obs)
            action, _, _, _, _ = model.act(obs_n, deterministic=True)
            obs, _, _, info = env.step(action)
            sids = info["succ_ids"]
            if len(sids) > 0:
                qpos_buf.append(env.qpos[sids].detach().cpu().numpy().copy())
                qvel_buf.append(env.qvel[sids].detach().cpu().numpy().copy())
                ctrl_buf.append(env.ctrl[sids].detach().cpu().numpy().copy())
            steps += 1
    if not qpos_buf:
        print(f"WARNING: collected 0 handover states in {steps} steps!", flush=True)
        return 0
    qpos_all = np.concatenate(qpos_buf, axis=0)[:n_target]
    qvel_all = np.concatenate(qvel_buf, axis=0)[:n_target]
    ctrl_all = np.concatenate(ctrl_buf, axis=0)[:n_target]
    np.savez_compressed(out_path, qpos=qpos_all, qvel=qvel_all, ctrl=ctrl_all)
    print(f"Collected {len(qpos_all)} handover states in {steps} steps -> {out_path}", flush=True)
    return len(qpos_all)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["reorient", "twist"], required=True)
    p.add_argument("--xml", default="/content/rubik/robot/shadow_rubik.xml")
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--steps", type=int, default=100_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--desired-kl", type=float, default=0.016)
    p.add_argument("--entropy", type=float, default=0.002)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hidden", default="512,256,128")
    p.add_argument("--restore", default=None)
    p.add_argument("--handover", default=None)
    p.add_argument("--handover-prob", type=float, default=0.5)
    p.add_argument("--start-deg", type=float, default=22.0)
    p.add_argument("--brake-settle", action="store_true")
    p.add_argument("--collect-handovers", type=int, default=4096)
    p.add_argument("--save-every", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    hidden = tuple(int(x) for x in args.hidden.split(","))

    env = RubikWarpEnv(
        xml_path=args.xml,
        task=args.task,
        num_envs=args.num_envs,
        seed=args.seed,
        handover_path=args.handover,
        handover_prob=args.handover_prob,
        twist_start_deg=args.start_deg,
        brake_settle=args.brake_settle,
    )
    device = env.device
    norm = RunningNorm(env.obs_dim, device)
    model = ActorCritic(env.obs_dim, env.act_dim, actor_hidden=hidden, critic_hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    start_step = 0
    if args.restore:
        ck = torch.load(args.restore, map_location=device, weights_only=False)
        model.actor.load_state_dict(ck["actor_state_dict"])
        model.critic.load_state_dict(ck["critic_state_dict"])
        model.log_std.data.copy_(ck["log_std"].to(device))
        norm.load_np(ck["obs_mean"], ck["obs_std"])
        print(f"Restored from {args.restore} (prev step={ck.get('step', 0)})", flush=True)

    N = args.num_envs
    T = args.horizon
    batch_size = N * T
    mb_size = batch_size // args.minibatches
    n_iters = max(1, args.steps // batch_size)

    # Rollout buffers on GPU
    buf_obs = torch.zeros((T, N, env.obs_dim), device=device)
    buf_act = torch.zeros((T, N, env.act_dim), device=device)
    buf_logp = torch.zeros((T, N), device=device)
    buf_rew = torch.zeros((T, N), device=device)
    buf_done = torch.zeros((T, N), device=device)
    buf_val = torch.zeros((T, N), device=device)
    buf_mean = torch.zeros((T, N, env.act_dim), device=device)
    buf_std = torch.zeros((T, N, env.act_dim), device=device)

    obs = env.reset_all()
    lr = args.lr
    t0 = time.time()
    total_steps = start_step
    best_succ_rate = -1.0
    history = []

    print(f"Starting PPO [{args.task}]: N={N}, T={T}, batch={batch_size}, iters={n_iters}, hidden={hidden}", flush=True)

    for it in range(1, n_iters + 1):
        iter_succ = 0
        iter_drop = 0
        iter_done = 0
        iter_err_deg = 0.0

        # 1. Collect rollout
        with torch.no_grad():
            for t in range(T):
                norm.update(obs)
                obs_n = norm.normalize(obs)
                act, logp, val, mean, std = model.act(obs_n, deterministic=False)
                next_obs, rew, done, info = env.step(act)

                buf_obs[t] = obs_n
                buf_act[t] = act
                buf_logp[t] = logp
                buf_rew[t] = rew
                buf_done[t] = done.float()
                buf_val[t] = val
                buf_mean[t] = mean
                buf_std[t] = std
                obs = next_obs

                iter_succ += info["n_succ"]
                iter_drop += info["n_drop"]
                iter_done += info["n_done"]
                iter_err_deg += info.get("mean_err_deg", 0.0)

            # Bootstrap value
            last_obs_n = norm.normalize(obs)
            last_val = model.critic(last_obs_n).squeeze(-1)

            # GAE
            adv = torch.zeros_like(buf_rew)
            last_gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                non_term = 1.0 - buf_done[t]
                next_v = last_val if t == T - 1 else buf_val[t + 1]
                delta = buf_rew[t] + args.gamma * next_v * non_term - buf_val[t]
                last_gae = delta + args.gamma * args.lam * non_term * last_gae
                adv[t] = last_gae
            ret = adv + buf_val

        total_steps += batch_size

        # 2. PPO update
        f_obs = buf_obs.reshape(batch_size, env.obs_dim)
        f_act = buf_act.reshape(batch_size, env.act_dim)
        f_logp = buf_logp.reshape(batch_size)
        f_adv = adv.reshape(batch_size)
        f_ret = ret.reshape(batch_size)
        f_val = buf_val.reshape(batch_size)
        f_mean = buf_mean.reshape(batch_size, env.act_dim)
        f_std = buf_std.reshape(batch_size, env.act_dim)

        kl_sum = 0.0
        ploss_sum = 0.0
        vloss_sum = 0.0
        ent_sum = 0.0
        n_mb = 0

        for _ in range(args.epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                idx = perm[start:start + mb_size]
                new_logp, ent, new_val, new_mean, new_std = model.evaluate(f_obs[idx], f_act[idx])

                # Analytic Gaussian KL for adaptive LR
                with torch.no_grad():
                    old_m, old_s = f_mean[idx], f_std[idx]
                    kl = (torch.log(new_std / old_s) + (old_s ** 2 + (old_m - new_mean) ** 2) / (2.0 * new_std ** 2) - 0.5).sum(dim=-1).mean()
                    kl_val = float(kl.item())
                    if kl_val > args.desired_kl * 2.0:
                        lr = max(1e-5, lr / 1.5)
                    elif kl_val < args.desired_kl * 0.5 and kl_val > 0.0:
                        lr = min(3e-3, lr * 1.5)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                mb_adv = f_adv[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                ratio = torch.exp(new_logp - f_logp[idx])
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = f_val[idx] + torch.clamp(new_val - f_val[idx], -0.2, 0.2)
                v_loss1 = (new_val - f_ret[idx]) ** 2
                v_loss2 = (v_clipped - f_ret[idx]) ** 2
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                entropy_loss = ent.mean()
                loss = policy_loss + 1.0 * value_loss - args.entropy * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                kl_sum += kl_val
                ploss_sum += float(policy_loss.item())
                vloss_sum += float(value_loss.item())
                ent_sum += float(entropy_loss.item())
                n_mb += 1

        if it == 1 or it % 10 == 0 or it == n_iters:
            elapsed = time.time() - t0
            fps = total_steps / max(elapsed, 1e-3)
            mean_rew = float(buf_rew.mean().item())
            succ_per_ep = iter_succ / max(iter_done, 1)
            succ_rate = iter_succ / max(iter_succ + iter_drop, 1)
            std_mean = float(torch.exp(model.log_std).mean().item())
            err_deg = iter_err_deg / T
            curr_deg = env.twist_curr_rad * 180.0 / math.pi
            # Automatic Twist angle curriculum: smooth ladder 22 -> 32 -> 45 -> 60 -> 75 -> 90 deg!
            if args.task == "twist" and env.twist_curr_rad < math.pi * 0.5 - 1e-3:
                if succ_rate >= 0.72 and succ_per_ep >= 0.70 and iter_succ >= 80:
                    stages = [22.0, 32.0, 45.0, 60.0, 75.0, 90.0]
                    nxt = next((s for s in stages if s > curr_deg + 1.0), 90.0)
                    env.twist_curr_rad = math.radians(nxt)
                    best_succ_rate = -1.0  # reset best tracker for the harder curriculum stage!
                    print(f"  >>> CURRICULUM ADVANCED: twist angle {curr_deg:.0f} deg -> {nxt:.0f} deg <<<", flush=True)
                    curr_deg = nxt

            rec = {
                "iter": it,
                "steps": total_steps,
                "t": round(elapsed, 1),
                "fps": int(fps),
                "rew": round(mean_rew, 3),
                "err_deg": round(err_deg, 1),
                "curr_deg": round(curr_deg, 1),
                "succ": iter_succ,
                "drop": iter_drop,
                "succ_rate": round(succ_rate, 3),
                "succ_per_ep": round(succ_per_ep, 2),
                "kl": round(kl_sum / n_mb, 4),
                "lr": round(lr, 6),
                "std": round(std_mean, 3),
            }
            history.append(rec)
            print(
                f"[{elapsed:6.1f}s] it={it:4d} steps={total_steps:10,d} fps={fps:7,.0f} "
                f"rew={mean_rew:+6.3f} err={err_deg:4.1f}d(c={curr_deg:2.0f}) succ={iter_succ:4d} drop={iter_drop:4d} "
                f"s_rate={succ_rate:.3f} s/ep={succ_per_ep:4.2f} kl={kl_sum/n_mb:.4f} lr={lr:.1e} std={std_mean:.3f}",
                flush=True,
            )
            with open(os.path.join(args.out, "log.json"), "w") as f:
                json.dump(history, f, indent=2)

            # Only save best.pt at full 90 deg for twist (or always for reorient)
            at_full = (args.task == "reorient") or (env.twist_curr_rad >= math.pi * 0.5 - 1e-3)
            if at_full and succ_rate > best_succ_rate and iter_succ >= 50:
                best_succ_rate = succ_rate
                best_path = os.path.join(args.out, "best.pt")
                save_ckpt(best_path, model, norm, env.obs_dim, env.act_dim, total_steps, rec)
                export_checkpoint(best_path, os.path.join(args.out, "exported_best"))

        if it % args.save_every == 0 or it == n_iters:
            latest_path = os.path.join(args.out, "latest.pt")
            save_ckpt(latest_path, model, norm, env.obs_dim, env.act_dim, total_steps, history[-1] if history else {})
            export_checkpoint(latest_path, os.path.join(args.out, "exported_latest"))

    # Collect handover states from best checkpoint
    best_path = os.path.join(args.out, "best.pt")
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.actor.load_state_dict(ck["actor_state_dict"])
        norm.load_np(ck["obs_mean"], ck["obs_std"])
    if args.collect_handovers > 0:
        collect_handovers(env, model, norm, n_target=args.collect_handovers,
                          out_path=os.path.join(args.out, "handovers.npz"))
    print(f"DONE {args.task} in {time.time() - t0:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
