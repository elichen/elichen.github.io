"""Online Alternating Closed-Loop PPO Co-Training v4 ("Iron-Grip Sim-to-WASM").

Warm-starts directly from exported policy.{json,bin} (reorient4 + twist6), runs safety-first
domain-randomized 25-move closed-loop co-training on MuJoCo-Warp GPU, anchors gently to the
reference policy to preserve 100% standalone skill, and validates LIVE every 25 iters on
native parallel C MuJoCo (exact WASM physics parity!) so best_pair is strictly WASM-verified!
"""
import argparse
import collections
import copy
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closedloop_env import ClosedLoopRubikEnv
from cpu_eval import eval_cpu_mujoco
from rubik_warp_env import OBS_REORIENT_DIM, OBS_TWIST_DIM
from train_ppo import ActorCritic, RunningNorm, save_ckpt
from export import export_checkpoint


def load_exported_into_model(export_dir: str, model: ActorCritic, norm: RunningNorm, device: torch.device, init_log_std: float = -1.9):
    """Load browser-exported policy.json + policy.bin directly into ActorCritic + RunningNorm."""
    with open(os.path.join(export_dir, "policy.json"), "r") as f:
        meta = json.load(f)
    weights = np.fromfile(os.path.join(export_dir, "policy.bin"), dtype="<f4")

    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    norm.load_np(mean, std)
    norm.count = 1e8

    sd = model.actor.state_dict()
    w_keys = [k for k in sd.keys() if k.endswith(".weight")]
    offset = 0
    for wk, layer in zip(w_keys, meta["layers"]):
        bk = wk[:-7] + ".bias"
        in_d, out_d = layer["in"], layer["out"]
        w_sz = out_d * in_d
        w = torch.from_numpy(weights[offset:offset + w_sz].reshape(out_d, in_d).copy()).to(device)
        offset += w_sz
        b = torch.from_numpy(weights[offset:offset + out_d].copy()).to(device)
        offset += out_d
        sd[wk] = w
        sd[bk] = b
    assert offset == len(weights), f"Weight size mismatch: consumed {offset} vs {len(weights)}"
    model.actor.load_state_dict(sd)
    model.log_std.data.fill_(init_log_std)
    print(f"Loaded warm-start actor from {export_dir} ({meta['layers']}, log_std={init_log_std})", flush=True)


def export_pair(out_dir: str, tag: str, re_model, re_norm, tw_model, tw_norm, step: int, stats: dict):
    pair_dir = os.path.join(out_dir, tag)
    os.makedirs(pair_dir, exist_ok=True)
    re_pt = os.path.join(pair_dir, "reorient.pt")
    tw_pt = os.path.join(pair_dir, "twist.pt")
    save_ckpt(re_pt, re_model, re_norm, OBS_REORIENT_DIM, 20, step, stats)
    save_ckpt(tw_pt, tw_model, tw_norm, OBS_TWIST_DIM, 20, step, stats)
    export_checkpoint(re_pt, os.path.join(pair_dir, "reorient"))
    export_checkpoint(tw_pt, os.path.join(pair_dir, "twist"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="/content/rubik/robot/shadow_rubik.xml")
    p.add_argument("--reorient-init", required=True)
    p.add_argument("--twist-init", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=1280)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--iters-twist", type=int, default=125)
    p.add_argument("--iters-reorient", type=int, default=150)
    p.add_argument("--critic-warmup", type=int, default=25)
    p.add_argument("--target-moves", type=int, default=25)
    p.add_argument("--hard-neg-prob", type=float, default=0.30)
    p.add_argument("--lr", type=float, default=3.0e-5)
    p.add_argument("--critic-lr", type=float, default=1.0e-3)
    p.add_argument("--desired-kl", type=float, default=0.004)
    p.add_argument("--entropy", type=float, default=0.0001)
    p.add_argument("--anchor-coef", type=float, default=0.015)
    p.add_argument("--blame", type=float, default=-100.0)
    p.add_argument("--cpu-eval-every", type=int, default=25)
    p.add_argument("--cpu-eval-episodes", type=int, default=48)
    p.add_argument("--reorient-first", action="store_true")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--hidden", default="512,256,128")
    p.add_argument("--seed", type=int, default=24)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    hidden = tuple(int(x) for x in args.hidden.split(","))

    env = ClosedLoopRubikEnv(
        xml_path=args.xml,
        num_envs=args.num_envs,
        eval_envs=args.eval_envs,
        target_moves=args.target_moves,
        hard_neg_prob=args.hard_neg_prob,
        dr_enabled=True,
        seed=args.seed,
    )
    device = env.device
    N = env.num_envs
    E = env.eval_envs
    NT = N - E
    T = args.horizon

    re_norm = RunningNorm(OBS_REORIENT_DIM, device)
    tw_norm = RunningNorm(OBS_TWIST_DIM, device)
    re_model = ActorCritic(OBS_REORIENT_DIM, 20, actor_hidden=hidden, critic_hidden=hidden).to(device)
    tw_model = ActorCritic(OBS_TWIST_DIM, 20, actor_hidden=hidden, critic_hidden=hidden).to(device)

    load_exported_into_model(args.reorient_init, re_model, re_norm, device, init_log_std=-1.9)
    load_exported_into_model(args.twist_init, tw_model, tw_norm, device, init_log_std=-1.9)

    # Frozen reference actors for behavioral KL anchoring (prevents forgetting standalone skills!)
    re_ref_actor = copy.deepcopy(re_model.actor).eval()
    tw_ref_actor = copy.deepcopy(tw_model.actor).eval()
    for prm in list(re_ref_actor.parameters()) + list(tw_ref_actor.parameters()):
        prm.requires_grad_(False)

    re_opt = torch.optim.Adam([
        {"params": list(re_model.actor.parameters()) + [re_model.log_std], "lr": args.lr},
        {"params": list(re_model.critic.parameters()), "lr": args.critic_lr},
    ], eps=1e-5)
    tw_opt = torch.optim.Adam([
        {"params": list(tw_model.actor.parameters()) + [tw_model.log_std], "lr": args.lr},
        {"params": list(tw_model.critic.parameters()), "lr": args.critic_lr},
    ], eps=1e-5)

    # Measure exact native C-MuJoCo baseline BEFORE any updates!
    t_b = time.time()
    base_cpu = eval_cpu_mujoco(args.xml, re_model, re_norm, tw_model, tw_norm, n_episodes=args.cpu_eval_episodes, target_moves=args.target_moves)
    best_cpu_solve = base_cpu["solve_pct"]
    print(
        f"=== BASELINE NATIVE C-MUJOCO ({time.time()-t_b:.1f}s): CPU_SOLVE25={base_cpu['solve_pct']:5.1f}% "
        f"({base_cpu['solved']}/{base_cpu['n']}, drops={base_cpu['dropped']}[r={base_cpu['re_drops']},t={base_cpu['tw_drops']}], "
        f"bails={base_cpu['bails']}, avg_mv={base_cpu['avg_moves']}, sim={base_cpu['avg_sim_s']}s) ===",
        flush=True,
    )

    max_obs_dim = max(OBS_REORIENT_DIM, OBS_TWIST_DIM)
    buf_obs = torch.zeros((T, NT, max_obs_dim), device=device)
    buf_act = torch.zeros((T, NT, 20), device=device)
    buf_logp = torch.zeros((T, NT), device=device)
    buf_rew = torch.zeros((T, NT), device=device)
    buf_done = torch.zeros((T, NT), device=device, dtype=torch.bool)
    buf_drop = torch.zeros((T, NT), device=device, dtype=torch.bool)
    buf_active = torch.zeros((T, NT), device=device, dtype=torch.bool)
    buf_val = torch.zeros((T, NT), device=device)
    buf_mean = torch.zeros((T, NT, 20), device=device)
    buf_std = torch.zeros((T, NT, 20), device=device)

    obs_re, obs_tw = env.reset_all()
    t0 = time.time()
    total_steps = 0
    global_it = 0
    history = []

    schedule = []
    for r in range(args.rounds):
        it_tw = args.iters_twist if r == 0 else max(80, int(args.iters_twist * 0.8))
        it_re = args.iters_reorient if r == 0 else max(100, int(args.iters_reorient * 0.8))
        if args.reorient_first:
            schedule.append((r + 1, "reorient", 0, it_re))
            schedule.append((r + 1, "twist", 1, it_tw))
        else:
            schedule.append((r + 1, "twist", 1, it_tw))
            schedule.append((r + 1, "reorient", 0, it_re))

    eval_hist_solved = collections.deque(maxlen=250)
    eval_hist_moves = collections.deque(maxlen=250)

    print(
        f"Starting Iron-Grip Sim-to-WASM Co-Training (cl4): N={N} (eval={E}, train={NT}), T={T}, "
        f"target_moves={args.target_moves}, hn_prob={args.hard_neg_prob}, anchor={args.anchor_coef}, blame={args.blame}",
        flush=True,
    )

    for round_idx, task_name, active_phase, n_iters in schedule:
        active_model = re_model if active_phase == 0 else tw_model
        active_ref = re_ref_actor if active_phase == 0 else tw_ref_actor
        active_opt = re_opt if active_phase == 0 else tw_opt
        obs_dim = OBS_REORIENT_DIM if active_phase == 0 else OBS_TWIST_DIM
        lr = args.lr
        active_opt.param_groups[0]["lr"] = lr
        active_opt.param_groups[1]["lr"] = args.critic_lr

        print(f"\n=== ROUND {round_idx} | ACTIVE={task_name.upper()} ({n_iters} iters, warmup={args.critic_warmup}) ===", flush=True)

        for block_it in range(1, n_iters + 1):
            global_it += 1
            warmup = (round_idx == 1 and block_it <= args.critic_warmup)

            iter_re_succ = iter_tw_succ = iter_tw_bail = iter_drop = iter_re_drop = iter_tw_drop = 0

            with torch.no_grad():
                for t in range(T):
                    obs_re_n = re_norm.normalize(obs_re)
                    obs_tw_n = tw_norm.normalize(obs_tw)

                    # Mild observation noise on training slice only (kills fragile open-loop memorization)
                    obs_re_tr = obs_re_n[E:] + torch.randn_like(obs_re_n[E:]) * 0.004
                    obs_tw_tr = obs_tw_n[E:] + torch.randn_like(obs_tw_n[E:]) * 0.004

                    re_ev, _, _, _, _ = re_model.act(obs_re_n[:E], deterministic=True)
                    tw_ev, _, _, _, _ = tw_model.act(obs_tw_n[:E], deterministic=True)

                    if active_phase == 0:
                        act_tr, logp_tr, val_tr, mean_tr, std_tr = re_model.act(obs_re_tr, deterministic=False)
                        tw_tr, _, _, _, _ = tw_model.act(obs_tw_tr, deterministic=True)
                        obs_tr = obs_re_tr
                        re_all = torch.cat([re_ev, act_tr], dim=0)
                        tw_all = torch.cat([tw_ev, tw_tr], dim=0)
                    else:
                        re_tr, _, _, _, _ = re_model.act(obs_re_tr, deterministic=True)
                        act_tr, logp_tr, val_tr, mean_tr, std_tr = tw_model.act(obs_tw_tr, deterministic=False)
                        obs_tr = obs_tw_tr
                        re_all = torch.cat([re_ev, re_tr], dim=0)
                        tw_all = torch.cat([tw_ev, act_tr], dim=0)

                    is_tw_now = (env.phase == 1).unsqueeze(-1)
                    action = torch.where(is_tw_now, tw_all, re_all)
                    next_obs_re, next_obs_tw, rew, done, info = env.step(action)

                    is_act_tr = (info["phase_before"][E:] == active_phase)
                    buf_obs[t, :, :obs_dim] = obs_tr
                    buf_act[t] = act_tr
                    buf_logp[t] = logp_tr
                    buf_rew[t] = rew[E:]
                    buf_done[t] = done[E:]
                    buf_drop[t] = info["dropped"][E:]
                    buf_active[t] = is_act_tr
                    buf_val[t] = val_tr
                    buf_mean[t] = mean_tr
                    buf_std[t] = std_tr

                    obs_re, obs_tw = next_obs_re, next_obs_tw

                    iter_re_succ += info["n_re_succ"]
                    iter_tw_succ += info["n_tw_succ"]
                    iter_tw_bail += info["n_tw_bail"]
                    iter_drop += info["n_drop"]
                    iter_re_drop += info["n_re_drop"]
                    iter_tw_drop += info["n_tw_drop"]

                    if info["eval_done"] > 0:
                        for _ in range(info["eval_solved"]):
                            eval_hist_solved.append(1)
                        for _ in range(info["eval_dropped"] + info["eval_timeout"]):
                            eval_hist_solved.append(0)
                        avg_m = info["eval_moves_sum"] / info["eval_done"]
                        for _ in range(info["eval_done"]):
                            eval_hist_moves.append(avg_m)

                last_obs_act = re_norm.normalize(obs_re[E:]) if active_phase == 0 else tw_norm.normalize(obs_tw[E:])
                next_v = active_model.critic(last_obs_act).squeeze(-1)

                adv = torch.zeros((T, NT), device=device)
                ret = torch.zeros((T, NT), device=device)
                last_gae = torch.zeros(NT, device=device)
                pending_blame = torch.zeros(NT, device=device)
                blame_done = torch.zeros(NT, device=device, dtype=torch.bool)

                for t in reversed(range(T)):
                    is_act = buf_active[t]
                    d = buf_done[t]
                    dr = buf_drop[t]

                    p_drop = (~is_act) & dr
                    p_end = (~is_act) & d & (~dr)
                    pending_blame = torch.where(p_drop, torch.full_like(pending_blame, args.blame), pending_blame)
                    blame_done = torch.where(p_drop | p_end, torch.ones_like(blame_done), blame_done)
                    last_gae = torch.where(p_drop | p_end, torch.zeros_like(last_gae), last_gae)
                    next_v = torch.where(p_drop | p_end, torch.zeros_like(next_v), next_v)

                    r_eff = buf_rew[t] + pending_blame
                    d_eff = d | blame_done
                    nt = 1.0 - d_eff.float()
                    delta = r_eff + args.gamma * next_v * nt - buf_val[t]
                    new_gae = delta + args.gamma * args.lam * nt * last_gae

                    adv[t] = torch.where(is_act, new_gae, adv[t])
                    ret[t] = torch.where(is_act, new_gae + buf_val[t], ret[t])

                    next_v = torch.where(is_act, buf_val[t], next_v)
                    last_gae = torch.where(is_act, new_gae, last_gae)
                    pending_blame = torch.where(is_act, torch.zeros_like(pending_blame), pending_blame)
                    blame_done = torch.where(is_act, torch.zeros_like(blame_done), blame_done)

            total_steps += N * T

            mask_flat = buf_active.reshape(-1)
            n_act = int(mask_flat.sum().item())
            if n_act < 512:
                continue

            f_obs = buf_obs[:, :, :obs_dim].reshape(-1, obs_dim)[mask_flat]
            f_act = buf_act.reshape(-1, 20)[mask_flat]
            f_logp = buf_logp.reshape(-1)[mask_flat]
            f_adv = adv.reshape(-1)[mask_flat]
            f_ret = ret.reshape(-1)[mask_flat]
            f_mean = buf_mean.reshape(-1, 20)[mask_flat]
            f_std = buf_std.reshape(-1, 20)[mask_flat]

            mb_size = max(256, n_act // args.minibatches)
            kl_sum = vloss_sum = anch_sum = 0.0
            n_mb = 0

            for _ in range(args.epochs):
                perm = torch.randperm(n_act, device=device)
                for start in range(0, n_act - mb_size + 1, mb_size):
                    idx = perm[start:start + mb_size]
                    new_logp, ent, new_val, new_mean, new_std = active_model.evaluate(f_obs[idx], f_act[idx])
                    value_loss = nn.functional.smooth_l1_loss(new_val, f_ret[idx], beta=15.0)

                    if warmup:
                        loss = value_loss
                        kl_val = anch_val = 0.0
                    else:
                        with torch.no_grad():
                            old_m, old_s = f_mean[idx], f_std[idx]
                            kl = (torch.log(new_std / old_s) + (old_s ** 2 + (old_m - new_mean) ** 2) / (2.0 * new_std ** 2) - 0.5).sum(dim=-1).mean()
                            kl_val = float(kl.item())
                            if kl_val > args.desired_kl * 2.0:
                                lr = max(5e-6, lr / 1.5)
                            elif kl_val < args.desired_kl * 0.5 and kl_val > 0.0:
                                lr = min(1.0e-4, lr * 1.25)
                            active_opt.param_groups[0]["lr"] = lr
                            ref_mean = torch.tanh(active_ref(f_obs[idx]))

                        mb_adv = (f_adv[idx] - f_adv[idx].mean()) / (f_adv[idx].std() + 1e-8)
                        ratio = torch.exp(new_logp - f_logp[idx])
                        policy_loss = -torch.min(ratio * mb_adv, torch.clamp(ratio, 0.85, 1.15) * mb_adv).mean()
                        anchor_loss = ((new_mean - ref_mean) ** 2).sum(dim=-1).mean()
                        anch_val = float(anchor_loss.item())
                        loss = policy_loss + 0.5 * value_loss + args.anchor_coef * anchor_loss - args.entropy * ent.mean()

                    active_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(active_model.parameters(), 1.0)
                    active_opt.step()
                    with torch.no_grad():
                        active_model.log_std.clamp_(-2.4, -1.4)

                    kl_sum += kl_val
                    vloss_sum += float(value_loss.item())
                    anch_sum += anch_val
                    n_mb += 1

            if block_it == 1 or block_it % 10 == 0 or block_it == n_iters:
                elapsed = time.time() - t0
                fps = total_steps / max(elapsed, 1e-3)
                n_eval = len(eval_hist_solved)
                gpu_solve = 100.0 * sum(eval_hist_solved) / max(n_eval, 1)
                avg_moves = sum(eval_hist_moves) / max(len(eval_hist_moves), 1)
                drop_per_move = 100.0 * iter_drop / max(iter_tw_succ + iter_drop, 1)
                std_m = float(torch.exp(active_model.log_std).mean().item())
                rec = {
                    "it": global_it, "round": round_idx, "active": task_name, "warmup": warmup,
                    "steps": total_steps, "t": round(elapsed, 1), "fps": int(fps),
                    "gpu_solve25": round(gpu_solve, 1), "avg_moves": round(avg_moves, 1),
                    "tw_succ": iter_tw_succ, "re_succ": iter_re_succ, "bail": iter_tw_bail,
                    "drop": iter_drop, "re_drop": iter_re_drop, "tw_drop": iter_tw_drop,
                    "d_mv": round(drop_per_move, 2), "hn": env.hn_count,
                    "kl": round(kl_sum / max(n_mb, 1), 4), "vl": round(vloss_sum / max(n_mb, 1), 1),
                    "anch": round(anch_sum / max(n_mb, 1), 3), "std": round(std_m, 3),
                }
                history.append(rec)
                print(
                    f"[{elapsed:6.1f}s] R{round_idx} {task_name[:2].upper()}{'*' if warmup else ' '} it={global_it:4d} "
                    f"fps={fps:6,.0f} | GPU25={gpu_solve:5.1f}% (mv={avg_moves:4.1f}) | "
                    f"tw={iter_tw_succ:4d} re={iter_re_succ:4d} bail={iter_tw_bail:3d} "
                    f"drop={iter_drop:3d}(r={iter_re_drop:2d},t={iter_tw_drop:2d}) d/mv={drop_per_move:4.2f}% "
                    f"hn={env.hn_count:4d} vl={vloss_sum/max(n_mb,1):5.1f} anch={anch_sum/max(n_mb,1):.3f}",
                    flush=True,
                )

            # Native C-MuJoCo evaluation (exact WASM physics parity!) every cpu_eval_every iters
            if (not warmup) and (block_it % args.cpu_eval_every == 0 or block_it == n_iters):
                tc = time.time()
                cpu_res = eval_cpu_mujoco(args.xml, re_model, re_norm, tw_model, tw_norm, n_episodes=args.cpu_eval_episodes, target_moves=args.target_moves)
                dt_c = time.time() - tc
                if history:
                    history[-1]["cpu_eval"] = cpu_res
                print(
                    f"  [CPU-MUJOCO {dt_c:4.1f}s] it={global_it:4d}: CPU_SOLVE25={cpu_res['solve_pct']:5.1f}% "
                    f"({cpu_res['solved']}/{cpu_res['n']}, drops={cpu_res['dropped']}[r={cpu_res['re_drops']},t={cpu_res['tw_drops']}], "
                    f"bails={cpu_res['bails']}, mv={cpu_res['avg_moves']}, sim={cpu_res['avg_sim_s']}s)",
                    flush=True,
                )
                with open(os.path.join(args.out, "log.json"), "w") as f:
                    json.dump(history, f, indent=2)
                if cpu_res["solve_pct"] > best_cpu_solve:
                    best_cpu_solve = cpu_res["solve_pct"]
                    export_pair(args.out, "best_pair", re_model, re_norm, tw_model, tw_norm, total_steps, cpu_res)
                    print(f"  >>> NEW BEST CPU-VERIFIED PAIR: CPU_SOLVE25={best_cpu_solve:.1f}% (drops={cpu_res['dropped']}, bails={cpu_res['bails']}) -> best_pair <<<", flush=True)

            if block_it % 50 == 0 or block_it == n_iters:
                export_pair(args.out, f"r{round_idx}_{task_name}_latest", re_model, re_norm, tw_model, tw_norm, total_steps, history[-1] if history else {})

    export_pair(args.out, "final_pair", re_model, re_norm, tw_model, tw_norm, total_steps, history[-1] if history else {})
    print(f"DONE cl4 IRON-GRIP CO-TRAINING in {time.time() - t0:.1f}s (best CPU_SOLVE25={best_cpu_solve:.1f}%) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
