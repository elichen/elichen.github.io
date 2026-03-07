#!/usr/bin/env python3

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from train_double_pendulum_sac import DoublePendulumSwingupEnv, EnvConfig, clamp, smoothstep


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


def load_json_actor_weights(model: Actor, json_path: Path) -> None:
    data = json.loads(json_path.read_text())
    with torch.no_grad():
        layer0 = model.net[0]
        layer1 = model.net[2]
        layer2 = model.net[4]
        layer0.weight.copy_(torch.tensor(np.asarray(data["layer_0"]["kernel"], dtype=np.float32).reshape(8, 256).T))
        layer0.bias.copy_(torch.tensor(data["layer_0"]["bias"], dtype=torch.float32))
        layer1.weight.copy_(torch.tensor(np.asarray(data["layer_1"]["kernel"], dtype=np.float32).reshape(256, 256).T))
        layer1.bias.copy_(torch.tensor(data["layer_1"]["bias"], dtype=torch.float32))
        layer2.weight.copy_(torch.tensor(np.asarray(data["mu"]["kernel"], dtype=np.float32).reshape(256, 1).T))
        layer2.bias.copy_(torch.tensor(data["mu"]["bias"], dtype=torch.float32))


def export_json_actor_weights(model: Actor, json_path: Path) -> None:
    layer0 = model.net[0]
    layer1 = model.net[2]
    layer2 = model.net[4]
    payload = {
        "layer_0": {
            "kernel": layer0.weight.detach().cpu().numpy().T.reshape(-1).tolist(),
            "bias": layer0.bias.detach().cpu().numpy().tolist(),
        },
        "layer_1": {
            "kernel": layer1.weight.detach().cpu().numpy().T.reshape(-1).tolist(),
            "bias": layer1.bias.detach().cpu().numpy().tolist(),
        },
        "mu": {
            "kernel": layer2.weight.detach().cpu().numpy().T.reshape(-1).tolist(),
            "bias": layer2.bias.detach().cpu().numpy().tolist(),
        },
    }
    json_path.write_text(json.dumps(payload))


def actor_action(model: Actor, obs: np.ndarray) -> float:
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return float(model(obs_tensor)[0, 0])


def teacher_action(model: Actor, env: DoublePendulumSwingupEnv, obs: np.ndarray, args: argparse.Namespace) -> float:
    base_action = actor_action(model, obs)
    x, xd, t1, t1d, t2, t2d = env.state
    uprightness = 0.5 * (math.cos(t1) + math.cos(t2))
    stillness = math.exp(-args.stillness_falloff * (t1d * t1d + t2d * t2d))
    blend = smoothstep(args.upright_min, args.upright_max, uprightness) * stillness
    assist = clamp(-args.kx * x - args.kv * xd, -args.assist_limit, args.assist_limit)
    return clamp(base_action + assist * blend, -1.0, 1.0)


def evaluate_actor(model: Actor, env_config: EnvConfig, episodes: int, seed: int) -> Dict[str, float]:
    returns = []
    mean_abs_xs = []
    final_abs_xs = []
    upright_steps = []
    first_upright_steps = []

    for episode in range(episodes):
        env = DoublePendulumSwingupEnv(env_config)
        obs, _ = env.reset(seed=seed + episode)
        done = False
        ep_upright = 0
        first_upright = env_config.max_steps

        while not done:
            action = actor_action(model, obs)
            obs, _, _, done, _ = env.step(np.array([action], dtype=np.float32))
            if env.both_upright():
                ep_upright += 1
                if first_upright == env_config.max_steps:
                    first_upright = env.steps

        returns.append(env.episode_return)
        mean_abs_xs.append(env.episode_abs_cart / env.steps)
        final_abs_xs.append(abs(env.state[0]))
        upright_steps.append(ep_upright)
        first_upright_steps.append(first_upright)

    return {
        "avg_return": float(np.mean(returns)),
        "avg_mean_abs_x": float(np.mean(mean_abs_xs)),
        "avg_final_abs_x": float(np.mean(final_abs_xs)),
        "avg_upright_steps": float(np.mean(upright_steps)),
        "min_upright_steps": float(np.min(upright_steps)),
        "avg_first_upright_step": float(np.mean(first_upright_steps)),
    }


def collect_dataset(student: Actor, teacher: Actor, env_config: EnvConfig, args: argparse.Namespace, round_index: int) -> Tuple[np.ndarray, np.ndarray]:
    observations = []
    actions = []

    for episode in range(args.collect_episodes):
        env = DoublePendulumSwingupEnv(env_config)
        obs, _ = env.reset(seed=args.seed + round_index * 1000 + episode)
        done = False
        while not done:
            observations.append(obs.copy())
            actions.append(teacher_action(teacher, env, obs, args))
            student_act = actor_action(student, obs)
            obs, _, _, done, _ = env.step(np.array([student_act], dtype=np.float32))

    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def fit_student(student: Actor, xs: np.ndarray, ys: np.ndarray, args: argparse.Namespace) -> None:
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    x_tensor = torch.tensor(xs, dtype=torch.float32)
    y_tensor = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

    for _ in range(args.epochs):
        permutation = torch.randperm(x_tensor.shape[0])
        for start in range(0, x_tensor.shape[0], args.batch_size):
            batch = permutation[start : start + args.batch_size]
            loss = loss_fn(student(x_tensor[batch]), y_tensor[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop policy distillation for the centered double pendulum demo.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--collect-episodes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--warm-start-json", type=Path, default=Path("trained-weights-double-sac.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("training-artifacts/closed-loop-distill"))
    parser.add_argument("--rod-length-1", type=float, default=1.0)
    parser.add_argument("--rod-length-2", type=float, default=1.0)
    parser.add_argument("--x-limit", type=float, default=5.0)
    parser.add_argument("--kx", type=float, default=0.10)
    parser.add_argument("--kv", type=float, default=0.18)
    parser.add_argument("--assist-limit", type=float, default=0.45)
    parser.add_argument("--upright-min", type=float, default=0.80)
    parser.add_argument("--upright-max", type=float, default=0.97)
    parser.add_argument("--stillness-falloff", type=float, default=0.04)
    parser.add_argument("--target-mean-abs-x", type=float, default=1.60)
    parser.add_argument("--target-final-abs-x", type=float, default=1.50)
    parser.add_argument("--target-upright-steps", type=float, default=425.0)
    parser.add_argument("--target-first-upright-step", type=float, default=85.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_config = EnvConfig(
        rod_length_1=args.rod_length_1,
        rod_length_2=args.rod_length_2,
        x_limit=args.x_limit,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(asdict(env_config), indent=2))

    teacher = Actor()
    student = Actor()
    load_json_actor_weights(teacher, args.warm_start_json)
    student.load_state_dict(teacher.state_dict())
    teacher.eval()

    baseline = evaluate_actor(student, env_config, episodes=args.eval_episodes, seed=args.seed)
    print(json.dumps({"phase": "baseline", **baseline}, sort_keys=True))

    best_metrics = baseline
    export_json_actor_weights(student, args.out_dir / "best_weights.json")
    (args.out_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2))

    for round_index in range(args.rounds):
        xs, ys = collect_dataset(student, teacher, env_config, args, round_index)
        fit_student(student, xs, ys, args)
        metrics = evaluate_actor(student, env_config, episodes=args.eval_episodes, seed=args.seed + 20_000 + round_index * 100)
        summary = {"round": round_index, **metrics}
        print(json.dumps(summary, sort_keys=True))

        current_key = (
            metrics["avg_mean_abs_x"],
            metrics["avg_final_abs_x"],
            -metrics["avg_upright_steps"],
            -metrics["avg_return"],
        )
        best_key = (
            best_metrics["avg_mean_abs_x"],
            best_metrics["avg_final_abs_x"],
            -best_metrics["avg_upright_steps"],
            -best_metrics["avg_return"],
        )
        if current_key < best_key:
            best_metrics = metrics
            export_json_actor_weights(student, args.out_dir / "best_weights.json")
            (args.out_dir / "best_metrics.json").write_text(json.dumps({"round": round_index, **metrics}, indent=2))

        mastered = (
            metrics["avg_mean_abs_x"] <= args.target_mean_abs_x
            and metrics["avg_final_abs_x"] <= args.target_final_abs_x
            and metrics["avg_upright_steps"] >= args.target_upright_steps
            and metrics["min_upright_steps"] >= args.target_upright_steps
            and metrics["avg_first_upright_step"] <= args.target_first_upright_step
        )
        if mastered:
            break

    final_metrics = evaluate_actor(student, env_config, episodes=max(args.eval_episodes * 2, 24), seed=args.seed + 50_000)
    export_json_actor_weights(student, args.out_dir / "final_weights.json")
    (args.out_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2))
    print(json.dumps({"phase": "final", **final_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
