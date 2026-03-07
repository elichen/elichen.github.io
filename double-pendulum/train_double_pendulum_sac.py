#!/usr/bin/env python3

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class EnvConfig:
    gravity: float = 9.8
    cart_mass: float = 1.0
    rod_mass_1: float = 0.5
    rod_mass_2: float = 0.5
    rod_length_1: float = 1.0
    rod_length_2: float = 1.0
    force_mag: float = 100.0
    dt: float = 0.02
    joint_damping: float = 1.0
    x_limit: float = 5.0
    max_steps: int = 500


class DoublePendulumSwingupEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        high = np.full(8, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state = np.zeros(6, dtype=np.float64)
        self.steps = 0
        self.episode_return = 0.0
        self.episode_abs_cart = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = np.array(
            [
                self.np_random.uniform(-0.05, 0.05),
                0.0,
                math.pi + self.np_random.uniform(-0.1, 0.1),
                0.0,
                math.pi + self.np_random.uniform(-0.1, 0.1),
                0.0,
            ],
            dtype=np.float64,
        )
        self.steps = 0
        self.episode_return = 0.0
        self.episode_abs_cart = 0.0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        x, xd, t1, t1d, t2, t2d = self.state
        return np.array(
            [x, xd, math.sin(t1), math.cos(t1), t1d, math.sin(t2), math.cos(t2), t2d],
            dtype=np.float32,
        )

    def uprightness(self) -> float:
        return 0.5 * (math.cos(self.state[2]) + math.cos(self.state[4]))

    def both_upright(self) -> bool:
        return math.cos(self.state[2]) > 0.9 and math.cos(self.state[4]) > 0.9

    def derivatives(self, state: np.ndarray, force: float) -> np.ndarray:
        x, xd, t1, t1d, t2, t2d = state
        cfg = self.config

        c1, s1 = math.cos(t1), math.sin(t1)
        c2, s2 = math.cos(t2), math.sin(t2)
        c12, s12 = math.cos(t1 - t2), math.sin(t1 - t2)

        d1 = cfg.cart_mass + cfg.rod_mass_1 + cfg.rod_mass_2
        d2 = (cfg.rod_mass_1 / 2.0 + cfg.rod_mass_2) * cfg.rod_length_1
        d3 = cfg.rod_mass_2 * cfg.rod_length_2 / 2.0
        d4 = (cfg.rod_mass_1 / 3.0 + cfg.rod_mass_2) * cfg.rod_length_1 * cfg.rod_length_1
        d5 = cfg.rod_mass_2 * cfg.rod_length_1 * cfg.rod_length_2 / 2.0
        d6 = cfg.rod_mass_2 * cfg.rod_length_2 * cfg.rod_length_2 / 3.0

        m11, m12, m13 = d1, d2 * c1, d3 * c2
        m21, m22, m23 = d2 * c1, d4, d5 * c12
        m31, m32, m33 = d3 * c2, d5 * c12, d6

        f1 = force + d2 * t1d * t1d * s1 + d3 * t2d * t2d * s2
        f2 = d5 * t2d * t2d * s12 + (cfg.rod_mass_1 / 2.0 + cfg.rod_mass_2) * cfg.gravity * cfg.rod_length_1 * s1 - cfg.joint_damping * t1d
        f3 = -d5 * t1d * t1d * s12 + cfg.rod_mass_2 * cfg.gravity * cfg.rod_length_2 * s2 / 2.0 - cfg.joint_damping * t2d

        det = (
            m11 * (m22 * m33 - m23 * m32)
            - m12 * (m21 * m33 - m23 * m31)
            + m13 * (m21 * m32 - m22 * m31)
        )

        if abs(det) < 1e-10:
            return np.array([xd, force / d1, t1d, 0.0, t2d, 0.0], dtype=np.float64)

        xdd = (
            f1 * (m22 * m33 - m23 * m32)
            - m12 * (f2 * m33 - m23 * f3)
            + m13 * (f2 * m32 - m22 * f3)
        ) / det
        t1dd = (
            m11 * (f2 * m33 - m23 * f3)
            - f1 * (m21 * m33 - m23 * m31)
            + m13 * (m21 * f3 - f2 * m31)
        ) / det
        t2dd = (
            m11 * (m22 * f3 - f2 * m32)
            - m12 * (m21 * f3 - f2 * m31)
            + f1 * (m21 * m32 - m22 * m31)
        ) / det
        return np.array([xd, xdd, t1d, t1dd, t2d, t2dd], dtype=np.float64)

    def compute_reward(self, action: float) -> float:
        x, xd, t1, t1d, t2, t2d = self.state
        upright_reward = 1.0 + 0.5 * math.cos(t1) + 0.5 * math.cos(t2)
        balance_blend = smoothstep(0.55, 0.92, self.uprightness())
        center_penalty = (0.06 + 0.42 * balance_blend) * (x / self.config.x_limit) ** 2
        cart_velocity_penalty = (0.004 + 0.018 * balance_blend) * min(xd * xd, 9.0)
        angular_velocity_penalty = 0.0025 * min(t1d * t1d + t2d * t2d, 49.0)
        control_penalty = (0.004 + 0.010 * balance_blend) * action * action
        rail_penalty = (0.04 + 0.14 * balance_blend) * smoothstep(0.72, 1.0, abs(x) / self.config.x_limit)
        recenter_bonus = 0.15 * balance_blend * math.exp(-0.5 * (x / 1.25) ** 2)
        return upright_reward - center_penalty - cart_velocity_penalty - angular_velocity_penalty - control_penalty - rail_penalty + recenter_bonus

    def step(self, action: np.ndarray):
        action_value = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        action_value = clamp(action_value, -1.0, 1.0)
        force = action_value * self.config.force_mag
        dt = self.config.dt
        state = self.state.copy()

        k1 = self.derivatives(state, force)
        k2 = self.derivatives(state + 0.5 * dt * k1, force)
        k3 = self.derivatives(state + 0.5 * dt * k2, force)
        k4 = self.derivatives(state + dt * k3, force)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        next_state[2] = ((next_state[2] + math.pi) % (2.0 * math.pi)) - math.pi
        next_state[4] = ((next_state[4] + math.pi) % (2.0 * math.pi)) - math.pi
        if next_state[2] < -math.pi:
            next_state[2] += 2.0 * math.pi
        if next_state[4] < -math.pi:
            next_state[4] += 2.0 * math.pi

        if abs(next_state[0]) > self.config.x_limit:
            next_state[0] = self.config.x_limit * math.copysign(1.0, next_state[0])
            next_state[1] = 0.0

        self.state = next_state
        self.steps += 1

        reward = self.compute_reward(action_value)
        self.episode_return += reward
        self.episode_abs_cart += abs(self.state[0])
        truncated = self.steps >= self.config.max_steps
        info: Dict[str, float] = {}
        if truncated:
            info["episode"] = {
                "r": self.episode_return,
                "l": self.steps,
                "mean_abs_x": self.episode_abs_cart / self.steps,
            }
        return self._obs(), reward, False, truncated, info


def load_actor_from_json(model: SAC, json_path: Path) -> None:
    data = json.loads(json_path.read_text())
    linear_layers = [module for module in model.actor.latent_pi if isinstance(module, nn.Linear)]
    if len(linear_layers) != 2:
        raise RuntimeError("Expected two linear layers in actor.latent_pi")

    source_layers = [
        ("layer_0", linear_layers[0]),
        ("layer_1", linear_layers[1]),
    ]
    for key, layer in source_layers:
        kernel = np.asarray(data[key]["kernel"], dtype=np.float32)
        bias = np.asarray(data[key]["bias"], dtype=np.float32)
        weight = kernel.reshape(layer.in_features, layer.out_features).T
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(weight))
            layer.bias.copy_(torch.from_numpy(bias))

    mu_kernel = np.asarray(data["mu"]["kernel"], dtype=np.float32).reshape(model.actor.mu.in_features, model.actor.mu.out_features).T
    mu_bias = np.asarray(data["mu"]["bias"], dtype=np.float32)
    with torch.no_grad():
        model.actor.mu.weight.copy_(torch.from_numpy(mu_kernel))
        model.actor.mu.bias.copy_(torch.from_numpy(mu_bias))
        nn.init.zeros_(model.actor.log_std.weight)
        model.actor.log_std.bias.fill_(-2.0)


def export_actor_to_json(model: SAC, json_path: Path) -> None:
    linear_layers = [module for module in model.actor.latent_pi if isinstance(module, nn.Linear)]
    if len(linear_layers) != 2:
        raise RuntimeError("Expected two linear layers in actor.latent_pi")

    payload = {}
    for index, layer in enumerate(linear_layers):
        weight = layer.weight.detach().cpu().numpy().T.reshape(-1).tolist()
        bias = layer.bias.detach().cpu().numpy().reshape(-1).tolist()
        payload[f"layer_{index}"] = {"kernel": weight, "bias": bias}

    payload["mu"] = {
        "kernel": model.actor.mu.weight.detach().cpu().numpy().T.reshape(-1).tolist(),
        "bias": model.actor.mu.bias.detach().cpu().numpy().reshape(-1).tolist(),
    }
    json_path.write_text(json.dumps(payload))


def evaluate_policy_metrics(model: SAC, env_config: EnvConfig, episodes: int, seed: int) -> Dict[str, float]:
    returns = []
    mean_abs_xs = []
    final_abs_xs = []
    upright_steps = []
    first_upright_steps = []

    for episode in range(episodes):
        env = DoublePendulumSwingupEnv(env_config)
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        ep_upright = 0
        first_upright = env_config.max_steps

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, done, _ = env.step(action)
            total_reward += reward
            if env.both_upright():
                ep_upright += 1
                if first_upright == env_config.max_steps:
                    first_upright = env.steps

        returns.append(total_reward)
        mean_abs_xs.append(env.episode_abs_cart / env.steps)
        final_abs_xs.append(abs(env.state[0]))
        upright_steps.append(ep_upright)
        first_upright_steps.append(first_upright)

    return {
        "avg_return": float(np.mean(returns)),
        "avg_mean_abs_x": float(np.mean(mean_abs_xs)),
        "avg_final_abs_x": float(np.mean(final_abs_xs)),
        "avg_upright_steps": float(np.mean(upright_steps)),
        "avg_first_upright_step": float(np.mean(first_upright_steps)),
        "min_upright_steps": float(np.min(upright_steps)),
    }


class MasteryCallback(BaseCallback):
    def __init__(
        self,
        out_dir: Path,
        env_config: EnvConfig,
        eval_episodes: int,
        eval_freq: int,
        seed: int,
        required_streak: int = 2,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.env_config = env_config
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.seed = seed
        self.required_streak = required_streak
        self.streak = 0
        self.best_metrics: Optional[Dict[str, float]] = None

    def _on_step(self) -> bool:
        if self.num_timesteps == 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        metrics = evaluate_policy_metrics(
            self.model,
            env_config=self.env_config,
            episodes=self.eval_episodes,
            seed=self.seed + self.num_timesteps,
        )
        summary = {
            "timesteps": self.num_timesteps,
            **metrics,
        }
        print(json.dumps(summary, sort_keys=True))

        current_key = (
            metrics["avg_mean_abs_x"],
            metrics["avg_final_abs_x"],
            -metrics["avg_upright_steps"],
            -metrics["avg_return"],
        )
        best_key = None
        if self.best_metrics is not None:
            best_key = (
                self.best_metrics["avg_mean_abs_x"],
                self.best_metrics["avg_final_abs_x"],
                -self.best_metrics["avg_upright_steps"],
                -self.best_metrics["avg_return"],
            )

        if best_key is None or current_key < best_key:
            self.best_metrics = summary
            self.model.save(self.out_dir / "best_model")
            export_actor_to_json(self.model, self.out_dir / "best_weights.json")
            (self.out_dir / "best_metrics.json").write_text(json.dumps(summary, indent=2))

        mastered = (
            metrics["avg_upright_steps"] >= 425.0
            and metrics["min_upright_steps"] >= 420.0
            and metrics["avg_mean_abs_x"] <= 1.8
            and metrics["avg_final_abs_x"] <= 1.8
            and metrics["avg_first_upright_step"] <= 85.0
        )
        self.streak = self.streak + 1 if mastered else 0
        return self.streak < self.required_streak


def build_model(args: argparse.Namespace, env, warm_start: Optional[Path]) -> SAC:
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [256, 256]},
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.1",
        verbose=1,
        seed=args.seed,
        device="cpu",
    )
    if warm_start is not None:
        load_actor_from_json(model, warm_start)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export a centered SAC policy for the double pendulum swing-up demo.")
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--rod-length-1", type=float, default=1.0)
    parser.add_argument("--rod-length-2", type=float, default=1.0)
    parser.add_argument("--x-limit", type=float, default=5.0)
    parser.add_argument("--warm-start-json", type=Path, default=Path("trained-weights-double-sac.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("training-artifacts/centered-sac"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)

    env_config = EnvConfig(
        rod_length_1=args.rod_length_1,
        rod_length_2=args.rod_length_2,
        x_limit=args.x_limit,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(asdict(env_config), indent=2))

    vec_env_cls = DummyVecEnv if args.vec_env == "dummy" or args.num_envs == 1 else SubprocVecEnv
    env = make_vec_env(
        lambda: DoublePendulumSwingupEnv(env_config),
        n_envs=args.num_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls,
    )

    warm_start = args.warm_start_json if args.warm_start_json.exists() else None
    model = build_model(args, env, warm_start)

    baseline_metrics = evaluate_policy_metrics(model, env_config, episodes=args.eval_episodes, seed=args.seed)
    print(json.dumps({"phase": "baseline", **baseline_metrics}, sort_keys=True))

    callback = MasteryCallback(
        out_dir=args.out_dir,
        env_config=env_config,
        eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)

    model.save(args.out_dir / "final_model")
    export_actor_to_json(model, args.out_dir / "final_weights.json")
    final_metrics = evaluate_policy_metrics(model, env_config, episodes=args.eval_episodes, seed=args.seed + 50_000)
    (args.out_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2))
    print(json.dumps({"phase": "final", **final_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
