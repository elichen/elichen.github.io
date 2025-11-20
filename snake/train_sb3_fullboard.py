#!/usr/bin/env python3
"""
Train a full-board Snake policy with Stable-Baselines3 (Recurrent PPO).

Usage:
    python train_sb3_fullboard.py --grid-sizes 7,10,14,20 --steps-per-stage 2000000
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from sb3_fullboard_env import FullBoardSnakeEnv, RewardConfig, make_env


class SnakeFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor that fuses CNN board embedding with scalar stats."""

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        board_shape = observation_space["board"].shape
        stats_dim = observation_space["stats"].shape[0]

        self.conv_net = nn.Sequential(
            nn.Conv2d(board_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_board = torch.as_tensor(observation_space["board"].sample()[None]).float()
            conv_dim = self.conv_net(dummy_board).shape[1]

        self.stats_net = nn.Sequential(
            nn.Linear(stats_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(conv_dim + 64, features_dim),
            nn.ReLU(),
        )
        self._features_dim = features_dim

    def forward(self, observations):
        board_tensor = observations["board"].float()
        stats_tensor = observations["stats"].float()
        board_feat = self.conv_net(board_tensor)
        stats_feat = self.stats_net(stats_tensor)
        concat = torch.cat([board_feat, stats_feat], dim=1)
        return self.fc(concat)


@dataclass
class StageConfig:
    """Definition of a curriculum stage."""

    grid_size: int
    timesteps: int


class StageTracker(BaseCallback):
    """Keeps track of curriculum transitions in TensorBoard/console."""

    def __init__(self, stage: StageConfig):
        super().__init__()
        self.stage = stage
        self.start_time = time.time()

    def _on_training_start(self) -> None:
        self.logger.record("stage/grid_size", self.stage.grid_size)
        self.logger.record("stage/timesteps", self.stage.timesteps)

    def _on_rollout_end(self) -> bool:
        elapsed = time.time() - self.start_time
        self.logger.record("stage/elapsed_seconds", elapsed)
        return True

    def _on_step(self) -> bool:
        # Nothing to do each step; callback required by BaseCallback.
        return True


def parse_grid_schedule(grid_argument: str, timesteps_per_stage: int) -> List[StageConfig]:
    sizes = [int(item) for item in grid_argument.split(",") if item.strip()]
    return [StageConfig(grid_size=size, timesteps=timesteps_per_stage) for size in sizes]


def build_vec_env(
    grid_size: int,
    n_envs: int,
    seed: int,
    log_dir: Path,
    reward_config: RewardConfig,
    observation_grid_size: int,
):
    env_fns = [
        make_env(
            grid_size=grid_size,
            seed=seed + i,
            reward_config=reward_config,
            observation_grid_size=observation_grid_size,
        )
        for i in range(n_envs)
    ]
    vec = DummyVecEnv(env_fns)
    monitor_file = log_dir / f"monitor_grid{grid_size}.csv"
    monitored = VecMonitor(vec, str(monitor_file))
    stacked = VecFrameStack(monitored, n_stack=4, channels_order="first")
    return stacked


def create_model(
    env,
    tensorboard_log: Path,
    device: str,
    feature_dim: int,
    lstm_hidden_size: int,
):
    policy_kwargs = dict(
        features_extractor_class=SnakeFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=feature_dim),
    )
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=2048,
        n_epochs=4,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(tensorboard_log),
        device=device,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    return model


def run_stage(
    model: PPO,
    stage: StageConfig,
    log_dir: Path,
    checkpoints_dir: Path,
    eval_dir: Path,
    n_envs: int,
    reward_config: RewardConfig,
    observation_grid_size: int,
) -> StageTracker:
    eval_env = build_vec_env(
        stage.grid_size,
        n_envs=1,
        seed=10_000,
        log_dir=eval_dir,
        reward_config=reward_config,
        observation_grid_size=observation_grid_size,
    )
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=50,
        eval_freq=max(5_000 // n_envs, 1),
        deterministic=True,
        render=False,
        best_model_save_path=str(eval_dir),
        log_path=str(eval_dir / "eval_logs"),
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=str(checkpoints_dir),
        name_prefix=f"grid{stage.grid_size}",
    )

    stage_tracker = StageTracker(stage)
    callback = CallbackList([checkpoint_callback, eval_callback, stage_tracker])
    model.learn(
        total_timesteps=stage.timesteps,
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=True,
    )
    eval_env.close()
    return stage_tracker


def export_sample_observation(grid_size: int, output_path: Path, observation_grid_size: int) -> None:
    """Save a reference observation for downstream export/debugging."""
    env = FullBoardSnakeEnv(grid_size=grid_size, observation_grid_size=observation_grid_size)
    obs, _ = env.reset(seed=42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(
            {"board": obs["board"].tolist(), "stats": obs["stats"].tolist()},
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="Train SB3 Recurrent PPO Snake agent.")
    parser.add_argument("--grid-sizes", type=str, default="7,10,14,20", help="Comma-delimited curriculum grid sizes.")
    parser.add_argument("--steps-per-stage", type=int, default=2_000_000, help="Timesteps per curriculum stage.")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--log-dir", type=Path, default=Path("sb3_fullmap_model"), help="Root logging directory.")
    parser.add_argument("--feature-dim", type=int, default=512, help="Feature extractor output dim.")
    parser.add_argument("--lstm-hidden", type=int, default=256, help="LSTM hidden state size.")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch device (auto, mps, cpu).")
    parser.add_argument("--export-obs", action="store_true", help="Dump a sample observation JSON for debugging.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Path to an SB3 checkpoint (.zip) to resume training from.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    set_random_seed(args.seed)
    log_root = args.log_dir
    checkpoints_dir = log_root / "checkpoints"
    eval_dir = log_root / "eval"
    tensorboard_dir = log_root / "tensorboard"
    for path in (checkpoints_dir, eval_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)

    reward_config = RewardConfig()
    stages = parse_grid_schedule(args.grid_sizes, args.steps_per_stage)
    max_grid_size = max(stage.grid_size for stage in stages)

    train_env = build_vec_env(
        grid_size=stages[0].grid_size,
        n_envs=args.n_envs,
        seed=args.seed,
        log_dir=log_root,
        reward_config=reward_config,
        observation_grid_size=max_grid_size,
    )
    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        model = PPO.load(args.resume_from, env=train_env, device=device)
    else:
        model = create_model(
            env=train_env,
            tensorboard_log=tensorboard_dir,
            device=device,
            feature_dim=args.feature_dim,
            lstm_hidden_size=args.lstm_hidden,
        )
    new_logger = configure(str(log_root / "sb3_logs"), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    for idx, stage in enumerate(stages):
        if idx > 0:
            train_env.close()
            train_env = build_vec_env(
                grid_size=stage.grid_size,
                n_envs=args.n_envs,
                seed=args.seed + idx * 100,
                log_dir=log_root,
                reward_config=reward_config,
                observation_grid_size=max_grid_size,
            )
            # Observation shape changes between stages, so we need to re-create the policy.
            model.set_env(train_env, force_reset=True)
        print(f"\n===== Stage {idx + 1}/{len(stages)} | Grid {stage.grid_size} | Timesteps {stage.timesteps:,} =====")
        run_stage(
            model=model,
            stage=stage,
            log_dir=log_root,
            checkpoints_dir=checkpoints_dir,
            eval_dir=eval_dir,
            n_envs=args.n_envs,
            reward_config=reward_config,
            observation_grid_size=max_grid_size,
        )

    final_path = checkpoints_dir / "final_fullboard"
    model.save(str(final_path))
    train_env.close()
    print(f"\nTraining complete! Final model stored at {final_path}.zip")

    if args.export_obs:
        export_sample_observation(
            stages[-1].grid_size,
            log_root / "sample_observation.json",
            observation_grid_size=max_grid_size,
        )
        print("Sample observation exported for conversion/debugging.")


if __name__ == "__main__":
    main()
