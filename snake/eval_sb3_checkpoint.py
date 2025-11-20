#!/usr/bin/env python3
"""
Evaluate a saved SB3 checkpoint on the full-board Snake environment.

Example:
    python eval_sb3_checkpoint.py \
        --model sb3_fullmap_model/checkpoints/final_fullboard.zip \
        --grid-size 20 \
        --episodes 10
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import List, Tuple

import numpy as np
from stable_baselines3 import PPO

from sb3_fullboard_env import FullBoardSnakeEnv, RewardConfig


def run_episode(
    model,
    env: FullBoardSnakeEnv,
    device: str,
    n_stack: int,
    channels: int,
) -> Tuple[int, int, float]:
    """Play a single deterministic episode and return (foods, length, grid_fill)."""
    obs, _ = env.reset()
    stacked_obs = np.zeros((n_stack * channels, *obs["board"].shape[1:]), dtype=np.float32)
    stacked_stats = np.zeros((n_stack * obs["stats"].shape[0],), dtype=np.float32)
    for i in range(n_stack):
        stacked_obs[i * channels : (i + 1) * channels] = obs["board"]
        stacked_stats[i * obs["stats"].shape[0] : (i + 1) * obs["stats"].shape[0]] = obs["stats"]
    state = None
    episode_start = np.array([True], dtype=bool)

    total_reward = 0.0
    done = False
    truncated = False

    while not (done or truncated):
        action, state = model.predict(
            {"board": stacked_obs, "stats": stacked_stats},
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += float(reward)
        episode_start[:] = done or truncated

        stacked_obs = np.roll(stacked_obs, shift=-channels, axis=0)
        stacked_obs[-channels:] = obs["board"]
        stacked_stats = np.roll(stacked_stats, shift=-obs["stats"].shape[0])
        stacked_stats[-obs["stats"].shape[0]:] = obs["stats"]

    foods = int(info.get("score", 0))
    length = int(info.get("length", 0))
    grid_fill = float(info.get("grid_fill", 0.0))
    return foods, length, grid_fill


def evaluate(
    model_path: Path,
    grid_size: int,
    episodes: int,
    device: str,
    seed: int,
    observation_grid_size: int,
    n_stack: int = 4,
) -> None:
    model = PPO.load(model_path, device=device)
    env = FullBoardSnakeEnv(
        grid_size=grid_size,
        seed=seed,
        reward_config=RewardConfig(),
        observation_grid_size=observation_grid_size,
    )
    channels = env.board_shape[0]

    food_counts: List[int] = []
    lengths: List[int] = []
    fills: List[float] = []

    for ep in range(episodes):
        foods, length, grid_fill = run_episode(model, env, device, n_stack, channels)
        food_counts.append(foods)
        lengths.append(length)
        fills.append(grid_fill)
        print(
            f"Episode {ep+1:02d}: foods={foods:3d}, steps={length:4d}, "
            f"grid_fill={grid_fill*100:5.2f}%"
        )

    mean_food = statistics.mean(food_counts)
    median_food = statistics.median(food_counts)
    best = max(food_counts)

    print("\nSummary")
    print("-------")
    print(f"Episodes      : {episodes}")
    print(f"Grid size     : {grid_size}x{grid_size}")
    print(f"Avg foods     : {mean_food:.2f}")
    print(f"Median foods  : {median_food:.2f}")
    print(f"Best foods    : {best}")
    print(f"Avg steps     : {statistics.mean(lengths):.1f}")
    print(f"Avg grid fill : {statistics.mean(fills)*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate an SB3 Snake checkpoint.")
    parser.add_argument("--model", type=Path, required=True, help="Path to .zip checkpoint")
    parser.add_argument("--grid-size", type=int, default=20, help="Grid size to evaluate on")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch device (auto/mps/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=2025, help="Environment seed")
    parser.add_argument(
        "--obs-grid-size",
        type=int,
        default=None,
        help="Observation grid size (defaults to grid size). Match training max grid when using curriculum.",
    )
    args = parser.parse_args()

    device = args.device
    if args.device == "auto":
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    obs_grid_size = args.obs_grid_size or args.grid_size
    evaluate(args.model, args.grid_size, args.episodes, device, args.seed, obs_grid_size)


if __name__ == "__main__":
    main()
