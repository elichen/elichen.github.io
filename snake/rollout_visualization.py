"""
Rollout Visualization for Snake RL Agent.
Visualize trained agent gameplay with optional video recording.
"""

import argparse
import time
import os
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

from snake_env import SnakeEnv
from make_env import make_eval_env
from evaluation_utils import print_evaluation_summary, evaluate


def run_visual_episode(
    model: PPO,
    env: SnakeEnv,
    deterministic: bool = True,
    delay: float = 0.1,
    max_steps: int = 10000,
) -> dict:
    """
    Run a single episode with ASCII visualization.

    Args:
        model: Trained model
        env: Snake environment
        deterministic: Whether to use deterministic policy
        delay: Delay between frames (seconds)
        max_steps: Maximum steps per episode

    Returns:
        Final episode info
    """
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0

    # Clear screen
    print("\033[2J\033[H", end="")

    while not done and step < max_steps:
        # Clear and render
        print("\033[H", end="")  # Move cursor to top
        env.render()

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))

        total_reward += reward
        done = terminated or truncated
        step += 1

        time.sleep(delay)

    # Final render
    print("\033[H", end="")
    env.render()

    print(f"\n{'='*40}")
    print(f"Episode Complete!")
    print(f"Score: {info.get('score', 0)}")
    print(f"Length: {info.get('length', 0)}")
    print(f"Steps: {step}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Reason: {info.get('reason', 'N/A')}")
    print(f"{'='*40}\n")

    return info


def run_multiple_episodes(
    model: PPO,
    env: SnakeEnv,
    n_episodes: int = 5,
    deterministic: bool = True,
    delay: float = 0.1,
    pause_between: bool = True,
) -> None:
    """
    Run multiple episodes with visualization.

    Args:
        model: Trained model
        env: Snake environment
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic policy
        delay: Delay between frames
        pause_between: Whether to pause between episodes
    """
    scores = []

    for ep in range(n_episodes):
        print(f"\n{'='*40}")
        print(f"Episode {ep + 1}/{n_episodes}")
        print(f"{'='*40}")

        if pause_between and ep > 0:
            input("Press Enter to start next episode...")

        info = run_visual_episode(model, env, deterministic, delay)
        scores.append(info.get("score", 0))

    print(f"\n{'='*40}")
    print(f"Summary of {n_episodes} episodes:")
    print(f"Scores: {scores}")
    print(f"Mean: {np.mean(scores):.2f}")
    print(f"Max: {max(scores)}")
    print(f"Min: {min(scores)}")
    print(f"{'='*40}\n")


def save_video(
    model: PPO,
    env: SnakeEnv,
    output_path: str,
    n_episodes: int = 1,
    deterministic: bool = True,
    fps: int = 10,
    max_steps: int = 10000,
) -> None:
    """
    Save episodes as video files.

    Args:
        model: Trained model
        env: Environment with render_mode="rgb_array"
        output_path: Output video path
        n_episodes: Number of episodes to record
        deterministic: Whether to use deterministic policy
        fps: Frames per second
        max_steps: Maximum steps per episode
    """
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio[ffmpeg]")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    frames = []

    for ep in range(n_episodes):
        print(f"Recording episode {ep + 1}/{n_episodes}...")
        obs, info = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            step += 1

        # Add final frame
        frame = env.render()
        if frame is not None:
            # Hold final frame for a bit
            for _ in range(fps):
                frames.append(frame)

        print(f"Episode {ep + 1}: Score={info.get('score', 0)}, Steps={step}")

    # Save video
    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved! ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Visualize Snake RL Agent")

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=20,
        help="Board size for evaluation",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between frames (seconds)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of deterministic",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Don't pause between episodes",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (no visualization)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Save episodes to video file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for video output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = PPO.load(args.model_path)

    deterministic = not args.stochastic

    if args.eval_only:
        # Run evaluation
        env = make_eval_env(n=args.board_size, seed=args.seed)
        mean_score, std_score, stats = evaluate(
            model, env, n_episodes=args.eval_episodes,
            deterministic=deterministic, verbose=True
        )
        print_evaluation_summary(stats, name=args.model_path)
        env.close()

    elif args.save_video:
        # Record video
        env = make_eval_env(n=args.board_size, seed=args.seed, render_mode="rgb_array")
        save_video(
            model, env,
            output_path=args.save_video,
            n_episodes=args.n_episodes,
            deterministic=deterministic,
            fps=args.fps,
        )
        env.close()

    else:
        # Visual playback
        env = make_eval_env(n=args.board_size, seed=args.seed, render_mode="human")
        run_multiple_episodes(
            model, env,
            n_episodes=args.n_episodes,
            deterministic=deterministic,
            delay=args.delay,
            pause_between=not args.no_pause,
        )
        env.close()


if __name__ == "__main__":
    main()
