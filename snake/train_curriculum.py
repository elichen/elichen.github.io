"""
Curriculum Training Script for Snake RL Agent.
Trains PPO agent through progressively larger board sizes.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure

from snake_env import SnakeEnv
from snake_features import SnakeFeatureExtractor, SnakeFeatureExtractorLarge
from make_env import make_vec_env, make_eval_env


# Curriculum configuration
BOARD_SIZES = [6, 8, 10, 14, 20]

SCORE_THRESHOLDS = {
    6: 4,
    8: 6,
    10: 8,
    14: 10,
    20: 12,
}

TIMESTEPS_PER_PHASE = {
    6: 500_000,
    8: 1_000_000,
    10: 1_500_000,
    14: 2_500_000,
    20: 5_000_000,
}


class CurriculumCallback(BaseCallback):
    """
    Callback to log curriculum-specific metrics.
    """

    def __init__(self, board_size: int, verbose: int = 0):
        super().__init__(verbose)
        self.board_size = board_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []

    def _on_step(self) -> bool:
        # Check for episode completion in infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
            if "score" in info:
                self.episode_scores.append(info["score"])

        return True

    def _on_rollout_end(self) -> None:
        if self.episode_scores:
            avg_score = np.mean(self.episode_scores[-100:])
            max_score = max(self.episode_scores[-100:]) if self.episode_scores else 0
            self.logger.record("curriculum/board_size", self.board_size)
            self.logger.record("curriculum/avg_score_100ep", avg_score)
            self.logger.record("curriculum/max_score_100ep", max_score)
            self.logger.record("curriculum/total_episodes", len(self.episode_scores))


def get_device() -> str:
    """Select best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_model(
    env,
    device: str,
    features_dim: int = 256,
    use_large_network: bool = False,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    gamma: float = 0.995,
    ent_coef: float = 0.01,
    tensorboard_log: Optional[str] = None,
) -> PPO:
    """
    Create a new PPO model with custom feature extractor.

    Args:
        env: Vectorized environment
        device: Device to use (mps, cuda, cpu)
        features_dim: Feature dimension for the extractor
        use_large_network: Whether to use larger network architecture
        learning_rate: Learning rate
        n_steps: Number of steps per rollout
        batch_size: Minibatch size
        gamma: Discount factor
        ent_coef: Entropy coefficient
        tensorboard_log: Path for tensorboard logs

    Returns:
        PPO model
    """
    extractor_class = SnakeFeatureExtractorLarge if use_large_network else SnakeFeatureExtractor

    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    return model


def train_curriculum(
    output_dir: str = "models",
    log_dir: str = "logs",
    n_envs: int = 8,
    use_subproc: bool = True,
    use_large_network: bool = False,
    seed: int = 42,
    resume_from: Optional[str] = None,
    start_size: int = 6,
) -> None:
    """
    Run curriculum training through all board sizes.

    Args:
        output_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        n_envs: Number of parallel environments
        use_subproc: Whether to use subprocess vectorization
        use_large_network: Whether to use larger network
        seed: Random seed
        resume_from: Path to model to resume from
        start_size: Board size to start curriculum from
    """
    device = get_device()
    print(f"Using device: {device}")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"snake_curriculum_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tb_log_dir = Path(log_dir) / f"snake_curriculum_{timestamp}"
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    model = None
    total_timesteps = 0

    # Filter board sizes based on start_size
    board_sizes = [s for s in BOARD_SIZES if s >= start_size]

    for board_size in board_sizes:
        print(f"\n{'='*60}")
        print(f"CURRICULUM PHASE: {board_size}x{board_size} Grid")
        print(f"{'='*60}\n")

        # Create environment
        env = make_vec_env(
            n=board_size,
            n_envs=n_envs,
            gamma=0.995,
            seed=seed,
            use_subproc=use_subproc,
        )

        # Create evaluation environment
        eval_env = make_vec_env(
            n=board_size,
            n_envs=1,
            gamma=0.995,
            seed=seed + 1000,
            use_subproc=False,
        )

        if model is None:
            if resume_from:
                print(f"Resuming from: {resume_from}")
                model = PPO.load(resume_from, env=env, device=device)
            else:
                model = create_model(
                    env=env,
                    device=device,
                    use_large_network=use_large_network,
                    tensorboard_log=str(tb_log_dir),
                )
        else:
            # Create new model for new grid size and transfer weights
            # This works because we use AdaptiveAvgPool2d in the feature extractor
            print("Transferring weights to new model for larger grid...")
            old_state_dict = model.policy.state_dict()

            model = create_model(
                env=env,
                device=device,
                use_large_network=use_large_network,
                tensorboard_log=str(tb_log_dir),
            )

            # Load weights from previous model (architecture is identical due to adaptive pooling)
            model.policy.load_state_dict(old_state_dict)
            print("Weights transferred successfully!")

        # Create callbacks
        curriculum_callback = CurriculumCallback(board_size)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / f"best_{board_size}x{board_size}"),
            log_path=str(run_dir / f"eval_{board_size}x{board_size}"),
            eval_freq=max(10000 // n_envs, 1000),
            n_eval_episodes=10,
            deterministic=True,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(50000 // n_envs, 5000),
            save_path=str(run_dir / f"checkpoints_{board_size}x{board_size}"),
            name_prefix="snake",
        )

        callbacks = CallbackList([
            curriculum_callback,
            eval_callback,
            checkpoint_callback,
        ])

        # Train
        timesteps = TIMESTEPS_PER_PHASE[board_size]
        print(f"Training for {timesteps:,} timesteps...")

        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=f"ppo_{board_size}x{board_size}",
        )

        total_timesteps += timesteps

        # Save model for this phase
        model_path = run_dir / f"snake_ppo_{board_size}x{board_size}.zip"
        model.save(str(model_path))
        print(f"Saved model: {model_path}")

        # Cleanup
        env.close()
        eval_env.close()

    # Save final model
    final_path = run_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\n{'='*60}")
    print(f"Training complete! Total timesteps: {total_timesteps:,}")
    print(f"Final model saved: {final_path}")
    print(f"{'='*60}\n")


def train_single_size(
    board_size: int = 20,
    timesteps: int = 1_000_000,
    output_dir: str = "models",
    log_dir: str = "logs",
    n_envs: int = 8,
    use_subproc: bool = True,
    use_large_network: bool = False,
    seed: int = 42,
    resume_from: Optional[str] = None,
) -> None:
    """
    Train on a single board size without curriculum.

    Args:
        board_size: Grid size to train on
        timesteps: Total training timesteps
        output_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        n_envs: Number of parallel environments
        use_subproc: Whether to use subprocess vectorization
        use_large_network: Whether to use larger network
        seed: Random seed
        resume_from: Path to model to resume from
    """
    device = get_device()
    print(f"Using device: {device}")
    print(f"Training on {board_size}x{board_size} grid for {timesteps:,} timesteps")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"snake_{board_size}x{board_size}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tb_log_dir = Path(log_dir) / f"snake_{board_size}x{board_size}_{timestamp}"

    # Create environment
    env = make_vec_env(
        n=board_size,
        n_envs=n_envs,
        gamma=0.995,
        seed=seed,
        use_subproc=use_subproc,
    )

    eval_env = make_vec_env(
        n=board_size,
        n_envs=1,
        gamma=0.995,
        seed=seed + 1000,
        use_subproc=False,
    )

    if resume_from:
        print(f"Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env, device=device)
    else:
        model = create_model(
            env=env,
            device=device,
            use_large_network=use_large_network,
            tensorboard_log=str(tb_log_dir),
        )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(10000 // n_envs, 1000),
        n_eval_episodes=10,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // n_envs, 10000),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="snake",
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name="ppo",
    )

    # Save
    model.save(str(run_dir / "final_model.zip"))
    print(f"\nTraining complete! Model saved to: {run_dir / 'final_model.zip'}")

    env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train Snake RL Agent")

    parser.add_argument(
        "--mode",
        type=str,
        default="curriculum",
        choices=["curriculum", "single"],
        help="Training mode: curriculum or single board size",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=20,
        help="Board size for single mode training",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total timesteps for single mode training",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for models",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for tensorboard logs",
    )
    parser.add_argument(
        "--use-subproc",
        action="store_true",
        help="Use SubprocVecEnv for parallelism",
    )
    parser.add_argument(
        "--use-large-network",
        action="store_true",
        help="Use larger network architecture",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to model to resume training from",
    )
    parser.add_argument(
        "--start-size",
        type=int,
        default=6,
        help="Board size to start curriculum from",
    )

    args = parser.parse_args()

    if args.mode == "curriculum":
        train_curriculum(
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            n_envs=args.n_envs,
            use_subproc=args.use_subproc,
            use_large_network=args.use_large_network,
            seed=args.seed,
            resume_from=args.resume_from,
            start_size=args.start_size,
        )
    else:
        train_single_size(
            board_size=args.board_size,
            timesteps=args.timesteps,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            n_envs=args.n_envs,
            use_subproc=args.use_subproc,
            use_large_network=args.use_large_network,
            seed=args.seed,
            resume_from=args.resume_from,
        )


if __name__ == "__main__":
    main()
