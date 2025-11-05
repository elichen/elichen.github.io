"""
Optimized PPO training with action masking
Based on best practices: 70% perfect opponent + action masking
"""

import numpy as np
from tictactoe_env import TicTacToeEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
import os


def mask_fn(env):
    """Function to get action mask from environment"""
    # Unwrap Monitor to get to actual env
    if hasattr(env, 'env'):
        return env.env.action_masks()
    return env.action_masks()


def train_with_masking():
    """Train PPO with action masking and optimal opponent mix"""

    print("=" * 60)
    print("MASKABLE PPO TRAINING (1M STEPS)")
    print("=" * 60)
    print()
    print("Optimizations:")
    print("  ✓ Action masking (no invalid moves)")
    print("  ✓ 70% perfect / 30% random opponent")
    print("  ✓ Clean reward: +1 win, 0 draw, -1 loss")
    print("  ✓ 1M timesteps for robust learning")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)

    # Create masked environments - ActionMasker must be outermost wrapper
    def make_env():
        # Use mixed opponent: 70% perfect, 30% random (as advertised!)
        base_env = TicTacToeEnv(opponent_type='mixed', perfect_ratio=0.7)
        masked_env = ActionMasker(base_env, mask_fn)
        return Monitor(masked_env)

    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    print(f"\nCreated {n_envs} parallel environments with action masking")

    # Create MaskablePPO model
    print("Creating MaskablePPO model...")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[128, 128],
            activation_fn=torch.nn.ReLU
        ),
        device='cpu'
    )

    # Train
    total_timesteps = 1_000_000
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("Progress shown every 50 iterations")
    print("-" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=50,
        progress_bar=False
    )

    print("-" * 60)
    print("\nTraining completed!")

    # Save model
    model.save("models/ppo_masked")
    print(f"Model saved to: models/ppo_masked.zip")

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Create non-masked env for evaluation
    eval_env_perfect = TicTacToeEnv(opponent_type='perfect')
    eval_env_random = TicTacToeEnv(opponent_type='random')

    print("\nVs Perfect Opponent (100 games)...")
    wins, draws, losses = 0, 0, 0

    for i in range(100):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/100")
        obs, _ = eval_env_perfect.reset()
        done = False

        while not done:
            # Get action mask and predict
            action_mask = eval_env_perfect.action_masks()
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_perfect.step(action)
            done = terminated or truncated

        if "winner" in info:
            if info["winner"] == "agent":
                wins += 1
            else:
                losses += 1
        elif "draw" in info:
            draws += 1

    print(f"\nResults vs Perfect:")
    print(f"  Wins:   {wins}")
    print(f"  Draws:  {draws}")
    print(f"  Losses: {losses}")
    print(f"  Never Lose: {(wins + draws)/100*100:.1f}%")

    # Vs Random
    print("\nVs Random Opponent (100 games)...")
    wins_r, draws_r = 0, 0

    for i in range(100):
        obs, _ = eval_env_random.reset()
        done = False

        while not done:
            action_mask = eval_env_random.action_masks()
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_random.step(action)
            done = terminated or truncated

        if "winner" in info and info["winner"] == "agent":
            wins_r += 1
        elif "draw" in info:
            draws_r += 1

    print(f"\nResults vs Random:")
    print(f"  Wins:   {wins_r}")
    print(f"  Draws:  {draws_r}")
    print(f"  Losses: {100 - wins_r - draws_r}")

    if (wins + draws) >= 95 and wins_r >= 80:
        print("\n✅ EXPERT LEVEL - Never loses AND exploits weak opponents!")
    elif (wins + draws) >= 95:
        print("\n🔶 Strong defensive play")
    else:
        print("\n⚠️ Needs more training")

    return model


if __name__ == "__main__":
    train_with_masking()
