# -*- coding: utf-8 -*-
"""
Simplified but Optimized SAC Training for Stick Swingup - Final Attempt

Goal: Train a successful swingup policy in 1M steps using SAC with best practices.
"""

import os
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stick_env import StickBalancingEnv


class SimpleProgressCallback(BaseCallback):
    """Minimal progress tracking."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        self.balancing_steps = 0
        self.total_steps = 0
        self.best_reward = -np.inf

    def _on_training_start(self):
        self.start_time = time.time()
        print("\n🚀 SAC Training Started (1M Steps)")
        print("=" * 60)

    def _on_step(self):
        # Track balancing performance
        if hasattr(self.training_env, 'get_attr'):
            angles = self.training_env.get_attr('angle')
            if len(angles) > 0:
                angle = angles[0]
                self.total_steps += 1
                if abs(angle) < np.pi / 12:  # Within 15 degrees
                    self.balancing_steps += 1

        # Print progress every 50k steps
        if self.n_calls % 50000 == 0:
            elapsed = time.time() - self.start_time
            balance_pct = 100.0 * self.balancing_steps / max(self.total_steps, 1)

            # Get episode info
            ep_info = self.locals.get("infos", [{}])[0].get("episode", {})
            if ep_info:
                ep_reward = ep_info.get("r", 0)
                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward
                    print(f"🎯 New best reward: {self.best_reward:.2f}")
            else:
                ep_reward = 0

            print(f"Step {self.n_calls:,}: Reward={ep_reward:.1f} | Balance={balance_pct:.1f}% | Time={elapsed/60:.1f}min")

        return True

    def _on_training_end(self):
        print("=" * 60)
        print(f"✅ Training Complete! Best reward: {self.best_reward:.2f}")


def make_env():
    """Create environment with shaped rewards."""
    def _init():
        env = StickBalancingEnv(use_shaped_reward=True)
        env = Monitor(env)
        return env
    return _init


def train_sac_final():
    """
    Final optimized SAC training.
    Key optimizations based on research:
    - SAC for better sample efficiency
    - Shaped rewards for faster convergence
    - Observation normalization for stability
    - Tuned hyperparameters for 1M step convergence
    """

    # Create environment
    print("📊 Setting up environment...")
    env = DummyVecEnv([make_env()])

    # Normalize observations for stability
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    # Policy network architecture
    policy_kwargs = dict(
        net_arch=[256, 256],  # Deeper network for complex control
    )

    print("🤖 Creating SAC model...")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        # Key hyperparameters for fast convergence
        learning_rate=3e-4,  # Optimal for pendulum tasks
        buffer_size=300000,  # Large enough for good sample efficiency
        learning_starts=1000,  # Start learning early
        batch_size=256,  # Good batch size for stability
        tau=0.005,  # Soft update coefficient
        gamma=0.99,  # Discount factor
        train_freq=1,  # Train after every step
        gradient_steps=1,  # Gradient steps per update
        ent_coef="auto",  # Automatic entropy tuning
        target_update_interval=1,
        target_entropy="auto",
        policy_kwargs=policy_kwargs,
        verbose=0,  # Reduce verbosity for speed
        seed=42  # Reproducibility
    )

    # Progress callback
    progress_callback = SimpleProgressCallback()

    # Train the model
    print("🏋️ Training for 1,000,000 steps...")
    print("Monitor the progress below:\n")

    model.learn(
        total_timesteps=1_000_000,
        callback=progress_callback,
        progress_bar=False
    )

    # Save model and stats
    os.makedirs("models_sac_final", exist_ok=True)
    model.save("models_sac_final/sac_stick_final")
    env.save("models_sac_final/vec_normalize.pkl")

    print("\n💾 Model saved to models_sac_final/")

    return model, env


def test_model():
    """Test the trained model."""
    print("\n🎮 Testing trained model...")

    # Load environment and model
    env = DummyVecEnv([make_env()])
    env = VecNormalize.load("models_sac_final/vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False

    model = SAC.load("models_sac_final/sac_stick_final")

    # Test for 5 episodes
    for episode in range(5):
        obs = env.reset()
        total_reward = 0
        balancing_steps = 0
        total_steps = 0

        for _ in range(1000):  # Max 1000 steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            total_steps += 1

            # Check if balancing
            angles = env.get_attr('angle')
            if len(angles) > 0 and abs(angles[0]) < np.pi / 12:
                balancing_steps += 1

            if done:
                break

        balance_pct = 100.0 * balancing_steps / total_steps
        print(f"Episode {episode+1}: Reward={total_reward:.1f} | Balance={balance_pct:.1f}% ({balancing_steps}/{total_steps} steps)")

    env.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_model()
    else:
        model, env = train_sac_final()
        print("\n🎯 To test the model, run: python train_sac_final.py --test")