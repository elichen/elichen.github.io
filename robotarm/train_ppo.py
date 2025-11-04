#!/usr/bin/env python3
"""
Train a PPO agent for the robot arm task using Stable Baselines3
PPO is better suited for this task than DQN
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from typing import Dict, Tuple, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch.nn as nn
import os
from datetime import datetime


class RobotArmEnv(gym.Env):
    """Custom Environment for robot arm task"""

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        # Environment dimensions
        self.width = 600
        self.height = 400
        self.block_size = 30
        self.ground_y = self.height - 20
        self.max_height = self.ground_y - 150

        # Robot arm parameters
        self.arm_base_x = 300
        self.arm_base_y = 380
        self.arm_length1 = 100
        self.arm_length2 = 100
        self.angle_step = 0.1

        # Episode limits
        self.max_steps = 500
        self.current_step = 0

        # Action space: 0-3 for arm movements, 4 for claw toggle
        self.action_space = spaces.Discrete(5)

        # Observation space: 11 features
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(11,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Reset robot arm state to match JavaScript demo (arm points straight up)
        self.angle1 = math.pi / 2  # Straight up, matching robotArm.js
        self.angle2 = 0.0
        self.is_claw_closed = False

        # Reset block position (ensure it's reachable)
        while True:
            min_x = self.arm_base_x - 200
            max_x = self.arm_base_x + 200
            self.block_x = self.np_random.uniform(min_x, max_x)
            self.block_y = self.ground_y - self.block_size / 2

            if self._is_position_reachable(self.block_x, self.block_y):
                break

        self.is_block_held = False
        self.current_step = 0
        self.last_distance = None
        self.last_block_y = None

        return self._get_observation(), {}

    def _is_position_reachable(self, x: float, y: float) -> bool:
        """Check if position is reachable by the arm"""
        dx = x - self.arm_base_x
        dy = self.arm_base_y - y
        distance = math.sqrt(dx * dx + dy * dy)

        max_reach = self.arm_length1 + self.arm_length2
        min_reach = abs(self.arm_length1 - self.arm_length2)
        min_distance_from_base = 50

        return (distance <= max_reach and
                distance >= min_reach and
                distance >= min_distance_from_base and
                abs(x - self.arm_base_x) >= min_distance_from_base)

    def _get_claw_position(self) -> Tuple[float, float]:
        """Calculate current claw position from arm angles"""
        x1 = self.arm_base_x + self.arm_length1 * math.cos(self.angle1)
        y1 = self.arm_base_y - self.arm_length1 * math.sin(self.angle1)

        x2 = x1 + self.arm_length2 * math.cos(self.angle1 + self.angle2)
        y2 = y1 - self.arm_length2 * math.sin(self.angle1 + self.angle2)

        return x2, y2

    def _check_valid_angles(self, angle1: float, angle2: float) -> bool:
        """Check if angles are within valid range"""
        if angle1 < -math.pi or angle1 > math.pi:
            return False

        max_angle2 = 150 * math.pi / 180
        if angle2 < -max_angle2 or angle2 > max_angle2:
            return False

        x1 = self.arm_base_x + self.arm_length1 * math.cos(angle1)
        y1 = self.arm_base_y - self.arm_length1 * math.sin(angle1)

        x2 = x1 + self.arm_length2 * math.cos(angle1 + angle2)
        y2 = y1 - self.arm_length2 * math.sin(angle1 + angle2)

        if y1 > self.ground_y or y2 > self.ground_y:
            return False

        return True

    def _get_valid_actions(self) -> np.ndarray:
        """Get mask of currently valid actions"""
        valid = np.ones(5, dtype=np.float32)

        if not self._check_valid_angles(self.angle1 + self.angle_step, self.angle2):
            valid[0] = 0
        if not self._check_valid_angles(self.angle1 - self.angle_step, self.angle2):
            valid[1] = 0
        if not self._check_valid_angles(self.angle1, self.angle2 + self.angle_step):
            valid[2] = 0
        if not self._check_valid_angles(self.angle1, self.angle2 - self.angle_step):
            valid[3] = 0

        valid[4] = 1  # Claw toggle always valid

        return valid

    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        claw_x, claw_y = self._get_claw_position()
        distance_to_block = math.sqrt(
            (claw_x - self.block_x) ** 2 +
            (claw_y - self.block_y) ** 2
        )

        valid_actions = self._get_valid_actions()[:4]

        return np.array([
            self.angle1 / math.pi,
            self.angle2 / (150 * math.pi / 180),
            (self.block_x - self.arm_base_x) / 200,
            (self.block_y - self.max_height) / (self.ground_y - self.max_height),
            1.0 if self.is_claw_closed else 0.0,
            min(distance_to_block / 200, 1.0),
            1.0 if self.is_block_held else 0.0,
            valid_actions[0],
            valid_actions[1],
            valid_actions[2],
            valid_actions[3],
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1

        # Get valid actions mask
        valid_actions = self._get_valid_actions()

        # Hard masking - penalize invalid actions
        if valid_actions[action] == 0:
            obs = self._get_observation()
            return obs, -10.0, False, False, {"invalid_action": True}

        # Execute action
        if action == 0:
            self.angle1 += self.angle_step
        elif action == 1:
            self.angle1 -= self.angle_step
        elif action == 2:
            self.angle2 += self.angle_step
        elif action == 3:
            self.angle2 -= self.angle_step
        elif action == 4:
            self.is_claw_closed = not self.is_claw_closed

        # Update block physics
        claw_x, claw_y = self._get_claw_position()
        distance_to_block = math.sqrt(
            (claw_x - self.block_x) ** 2 +
            (claw_y - self.block_y) ** 2
        )

        # Check if block should be held
        if not self.is_block_held and self.is_claw_closed and distance_to_block < self.block_size:
            self.is_block_held = True
        elif self.is_block_held and not self.is_claw_closed:
            self.is_block_held = False
            self.block_y = self.ground_y - self.block_size / 2

        # Move block with claw if held
        if self.is_block_held:
            self.block_x = claw_x
            self.block_y = claw_y

        # Incremental progress rewards
        reward = 0.0
        terminated = False

        if not self.is_block_held:
            # Reward for getting closer to block
            if self.last_distance is not None:
                distance_improvement = self.last_distance - distance_to_block
                reward += distance_improvement * 10.0
            self.last_distance = distance_to_block
        else:
            # Reward for holding block (not dropping)
            reward += 1.0

            # Reward for lifting block higher
            if self.last_block_y is not None:
                height_improvement = self.last_block_y - self.block_y
                reward += height_improvement * 5.0
            self.last_block_y = self.block_y

        # Success: block lifted above target
        if self.is_block_held and self.block_y < self.max_height:
            reward += 100.0
            terminated = True

        # Episode timeout
        truncated = self.current_step >= self.max_steps

        # Reset tracking on episode end
        if terminated or truncated:
            self.last_distance = None
            self.last_block_y = None

        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}


class TrainingMonitorCallback(BaseCallback):
    """Callback for monitoring training progress"""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.last_log_step = 0

    def _on_step(self) -> bool:
        # Log every check_freq steps
        if self.n_calls - self.last_log_step >= self.check_freq:
            self.last_log_step = self.n_calls

            # Get recent episode info
            if hasattr(self.locals, 'infos'):
                for info in self.locals['infos']:
                    if 'episode' in info:
                        ep_reward = info['episode']['r']
                        ep_length = info['episode']['l']

                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        self.successes.append(1 if ep_reward > 50 else 0)

                        # Calculate success rate
                        if len(self.successes) >= 10:
                            recent_success_rate = np.mean(self.successes[-100:])
                            if self.verbose:
                                print(f"Step: {self.n_calls}, "
                                      f"Episode Reward: {ep_reward:.2f}, "
                                      f"Length: {ep_length}, "
                                      f"Success Rate (last 100): {recent_success_rate:.2%}")

                                # Check for signs of life
                                if recent_success_rate > 0 and len(self.successes) <= 110:
                                    print("✓ Signs of life detected! Agent is learning to succeed.")

        return True


def make_env(rank: int, seed: int = 0):
    """Utility function for multiprocessed env"""
    def _init():
        env = RobotArmEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_robot_arm_ppo(total_timesteps: int = 1000000, n_envs: int = 4):
    """Train PPO agent on robot arm task"""

    print("Setting up PPO training...")
    print(f"Using {n_envs} parallel environments")

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create vectorized environments for parallel training
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix="robot_arm_ppo"
    )

    monitor_callback = TrainingMonitorCallback(check_freq=2000)

    # Create PPO model with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=nn.ReLU
        ),
        verbose=1,
        device="auto",
        tensorboard_log=None
    )

    print(f"Starting PPO training for {total_timesteps} timesteps...")
    print("PPO advantages over DQN:")
    print("  - Better for discrete actions with continuous-like control")
    print("  - More sample efficient")
    print("  - More stable training")
    print("  - Better handling of action masking")

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, monitor_callback],
        progress_bar=False
    )

    # Save final model
    model.save("models/robot_arm_ppo_final")
    print(f"Training complete! Model saved to models/robot_arm_ppo_final")

    # Print final statistics
    if monitor_callback.successes:
        final_success_rate = np.mean(monitor_callback.successes[-100:])
        print(f"Final success rate (last 100 episodes): {final_success_rate:.2%}")

    return model, monitor_callback


def test_model(model_path: str = "models/robot_arm_ppo_final", n_episodes: int = 10):
    """Test a trained PPO model"""
    env = RobotArmEnv()
    model = PPO.load(model_path)

    successes = []
    rewards = []
    lengths = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        successes.append(1 if episode_reward > 50 else 0)
        rewards.append(episode_reward)
        lengths.append(episode_length)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}, Success = {successes[-1]}")

    print(f"\nTest Results:")
    print(f"  Success rate: {np.mean(successes):.2%}")
    print(f"  Average reward: {np.mean(rewards):.2f}")
    print(f"  Average length: {np.mean(lengths):.1f}")

    return successes, rewards, lengths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO for robot arm task")
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="Total training timesteps (default: 1M)")
    parser.add_argument("--envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--test", action="store_true",
                        help="Test the trained model")
    parser.add_argument("--model", type=str, default="models/robot_arm_ppo_final",
                        help="Model path for testing")

    args = parser.parse_args()

    if args.test:
        test_model(args.model)
    else:
        print(f"Training PPO for {args.timesteps} timesteps with {args.envs} parallel environments...")
        model, callback = train_robot_arm_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.envs
        )

        # Run a quick test
        print("\nRunning quick test with trained model...")
        test_model("models/robot_arm_ppo_final", n_episodes=5)