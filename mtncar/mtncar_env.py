"""
Custom Mountain Car Gym Environment
Matches the physics and reward structure from the web app implementation.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MountainCarEnv(gym.Env):
    """
    Custom Mountain Car environment matching the web app's physics.

    Observation Space: Box(2) - [position, velocity]
    Action Space: Discrete(3) - [left, none, right]

    Goal: Reach position >= 0.5
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # Physics parameters (matching mountainCar.js)
        self.min_position = -1.2
        self.max_position = 0.6
        self.goal_position = 0.5
        self.max_velocity = 0.07
        self.min_velocity = -0.07
        self.force = 0.0008
        self.gravity = 0.0025

        # Action space: 0=left, 1=none, 2=right
        self.action_space = spaces.Discrete(3)

        # Observation space: [position, velocity]
        self.observation_space = spaces.Box(
            low=np.array([self.min_position, self.min_velocity], dtype=np.float32),
            high=np.array([self.max_position, self.max_velocity], dtype=np.float32),
            dtype=np.float32
        )

        # Rendering
        self.render_mode = render_mode

        # State
        self.state = None
        self.last_position = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start at bottom of hill with zero velocity (matching web app)
        self.state = np.array([-0.5, 0.0], dtype=np.float32)
        self.last_position = self.state[0]
        self.steps = 0

        return self.state, {}

    def step(self, action):
        position, velocity = self.state

        # Map action to force
        if action == 0:  # left
            force = -self.force
        elif action == 2:  # right
            force = self.force
        else:  # none
            force = 0.0

        # Update velocity (matching mountainCar.js step function)
        velocity += force - self.gravity * np.cos(3 * position)
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity)

        # Update position
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        # Stop at left boundary
        if position <= self.min_position and velocity < 0:
            velocity = 0.0

        # Standard Mountain Car sparse reward (Gymnasium MountainCar-v0 compatible)
        done = False
        truncated = False

        # Check if goal reached
        if position >= self.goal_position:
            reward = 0.0  # Sparse reward: 0 for success
            done = True
        else:
            # Sparse reward: -1 for each step (encourages reaching goal quickly)
            reward = -1.0

        # Update state
        self.last_position = position
        self.state = np.array([position, velocity], dtype=np.float32)
        self.steps += 1

        # Don't terminate on left boundary - this was the key bug!
        # The car can bounce off the left wall, that's part of the strategy

        # Truncate after too many steps (prevent infinite episodes)
        if self.steps >= 1000:
            truncated = True

        info = {
            "steps": self.steps,
            "success": position >= self.goal_position
        }

        return self.state, reward, done, truncated, info

    def render(self):
        if self.render_mode == "human":
            # Simple text rendering
            position, velocity = self.state
            print(f"Step {self.steps}: pos={position:.3f}, vel={velocity:.4f}")


def make_env():
    """Factory function to create the environment."""
    return MountainCarEnv()


if __name__ == "__main__":
    # Test the environment
    env = MountainCarEnv(render_mode="human")

    print("Testing Mountain Car Environment")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run a random episode
    obs, _ = env.reset()
    print(f"\nInitial state: {obs}")

    total_reward = 0
    done = False
    truncated = False

    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if info["success"]:
            print(f"\nSuccess! Reached goal in {info['steps']} steps")
            print(f"Total reward: {total_reward:.2f}")
            break

    if truncated:
        print(f"\nTruncated after {info['steps']} steps")
        print(f"Total reward: {total_reward:.2f}")
