# -*- coding: utf-8 -*-
"""
Stick Balancing Environment - Python Port
Exact reimplementation of the JavaScript stick balancing environment for training with stable-baselines3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple


class StickBalancingEnv(gym.Env):
    """
    Stick Balancing Environment - a pendulum on a cart that needs to be swung up and balanced.

    State Space (4D):
        - position: Cart position on rail [-2.4, 2.4] meters
        - velocity: Cart velocity [-6, 6] m/s
        - angle: Stick angle in radians (π = downward)
        - angular_velocity: Angular velocity in rad/s

    Action Space (Continuous Box):
        - Single continuous value in [-1, 1]
        - Maps to target velocity: action * 5.0 m/s
        - -1: Move left at max speed (-5 m/s)
        -  0: No movement (0 m/s)
        - +1: Move right at max speed (+5 m/s)

    Reward:
        - cos(angle) - 0.001 * (velocity^2 + angular_velocity^2)
        - Maximum reward when stick is upright (angle = 0)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, render_mode: Optional[str] = None, use_shaped_reward: bool = False):
        super().__init__()

        # Reward shaping option for easier swing-up learning
        self.use_shaped_reward = use_shaped_reward

        # Physics parameters (matching JavaScript implementation exactly)
        self.gravity = 9.81  # m/s^2
        self.rail_length = 4.8  # meters
        self.max_position = self.rail_length / 2  # ±2.4 meters
        self.stick_length = 1.8  # meters
        self.dt = 0.02  # time step in seconds (50 Hz)

        # Mass properties
        self.stick_mass = 0.1  # kg
        self.weight_mass = 0.3  # kg (at stick tip)
        self.total_mass = self.stick_mass + self.weight_mass  # 0.4 kg

        # Calculate center of mass (weighted average)
        # Stick COM at L/2, weight at L
        self.center_of_mass = (
            (self.stick_mass * self.stick_length / 2 + self.weight_mass * self.stick_length) /
            self.total_mass
        )  # 1.575 meters

        # Moment of inertia
        # Rod: (1/3)*m*L^2 about one end
        # Point mass: m*L^2
        self.moment_of_inertia = (
            (self.stick_mass * self.stick_length * self.stick_length / 3) +
            (self.weight_mass * self.stick_length * self.stick_length)
        )  # 1.08 kg⋅m²

        # Control parameters
        self.acceleration_gain = 25.0  # How quickly cart reaches target velocity
        self.max_velocity = 6.0  # m/s
        self.angular_damping = 0.10  # Angular velocity damping

        # Define action and observation spaces
        # Continuous action space: [-1, 1] mapped to target velocity
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space bounds
        self.observation_space = spaces.Box(
            low=np.array([-self.max_position, -self.max_velocity, -np.pi, -np.inf]),
            high=np.array([self.max_position, self.max_velocity, np.pi, np.inf]),
            dtype=np.float32
        )

        # State variables
        self.position = 0.0
        self.velocity = 0.0
        self.angle = np.pi  # Start pointing down
        self.angular_velocity = 0.0

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state with small random perturbations."""
        super().reset(seed=seed)

        # Initialize with small random values (matching JavaScript)
        self.position = (self.np_random.random() - 0.5) * 0.1  # ±0.05 m
        self.velocity = (self.np_random.random() - 0.5) * 0.1  # ±0.05 m/s
        self.angle = np.pi + (self.np_random.random() - 0.5) * 0.3  # π ± 0.15 rad
        self.angular_velocity = (self.np_random.random() - 0.5) * 0.1  # ±0.05 rad/s

        # Normalize angle to [-π, π]
        self.angle = self._normalize_angle(self.angle)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step within the environment."""
        # Handle both single values and arrays
        if isinstance(action, (int, float)):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, np.ndarray) and action.shape == ():
            action = np.array([action], dtype=np.float32)

        # Extract scalar value from array and clip to valid range
        action_value = action[0] if isinstance(action, np.ndarray) else action
        action_value = np.clip(action_value, -1.0, 1.0)

        # Map continuous action [-1, 1] to target velocity [-5, 5] m/s
        target_velocity = action_value * 5.0  # Maps to [-5, 5] m/s

        # Check if we're at a boundary and trying to move further into it
        at_left_boundary = self.position <= -self.max_position and target_velocity < 0
        at_right_boundary = self.position >= self.max_position and target_velocity > 0

        # Calculate cart acceleration (matching JavaScript implementation)
        cart_acceleration = 0.0
        if not at_left_boundary and not at_right_boundary:
            # Only accelerate if not pushing into a boundary
            cart_acceleration = (target_velocity - self.velocity) * self.acceleration_gain
        elif (at_left_boundary and target_velocity > 0) or (at_right_boundary and target_velocity < 0):
            # Allow moving away from boundary
            cart_acceleration = (target_velocity - self.velocity) * self.acceleration_gain
        else:
            # At boundary and trying to push into it - stop completely
            self.velocity = 0

        # Update cart position and velocity
        self.velocity += cart_acceleration * self.dt
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)
        self.position += self.velocity * self.dt

        # Hard stop at rail boundaries
        if self.position < -self.max_position:
            self.position = -self.max_position
            self.velocity = max(0, self.velocity)  # Only allow positive velocity
        elif self.position > self.max_position:
            self.position = self.max_position
            self.velocity = min(0, self.velocity)  # Only allow negative velocity

        # Pendulum dynamics (matching JavaScript implementation)
        sin_theta = np.sin(self.angle)
        cos_theta = np.cos(self.angle)

        # Torque from gravity acting on center of mass
        gravity_torque = self.total_mass * self.gravity * self.center_of_mass * sin_theta

        # Torque from cart acceleration (pseudo-force in accelerating frame)
        accel_torque = -self.total_mass * self.center_of_mass * cart_acceleration * cos_theta

        # Total torque
        total_torque = gravity_torque + accel_torque

        # Angular acceleration = torque / moment of inertia
        angular_acceleration = total_torque / self.moment_of_inertia

        # Apply damping (proportional to angular velocity)
        damping_acceleration = -self.angular_damping * self.angular_velocity
        total_angular_acceleration = angular_acceleration + damping_acceleration

        # Update angle and angular velocity
        self.angular_velocity += total_angular_acceleration * self.dt
        self.angle += self.angular_velocity * self.dt

        # Normalize angle to [-π, π]
        while self.angle > np.pi:
            self.angle -= 2 * np.pi
        while self.angle < -np.pi:
            self.angle += 2 * np.pi

        # Calculate reward
        if self.use_shaped_reward:
            # Shaped reward for easier swing-up learning

            # 1. Primary: Upright reward (maximize when vertical)
            upright_reward = np.cos(self.angle)

            # 2. Energy reward - encourage building energy to reach upright
            # Energy: E = 0.5 * I * ω² - m * g * L * cos(θ)
            kinetic_energy = 0.5 * self.moment_of_inertia * self.angular_velocity ** 2
            potential_energy = -self.total_mass * self.gravity * self.center_of_mass * np.cos(self.angle)
            total_energy = kinetic_energy + potential_energy

            # Target energy at upright position (all potential)
            target_energy = self.total_mass * self.gravity * self.center_of_mass

            # Reward for energy close to target (normalized to [-1, 1] range)
            max_reasonable_energy = 2 * target_energy
            energy_error = abs(total_energy - target_energy)
            energy_reward = 1.0 - (energy_error / max_reasonable_energy)
            energy_reward = np.clip(energy_reward, -1, 1)

            # 3. Height reward - encourage getting stick tip higher
            # Height of stick tip above pivot point (cos gives height directly)
            tip_height = self.stick_length * np.cos(self.angle)  # Positive when up, negative when down
            max_height = self.stick_length  # Maximum when upright
            height_reward = tip_height / max_height  # Ranges from -1 (down) to +1 (up)

            # 4. Velocity penalty (same as before)
            velocity_penalty = 0.001 * (self.velocity ** 2 + self.angular_velocity ** 2)

            # Combine rewards with weights
            # Strong weight on upright (final goal), moderate on energy/height (intermediate)
            reward = (
                5.0 * upright_reward +      # Primary: be upright
                1.0 * energy_reward +        # Secondary: have right energy
                2.0 * height_reward -        # Secondary: get stick high
                velocity_penalty             # Penalty: be smooth
            )
        else:
            # Original sparse reward
            upright_reward = np.cos(self.angle)
            velocity_penalty = 0.001 * (self.velocity ** 2 + self.angular_velocity ** 2)
            reward = upright_reward - velocity_penalty

        # Episode never terminates (continuous task)
        terminated = False
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π] range."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return np.array([
            self.position,
            self.velocity,
            self.angle,
            self.angular_velocity
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        """Get additional info about the current state."""
        return {
            'upright_reward': np.cos(self.angle),
            'velocity_penalty': 0.001 * (self.velocity ** 2 + self.angular_velocity ** 2),
            'is_upright': abs(self.angle) < np.pi / 6,  # Within 30 degrees of vertical
        }

    def render(self):
        """Render the environment (optional, for visualization during training)."""
        if self.render_mode == "human":
            try:
                import pygame

                if self.screen is None:
                    pygame.init()
                    pygame.display.init()
                    self.screen = pygame.display.set_mode((900, 600))
                    pygame.display.set_caption("Stick Balancing Environment")

                if self.clock is None:
                    self.clock = pygame.time.Clock()

                # Clear screen
                self.screen.fill((255, 255, 255))

                # Scale factors
                scale = 150  # pixels per meter
                cart_y = 400  # vertical position

                # Draw rail
                pygame.draw.line(self.screen, (100, 100, 100),
                               (50, cart_y), (850, cart_y), 3)

                # Draw boundaries
                pygame.draw.line(self.screen, (200, 0, 0),
                               (50, cart_y - 20), (50, cart_y + 20), 5)
                pygame.draw.line(self.screen, (200, 0, 0),
                               (850, cart_y - 20), (850, cart_y + 20), 5)

                # Draw cart
                cart_x = 450 + self.position * scale
                cart_rect = pygame.Rect(cart_x - 35, cart_y - 11, 70, 22)
                pygame.draw.rect(self.screen, (0, 0, 255), cart_rect)

                # Draw stick
                stick_end_x = cart_x + self.stick_length * scale * np.sin(self.angle)
                stick_end_y = cart_y - self.stick_length * scale * np.cos(self.angle)
                pygame.draw.line(self.screen, (139, 69, 19),
                               (cart_x, cart_y - 11),
                               (stick_end_x, stick_end_y), 7)

                # Draw weight
                pygame.draw.circle(self.screen, (128, 128, 128),
                                 (int(stick_end_x), int(stick_end_y)), 10)

                pygame.display.flip()
                self.clock.tick(50)

            except ImportError:
                pass  # Pygame not available, skip rendering

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()