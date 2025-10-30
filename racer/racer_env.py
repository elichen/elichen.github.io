"""
Racing Environment for Stable Baselines3 Training
A custom Gymnasium environment that simulates the JavaScript racing game
with ray-based sensors and continuous control for PPO training.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

class RacerEnv(gym.Env):
    """Custom Environment for racing AI training with ray sensors"""

    def __init__(self, max_steps=2000, render_mode=None):
        super().__init__()

        # Environment parameters
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode

        # Track definition (matching JavaScript track.js - bean shape with upward arch)
        # Track centered at (0, 0) for simplicity
        self.outer_points = np.array([
            [-400, -250],  # Top left
            [400, -250],   # Top right
            [500, -150],   # Top right corner
            [500, 50],     # Right side upper
            [480, 150],    # Right side lower
            [400, 230],    # Bottom right curve
            [250, 260],    # Bottom right approaching arch
            [100, 250],    # Bottom right side of arch
            [0, 120],      # Bottom center (deep upward arch)
            [-100, 250],   # Bottom left side of arch (symmetric)
            [-250, 260],   # Bottom left approaching arch (symmetric)
            [-400, 230],   # Bottom left curve (symmetric)
            [-480, 150],   # Left side lower (symmetric)
            [-500, 50],    # Left side upper (symmetric)
            [-500, -150]   # Top left corner
        ])

        self.inner_points = np.array([
            [-350, -130],  # Top left
            [-380, -100],  # Top left corner
            [-380, 38],    # Left side upper
            [-369, 91],    # Left side lower
            [-341, 119],   # Bottom left curve
            [-242, 139],   # Bottom left approaching arch
            [-162, 134],   # Bottom left side of arch
            [0, -77],      # Bottom center (upward arch)
            [162, 134],    # Bottom right side of arch
            [242, 139],    # Bottom right approaching arch
            [341, 119],    # Bottom right curve
            [369, 91],     # Right side lower
            [380, 38],     # Right side upper
            [380, -100],   # Top right corner
            [350, -130]    # Top right
        ])

        # Finish line
        self.finish_line = {
            'x1': 0, 'y1': -250,
            'x2': 0, 'y2': -130,  # Extended to new inner edge
            'startX': 0,
            'startY': -190,       # Halfway between y1 and y2
            'startAngle': math.pi  # Pointing left (perpendicular to finish line)
        }

        # Car physics constants (matching car.js)
        self.max_speed = 50  # 5x original speed for extreme racing
        self.max_reverse_speed = -5
        self.acceleration = 0.3  # Moderate acceleration for control
        self.brake_force = 1.0  # Much stronger braking needed at high speeds
        self.reverse_acceleration = 0.1
        # Drag removed - no air resistance

        # Speed-dependent turning physics
        self.min_turn_radius = 30  # Minimum turning radius at low speed (pixels)
        self.max_turn_radius = 200  # Maximum turning radius at max speed (pixels)
        self.turn_speed_base = 0.03  # Base turn rate at zero speed
        self.turn_speed_min = 0.005  # Minimum turn rate at max speed
        self.car_width = 20
        self.car_height = 40

        # Ray sensor configuration
        self.num_rays = 9  # Number of ray sensors
        self.max_ray_distance = 300  # Maximum sensing distance
        self.ray_angles = np.linspace(-90, 90, self.num_rays)  # Degrees relative to car

        # Define action and observation spaces
        # Actions: [steering, throttle/brake]
        # steering: -1 (left) to 1 (right)
        # throttle: -1 (brake/reverse) to 1 (accelerate)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observations: ray distances + speed + angular velocity
        # All normalized to [0, 1] or [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_rays + 2,),  # rays + speed + angular_velocity
            dtype=np.float32
        )

        # State variables
        self.reset()

        # Track progress tracking
        self.checkpoints = self._generate_checkpoints()
        self.last_checkpoint = 0
        self.laps_completed = 0
        self.lap_start_time = 0
        self.best_lap_time = float('inf')

    def _generate_checkpoints(self):
        """Generate checkpoints around the track for progress tracking (counter-clockwise)"""
        checkpoints = []
        # Create gates at regular intervals around the track
        # Track points go counter-clockwise, starting from finish line (top center)
        num_gates = 16

        # Calculate center-based checkpoints going counter-clockwise
        # Starting from finish line at top (270 degrees / 3π/2), going left
        for i in range(num_gates):
            # Start at 180 degrees (left) and go counter-clockwise
            angle = math.pi - (2 * math.pi * i) / num_gates

            # Use angle to find nearest track points
            outer_idx = int(i * len(self.outer_points) / num_gates) % len(self.outer_points)
            inner_idx = int(i * len(self.inner_points) / num_gates) % len(self.inner_points)

            gate = {
                'x1': self.outer_points[outer_idx][0],
                'y1': self.outer_points[outer_idx][1],
                'x2': self.inner_points[inner_idx][0],
                'y2': self.inner_points[inner_idx][1],
                'index': i
            }
            checkpoints.append(gate)
        return checkpoints

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Reset car state
        self.x = self.finish_line['startX']
        self.y = self.finish_line['startY']
        self.angle = self.finish_line['startAngle']
        self.speed = 0
        self.last_x = self.x
        self.last_y = self.y
        self.angular_velocity = 0

        # Reset tracking variables
        self.current_step = 0
        self.last_checkpoint = 0
        self.laps_completed = 0
        self.lap_start_time = 0
        self.collisions = 0
        self.total_distance = 0

        return self._get_observation(), {}

    def _cast_ray(self, angle_offset):
        """Cast a ray from the car and return distance to track boundary"""
        # Convert angle from degrees to radians and add to car angle
        ray_angle = self.angle + math.radians(angle_offset)

        # Ray direction
        dx = math.cos(ray_angle)
        dy = math.sin(ray_angle)

        # Cast ray step by step
        distance = 0
        step_size = 5
        max_steps = int(self.max_ray_distance / step_size)

        for i in range(max_steps):
            distance = i * step_size
            ray_x = self.x + dx * distance
            ray_y = self.y + dy * distance

            # Check if point is outside track
            if not self._is_point_inside_track(ray_x, ray_y):
                break

        return distance

    def _is_point_inside_track(self, x, y):
        """Check if a point is inside the track (between outer and inner boundaries)"""
        return (self._is_point_inside_polygon(x, y, self.outer_points) and
                not self._is_point_inside_polygon(x, y, self.inner_points))

    def _is_point_inside_polygon(self, x, y, points):
        """Ray casting algorithm for point in polygon test"""
        inside = False
        p1x, p1y = points[-1]
        for p2x, p2y in points:
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _get_observation(self):
        """Get current observation (ray distances + speed + angular velocity)"""
        observations = []

        # Cast rays and normalize distances
        for angle in self.ray_angles:
            distance = self._cast_ray(angle)
            normalized_distance = distance / self.max_ray_distance
            observations.append(normalized_distance)

        # Add normalized speed
        normalized_speed = self.speed / self.max_speed
        observations.append(np.clip(normalized_speed, -1, 1))

        # Add normalized angular velocity
        observations.append(np.clip(self.angular_velocity, -1, 1))

        return np.array(observations, dtype=np.float32)

    def step(self, action):
        """Execute one time step within the environment"""
        self.current_step += 1

        # Parse actions
        steering = np.clip(action[0], -1, 1)
        throttle = np.clip(action[1], -1, 1)

        # Store last position
        self.last_x = self.x
        self.last_y = self.y
        last_angle = self.angle

        # No drag - removed air resistance

        # Apply throttle/brake
        if throttle > 0:
            # Accelerate forward
            if self.speed >= 0:
                self.speed = min(self.max_speed, self.speed + throttle * self.acceleration)
            else:
                # Braking from reverse
                self.speed = min(0, self.speed + throttle * self.brake_force)
        elif throttle < 0:
            # Brake or reverse
            if self.speed <= 0:
                # Reverse acceleration
                self.speed = max(self.max_reverse_speed,
                               self.speed + throttle * self.reverse_acceleration)
            else:
                # Braking from forward
                self.speed = max(0, self.speed - abs(throttle) * self.brake_force)

        # Speed-dependent turning - realistic physics
        # At higher speeds, turning radius increases (less sharp turns)
        abs_speed = abs(self.speed)
        speed_ratio = abs_speed / self.max_speed  # 0 to 1

        # Calculate effective turn speed based on current speed
        # Interpolate between turn_speed_base (at 0 speed) and turn_speed_min (at max speed)
        effective_turn_speed = self.turn_speed_base * (1 - speed_ratio) + self.turn_speed_min * speed_ratio

        # Apply turning only if moving (slight turning allowed at very low speeds for maneuvering)
        turn_multiplier = abs_speed * 2 if abs_speed < 0.5 else 1  # Gradual turn activation at very low speeds

        self.angle += steering * effective_turn_speed * turn_multiplier

        # Calculate angular velocity for observation
        self.angular_velocity = (self.angle - last_angle) / 0.016  # Assuming 60 FPS

        # Calculate new position
        new_x = self.x + math.cos(self.angle) * self.speed
        new_y = self.y + math.sin(self.angle) * self.speed

        # Check collision with multiple points around car
        car_points = [
            (new_x - 15, new_y - 15),  # Front left
            (new_x + 15, new_y - 15),  # Front right
            (new_x - 15, new_y + 15),  # Rear left
            (new_x + 15, new_y + 15),  # Rear right
            (new_x, new_y)              # Center
        ]

        collision = False
        for point in car_points:
            if not self._is_point_inside_track(point[0], point[1]):
                collision = True
                break

        if not collision:
            self.x = new_x
            self.y = new_y
        else:
            # Collision response
            self.speed *= 0.5
            self.collisions += 1

        # Track total distance traveled
        distance_moved = math.sqrt((self.x - self.last_x)**2 + (self.y - self.last_y)**2)
        self.total_distance += distance_moved

        # Check checkpoint progress
        current_checkpoint = self._get_nearest_checkpoint()
        checkpoint_progress = 0
        if current_checkpoint != self.last_checkpoint:
            # Check if we're going forward (not backward)
            expected_next = (self.last_checkpoint + 1) % len(self.checkpoints)
            if current_checkpoint == expected_next:
                checkpoint_progress = 100  # Reward for checkpoint
                self.last_checkpoint = current_checkpoint

        # Check lap completion
        lap_completed = self._check_lap_completion()
        lap_time = 0
        if lap_completed:
            self.laps_completed += 1
            lap_time = self.current_step - self.lap_start_time
            self.lap_start_time = self.current_step
            if lap_time < self.best_lap_time:
                self.best_lap_time = lap_time

        # Calculate reward (lap time focused)
        reward = self._calculate_reward(
            collision=collision,
            checkpoint_progress=checkpoint_progress,
            lap_completed=lap_completed,
            lap_time=lap_time,
            distance_moved=distance_moved
        )

        # Check if episode is done
        done = False
        truncated = False

        if lap_completed and self.laps_completed >= 1:
            done = True  # Complete after 1 full lap
        elif self.current_step >= self.max_steps:
            truncated = True
        elif self.speed < 0.1 and self.current_step > 100:
            # Stuck detection
            if self.total_distance / self.current_step < 0.5:
                done = True

        info = {
            'laps_completed': self.laps_completed,
            'lap_time': lap_time if lap_completed else 0,
            'best_lap_time': self.best_lap_time,
            'collisions': self.collisions,
            'checkpoint': self.last_checkpoint,
            'total_distance': self.total_distance
        }

        return self._get_observation(), reward, done, truncated, info

    def _calculate_reward(self, collision, checkpoint_progress, lap_completed,
                         lap_time, distance_moved):
        """Calculate reward with focus on lap time optimization"""
        reward = 0

        # Large reward for lap completion (inversely proportional to lap time)
        if lap_completed:
            if lap_time > 0:
                # Bonus scaled by how fast the lap was (encourage speed)
                time_bonus = 1000 / (lap_time / 60)  # Assuming 60 FPS
                reward += time_bonus
            else:
                reward += 500  # Default lap bonus

        # Checkpoint progress reward
        if checkpoint_progress > 0:
            reward += checkpoint_progress * 0.5

        # Reward for forward movement, penalize reverse
        if self.speed > 0:
            reward += distance_moved * 0.1  # Strong reward for going forward
        else:
            reward -= distance_moved * 0.5  # Penalty for going backward

        # Collision penalty (mild to encourage aggressive driving)
        if collision:
            reward -= 10

        # Small time penalty to encourage speed
        reward -= 0.1

        return reward

    def _get_nearest_checkpoint(self):
        """Find which checkpoint the car is nearest to"""
        min_dist = float('inf')
        nearest = self.last_checkpoint

        for i, checkpoint in enumerate(self.checkpoints):
            # Distance to checkpoint gate center
            gate_center_x = (checkpoint['x1'] + checkpoint['x2']) / 2
            gate_center_y = (checkpoint['y1'] + checkpoint['y2']) / 2
            dist = math.sqrt((self.x - gate_center_x)**2 + (self.y - gate_center_y)**2)

            if dist < min_dist:
                min_dist = dist
                nearest = i

        return nearest

    def _check_lap_completion(self):
        """Check if the car has crossed the finish line"""
        # Check if we crossed finish line from right to left (completing a lap)
        # Must have made significant progress (at least halfway around)
        if (self.last_x >= self.finish_line['x1'] and
            self.x < self.finish_line['x1'] and
            self.y >= self.finish_line['y1'] and
            self.y <= self.finish_line['y2'] and
            self.last_checkpoint >= len(self.checkpoints) // 2):  # At least halfway around
            return True
        return False

    def render(self):
        """Render the environment (optional, for debugging)"""
        if self.render_mode == "human":
            # Simple text output for debugging
            print(f"Step: {self.current_step}, Pos: ({self.x:.1f}, {self.y:.1f}), "
                  f"Speed: {self.speed:.2f}, Checkpoint: {self.last_checkpoint}, "
                  f"Laps: {self.laps_completed}")

    def close(self):
        """Clean up resources"""
        pass


# Test the environment
if __name__ == "__main__":
    env = RacerEnv(render_mode="human")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation: {obs}")

    # Test a few random steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            break

    env.close()