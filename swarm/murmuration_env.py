"""
Starling Murmuration Environment with Moving Attractor

Birds learn to flock through:
- Following a slowly moving attractor point (creates swirling)
- Staying near neighbors (cohesion)
- Not colliding (separation)
- Matching neighbor velocity (alignment)
- Penalty for losing neighbors from view (keeps flock tight)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict


class MurmurationEnv(gym.Env):
    def __init__(
        self,
        num_birds: int = 50,
        num_neighbors: int = 7,
        max_steps: int = 500,
        arena_size: float = 1.0,
        bird_speed: float = 0.02,
        bird_acceleration: float = 0.008,
        ideal_separation: float = 0.03,
        cohesion_radius: float = 0.12,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.num_birds = num_birds
        self.num_neighbors = min(num_neighbors, num_birds - 1)
        self.max_steps = max_steps
        self.arena_size = arena_size
        self.bird_speed = bird_speed
        self.bird_acceleration = bird_acceleration
        self.ideal_separation = ideal_separation
        self.cohesion_radius = cohesion_radius
        self.render_mode = render_mode

        # Observation: velocity(2) + attractor_dir(2) + neighbors(K*4)
        self.obs_dim = 2 + 2 + num_neighbors * 4

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(num_birds * self.obs_dim,),
            dtype=np.float32
        )

        # 9 discrete actions: 8 directions + stay
        self.action_space = spaces.MultiDiscrete([9] * num_birds)

        # Direction vectors for actions
        angles = np.linspace(0, 2 * np.pi, 9)[:-1]
        self.action_dirs = np.array([
            [np.cos(a), np.sin(a)] for a in angles
        ] + [[0, 0]])

        # State
        self.positions = None
        self.velocities = None
        self.attractor = None
        self.attractor_angle = 0
        self.step_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        # Spawn birds in a cluster at center
        center = self.arena_size / 2
        self.positions = self.np_random.uniform(
            center - 0.12, center + 0.12,
            size=(self.num_birds, 2)
        )

        # Random initial velocities
        angles = self.np_random.uniform(0, 2 * np.pi, self.num_birds)
        speeds = self.np_random.uniform(0.5, 1.0, self.num_birds) * self.bird_speed
        self.velocities = np.stack([
            np.cos(angles) * speeds,
            np.sin(angles) * speeds
        ], axis=1)

        # Initialize attractor at center
        self.attractor = np.array([center, center])
        self.attractor_angle = self.np_random.uniform(0, 2 * np.pi)

        self.step_count = 0
        return self._get_obs(), self._get_info()

    def _move_attractor(self):
        """Move attractor in a smooth, interesting pattern"""
        # Slowly rotating figure-8 / lemniscate pattern
        self.attractor_angle += 0.02

        center = self.arena_size / 2
        radius = 0.25

        # Figure-8 pattern
        t = self.attractor_angle
        self.attractor[0] = center + radius * np.sin(t)
        self.attractor[1] = center + radius * np.sin(t) * np.cos(t)

    def _get_neighbors(self, bird_idx: int) -> np.ndarray:
        pos = self.positions[bird_idx]
        distances = np.linalg.norm(self.positions - pos, axis=1)
        distances[bird_idx] = np.inf
        return np.argsort(distances)[:self.num_neighbors]

    def _get_obs(self) -> np.ndarray:
        all_obs = []

        for i in range(self.num_birds):
            pos = self.positions[i]
            vel = self.velocities[i]

            # My velocity (normalized)
            my_vel = vel / self.bird_speed

            # Direction to attractor (normalized)
            to_attractor = self.attractor - pos
            dist_to_attractor = np.linalg.norm(to_attractor)
            if dist_to_attractor > 0.01:
                attractor_dir = to_attractor / dist_to_attractor
            else:
                attractor_dir = np.array([0.0, 0.0])

            # Neighbors
            neighbors = self._get_neighbors(i)
            neighbor_obs = []

            for n in neighbors:
                n_rel_pos = (self.positions[n] - pos) / self.arena_size
                n_rel_vel = (self.velocities[n] - vel) / self.bird_speed
                neighbor_obs.extend([n_rel_pos[0], n_rel_pos[1], n_rel_vel[0], n_rel_vel[1]])

            obs = np.concatenate([my_vel, attractor_dir, neighbor_obs]).astype(np.float32)
            all_obs.append(obs)

        return np.concatenate(all_obs)

    def _get_info(self):
        center = np.mean(self.positions, axis=0)
        avg_dist_to_center = np.mean(np.linalg.norm(self.positions - center, axis=1))

        avg_vel = np.mean(self.velocities, axis=0)
        avg_vel_norm = avg_vel / (np.linalg.norm(avg_vel) + 1e-8)
        alignments = []
        for v in self.velocities:
            v_norm = v / (np.linalg.norm(v) + 1e-8)
            alignments.append(np.dot(v_norm, avg_vel_norm))
        avg_alignment = np.mean(alignments)

        return {
            "cohesion": float(avg_dist_to_center),
            "alignment": float(avg_alignment),
            "step": self.step_count,
        }

    def step(self, actions: np.ndarray):
        # Move the attractor
        self._move_attractor()

        rewards = np.zeros(self.num_birds, dtype=np.float32)

        # Apply actions
        for i in range(self.num_birds):
            action_dir = self.action_dirs[actions[i]]
            acceleration = action_dir * self.bird_acceleration
            self.velocities[i] += acceleration

            # Clamp speed
            speed = np.linalg.norm(self.velocities[i])
            if speed > self.bird_speed:
                self.velocities[i] = self.velocities[i] / speed * self.bird_speed
            elif speed < self.bird_speed * 0.3:
                if speed > 0:
                    self.velocities[i] = self.velocities[i] / speed * self.bird_speed * 0.3

        # Update positions
        self.positions += self.velocities

        # Soft boundary steering
        margin = 0.15
        turn_force = 0.004
        for i in range(self.num_birds):
            if self.positions[i, 0] < margin:
                self.velocities[i, 0] += turn_force
            if self.positions[i, 0] > self.arena_size - margin:
                self.velocities[i, 0] -= turn_force
            if self.positions[i, 1] < margin:
                self.velocities[i, 1] += turn_force
            if self.positions[i, 1] > self.arena_size - margin:
                self.velocities[i, 1] -= turn_force

        # Clamp to arena
        self.positions = np.clip(self.positions, 0.02, self.arena_size - 0.02)

        # Calculate rewards
        flock_center = np.mean(self.positions, axis=0)

        for i in range(self.num_birds):
            neighbors = self._get_neighbors(i)

            # 1. Attractor following - reward for moving toward attractor
            to_attractor = self.attractor - self.positions[i]
            dist_to_attractor = np.linalg.norm(to_attractor)
            attractor_reward = max(0, 0.3 - dist_to_attractor) * 0.5

            # 2. Cohesion - reward for being near flock center
            dist_to_flock = np.linalg.norm(self.positions[i] - flock_center)
            cohesion_reward = max(0, 0.15 - dist_to_flock) * 0.3

            # 3. Neighbor proximity - reward for having close neighbors
            neighbor_reward = 0
            separation_penalty = 0
            alignment_reward = 0

            for n in neighbors:
                dist = np.linalg.norm(self.positions[n] - self.positions[i])

                # Reward for neighbors within cohesion radius
                if dist < self.cohesion_radius:
                    neighbor_reward += 0.08

                # Penalty for being too close
                if dist < self.ideal_separation:
                    separation_penalty += 0.4 * (self.ideal_separation - dist) / self.ideal_separation

                # Alignment reward
                my_vel_norm = self.velocities[i] / (np.linalg.norm(self.velocities[i]) + 1e-8)
                n_vel_norm = self.velocities[n] / (np.linalg.norm(self.velocities[n]) + 1e-8)
                alignment_reward += 0.03 * np.dot(my_vel_norm, n_vel_norm)

            rewards[i] = attractor_reward + cohesion_reward + neighbor_reward + alignment_reward - separation_penalty

        self.step_count += 1
        info = self._get_info()
        info["bird_rewards"] = rewards
        info["mean_reward"] = float(np.mean(rewards))

        terminated = False
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(np.mean(rewards)), terminated, truncated, info


def make_murmuration_env(**kwargs):
    return MurmurationEnv(**kwargs)


if __name__ == "__main__":
    env = make_murmuration_env(num_birds=50, num_neighbors=7)
    obs, info = env.reset(seed=42)

    print(f"Obs dim per bird: {env.obs_dim}")
    print(f"Initial cohesion: {info['cohesion']:.3f}")

    for step in range(100):
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)

        if step % 20 == 0:
            print(f"Step {step}: cohesion={info['cohesion']:.3f}, reward={reward:.3f}")
