"""
Starling Murmuration Environment

Birds learn to flock through:
- Staying near neighbors (cohesion)
- Not colliding (separation)
- Matching neighbor velocity (alignment)
- Following a moving attractor (creates swirling patterns)
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
        ideal_separation: float = 0.025,
        cohesion_radius: float = 0.10,
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

        # Attractor settings (moving point that creates swirling)
        self.attractor_speed = bird_speed * 1.5

        # Observation: velocity(2) + attractor_dir(2) + neighbors(K*4)
        self.obs_dim = 2 + 2 + num_neighbors * 4

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(num_birds * self.obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete([9] * num_birds)

        angles = np.linspace(0, 2 * np.pi, 9)[:-1]
        self.action_dirs = np.array([
            [np.cos(a), np.sin(a)] for a in angles
        ] + [[0, 0]])

        self.positions = None
        self.velocities = None
        self.predator_pos = None
        self.predator_vel = None
        self.step_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        center = self.arena_size / 2
        self.positions = self.np_random.uniform(
            center - 0.12, center + 0.12,
            size=(self.num_birds, 2)
        )

        angles = self.np_random.uniform(0, 2 * np.pi, self.num_birds)
        speeds = self.np_random.uniform(0.5, 1.0, self.num_birds) * self.bird_speed
        self.velocities = np.stack([
            np.cos(angles) * speeds,
            np.sin(angles) * speeds
        ], axis=1)

        # Attractor moves in figure-8 pattern around center
        self.attractor_pos = np.array([self.arena_size / 2, self.arena_size / 2])
        self.attractor_angle = self.np_random.uniform(0, 2 * np.pi)
        self.step_count = 0
        return self._get_obs(), self._get_info()

    def _move_attractor(self):
        """Attractor moves in a smooth figure-8 pattern to create swirling"""
        self.attractor_angle += 0.02

        # Figure-8 (lemniscate) pattern
        center = self.arena_size / 2
        radius = 0.25
        t = self.attractor_angle
        self.attractor_pos[0] = center + radius * np.sin(t)
        self.attractor_pos[1] = center + radius * np.sin(t) * np.cos(t)

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

            my_vel = vel / self.bird_speed

            # Attractor direction (normalized)
            to_attractor = self.attractor_pos - pos
            attractor_dist = np.linalg.norm(to_attractor)
            if attractor_dist > 0.01:
                attractor_dir = to_attractor / attractor_dist
            else:
                attractor_dir = np.array([0.0, 0.0])

            neighbors = self._get_neighbors(i)
            neighbor_obs = []

            for n in neighbors:
                n_rel_pos = (self.positions[n] - pos) / self.arena_size
                n_rel_vel = (self.velocities[n] - vel) / self.bird_speed
                neighbor_obs.extend([n_rel_pos[0], n_rel_pos[1], n_rel_vel[0], n_rel_vel[1]])

            obs = np.concatenate([
                my_vel,
                attractor_dir,
                neighbor_obs
            ]).astype(np.float32)
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

        return {
            "cohesion": float(avg_dist_to_center),
            "alignment": float(np.mean(alignments)),
            "step": self.step_count,
        }

    def step(self, actions: np.ndarray):
        self._move_attractor()

        rewards = np.zeros(self.num_birds, dtype=np.float32)

        # Apply actions
        for i in range(self.num_birds):
            action_dir = self.action_dirs[actions[i]]
            self.velocities[i] += action_dir * self.bird_acceleration

            speed = np.linalg.norm(self.velocities[i])
            if speed > self.bird_speed:
                self.velocities[i] = self.velocities[i] / speed * self.bird_speed
            elif speed < self.bird_speed * 0.3 and speed > 0:
                self.velocities[i] = self.velocities[i] / speed * self.bird_speed * 0.3

        self.positions += self.velocities

        # Soft boundary
        margin = 0.12
        turn_force = 0.005
        for i in range(self.num_birds):
            if self.positions[i, 0] < margin:
                self.velocities[i, 0] += turn_force
            if self.positions[i, 0] > self.arena_size - margin:
                self.velocities[i, 0] -= turn_force
            if self.positions[i, 1] < margin:
                self.velocities[i, 1] += turn_force
            if self.positions[i, 1] > self.arena_size - margin:
                self.velocities[i, 1] -= turn_force

        self.positions = np.clip(self.positions, 0.02, self.arena_size - 0.02)

        # Calculate rewards
        flock_center = np.mean(self.positions, axis=0)

        for i in range(self.num_birds):
            neighbors = self._get_neighbors(i)

            # 1. Attractor following (gentle pull toward attractor)
            dist_to_attractor = np.linalg.norm(self.positions[i] - self.attractor_pos)
            attractor_reward = max(0, 0.3 - dist_to_attractor) * 0.3

            # 2. Cohesion - stay with flock
            dist_to_flock = np.linalg.norm(self.positions[i] - flock_center)
            cohesion_reward = max(0, 0.2 - dist_to_flock) * 0.5

            # 3. Separation penalty
            separation_penalty = 0
            for n in neighbors:
                dist = np.linalg.norm(self.positions[n] - self.positions[i])
                if dist < self.ideal_separation:
                    separation_penalty += 0.3 * (self.ideal_separation - dist) / self.ideal_separation

            # 4. Alignment with neighbors
            alignment_reward = 0
            my_vel_norm = self.velocities[i] / (np.linalg.norm(self.velocities[i]) + 1e-8)
            for n in neighbors:
                n_vel_norm = self.velocities[n] / (np.linalg.norm(self.velocities[n]) + 1e-8)
                alignment_reward += 0.03 * np.dot(my_vel_norm, n_vel_norm)

            rewards[i] = attractor_reward + cohesion_reward + alignment_reward - separation_penalty

        self.step_count += 1
        info = self._get_info()
        info["bird_rewards"] = rewards
        info["mean_reward"] = float(np.mean(rewards))

        terminated = False
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(np.mean(rewards)), terminated, truncated, info


def make_murmuration_env(**kwargs):
    return MurmurationEnv(**kwargs)
