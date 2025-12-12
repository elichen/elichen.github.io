"""
Predator Escape Flocking Environment

Fish learn to school together through survival pressure.
The predator confusion effect makes it harder to catch fish in groups.

Key insight: Fish near other fish survive longer, creating a learnable signal
for emergent schooling behavior.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict


class FlockingEnv(gym.Env):
    """
    N fish try to survive against 1 predator.

    Observation (per fish):
        - my_position (2): x, y normalized
        - my_velocity (2): vx, vy normalized
        - predator_relative (2): direction to predator
        - predator_distance (1): distance to predator
        - nearest_neighbors (K × 4): relative_pos(2) + relative_vel(2) for K neighbors

    Action (discrete):
        - 8 directions + stay (9 actions total)

    Reward:
        - +0.1 per timestep alive
        - -5.0 when eaten
        - Bonus for survivors at episode end
    """

    def __init__(
        self,
        num_fish: int = 40,
        num_neighbors: int = 5,
        max_steps: int = 500,
        arena_size: float = 1.0,
        fish_max_speed: float = 0.015,
        fish_acceleration: float = 0.004,
        predator_speed: float = 0.025,  # Much faster than fish - can't outrun
        attack_range: float = 0.03,
        base_catch_rate: float = 0.5,   # Very high base rate
        confusion_alpha: float = 2.0,    # Strong confusion effect
        neighbor_radius: float = 0.08,   # Need to be close for protection
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.num_fish = num_fish
        self.num_neighbors = min(num_neighbors, num_fish - 1)
        self.max_steps = max_steps
        self.arena_size = arena_size
        self.fish_max_speed = fish_max_speed
        self.fish_acceleration = fish_acceleration
        self.predator_speed = predator_speed
        self.attack_range = attack_range
        self.base_catch_rate = base_catch_rate
        self.confusion_alpha = confusion_alpha
        self.neighbor_radius = neighbor_radius
        self.render_mode = render_mode

        # Observation: position(2) + velocity(2) + predator_rel(2) + pred_dist(1) + neighbors(K*4)
        self.obs_dim = 2 + 2 + 2 + 1 + num_neighbors * 4

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(num_fish * self.obs_dim,),
            dtype=np.float32
        )

        # 9 discrete actions: 8 directions + stay
        self.action_space = spaces.MultiDiscrete([9] * num_fish)

        # Direction vectors for actions (8 directions + stay)
        angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 directions
        self.action_dirs = np.array([
            [np.cos(a), np.sin(a)] for a in angles
        ] + [[0, 0]])  # Action 8 = stay

        # State
        self.fish_positions = None
        self.fish_velocities = None
        self.fish_alive = None
        self.predator_pos = None
        self.predator_vel = None
        self.step_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        # Spawn fish in center cluster
        center = self.arena_size / 2
        self.fish_positions = self.np_random.uniform(
            center - 0.2, center + 0.2,
            size=(self.num_fish, 2)
        )
        self.fish_velocities = self.np_random.uniform(
            -0.005, 0.005,
            size=(self.num_fish, 2)
        )
        self.fish_alive = np.ones(self.num_fish, dtype=bool)

        # Spawn predator at edge
        edge = self.np_random.choice([0, 1])
        self.predator_pos = np.array([
            self.np_random.uniform(0.1, 0.9) if self.np_random.random() > 0.5 else edge * self.arena_size,
            self.np_random.uniform(0.1, 0.9) if self.np_random.random() > 0.5 else edge * self.arena_size,
        ])
        self.predator_vel = np.zeros(2)

        self.step_count = 0
        return self._get_obs(), self._get_info()

    def _get_neighbors(self, fish_idx: int) -> np.ndarray:
        """Get indices of K nearest alive neighbors."""
        if not self.fish_alive[fish_idx]:
            return np.zeros(self.num_neighbors, dtype=int)

        pos = self.fish_positions[fish_idx]
        distances = np.full(self.num_fish, np.inf)

        for i in range(self.num_fish):
            if i != fish_idx and self.fish_alive[i]:
                distances[i] = np.linalg.norm(self.fish_positions[i] - pos)

        # Get K nearest
        nearest = np.argsort(distances)[:self.num_neighbors]
        return nearest

    def _get_obs(self) -> np.ndarray:
        all_obs = []

        for i in range(self.num_fish):
            if not self.fish_alive[i]:
                # Dead fish get zero observation
                all_obs.append(np.zeros(self.obs_dim, dtype=np.float32))
                continue

            pos = self.fish_positions[i]
            vel = self.fish_velocities[i]

            # My position (normalized to [0, 1])
            my_pos = pos / self.arena_size

            # My velocity (normalized)
            my_vel = vel / self.fish_max_speed

            # Predator relative position and distance
            pred_rel = self.predator_pos - pos
            pred_dist = np.linalg.norm(pred_rel)
            if pred_dist > 0:
                pred_dir = pred_rel / pred_dist
            else:
                pred_dir = np.zeros(2)
            pred_dist_norm = min(pred_dist / self.arena_size, 1.0)

            # Neighbors
            neighbors = self._get_neighbors(i)
            neighbor_obs = []

            for n in neighbors:
                if self.fish_alive[n]:
                    n_rel_pos = (self.fish_positions[n] - pos) / self.arena_size
                    n_rel_vel = (self.fish_velocities[n] - vel) / self.fish_max_speed
                    neighbor_obs.extend([n_rel_pos[0], n_rel_pos[1], n_rel_vel[0], n_rel_vel[1]])
                else:
                    neighbor_obs.extend([0, 0, 0, 0])

            obs = np.concatenate([
                my_pos,
                my_vel,
                pred_dir,
                [pred_dist_norm],
                neighbor_obs
            ]).astype(np.float32)

            all_obs.append(obs)

        return np.concatenate(all_obs)

    def _get_info(self):
        alive_count = np.sum(self.fish_alive)
        survival_rate = alive_count / self.num_fish

        # Calculate school cohesion (average distance to center of mass)
        if alive_count > 1:
            alive_positions = self.fish_positions[self.fish_alive]
            center = np.mean(alive_positions, axis=0)
            avg_dist = np.mean(np.linalg.norm(alive_positions - center, axis=1))
        else:
            avg_dist = 0

        return {
            "alive_count": int(alive_count),
            "survival_rate": float(survival_rate),
            "school_cohesion": float(avg_dist),
            "step": self.step_count,
        }

    def _count_nearby_fish(self, target_idx: int) -> int:
        """Count fish near the target (for confusion effect)."""
        if not self.fish_alive[target_idx]:
            return 0

        target_pos = self.fish_positions[target_idx]
        count = 0

        for i in range(self.num_fish):
            if i != target_idx and self.fish_alive[i]:
                dist = np.linalg.norm(self.fish_positions[i] - target_pos)
                if dist < self.neighbor_radius:
                    count += 1

        return count

    def _get_isolation_score(self, fish_idx: int) -> float:
        """Higher score = more isolated (fewer neighbors nearby)."""
        nearby = self._count_nearby_fish(fish_idx)
        return 1.0 / (1.0 + nearby)  # 1.0 if alone, ~0.1 if 9 neighbors

    def step(self, actions: np.ndarray):
        rewards = np.zeros(self.num_fish, dtype=np.float32)

        # Apply fish actions
        for i in range(self.num_fish):
            if not self.fish_alive[i]:
                continue

            # Get acceleration from action
            action_dir = self.action_dirs[actions[i]]
            acceleration = action_dir * self.fish_acceleration

            # Update velocity with momentum
            self.fish_velocities[i] += acceleration

            # Clamp speed
            speed = np.linalg.norm(self.fish_velocities[i])
            if speed > self.fish_max_speed:
                self.fish_velocities[i] = self.fish_velocities[i] / speed * self.fish_max_speed

            # Update position
            self.fish_positions[i] += self.fish_velocities[i]

            # Bounce off walls
            for d in range(2):
                if self.fish_positions[i, d] < 0:
                    self.fish_positions[i, d] = 0
                    self.fish_velocities[i, d] *= -0.5
                elif self.fish_positions[i, d] > self.arena_size:
                    self.fish_positions[i, d] = self.arena_size
                    self.fish_velocities[i, d] *= -0.5

        # Predator targets MOST ISOLATED fish (not just nearest)
        alive_indices = np.where(self.fish_alive)[0]
        if len(alive_indices) > 0:
            # Score each fish: isolated fish are preferred targets
            scores = []
            for idx in alive_indices:
                dist = np.linalg.norm(self.fish_positions[idx] - self.predator_pos)
                isolation = self._get_isolation_score(idx)
                # Combine distance and isolation (isolated fish are very attractive)
                score = isolation * 3.0 + 1.0 / (dist + 0.1)
                scores.append(score)

            target_idx = alive_indices[np.argmax(scores)]
            target_pos = self.fish_positions[target_idx]

            # Chase target
            direction = target_pos - self.predator_pos
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
            self.predator_vel = direction * self.predator_speed
            self.predator_pos += self.predator_vel

            # Keep predator in bounds
            self.predator_pos = np.clip(self.predator_pos, 0, self.arena_size)

            # Check for catch attempt
            if dist < self.attack_range:
                nearby_count = self._count_nearby_fish(target_idx)

                # Confusion effect: harder to catch fish in groups
                catch_prob = self.base_catch_rate / (1 + self.confusion_alpha * nearby_count)

                if self.np_random.random() < catch_prob:
                    # Fish is caught!
                    self.fish_alive[target_idx] = False
                    rewards[target_idx] = -5.0

        # Survival reward for alive fish
        rewards[self.fish_alive] += 0.1

        self.step_count += 1
        info = self._get_info()

        # Episode ends if all fish dead or time limit
        all_dead = not np.any(self.fish_alive)
        truncated = self.step_count >= self.max_steps
        terminated = all_dead

        # Survival bonus at end
        if truncated and not all_dead:
            rewards[self.fish_alive] += 5.0

        # Mean reward for PPO (treating swarm as single agent)
        mean_reward = float(np.mean(rewards))

        info["fish_rewards"] = rewards
        info["mean_reward"] = mean_reward
        info["deaths_this_step"] = int(np.sum(rewards == -5.0))

        return self._get_obs(), mean_reward, terminated, truncated, info


def make_flocking_env(**kwargs):
    return FlockingEnv(**kwargs)


if __name__ == "__main__":
    # Test the environment
    env = make_flocking_env(num_fish=30, num_neighbors=5)
    obs, info = env.reset(seed=42)

    print(f"Obs dim per fish: {env.obs_dim}")
    print(f"Total obs dim: {len(obs)}")
    print(f"Initial alive: {info['alive_count']}")

    # Run random actions
    total_reward = 0
    for step in range(200):
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward

        if step % 50 == 0:
            print(f"Step {step}: alive={info['alive_count']}, cohesion={info['school_cohesion']:.3f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            print(f"Final survival rate: {info['survival_rate']:.1%}")
            print(f"Total reward: {total_reward:.1f}")
            break
