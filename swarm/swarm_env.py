"""
Swarm Nest Selection Environment v4 - Dancing Signals

Bees discover nest quality by staying and "sampling" it.
After sampling, they broadcast a "dance intensity" proportional to quality.
Other bees can see neighbor dance intensities and nest choices.

The insight: bees should learn to follow neighbors who are dancing intensely.

Based on: "The Hive Mind is a Single Reinforcement Learning Agent" (arXiv:2410.17517)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict


class SwarmNestEnv(gym.Env):
    """
    Observation (per bee):
        - current_nest one-hot (2)
        - my_dance_intensity (1) - how excited I am about my nest
        - neighbor_nests one-hot (num_neighbors * 2)
        - neighbor_dance_intensities (num_neighbors)

    Dynamics:
        - When a bee stays at a nest, it "samples" and builds dance intensity
        - Dance intensity = time_at_nest * nest_quality (so good nests = higher intensity)
        - When a bee switches, intensity resets to 0

    The signal: neighbors at good nests will have HIGH dance intensity.
    """

    def __init__(
        self,
        num_bees: int = 30,
        num_nests: int = 2,
        num_neighbors: int = 5,
        max_steps: int = 100,
        quality_ratio: float = 2.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.num_bees = num_bees
        self.num_nests = num_nests
        self.num_neighbors = min(num_neighbors, num_bees - 1)
        self.max_steps = max_steps
        self.quality_ratio = quality_ratio
        self.render_mode = render_mode

        # Observation: nest_onehot + my_dance + neighbor_nests + neighbor_dances
        self.obs_dim = num_nests + 1 + num_neighbors * num_nests + num_neighbors

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(num_bees * self.obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete([num_nests] * num_bees)

        self.nest_qualities = None
        self.best_nest = None
        self.bee_preferences = None
        self.bee_time_at_nest = None
        self.bee_dance_intensity = None  # Quality signal they're broadcasting
        self.bee_positions = None
        self.step_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        self.nest_qualities = np.array([self.quality_ratio, 1.0])
        if self.np_random.random() < 0.5:
            self.nest_qualities = self.nest_qualities[::-1]
        self.best_nest = int(np.argmax(self.nest_qualities))

        self.bee_preferences = self.np_random.integers(0, self.num_nests, size=self.num_bees)
        self.bee_time_at_nest = np.zeros(self.num_bees)
        self.bee_dance_intensity = np.zeros(self.num_bees)
        self.bee_positions = self.np_random.random((self.num_bees, 2))

        self.step_count = 0
        return self._get_obs(), self._get_info()

    def _get_neighbors(self, bee_idx: int) -> np.ndarray:
        pos = self.bee_positions[bee_idx]
        distances = np.linalg.norm(self.bee_positions - pos, axis=1)
        distances[bee_idx] = np.inf
        return np.argsort(distances)[:self.num_neighbors]

    def _get_obs(self) -> np.ndarray:
        # Normalize dance intensities for observation
        max_dance = max(1.0, np.max(self.bee_dance_intensity))

        all_obs = []
        for i in range(self.num_bees):
            # My nest
            my_nest_oh = np.zeros(self.num_nests, dtype=np.float32)
            my_nest_oh[self.bee_preferences[i]] = 1.0

            # My dance intensity (normalized)
            my_dance = self.bee_dance_intensity[i] / max_dance

            # Neighbors
            neighbors = self._get_neighbors(i)

            neighbor_nests = np.zeros((self.num_neighbors, self.num_nests), dtype=np.float32)
            neighbor_dances = np.zeros(self.num_neighbors, dtype=np.float32)

            for j, n in enumerate(neighbors):
                neighbor_nests[j, self.bee_preferences[n]] = 1.0
                neighbor_dances[j] = self.bee_dance_intensity[n] / max_dance

            obs = np.concatenate([
                my_nest_oh,
                [my_dance],
                neighbor_nests.flatten(),
                neighbor_dances
            ])
            all_obs.append(obs)

        return np.concatenate(all_obs).astype(np.float32)

    def _get_info(self):
        counts = np.bincount(self.bee_preferences, minlength=self.num_nests)
        prop_at_best = counts[self.best_nest] / self.num_bees
        convergence = np.max(counts) / self.num_bees

        return {
            "prop_at_best": float(prop_at_best),
            "convergence": float(convergence),
            "counts": counts.tolist(),
            "best_nest": self.best_nest,
            "qualities": self.nest_qualities.tolist(),
            "mean_dance": float(np.mean(self.bee_dance_intensity)),
        }

    def step(self, actions: np.ndarray):
        old_prefs = self.bee_preferences.copy()
        self.bee_preferences = np.clip(actions, 0, self.num_nests - 1)

        # Update dance intensity
        for i in range(self.num_bees):
            if self.bee_preferences[i] == old_prefs[i]:
                # Stayed - dance intensity grows proportional to nest quality
                quality = self.nest_qualities[self.bee_preferences[i]]
                self.bee_time_at_nest[i] += 1
                # Dance intensity = accumulated quality signal
                self.bee_dance_intensity[i] = self.bee_time_at_nest[i] * quality
            else:
                # Switched - reset
                self.bee_time_at_nest[i] = 0
                self.bee_dance_intensity[i] = 0

        # Position drift
        self.bee_positions += self.np_random.normal(0, 0.02, self.bee_positions.shape)
        self.bee_positions = np.clip(self.bee_positions, 0, 1)

        self.step_count += 1
        info = self._get_info()

        bee_rewards = np.zeros(self.num_bees, dtype=float)

        for i in range(self.num_bees):
            neighbors = self._get_neighbors(i)

            # Find the highest-intensity dancer among neighbors
            best_dancer_idx = max(neighbors, key=lambda n: self.bee_dance_intensity[n])
            best_dancer_intensity = self.bee_dance_intensity[best_dancer_idx]
            best_dancer_nest = self.bee_preferences[best_dancer_idx]

            # REWARD SHAPING: bonus for matching the most intense dancer
            # This teaches: "follow whoever is dancing hardest"
            if best_dancer_intensity > 5:  # Only if dancer has some confidence
                if self.bee_preferences[i] == best_dancer_nest:
                    bee_rewards[i] += 0.5  # Matched the intense dancer!

            # Main reward: being at the objectively best nest
            if self.bee_preferences[i] == self.best_nest:
                bee_rewards[i] += 1.0

        # Collective bonus
        if info["prop_at_best"] > 0.85:
            bee_rewards += 1.0

        mean_reward = float(np.mean(bee_rewards))
        info["bee_rewards"] = bee_rewards
        info["mean_reward"] = mean_reward

        terminated = False
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), mean_reward, terminated, truncated, info


def make_swarm_env(**kwargs):
    return SwarmNestEnv(**kwargs)


if __name__ == "__main__":
    env = make_swarm_env(num_bees=20, num_neighbors=5, quality_ratio=2.0)
    obs, info = env.reset(seed=42)
    print(f"Obs dim per bee: {env.obs_dim}")
    print(f"Best nest: {info['best_nest']} (Q={env.nest_qualities[info['best_nest']]})")

    # Simulate and watch dance intensities
    for step in range(20):
        # All bees stay at their nest
        actions = env.bee_preferences.copy()
        obs, reward, _, _, info = env.step(actions)
        print(f"Step {step+1}: mean_dance={info['mean_dance']:.1f}")
