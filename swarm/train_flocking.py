"""
PPO Training for Predator Escape Flocking

Trains a shared policy network that all fish use to escape the predator.
Fish learn to school together through the predator confusion effect.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import json
import os
from flocking_env import make_flocking_env


class FishPolicy(nn.Module):
    """
    Shared policy network for all fish.
    Takes a single fish's observation and outputs action probabilities.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.actor(obs)

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)


class PPOFlockingTrainer:
    """PPO trainer for flocking environment with shared policy."""

    def __init__(
        self,
        num_envs: int = 16,
        num_fish: int = 40,
        num_neighbors: int = 5,
        max_steps: int = 300,
        # PPO hyperparameters
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.02,  # Higher entropy for exploration
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        hidden_dim: int = 64,
    ):
        self.num_envs = num_envs
        self.num_fish = num_fish
        self.num_neighbors = num_neighbors
        self.max_steps = max_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches

        # Create environments
        self.envs = [
            make_flocking_env(
                num_fish=num_fish,
                num_neighbors=num_neighbors,
                max_steps=max_steps,
            )
            for _ in range(num_envs)
        ]

        # Observation dimension per fish
        self.obs_dim = 2 + 2 + 2 + 1 + num_neighbors * 4  # 27 for K=5

        # 9 discrete actions
        self.num_actions = 9

        # Create policy network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = FishPolicy(self.obs_dim, self.num_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)

        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_survival = deque(maxlen=100)
        self.episode_cohesion = deque(maxlen=100)

        # Persistent state across rollouts
        self.current_obs = None
        self.current_alive = None
        self._init_envs()

    def _init_envs(self):
        """Initialize environment state."""
        all_obs = []
        all_alive = []
        for env in self.envs:
            flat_obs, _ = env.reset()
            bee_obs = flat_obs.reshape(self.num_fish, self.obs_dim)
            all_obs.append(bee_obs)
            all_alive.append(env.fish_alive.copy())
        self.current_obs = np.array(all_obs)
        self.current_alive = np.array(all_alive)

    def collect_rollouts(self, num_steps: int):
        """Collect experience from all environments."""

        obs_storage = []
        action_storage = []
        logprob_storage = []
        reward_storage = []
        done_storage = []
        value_storage = []
        alive_storage = []  # Track which fish are alive

        # Use persistent state (don't reset every rollout)
        all_obs = self.current_obs
        all_alive = self.current_alive

        for step in range(num_steps):
            # Flatten for batch processing
            flat_obs = all_obs.reshape(-1, self.obs_dim)
            obs_tensor = torch.FloatTensor(flat_obs).to(self.device)

            with torch.no_grad():
                actions, logprobs, _, values = self.policy.get_action_and_value(obs_tensor)

            actions_np = actions.cpu().numpy().reshape(self.num_envs, self.num_fish)
            logprobs_np = logprobs.cpu().numpy().reshape(self.num_envs, self.num_fish)
            values_np = values.cpu().numpy().reshape(self.num_envs, self.num_fish)

            # Store
            obs_storage.append(all_obs.copy())
            action_storage.append(actions_np)
            logprob_storage.append(logprobs_np)
            value_storage.append(values_np)
            alive_storage.append(all_alive.copy())

            # Step all environments
            rewards = []
            dones = []
            next_obs = []
            next_alive = []

            for i, env in enumerate(self.envs):
                flat_obs, mean_reward, terminated, truncated, info = env.step(actions_np[i])
                done = terminated or truncated

                # Use per-fish rewards
                if "fish_rewards" in info:
                    fish_rewards = info["fish_rewards"]
                else:
                    fish_rewards = np.full(self.num_fish, mean_reward)

                rewards.append(fish_rewards)
                dones.append(np.full(self.num_fish, done))

                if done:
                    # Log episode stats
                    self.episode_rewards.append(info.get("mean_reward", mean_reward) * self.max_steps)
                    self.episode_survival.append(info["survival_rate"])
                    self.episode_cohesion.append(info["school_cohesion"])

                    flat_obs, _ = env.reset()

                bee_obs = flat_obs.reshape(self.num_fish, self.obs_dim)
                next_obs.append(bee_obs)
                next_alive.append(env.fish_alive.copy())

            reward_storage.append(np.array(rewards))
            done_storage.append(np.array(dones))
            all_obs = np.array(next_obs)
            all_alive = np.array(next_alive)

        # Save state for next rollout
        self.current_obs = all_obs
        self.current_alive = all_alive

        # Get final values for GAE
        flat_obs = all_obs.reshape(-1, self.obs_dim)
        obs_tensor = torch.FloatTensor(flat_obs).to(self.device)
        with torch.no_grad():
            _, _, _, final_values = self.policy.get_action_and_value(obs_tensor)
        final_values_np = final_values.cpu().numpy().reshape(self.num_envs, self.num_fish)

        # Convert to numpy arrays
        obs_storage = np.array(obs_storage)
        action_storage = np.array(action_storage)
        logprob_storage = np.array(logprob_storage)
        reward_storage = np.array(reward_storage)
        done_storage = np.array(done_storage)
        value_storage = np.array(value_storage)
        alive_storage = np.array(alive_storage)

        # Compute advantages using GAE
        advantages = np.zeros_like(reward_storage)
        lastgaelam = 0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_values = final_values_np
                next_nonterminal = 1.0 - done_storage[t]
            else:
                next_values = value_storage[t + 1]
                next_nonterminal = 1.0 - done_storage[t]

            delta = reward_storage[t] + self.gamma * next_values * next_nonterminal - value_storage[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam

        returns = advantages + value_storage

        return obs_storage, action_storage, logprob_storage, advantages, returns, alive_storage

    def update(self, obs, actions, old_logprobs, advantages, returns, alive_mask):
        """Perform PPO update."""

        # Flatten everything
        batch_size = obs.shape[0] * obs.shape[1] * obs.shape[2]
        obs_flat = obs.reshape(batch_size, self.obs_dim)
        actions_flat = actions.reshape(batch_size)
        old_logprobs_flat = old_logprobs.reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)
        returns_flat = returns.reshape(batch_size)
        alive_flat = alive_mask.reshape(batch_size)

        # Only train on alive fish (dead fish have zero observations)
        valid_indices = np.where(alive_flat)[0]
        if len(valid_indices) == 0:
            return 0, 0, 0

        obs_flat = obs_flat[valid_indices]
        actions_flat = actions_flat[valid_indices]
        old_logprobs_flat = old_logprobs_flat[valid_indices]
        advantages_flat = advantages_flat[valid_indices]
        returns_flat = returns_flat[valid_indices]

        batch_size = len(valid_indices)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_flat).to(self.device)
        actions_tensor = torch.LongTensor(actions_flat).to(self.device)
        old_logprobs_tensor = torch.FloatTensor(old_logprobs_flat).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages_flat).to(self.device)
        returns_tensor = torch.FloatTensor(returns_flat).to(self.device)

        # Minibatch training
        minibatch_size = max(batch_size // self.num_minibatches, 1)
        indices = np.arange(batch_size)

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                mb_obs = obs_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_logprobs = old_logprobs_tensor[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]

                _, new_logprobs, entropy, new_values = self.policy.get_action_and_value(
                    mb_obs, mb_actions
                )

                # Policy loss
                log_ratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(log_ratio)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_values.squeeze() - mb_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return pg_loss.item(), v_loss.item(), entropy_loss.item()

    def train(self, total_timesteps: int, rollout_steps: int = 128, log_interval: int = 10):
        """Main training loop."""

        num_updates = total_timesteps // (rollout_steps * self.num_envs * self.num_fish)

        print(f"Training for {total_timesteps} timesteps ({num_updates} updates)")
        print(f"Device: {self.device}")
        print(f"Envs: {self.num_envs}, Fish: {self.num_fish}")
        print()

        for update in range(1, num_updates + 1):
            obs, actions, logprobs, advantages, returns, alive_mask = self.collect_rollouts(rollout_steps)
            pg_loss, v_loss, ent_loss = self.update(obs, actions, logprobs, advantages, returns, alive_mask)

            if update % log_interval == 0:
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                mean_survival = np.mean(self.episode_survival) if self.episode_survival else 0
                mean_cohesion = np.mean(self.episode_cohesion) if self.episode_cohesion else 0

                timesteps = update * rollout_steps * self.num_envs * self.num_fish
                print(f"Update {update}/{num_updates} | "
                      f"Timesteps: {timesteps:,} | "
                      f"Survival: {mean_survival:.1%} | "
                      f"Cohesion: {mean_cohesion:.3f} | "
                      f"PG Loss: {pg_loss:.4f}")

        print("\nTraining complete!")
        return self.policy

    def save_model(self, path: str):
        """Save model weights as JSON for TensorFlow.js."""
        weights = {}

        for name, param in self.policy.named_parameters():
            weights[name] = param.detach().cpu().numpy().tolist()

        weights["_architecture"] = {
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "hidden_dim": 64,
        }

        with open(path, "w") as f:
            json.dump(weights, f)

        print(f"Model saved to {path}")

    def save_pytorch_model(self, path: str):
        """Save PyTorch model checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "obs_dim": self.obs_dim,
                "num_actions": self.num_actions,
                "num_fish": self.num_fish,
            }
        }, path)
        print(f"PyTorch model saved to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-fish", type=int, default=40)
    parser.add_argument("--num-neighbors", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=str, default="flocking_weights.json")
    args = parser.parse_args()

    trainer = PPOFlockingTrainer(
        num_envs=args.num_envs,
        num_fish=args.num_fish,
        num_neighbors=args.num_neighbors,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
    )

    trainer.train(total_timesteps=args.timesteps)

    trainer.save_model(args.output)
    trainer.save_pytorch_model("flocking_checkpoint.pt")


if __name__ == "__main__":
    main()
