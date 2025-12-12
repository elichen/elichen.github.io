"""
PPO Training for Swarm Nest Selection

Trains a shared policy network that all bees use to make decisions.
Each bee observes its local neighborhood and decides which nest to prefer.
The collective reward is based on convergence to the best nest.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import json
import os
from swarm_env import make_swarm_env


class BeePolicy(nn.Module):
    """
    Shared policy network for all bees.
    Takes a single bee's observation and outputs action probabilities.
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


class PPOTrainer:
    """PPO trainer for swarm environment with shared policy."""

    def __init__(
        self,
        num_envs: int = 16,
        num_bees: int = 30,
        num_nests: int = 2,
        num_neighbors: int = 3,
        quality_ratio: float = 2.0,
        max_steps: int = 50,
        # PPO hyperparameters
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        hidden_dim: int = 64,
    ):
        self.num_envs = num_envs
        self.num_bees = num_bees
        self.num_nests = num_nests
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
            make_swarm_env(
                num_bees=num_bees,
                num_nests=num_nests,
                num_neighbors=num_neighbors,
                quality_ratio=quality_ratio,
                max_steps=max_steps,
            )
            for _ in range(num_envs)
        ]

        # Observation dimension per bee (v3 - no direct quality):
        # nest_onehot + my_time + neighbor_nests + neighbor_confidence
        self.obs_dim = num_nests + 1 + num_neighbors * num_nests + num_neighbors

        # Create policy network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = BeePolicy(self.obs_dim, num_nests, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)

        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_prop_at_best = deque(maxlen=100)

    def collect_rollouts(self, num_steps: int):
        """Collect experience from all environments."""

        # Storage - shape: (num_steps, num_envs, num_bees, ...)
        obs_storage = []
        action_storage = []
        logprob_storage = []
        reward_storage = []
        done_storage = []
        value_storage = []

        # Reset all environments
        all_obs = []
        for env in self.envs:
            flat_obs, _ = env.reset()
            # Reshape to (num_bees, obs_dim)
            bee_obs = flat_obs.reshape(self.num_bees, self.obs_dim)
            all_obs.append(bee_obs)

        # all_obs shape: (num_envs, num_bees, obs_dim)
        all_obs = np.array(all_obs)

        for step in range(num_steps):
            # Flatten for batch processing: (num_envs * num_bees, obs_dim)
            flat_obs = all_obs.reshape(-1, self.obs_dim)
            obs_tensor = torch.FloatTensor(flat_obs).to(self.device)

            with torch.no_grad():
                actions, logprobs, _, values = self.policy.get_action_and_value(obs_tensor)

            # Reshape back: (num_envs, num_bees)
            actions_np = actions.cpu().numpy().reshape(self.num_envs, self.num_bees)
            logprobs_np = logprobs.cpu().numpy().reshape(self.num_envs, self.num_bees)
            values_np = values.cpu().numpy().reshape(self.num_envs, self.num_bees)

            # Store
            obs_storage.append(all_obs.copy())
            action_storage.append(actions_np)
            logprob_storage.append(logprobs_np)
            value_storage.append(values_np)

            # Step all environments
            rewards = []
            dones = []
            next_obs = []

            for i, env in enumerate(self.envs):
                flat_obs, mean_reward, terminated, truncated, info = env.step(actions_np[i])
                done = terminated or truncated

                # Use per-bee rewards if available, otherwise broadcast mean reward
                if "bee_rewards" in info:
                    bee_rewards = info["bee_rewards"]
                else:
                    bee_rewards = np.full(self.num_bees, mean_reward)

                rewards.append(bee_rewards)
                dones.append(np.full(self.num_bees, done))

                if done:
                    # Log episode stats
                    self.episode_rewards.append(info.get("mean_reward", mean_reward) * self.max_steps)
                    self.episode_prop_at_best.append(info["prop_at_best"])

                    # Reset
                    flat_obs, _ = env.reset()

                bee_obs = flat_obs.reshape(self.num_bees, self.obs_dim)
                next_obs.append(bee_obs)

            reward_storage.append(np.array(rewards))
            done_storage.append(np.array(dones))
            all_obs = np.array(next_obs)

        # Get final values for GAE
        flat_obs = all_obs.reshape(-1, self.obs_dim)
        obs_tensor = torch.FloatTensor(flat_obs).to(self.device)
        with torch.no_grad():
            _, _, _, final_values = self.policy.get_action_and_value(obs_tensor)
        final_values_np = final_values.cpu().numpy().reshape(self.num_envs, self.num_bees)

        # Convert to numpy arrays
        obs_storage = np.array(obs_storage)  # (num_steps, num_envs, num_bees, obs_dim)
        action_storage = np.array(action_storage)  # (num_steps, num_envs, num_bees)
        logprob_storage = np.array(logprob_storage)  # (num_steps, num_envs, num_bees)
        reward_storage = np.array(reward_storage)  # (num_steps, num_envs, num_bees)
        done_storage = np.array(done_storage)  # (num_steps, num_envs, num_bees)
        value_storage = np.array(value_storage)  # (num_steps, num_envs, num_bees)

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

        return obs_storage, action_storage, logprob_storage, advantages, returns

    def update(self, obs, actions, old_logprobs, advantages, returns):
        """Perform PPO update."""

        # Flatten everything: (num_steps * num_envs * num_bees, ...)
        batch_size = obs.shape[0] * obs.shape[1] * obs.shape[2]
        obs_flat = obs.reshape(batch_size, self.obs_dim)
        actions_flat = actions.reshape(batch_size)
        old_logprobs_flat = old_logprobs.reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)
        returns_flat = returns.reshape(batch_size)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_flat).to(self.device)
        actions_tensor = torch.LongTensor(actions_flat).to(self.device)
        old_logprobs_tensor = torch.FloatTensor(old_logprobs_flat).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages_flat).to(self.device)
        returns_tensor = torch.FloatTensor(returns_flat).to(self.device)

        # Minibatch training
        minibatch_size = batch_size // self.num_minibatches
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

        num_updates = total_timesteps // (rollout_steps * self.num_envs * self.num_bees)

        print(f"Training for {total_timesteps} timesteps ({num_updates} updates)")
        print(f"Device: {self.device}")
        print(f"Envs: {self.num_envs}, Bees: {self.num_bees}, Nests: {self.num_nests}")
        print()

        for update in range(1, num_updates + 1):
            # Collect rollouts
            obs, actions, logprobs, advantages, returns = self.collect_rollouts(rollout_steps)

            # Update policy
            pg_loss, v_loss, ent_loss = self.update(obs, actions, logprobs, advantages, returns)

            # Logging
            if update % log_interval == 0:
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                mean_prop = np.mean(self.episode_prop_at_best) if self.episode_prop_at_best else 0

                timesteps = update * rollout_steps * self.num_envs * self.num_bees
                print(f"Update {update}/{num_updates} | "
                      f"Timesteps: {timesteps:,} | "
                      f"Reward: {mean_reward:.2f} | "
                      f"Prop@Best: {mean_prop:.1%} | "
                      f"PG Loss: {pg_loss:.4f} | "
                      f"V Loss: {v_loss:.4f}")

        print("\nTraining complete!")
        return self.policy

    def save_model(self, path: str):
        """Save model weights as JSON for TensorFlow.js."""
        weights = {}

        for name, param in self.policy.named_parameters():
            weights[name] = param.detach().cpu().numpy().tolist()

        # Also save architecture info
        weights["_architecture"] = {
            "obs_dim": self.obs_dim,
            "num_actions": self.num_nests,
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
                "num_nests": self.num_nests,
                "num_bees": self.num_bees,
            }
        }, path)
        print(f"PyTorch model saved to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-bees", type=int, default=30)
    parser.add_argument("--num-nests", type=int, default=2)
    parser.add_argument("--num-neighbors", type=int, default=3)
    parser.add_argument("--quality-ratio", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=str, default="model_weights.json")
    args = parser.parse_args()

    trainer = PPOTrainer(
        num_envs=args.num_envs,
        num_bees=args.num_bees,
        num_nests=args.num_nests,
        num_neighbors=args.num_neighbors,
        quality_ratio=args.quality_ratio,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
    )

    trainer.train(total_timesteps=args.timesteps)

    # Save model
    trainer.save_model(args.output)
    trainer.save_pytorch_model("model_checkpoint.pt")


if __name__ == "__main__":
    main()
