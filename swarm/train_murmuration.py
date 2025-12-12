"""
PPO Training for Starling Murmuration

Trains birds to flock together through local rewards for
cohesion, separation, and alignment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import json
from murmuration_env import make_murmuration_env


class BirdPolicy(nn.Module):
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


class PPOMurmurationTrainer:
    def __init__(
        self,
        num_envs: int = 16,
        num_birds: int = 50,
        num_neighbors: int = 7,
        max_steps: int = 200,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.05,  # Higher entropy for more varied behavior
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        hidden_dim: int = 64,
    ):
        self.num_envs = num_envs
        self.num_birds = num_birds
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

        self.envs = [
            make_murmuration_env(
                num_birds=num_birds,
                num_neighbors=num_neighbors,
                max_steps=max_steps,
            )
            for _ in range(num_envs)
        ]

        self.obs_dim = 2 + 2 + num_neighbors * 4  # velocity + attractor_dir + neighbors
        self.num_actions = 9

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = BirdPolicy(self.obs_dim, self.num_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)

        self.episode_cohesion = deque(maxlen=100)
        self.episode_alignment = deque(maxlen=100)

        self.current_obs = None
        self._init_envs()

    def _init_envs(self):
        all_obs = []
        for env in self.envs:
            flat_obs, _ = env.reset()
            bird_obs = flat_obs.reshape(self.num_birds, self.obs_dim)
            all_obs.append(bird_obs)
        self.current_obs = np.array(all_obs)

    def collect_rollouts(self, num_steps: int):
        obs_storage = []
        action_storage = []
        logprob_storage = []
        reward_storage = []
        done_storage = []
        value_storage = []

        all_obs = self.current_obs

        for step in range(num_steps):
            flat_obs = all_obs.reshape(-1, self.obs_dim)
            obs_tensor = torch.FloatTensor(flat_obs).to(self.device)

            with torch.no_grad():
                actions, logprobs, _, values = self.policy.get_action_and_value(obs_tensor)

            actions_np = actions.cpu().numpy().reshape(self.num_envs, self.num_birds)
            logprobs_np = logprobs.cpu().numpy().reshape(self.num_envs, self.num_birds)
            values_np = values.cpu().numpy().reshape(self.num_envs, self.num_birds)

            obs_storage.append(all_obs.copy())
            action_storage.append(actions_np)
            logprob_storage.append(logprobs_np)
            value_storage.append(values_np)

            rewards = []
            dones = []
            next_obs = []

            for i, env in enumerate(self.envs):
                flat_obs, mean_reward, terminated, truncated, info = env.step(actions_np[i])
                done = terminated or truncated

                if "bird_rewards" in info:
                    bird_rewards = info["bird_rewards"]
                else:
                    bird_rewards = np.full(self.num_birds, mean_reward)

                rewards.append(bird_rewards)
                dones.append(np.full(self.num_birds, done))

                if done:
                    self.episode_cohesion.append(info["cohesion"])
                    self.episode_alignment.append(info["alignment"])
                    flat_obs, _ = env.reset()

                bird_obs = flat_obs.reshape(self.num_birds, self.obs_dim)
                next_obs.append(bird_obs)

            reward_storage.append(np.array(rewards))
            done_storage.append(np.array(dones))
            all_obs = np.array(next_obs)

        self.current_obs = all_obs

        flat_obs = all_obs.reshape(-1, self.obs_dim)
        obs_tensor = torch.FloatTensor(flat_obs).to(self.device)
        with torch.no_grad():
            _, _, _, final_values = self.policy.get_action_and_value(obs_tensor)
        final_values_np = final_values.cpu().numpy().reshape(self.num_envs, self.num_birds)

        obs_storage = np.array(obs_storage)
        action_storage = np.array(action_storage)
        logprob_storage = np.array(logprob_storage)
        reward_storage = np.array(reward_storage)
        done_storage = np.array(done_storage)
        value_storage = np.array(value_storage)

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
        batch_size = obs.shape[0] * obs.shape[1] * obs.shape[2]
        obs_flat = obs.reshape(batch_size, self.obs_dim)
        actions_flat = actions.reshape(batch_size)
        old_logprobs_flat = old_logprobs.reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)
        returns_flat = returns.reshape(batch_size)

        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        obs_tensor = torch.FloatTensor(obs_flat).to(self.device)
        actions_tensor = torch.LongTensor(actions_flat).to(self.device)
        old_logprobs_tensor = torch.FloatTensor(old_logprobs_flat).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages_flat).to(self.device)
        returns_tensor = torch.FloatTensor(returns_flat).to(self.device)

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

                _, new_logprobs, entropy, new_values = self.policy.get_action_and_value(mb_obs, mb_actions)

                log_ratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(log_ratio)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_values.squeeze() - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return pg_loss.item(), v_loss.item(), entropy_loss.item()

    def train(self, total_timesteps: int, rollout_steps: int = 128, log_interval: int = 10):
        num_updates = total_timesteps // (rollout_steps * self.num_envs * self.num_birds)

        print(f"Training for {total_timesteps} timesteps ({num_updates} updates)")
        print(f"Device: {self.device}")
        print(f"Envs: {self.num_envs}, Birds: {self.num_birds}")
        print()

        for update in range(1, num_updates + 1):
            obs, actions, logprobs, advantages, returns = self.collect_rollouts(rollout_steps)
            pg_loss, v_loss, ent_loss = self.update(obs, actions, logprobs, advantages, returns)

            if update % log_interval == 0:
                mean_cohesion = np.mean(self.episode_cohesion) if self.episode_cohesion else 0
                mean_alignment = np.mean(self.episode_alignment) if self.episode_alignment else 0

                timesteps = update * rollout_steps * self.num_envs * self.num_birds
                print(f"Update {update}/{num_updates} | "
                      f"Timesteps: {timesteps:,} | "
                      f"Cohesion: {mean_cohesion:.3f} | "
                      f"Alignment: {mean_alignment:.2f} | "
                      f"PG Loss: {pg_loss:.4f}")

        print("\nTraining complete!")
        return self.policy

    def save_model(self, path: str):
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-birds", type=int, default=50)
    parser.add_argument("--num-neighbors", type=int, default=7)
    parser.add_argument("--output", type=str, default="murmuration_weights.json")
    args = parser.parse_args()

    trainer = PPOMurmurationTrainer(
        num_envs=args.num_envs,
        num_birds=args.num_birds,
        num_neighbors=args.num_neighbors,
    )

    trainer.train(total_timesteps=args.timesteps)
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
