#!/usr/bin/env python3
"""
Stream Q(λ) Training Script for CartPole Swingup
Based on "Streaming Deep Reinforcement Learning Finally Works" (arXiv:2410.14606)

The swingup task is harder than balancing:
- Pole starts hanging DOWN (θ = π)
- Agent must swing it up and balance
- Reward based on pole height: cos(θ) shifted to [0, 2]
"""

import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# =============================================================================
# CartPole Swingup Environment
# =============================================================================

class CartPoleSwingup:
    """CartPole Swingup - pole starts hanging down, must swing up and balance"""

    def __init__(self):
        # Physics constants (same as standard CartPole)
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.length = 0.5  # half pole length
        self.pole_mass_length = self.pole_mass * self.length
        self.force_mag = 10.0
        self.dt = 0.02

        # Boundaries
        self.x_limit = 2.4
        # No theta limit for swingup - pole can rotate freely

        self.max_steps = 500
        self.steps = 0
        self.state = None

    def reset(self):
        # Start with pole hanging DOWN (θ = π) with small random perturbation
        self.state = np.array([
            np.random.uniform(-0.05, 0.05),  # x
            0.0,                              # x_dot
            np.pi + np.random.uniform(-0.05, 0.05),  # theta (hanging down)
            0.0                               # theta_dot
        ], dtype=np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        self.steps += 1

        x, x_dot, theta, theta_dot = self.state

        # Force from action (0: left, 1: right)
        force = (-1 if action == 0 else 1) * self.force_mag

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Physics equations (same as standard CartPole)
        temp = (force + self.pole_mass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0/3.0 - self.pole_mass * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass

        # Euler integration
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc

        # Normalize theta to [-π, π]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # Reward: height of pole tip = cos(theta)
        # cos(0) = 1 (upright), cos(π) = -1 (down)
        # Shift to [0, 2] range: 1 + cos(theta)
        reward = 1.0 + np.cos(theta)

        # Episode ends if cart goes out of bounds or max steps
        done = abs(x) > self.x_limit or self.steps >= self.max_steps

        # Penalty for going out of bounds
        if abs(x) > self.x_limit:
            reward = 0.0

        return self.state.copy(), reward, done, {}


# =============================================================================
# Network Architecture (same as balance version)
# =============================================================================

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


class StreamingNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_actions=2, device='cpu'):
        super().__init__()
        self.device = device

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.ln1 = LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ln2 = LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions, bias=True)

        self.leaky_relu = nn.LeakyReLU(0.01)

        self.to(device)
        self.sparse_init()

    def sparse_init(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            with torch.no_grad():
                fan_in = layer.weight.shape[1]
                bound = 1.0 / math.sqrt(fan_in)
                layer.weight.uniform_(-bound, bound)
                num_zeros = min(int(0.9 * fan_in), fan_in - 1)
                zero_indices = torch.randperm(fan_in)[:num_zeros]
                layer.weight[:, zero_indices] = 0
                layer.bias.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        return x

    def export_weights(self):
        weights = {}
        layer_map = [
            (self.fc1, 'fc1'),
            (self.fc2, 'hidden'),
            (self.fc3, 'output')
        ]
        for layer, name in layer_map:
            kernel = layer.weight.detach().cpu().numpy().T
            bias = layer.bias.detach().cpu().numpy()
            weights[name] = {
                'kernel': kernel.flatten().tolist(),
                'kernelShape': list(kernel.shape),
                'bias': bias.tolist(),
                'biasShape': list(bias.shape)
            }
        return weights


# =============================================================================
# ObGD Optimizer
# =============================================================================

class ObGD:
    def __init__(self, params, lr=1.0, gamma=0.99, lambda_=0.8, kappa=2.0):
        self.params = list(params)
        self.lr = lr
        self.gamma = gamma
        self.lambda_ = lambda_
        self.kappa = kappa
        self.traces = [torch.zeros_like(p) for p in self.params]
        self.last_stats = None

    def step(self, delta, grads, reset_after=False):
        gamma_lambda = self.gamma * self.lambda_
        z_sum = 0.0

        for i, (trace, grad) in enumerate(zip(self.traces, grads)):
            trace.mul_(gamma_lambda).add_(grad)
            z_sum += trace.abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * self.lr * self.kappa
        step_size = self.lr / dot_product if dot_product > 1 else self.lr

        with torch.no_grad():
            for param, trace in zip(self.params, self.traces):
                param.add_(trace, alpha=-step_size * delta)

        if reset_after:
            for trace in self.traces:
                trace.zero_()

        self.last_stats = {
            'delta': delta,
            'delta_bar': delta_bar,
            'z_sum': z_sum,
            'step_size': step_size
        }


# =============================================================================
# Normalization
# =============================================================================

class RunningMeanStd:
    def __init__(self, shape, device='cpu'):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 0
        self.device = device
        self.frozen = False

    def update(self, x):
        if self.frozen:
            return
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        self.count += 1
        if self.count == 1:
            self.mean = x.clone()
            self.var = torch.zeros_like(x)
        else:
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

    def export(self):
        return {
            'mean': self.mean.cpu().numpy().tolist(),
            'var': self.var.cpu().numpy().tolist(),
            'count': self.count
        }


class NormalizedEnv:
    def __init__(self, env, gamma=0.99, device='cpu'):
        self.env = env
        self.gamma = gamma
        self.device = device

        self.obs_normalizer = RunningMeanStd((4,), device=device)
        self.reward_normalizer = RunningMeanStd((1,), device=device)
        self.reward_trace = 0.0

        self.time_limit = 500
        self.episode_time = -0.5
        self.steps = 0
        self.episode_return = 0.0

    def reset(self):
        obs = self.env.reset()
        self.episode_time = -0.5
        self.steps = 0
        self.episode_return = 0.0
        self.reward_trace = 0.0

        obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
        norm_obs = self.obs_normalizer.normalize(obs_tensor)
        state = torch.cat([norm_obs, torch.tensor([self.episode_time], device=self.device)])
        return state

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.steps += 1
        self.episode_return += reward
        self.episode_time += 1.0 / self.time_limit

        obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
        self.obs_normalizer.update(obs_tensor)
        norm_obs = self.obs_normalizer.normalize(obs_tensor)

        self.reward_trace = self.reward_trace * self.gamma * (0 if done else 1) + reward
        self.reward_normalizer.update(torch.tensor([self.reward_trace], device=self.device))

        var_sqrt = torch.sqrt(self.reward_normalizer.var + 1e-8).item()
        scaled_reward = reward / var_sqrt if var_sqrt > 0 else reward

        state = torch.cat([norm_obs, torch.tensor([self.episode_time], device=self.device)])

        return state, scaled_reward, done, {
            'episode': {'r': self.episode_return, 'steps': self.steps}
        }

    def export_normalization(self):
        return {
            'observation': self.obs_normalizer.export(),
            'reward': {
                'var': self.reward_normalizer.var.cpu().numpy().tolist(),
                'count': self.reward_normalizer.count
            }
        }


# =============================================================================
# Stream Q(λ) Agent
# =============================================================================

class StreamQAgent:
    def __init__(self, input_size=5, hidden_size=64, num_actions=2,
                 gamma=0.99, lambda_=0.8, lr=1.0, kappa=2.0,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=10000,
                 device='cpu'):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.step_count = 0

        self.network = StreamingNetwork(input_size, hidden_size, num_actions, device)
        self.optimizer = ObGD(self.network.parameters(), lr, gamma, lambda_, kappa)

    def select_action(self, state):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.step_count / self.epsilon_decay_steps
        )
        self.step_count += 1

        is_random = np.random.random() < self.epsilon

        if is_random:
            return np.random.randint(self.num_actions), True
        else:
            with torch.no_grad():
                q_values = self.network(state.unsqueeze(0))
                return q_values.argmax(dim=1).item(), False

    def update(self, state, action, reward, next_state, done, is_non_greedy):
        q_values = self.network(state.unsqueeze(0))
        selected_q = q_values[0, action]

        with torch.no_grad():
            next_q_values = self.network(next_state.unsqueeze(0))
            max_next_q = next_q_values.max(dim=1)[0]
            done_mask = 0.0 if done else 1.0
            td_target = reward + self.gamma * max_next_q * done_mask
            td_error = (td_target - selected_q).item()

        self.network.zero_grad()
        (-selected_q).backward()
        grads = [p.grad.clone() for p in self.network.parameters()]
        self.optimizer.step(td_error, grads, reset_after=(done or is_non_greedy))


# =============================================================================
# Training Loop
# =============================================================================

def train(total_steps=500000, hidden_size=64, save_path='.'):
    device = torch.device("cpu")
    print("Using CPU")

    env = NormalizedEnv(CartPoleSwingup(), gamma=0.99, device=device)

    agent = StreamQAgent(
        input_size=5,
        hidden_size=hidden_size,
        num_actions=2,
        gamma=0.99,
        lambda_=0.9,  # Higher lambda for longer credit assignment
        lr=1.0,
        kappa=2.0,
        epsilon_start=1.0,
        epsilon_end=0.001,  # Very low final epsilon for fine control
        epsilon_decay_steps=total_steps * 2 // 3,  # Decay faster to exploit longer
        device=device
    )

    episode_returns = []
    episode_lengths = []
    steps = 0
    episode_count = 0

    # Milestones for swingup (based on episode return, not length)
    # Max possible return per episode: 2.0 * 500 = 1000 (always upright)
    first_500 = None
    first_800 = None
    first_900 = None

    print(f"\nTraining CartPole Swingup for {total_steps:,} steps...")
    print("Reward: 1 + cos(θ), range [0, 2], max episode return = 1000")
    print("-" * 60)

    start_time = time.time()
    state = env.reset()

    while steps < total_steps:
        action, is_random = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.update(state, action, reward, next_state, done, is_random)

        state = next_state
        steps += 1

        if done:
            episode_count += 1
            ep_return = info['episode']['r']
            ep_length = info['episode']['steps']

            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)

            # Track milestones
            if first_500 is None and ep_return >= 500:
                first_500 = (episode_count, steps)
            if first_800 is None and ep_return >= 800:
                first_800 = (episode_count, steps)
            if first_900 is None and ep_return >= 900:
                first_900 = (episode_count, steps)

            if episode_count % 200 == 0:
                recent = episode_returns[-20:]
                avg_ret = sum(recent) / len(recent)
                max_ret = max(recent)
                elapsed = time.time() - start_time
                steps_per_sec = steps / elapsed
                print(f"Episode {episode_count:5d} | Steps: {steps:7d} | "
                      f"Avg Return: {avg_ret:6.1f} | Max: {max_ret:6.1f} | "
                      f"ε: {agent.epsilon:.3f} | {steps_per_sec:.0f} steps/s")

            state = env.reset()

    elapsed = time.time() - start_time

    print("-" * 60)
    print(f"\nTraining completed in {elapsed:.1f}s ({steps/elapsed:.0f} steps/s)")
    print(f"Total episodes: {episode_count}")

    if first_500:
        print(f"First 500+ return: Episode {first_500[0]} (step {first_500[1]})")
    if first_800:
        print(f"First 800+ return: Episode {first_800[0]} (step {first_800[1]})")
    if first_900:
        print(f"First 900+ return: Episode {first_900[0]} (step {first_900[1]})")

    # Final performance
    last_20 = episode_returns[-20:]
    avg_final = sum(last_20) / len(last_20)
    max_final = max(last_20)
    above_900 = sum(1 for r in last_20 if r >= 900)
    above_950 = sum(1 for r in last_20 if r >= 950)

    print(f"\nFinal 20 episodes: Avg={avg_final:.1f}, Max={max_final:.1f}")
    print(f"Episodes with 900+ return: {above_900}/20")
    print(f"Episodes with 950+ return: {above_950}/20")

    # Save weights
    save_path = Path(save_path)

    weights = agent.network.export_weights()
    weights_file = save_path / 'trained-weights-swingup.json'
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"\nWeights saved to: {weights_file}")

    norm_stats = env.export_normalization()
    norm_file = save_path / 'trained-normalization-swingup.json'
    with open(norm_file, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Normalization saved to: {norm_file}")

    return agent, env, episode_returns


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Stream Q(λ) for CartPole Swingup')
    parser.add_argument('--steps', type=int, default=500000, help='Total training steps')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden layer size')
    args = parser.parse_args()

    train(total_steps=args.steps, hidden_size=args.hidden)
