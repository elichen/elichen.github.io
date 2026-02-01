#!/usr/bin/env python3
"""
Stream Q(lambda) Training Script for Double Pendulum Swingup
Based on "Streaming Deep Reinforcement Learning Finally Works" (arXiv:2410.14606)

The double pendulum swingup is significantly harder than single:
- Two linked segments both start hanging DOWN (theta1 = theta2 = pi)
- Agent must swing both up and balance
- Chaotic dynamics make credit assignment difficult
- Reward based on height of both segments
"""

import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# =============================================================================
# Double Pendulum on Cart Environment
# =============================================================================

class CartPoleDouble:
    """Double Pendulum on Cart - both segments start hanging down, must swing up and balance"""

    def __init__(self):
        # Physics constants
        self.gravity = 9.8
        self.cart_mass = 1.0    # M: cart mass
        self.m1 = 0.1           # m1: first segment mass
        self.m2 = 0.1           # m2: second segment mass
        self.L1 = 0.5           # L1: first segment half-length
        self.L2 = 0.5           # L2: second segment half-length
        self.force_mag = 10.0
        self.dt = 0.02

        # Boundaries
        self.x_limit = 2.4

        self.max_steps = 500
        self.steps = 0
        self.state = None

    def reset(self):
        # Start with both segments hanging DOWN (theta = pi) with small random perturbation
        self.state = np.array([
            np.random.uniform(-0.05, 0.05),          # x
            0.0,                                      # x_dot
            np.pi + np.random.uniform(-0.05, 0.05),  # theta1 (hanging down)
            0.0,                                      # theta1_dot
            np.pi + np.random.uniform(-0.05, 0.05),  # theta2 (hanging down)
            0.0                                       # theta2_dot
        ], dtype=np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        self.steps += 1

        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = self.state

        # Force from action (0: left, 1: right)
        F = (-1 if action == 0 else 1) * self.force_mag

        # Physics parameters
        g = self.gravity
        M = self.cart_mass
        m1 = self.m1
        m2 = self.m2
        l1 = self.L1
        l2 = self.L2

        # Trig functions
        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        c2 = np.cos(theta2)
        s2 = np.sin(theta2)
        c12 = np.cos(theta1 - theta2)
        s12 = np.sin(theta1 - theta2)

        # Full lengths
        L1full = 2 * l1
        L2full = 2 * l2

        # Mass matrix coefficients (derived from Lagrangian)
        d1 = M + m1 + m2
        d2 = (m1/2 + m2) * L1full
        d3 = m2 * L2full / 2
        d4 = (m1/3 + m2) * L1full * L1full
        d5 = m2 * L1full * L2full / 2
        d6 = m2 * L2full * L2full / 3

        # Build mass matrix
        M11, M12, M13 = d1, d2 * c1, d3 * c2
        M21, M22, M23 = d2 * c1, d4, d5 * c12
        M31, M32, M33 = d3 * c2, d5 * c12, d6

        # Build forcing vector (Coriolis/centrifugal + gravity + external force)
        f1 = F + d2 * theta1_dot**2 * s1 + d3 * theta2_dot**2 * s2
        f2 = d5 * theta2_dot**2 * s12 + (m1/2 + m2) * g * L1full * s1
        f3 = -d5 * theta1_dot**2 * s12 + m2 * g * L2full * s2 / 2

        # Solve 3x3 system using Cramer's rule
        det = (M11 * (M22 * M33 - M23 * M32)
             - M12 * (M21 * M33 - M23 * M31)
             + M13 * (M21 * M32 - M22 * M31))

        if abs(det) < 1e-10:
            # Fallback for singular matrix
            x_acc = F / d1
            theta1_acc = 0.0
            theta2_acc = 0.0
        else:
            x_acc = (f1 * (M22 * M33 - M23 * M32)
                   - M12 * (f2 * M33 - M23 * f3)
                   + M13 * (f2 * M32 - M22 * f3)) / det

            theta1_acc = (M11 * (f2 * M33 - M23 * f3)
                        - f1 * (M21 * M33 - M23 * M31)
                        + M13 * (M21 * f3 - f2 * M31)) / det

            theta2_acc = (M11 * (M22 * f3 - f2 * M32)
                        - M12 * (M21 * f3 - f2 * M31)
                        + f1 * (M21 * M32 - M22 * M31)) / det

        # Euler integration
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta1 = theta1 + self.dt * theta1_dot
        theta1_dot = theta1_dot + self.dt * theta1_acc
        theta2 = theta2 + self.dt * theta2_dot
        theta2_dot = theta2_dot + self.dt * theta2_acc

        # Normalize angles to [-pi, pi]
        theta1 = ((theta1 + np.pi) % (2 * np.pi)) - np.pi
        theta2 = ((theta2 + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([x, x_dot, theta1, theta1_dot, theta2, theta2_dot], dtype=np.float32)

        # Reward: height-based for both segments
        # cos(0) = 1 (upright), cos(pi) = -1 (down)
        # Range [0, 2] total: 1 + 0.5*cos(theta1) + 0.5*cos(theta2)
        reward = 1.0 + 0.5 * np.cos(theta1) + 0.5 * np.cos(theta2)

        # Episode ends if cart goes out of bounds or max steps
        done = abs(x) > self.x_limit or self.steps >= self.max_steps

        # Penalty for going out of bounds
        if abs(x) > self.x_limit:
            reward = 0.0

        return self.state.copy(), reward, done, {}


# =============================================================================
# Network Architecture
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
    def __init__(self, input_size=7, hidden_size=128, num_actions=2, device='cpu'):
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
    def __init__(self, params, lr=1.0, gamma=0.99, lambda_=0.95, kappa=2.0):
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

        # 6D observation for double pendulum
        self.obs_normalizer = RunningMeanStd((6,), device=device)
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
# Stream Q(lambda) Agent
# =============================================================================

class StreamQAgent:
    def __init__(self, input_size=7, hidden_size=128, num_actions=2,
                 gamma=0.99, lambda_=0.95, lr=1.0, kappa=2.0,
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

def train(total_steps=2000000, hidden_size=128, save_path='.'):
    device = torch.device("cpu")
    print("Using CPU")

    env = NormalizedEnv(CartPoleDouble(), gamma=0.99, device=device)

    agent = StreamQAgent(
        input_size=7,       # 6 obs + 1 time
        hidden_size=hidden_size,
        num_actions=2,
        gamma=0.99,
        lambda_=0.95,       # Higher lambda for longer credit assignment (chaotic system)
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

    # Milestones for double pendulum swingup
    # Max possible return per episode: 2.0 * 500 = 1000 (both segments always upright)
    first_400 = None
    first_600 = None
    first_800 = None

    print(f"\nTraining Double Pendulum Swingup for {total_steps:,} steps...")
    print("Reward: 1 + 0.5*cos(theta1) + 0.5*cos(theta2), range [0, 2], max episode return = 1000")
    print("This is significantly harder than single pendulum due to chaotic dynamics!")
    print("-" * 70)

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
            if first_400 is None and ep_return >= 400:
                first_400 = (episode_count, steps)
            if first_600 is None and ep_return >= 600:
                first_600 = (episode_count, steps)
            if first_800 is None and ep_return >= 800:
                first_800 = (episode_count, steps)

            if episode_count % 200 == 0:
                recent = episode_returns[-20:]
                avg_ret = sum(recent) / len(recent)
                max_ret = max(recent)
                elapsed = time.time() - start_time
                steps_per_sec = steps / elapsed
                print(f"Episode {episode_count:5d} | Steps: {steps:8d} | "
                      f"Avg Return: {avg_ret:6.1f} | Max: {max_ret:6.1f} | "
                      f"eps: {agent.epsilon:.3f} | {steps_per_sec:.0f} steps/s")

            state = env.reset()

    elapsed = time.time() - start_time

    print("-" * 70)
    print(f"\nTraining completed in {elapsed:.1f}s ({steps/elapsed:.0f} steps/s)")
    print(f"Total episodes: {episode_count}")

    if first_400:
        print(f"First 400+ return: Episode {first_400[0]} (step {first_400[1]:,})")
    if first_600:
        print(f"First 600+ return: Episode {first_600[0]} (step {first_600[1]:,})")
    if first_800:
        print(f"First 800+ return: Episode {first_800[0]} (step {first_800[1]:,})")

    # Final performance
    last_20 = episode_returns[-20:]
    avg_final = sum(last_20) / len(last_20)
    max_final = max(last_20)
    above_600 = sum(1 for r in last_20 if r >= 600)
    above_800 = sum(1 for r in last_20 if r >= 800)

    print(f"\nFinal 20 episodes: Avg={avg_final:.1f}, Max={max_final:.1f}")
    print(f"Episodes with 600+ return: {above_600}/20")
    print(f"Episodes with 800+ return: {above_800}/20")

    # Save weights
    save_path = Path(save_path)

    weights = agent.network.export_weights()
    weights_file = save_path / 'trained-weights-double.json'
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"\nWeights saved to: {weights_file}")

    norm_stats = env.export_normalization()
    norm_file = save_path / 'trained-normalization-double.json'
    with open(norm_file, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Normalization saved to: {norm_file}")

    return agent, env, episode_returns


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Stream Q(lambda) for Double Pendulum Swingup')
    parser.add_argument('--steps', type=int, default=2000000, help='Total training steps (default: 2M)')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden layer size (default: 128)')
    args = parser.parse_args()

    train(total_steps=args.steps, hidden_size=args.hidden)
