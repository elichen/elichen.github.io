#!/usr/bin/env python3
"""
Stream Q(λ) Training Script for CartPole
Based on "Streaming Deep Reinforcement Learning Finally Works" (arXiv:2410.14606)

Uses PyTorch with MPS (Apple Silicon) or CUDA acceleration.
Exports weights compatible with the TensorFlow.js web app.
"""

import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from pathlib import Path


# =============================================================================
# Device Selection
# =============================================================================

def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


# =============================================================================
# Network Architecture (matches TensorFlow.js version)
# =============================================================================

class LayerNorm(nn.Module):
    """LayerNorm without learnable parameters (matches paper)"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


class StreamingNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, num_actions=2, device='cpu'):
        super().__init__()
        self.device = device

        # Architecture: Dense -> LayerNorm -> LeakyReLU -> Dense -> LayerNorm -> LeakyReLU -> Dense
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.ln1 = LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ln2 = LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions, bias=True)

        self.leaky_relu = nn.LeakyReLU(0.01)

        self.to(device)
        self.sparse_init()

    def sparse_init(self):
        """SparseInit from the paper - 90% sparsity with LeCun bounds"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            with torch.no_grad():
                fan_in = layer.weight.shape[1]
                fan_out = layer.weight.shape[0]

                # LeCun uniform initialization
                bound = 1.0 / math.sqrt(fan_in)
                layer.weight.uniform_(-bound, bound)

                # Sparse mask - zero 90% of input dimensions (but keep at least 1)
                num_zeros = min(int(0.9 * fan_in), fan_in - 1)
                zero_indices = torch.randperm(fan_in)[:num_zeros]
                layer.weight[:, zero_indices] = 0

                # Zero bias
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
        """Export weights in format compatible with TensorFlow.js web app"""
        weights = {}

        # Map PyTorch layers to TF.js layer names
        layer_map = [
            (self.fc1, 'fc1'),
            (self.fc2, 'hidden'),
            (self.fc3, 'output')
        ]

        for layer, name in layer_map:
            # PyTorch: [out, in], TF.js: [in, out] - need to transpose
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
# ObGD Optimizer (Algorithm 3 from paper)
# =============================================================================

class ObGD:
    def __init__(self, params, lr=1.0, gamma=0.99, lambda_=0.8, kappa=2.0):
        self.params = list(params)
        self.lr = lr
        self.gamma = gamma
        self.lambda_ = lambda_
        self.kappa = kappa

        # Initialize eligibility traces
        self.traces = [torch.zeros_like(p) for p in self.params]

        self.last_stats = None

    def step(self, delta, grads, reset_after=False):
        """
        ObGD update step (matches JS implementation)
        delta: TD error (scalar)
        grads: list of gradients for each parameter
        reset_after: whether to reset traces AFTER update (on episode end or non-greedy action)
        """
        # Update traces: z = γλz + grad
        gamma_lambda = self.gamma * self.lambda_
        z_sum = 0.0

        for i, (trace, grad) in enumerate(zip(self.traces, grads)):
            trace.mul_(gamma_lambda).add_(grad)
            z_sum += trace.abs().sum().item()

        # ObGD step size calculation
        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * self.lr * self.kappa
        step_size = self.lr / dot_product if dot_product > 1 else self.lr

        # Update parameters: w = w - α * δ * z
        with torch.no_grad():
            for param, trace in zip(self.params, self.traces):
                param.add_(trace, alpha=-step_size * delta)

        # Reset traces AFTER the update (matches JS behavior)
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
# Normalization Wrappers
# =============================================================================

class RunningMeanStd:
    """Welford's online algorithm for computing running mean and variance"""
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
    """Wraps environment with observation normalization, reward scaling, and time info"""
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
        obs, _ = self.env.reset()
        self.episode_time = -0.5
        self.steps = 0
        self.episode_return = 0.0
        self.reward_trace = 0.0

        # Normalize observation and add time
        obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
        norm_obs = self.obs_normalizer.normalize(obs_tensor)

        # Add time info
        state = torch.cat([norm_obs, torch.tensor([self.episode_time], device=self.device)])
        return state

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.steps += 1
        self.episode_return += reward
        self.episode_time += 1.0 / self.time_limit

        # Update observation normalizer
        obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
        self.obs_normalizer.update(obs_tensor)
        norm_obs = self.obs_normalizer.normalize(obs_tensor)

        # Scale reward using return variance
        self.reward_trace = self.reward_trace * self.gamma * (0 if done else 1) + reward
        self.reward_normalizer.update(torch.tensor([self.reward_trace], device=self.device))

        var_sqrt = torch.sqrt(self.reward_normalizer.var + 1e-8).item()
        scaled_reward = reward / var_sqrt if var_sqrt > 0 else reward

        # Add time info
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
    def __init__(self, input_size=5, hidden_size=32, num_actions=2,
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
        """Epsilon-greedy action selection"""
        # Linear epsilon decay
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
        """Stream Q(λ) update (matches JS implementation)"""
        # Compute Q-values
        q_values = self.network(state.unsqueeze(0))
        selected_q = q_values[0, action]

        with torch.no_grad():
            next_q_values = self.network(next_state.unsqueeze(0))
            max_next_q = next_q_values.max(dim=1)[0]

            done_mask = 0.0 if done else 1.0
            td_target = reward + self.gamma * max_next_q * done_mask
            td_error = (td_target - selected_q).item()

        # Compute gradients (for the selected Q-value, negated for gradient ascent)
        self.network.zero_grad()
        (-selected_q).backward()

        grads = [p.grad.clone() for p in self.network.parameters()]

        # ObGD update - reset traces AFTER update on episode end or non-greedy action
        # This matches JS behavior: always do the update, then reset if needed
        self.optimizer.step(td_error, grads, reset_after=(done or is_non_greedy))


# =============================================================================
# Training Loop
# =============================================================================

def train(total_steps=150000, hidden_size=32, save_path='.', force_cpu=False):
    if force_cpu:
        device = torch.device("cpu")
        print("Using CPU (forced)")
    else:
        device = get_device()

    # Create environment (standard 12° limit for training)
    base_env = gym.make('CartPole-v1')
    env = NormalizedEnv(base_env, gamma=0.99, device=device)

    # Create agent
    agent = StreamQAgent(
        input_size=5,  # 4 obs + 1 time
        hidden_size=hidden_size,
        num_actions=2,
        gamma=0.99,
        lambda_=0.8,
        lr=1.0,
        kappa=2.0,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=total_steps,
        device=device
    )

    # Training stats
    episode_returns = []
    episode_lengths = []
    steps = 0
    episode_count = 0

    # Milestones
    first_100 = None
    first_200 = None
    first_500 = None

    print(f"\nTraining for {total_steps:,} steps...")
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
            if first_100 is None and ep_length >= 100:
                first_100 = (episode_count, steps)
            if first_200 is None and ep_length >= 200:
                first_200 = (episode_count, steps)
            if first_500 is None and ep_length >= 500:
                first_500 = (episode_count, steps)

            # Progress logging
            if episode_count % 100 == 0:
                recent = episode_lengths[-20:]
                avg_len = sum(recent) / len(recent)
                max_len = max(recent)
                elapsed = time.time() - start_time
                steps_per_sec = steps / elapsed
                print(f"Episode {episode_count:4d} | Steps: {steps:6d} | "
                      f"Avg: {avg_len:5.1f} | Max: {max_len:3d} | "
                      f"ε: {agent.epsilon:.3f} | {steps_per_sec:.0f} steps/s")

            state = env.reset()

    elapsed = time.time() - start_time

    # Final summary
    print("-" * 60)
    print(f"\nTraining completed in {elapsed:.1f}s ({steps/elapsed:.0f} steps/s)")
    print(f"Total episodes: {episode_count}")

    if first_100:
        print(f"First 100+ steps: Episode {first_100[0]} (step {first_100[1]})")
    if first_200:
        print(f"First 200+ steps: Episode {first_200[0]} (step {first_200[1]})")
    if first_500:
        print(f"First 500 steps:  Episode {first_500[0]} (step {first_500[1]})")

    # Final performance
    last_20 = episode_lengths[-20:]
    avg_final = sum(last_20) / len(last_20)
    perfect_count = sum(1 for l in last_20 if l >= 500)
    print(f"\nFinal 20 episodes: Avg={avg_final:.1f}, Perfect={perfect_count}/20")

    # Save weights
    save_path = Path(save_path)

    weights = agent.network.export_weights()
    weights_file = save_path / 'trained-weights.json'
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"\nWeights saved to: {weights_file}")

    norm_stats = env.export_normalization()
    norm_file = save_path / 'trained-normalization.json'
    with open(norm_file, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Normalization saved to: {norm_file}")

    return agent, env


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Stream Q(λ) agent')
    parser.add_argument('--steps', type=int, default=150000, help='Total training steps')
    parser.add_argument('--hidden', type=int, default=32, help='Hidden layer size')
    parser.add_argument('--cpu', action='store_true', help='Force CPU (faster for small networks)')
    args = parser.parse_args()

    # CPU is faster for this tiny network due to MPS overhead
    train(total_steps=args.steps, hidden_size=args.hidden, force_cpu=args.cpu or True)
