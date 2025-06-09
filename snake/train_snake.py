#!/usr/bin/env python3
"""
Advanced Snake AI Training Script
Using Rainbow DQN with PyTorch for optimal performance on Apple Silicon
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import json
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Enable MPS (Metal Performance Shaders) for M4 GPU acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

@dataclass
class Config:
    """Training configuration"""
    # Game settings
    grid_size: int = 20
    max_steps_without_food: int = 100
    
    # State/Action space
    state_size: int = 12  # Matching the web implementation
    action_size: int = 4
    
    # Network architecture
    hidden_sizes: List[int] = (512, 512, 256)  # Larger network for better learning
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 128
    memory_size: int = 200000
    gamma: float = 0.99
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50000
    
    # Advanced DQN features
    use_double_dqn: bool = True
    use_dueling_dqn: bool = True
    use_noisy_nets: bool = True
    use_prioritized_replay: bool = True
    
    # PER hyperparameters
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 100000
    
    # Training settings
    target_update_freq: int = 1000
    save_freq: int = 10000
    eval_freq: int = 5000
    num_eval_episodes: int = 100
    
    # Reward shaping
    reward_food: float = 10.0
    reward_death: float = -10.0
    reward_move: float = -0.01
    reward_closer_to_food: float = 0.1
    reward_away_from_food: float = -0.1


class SnakeEnv:
    """Snake environment matching the JavaScript implementation"""
    
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset the game and return initial state"""
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, -1)  # Start moving up
        self.food = self._generate_food()
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.game_over = False
        self.prev_distance = self._manhattan_distance(self.snake[0], self.food)
        return self._get_state()
    
    def _generate_food(self) -> Tuple[int, int]:
        """Generate food at random position not occupied by snake"""
        while True:
            food = (random.randint(0, self.grid_size - 1), 
                   random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_state(self) -> np.ndarray:
        """Get current state matching JavaScript implementation"""
        head = self.snake[0]
        
        # One-hot encode snake direction (NSEW)
        snake_dir = [0, 0, 0, 0]
        if self.direction == (0, -1): snake_dir[0] = 1  # North
        elif self.direction == (0, 1): snake_dir[1] = 1  # South
        elif self.direction == (1, 0): snake_dir[2] = 1  # East
        elif self.direction == (-1, 0): snake_dir[3] = 1  # West
        
        # One-hot encode food direction (NSEW)
        food_dir = [0, 0, 0, 0]
        if self.food[1] < head[1]: food_dir[0] = 1  # North
        elif self.food[1] > head[1]: food_dir[1] = 1  # South
        if self.food[0] > head[0]: food_dir[2] = 1  # East
        elif self.food[0] < head[0]: food_dir[3] = 1  # West
        
        # One-hot encode immediate danger (NSEW)
        danger = [0, 0, 0, 0]
        danger[0] = self._is_danger(head[0], head[1] - 1)  # North
        danger[1] = self._is_danger(head[0], head[1] + 1)  # South
        danger[2] = self._is_danger(head[0] + 1, head[1])  # East
        danger[3] = self._is_danger(head[0] - 1, head[1])  # West
        
        return np.array(snake_dir + food_dir + danger, dtype=np.float32)
    
    def _is_danger(self, x: int, y: int) -> int:
        """Check if position is dangerous (wall or snake body)"""
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return 1
        if (x, y) in self.snake:
            return 1
        return 0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return (next_state, reward, done, info)"""
        if self.game_over:
            return self._get_state(), 0, True, {}
        
        # Map action to direction
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        self.direction = directions[action]
        
        # Move snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.game_over = True
            return self._get_state(), -10, True, {"collision": "wall"}
        
        # Check self collision
        if new_head in self.snake:
            self.game_over = True
            return self._get_state(), -10, True, {"collision": "self"}
        
        # Move snake
        self.snake.insert(0, new_head)
        self.steps += 1
        self.steps_since_food += 1
        
        # Calculate reward with shaping
        reward = -0.01  # Small penalty for each move
        
        # Distance-based reward shaping
        curr_distance = self._manhattan_distance(new_head, self.food)
        if curr_distance < self.prev_distance:
            reward += 0.1  # Reward for getting closer
        elif curr_distance > self.prev_distance:
            reward -= 0.1  # Penalty for moving away
        self.prev_distance = curr_distance
        
        # Check food consumption
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.steps_since_food = 0
            self.food = self._generate_food()
            self.prev_distance = self._manhattan_distance(new_head, self.food)
        else:
            self.snake.pop()  # Remove tail if no food eaten
        
        # Check starvation
        if self.steps_since_food >= self.grid_size * 5:  # More lenient than JS version
            self.game_over = True
            return self._get_state(), -10, True, {"collision": "starvation"}
        
        return self._get_state(), reward, False, {}
    
    def render(self):
        """Simple text rendering for debugging"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place snake
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                grid[y][x] = 'H'  # Head
            else:
                grid[y][x] = 'B'  # Body
        
        # Place food
        grid[self.food[1]][self.food[0]] = 'F'
        
        # Print grid
        print("\n" + "=" * (self.grid_size + 2))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("=" * (self.grid_size + 2))
        print(f"Score: {self.score}, Steps: {self.steps}")


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """Dueling DQN with Noisy Networks"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: Tuple[int], use_noisy: bool = True):
        super().__init__()
        self.use_noisy = use_noisy
        
        # Shared layers
        layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes[:-1]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        self.shared = nn.Sequential(*layers)
        
        # Value stream
        if use_noisy:
            self.value = nn.Sequential(
                NoisyLinear(prev_size, hidden_sizes[-1]),
                nn.ReLU(),
                NoisyLinear(hidden_sizes[-1], 1)
            )
        else:
            self.value = nn.Sequential(
                nn.Linear(prev_size, hidden_sizes[-1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[-1], 1)
            )
        
        # Advantage stream
        if use_noisy:
            self.advantage = nn.Sequential(
                NoisyLinear(prev_size, hidden_sizes[-1]),
                nn.ReLU(),
                NoisyLinear(hidden_sizes[-1], action_size)
            )
        else:
            self.advantage = nn.Sequential(
                nn.Linear(prev_size, hidden_sizes[-1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[-1], action_size)
            )
    
    def forward(self, x):
        shared = self.shared(x)
        value = self.value(shared)
        advantage = self.advantage(shared)
        # Combine value and advantage streams
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        states = torch.FloatTensor([e[0] for e in experiences]).to(device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(device)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(device)
        dones = torch.FloatTensor([e[4] for e in experiences]).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self):
        return len(self.buffer)


class RainbowDQNAgent:
    """Rainbow DQN Agent combining multiple improvements"""
    
    def __init__(self, config: Config):
        self.config = config
        self.steps = 0
        
        # Networks
        self.q_network = DuelingDQN(
            config.state_size, 
            config.action_size, 
            config.hidden_sizes,
            config.use_noisy_nets
        ).to(device)
        
        self.target_network = DuelingDQN(
            config.state_size, 
            config.action_size, 
            config.hidden_sizes,
            config.use_noisy_nets
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        if config.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(config.memory_size, config.per_alpha)
        else:
            self.memory = deque(maxlen=config.memory_size)
        
        # Epsilon for exploration (not used if noisy nets are enabled)
        self.epsilon = config.epsilon_start
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy or noisy networks"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        if training and self.config.use_noisy_nets:
            # Noisy networks handle exploration
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            self.q_network.train()
            return q_values.argmax().item()
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.randrange(self.config.action_size)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        if self.config.use_prioritized_replay:
            self.memory.push(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if self.config.use_prioritized_replay:
            if len(self.memory) < self.config.batch_size:
                return
            
            # Calculate beta for importance sampling
            beta = min(1.0, self.config.per_beta_start + 
                      (1.0 - self.config.per_beta_start) * self.steps / self.config.per_beta_steps)
            
            states, actions, rewards, next_states, dones, indices, weights = \
                self.memory.sample(self.config.batch_size, beta)
        else:
            if len(self.memory) < self.config.batch_size:
                return
            
            batch = random.sample(self.memory, self.config.batch_size)
            states = torch.FloatTensor([e[0] for e in batch]).to(device)
            actions = torch.LongTensor([e[1] for e in batch]).to(device)
            rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
            next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
            dones = torch.FloatTensor([e[4] for e in batch]).to(device)
            weights = torch.ones_like(rewards).to(device)
        
        # Reset noise if using noisy networks
        if self.config.use_noisy_nets:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values using Double DQN
        if self.config.use_double_dqn:
            # Use policy network to select actions
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            # Use target network to evaluate actions
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_q_values = self.target_network(next_states).max(1)[0]
        
        # Compute target values
        target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values
        
        # Compute loss with importance sampling weights
        td_errors = current_q_values.squeeze(1) - target_q_values.detach()
        loss = (weights * td_errors.pow(2)).mean()
        
        # Update priorities if using PER
        if self.config.use_prioritized_replay:
            priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update epsilon
        if not self.config.use_noisy_nets:
            self.epsilon = max(
                self.config.epsilon_end,
                self.config.epsilon_start - self.steps / self.config.epsilon_decay_steps
            )
        
        # Update target network
        if self.steps % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps += 1
    
    def save(self, path: str):
        """Save model and training state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
            'config': self.config.__dict__
        }, path)
    
    def load(self, path: str):
        """Load model and training state"""
        checkpoint = torch.load(path, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']


def train_agent(config: Config, num_episodes: int = 10000):
    """Train the Snake AI agent"""
    env = SnakeEnv(config.grid_size)
    agent = RainbowDQNAgent(config)
    
    # Training metrics
    scores = deque(maxlen=100)
    best_avg_score = 0
    training_start = time.time()
    
    # Create output directory
    output_dir = Path("snake_ai_output")
    output_dir.mkdir(exist_ok=True)
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Select action
            action = agent.act(state, training=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train
            if len(agent.memory) > config.batch_size:
                agent.replay()
            
            state = next_state
            
            if done:
                break
        
        scores.append(env.score)
        avg_score = np.mean(scores)
        
        # Logging
        if episode % 100 == 0:
            elapsed = time.time() - training_start
            print(f"Episode {episode}, Score: {env.score}, Avg Score: {avg_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Steps: {agent.steps}, "
                  f"Time: {elapsed:.1f}s")
        
        # Evaluation
        if episode % config.eval_freq == 0 and episode > 0:
            eval_scores = evaluate_agent(agent, config.num_eval_episodes, config.grid_size)
            eval_avg = np.mean(eval_scores)
            eval_max = np.max(eval_scores)
            print(f"Evaluation - Avg: {eval_avg:.2f}, Max: {eval_max}, "
                  f"Perfect games: {sum(s >= 397 for s in eval_scores)}/{config.num_eval_episodes}")
            
            # Save best model
            if eval_avg > best_avg_score:
                best_avg_score = eval_avg
                agent.save(output_dir / "best_model.pth")
                print(f"New best model saved! Avg score: {eval_avg:.2f}")
        
        # Regular checkpoint
        if episode % config.save_freq == 0 and episode > 0:
            agent.save(output_dir / f"checkpoint_{episode}.pth")
    
    # Final save
    agent.save(output_dir / "final_model.pth")
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title("Training Scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    
    plt.subplot(1, 2, 2)
    window = 100
    avg_scores = [np.mean(list(scores)[max(0, i-window):i+1]) for i in range(len(scores))]
    plt.plot(avg_scores)
    plt.title("Average Score (100 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_progress.png")
    print(f"Training complete! Results saved to {output_dir}")


def evaluate_agent(agent: RainbowDQNAgent, num_episodes: int, grid_size: int) -> List[int]:
    """Evaluate agent performance"""
    env = SnakeEnv(grid_size)
    scores = []
    
    for _ in range(num_episodes):
        state = env.reset()
        while True:
            action = agent.act(state, training=False)
            state, _, done, _ = env.step(action)
            if done:
                scores.append(env.score)
                break
    
    return scores


def export_to_tfjs(model_path: str, output_dir: str):
    """Export PyTorch model to TensorFlow.js format"""
    import onnx
    import tf2onnx
    import tensorflow as tf
    import tensorflowjs as tfjs
    
    # Load PyTorch model
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint['config']
    
    # Recreate model
    model = DuelingDQN(
        config_dict['state_size'],
        config_dict['action_size'],
        config_dict['hidden_sizes'],
        config_dict['use_noisy_nets']
    )
    model.load_state_dict(checkpoint['q_network_state_dict'])
    model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, config_dict['state_size'])
    onnx_path = "snake_model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, 
                     input_names=['input'], output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    # Convert ONNX to TensorFlow
    onnx_model = onnx.load(onnx_path)
    tf_rep = tf2onnx.convert.from_onnx(onnx_model)
    tf_model = tf_rep[0]
    
    # Save as TensorFlow SavedModel
    tf.saved_model.save(tf_model, "snake_tf_model")
    
    # Convert to TensorFlow.js
    tfjs.converters.convert_tf_saved_model("snake_tf_model", output_dir)
    
    # Save config for web app
    with open(f"{output_dir}/model_config.json", "w") as f:
        json.dump({
            'state_size': config_dict['state_size'],
            'action_size': config_dict['action_size'],
            'hidden_sizes': config_dict['hidden_sizes'],
            'grid_size': config_dict['grid_size']
        }, f)
    
    print(f"Model exported to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Snake AI with Rainbow DQN")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")
    parser.add_argument("--export", action="store_true", help="Export best model to TensorFlow.js")
    parser.add_argument("--test", action="store_true", help="Test the trained model")
    args = parser.parse_args()
    
    # Create optimized config
    config = Config()
    
    if args.test:
        # Test existing model
        agent = RainbowDQNAgent(config)
        agent.load("snake_ai_output/best_model.pth")
        scores = evaluate_agent(agent, 100, config.grid_size)
        print(f"Test Results - Avg: {np.mean(scores):.2f}, Max: {np.max(scores)}, "
              f"Min: {np.min(scores)}, Perfect games: {sum(s >= 397 for s in scores)}/100")
    elif args.export:
        # Export model
        export_to_tfjs("snake_ai_output/best_model.pth", "snake_tfjs_model")
    else:
        # Train model
        train_agent(config, args.episodes)