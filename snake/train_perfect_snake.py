#!/usr/bin/env python3
"""
Perfect Snake AI Training - Achieve 400 cells (full grid) completion
Uses advanced state representation, hybrid neural architecture, and curriculum learning
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
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Metrics tracking
class MetricsTracker:
    def __init__(self):
        self.episode_scores = []
        self.episode_lengths = []
        self.loss_history = []
        self.epsilon_history = []
        self.best_score = 0
        self.perfect_games = 0
        self.grid_fill_rates = []
        self.training_start_time = time.time()
        self.metrics_file = 'training_metrics.json'
        
    def log_episode(self, episode, score, length, loss, epsilon, grid_fill_rate):
        self.episode_scores.append(score)
        self.episode_lengths.append(length)
        self.loss_history.append(loss)
        self.epsilon_history.append(epsilon)
        self.grid_fill_rates.append(grid_fill_rate)
        
        if score > self.best_score:
            self.best_score = score
            logger.info(f"🎯 NEW BEST SCORE: {score} at episode {episode}")
            
        if score >= 399:
            self.perfect_games += 1
            logger.info(f"🏆 PERFECT GAME #{self.perfect_games} achieved at episode {episode}!")
            
        # Log every 100 episodes
        if episode % 100 == 0:
            avg_score = np.mean(self.episode_scores[-100:]) if len(self.episode_scores) >= 100 else np.mean(self.episode_scores)
            avg_length = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
            avg_fill = np.mean(self.grid_fill_rates[-100:]) if len(self.grid_fill_rates) >= 100 else np.mean(self.grid_fill_rates)
            
            logger.info(f"Episode {episode:5d} | Score: {score:3d} | Best: {self.best_score:3d} | "
                       f"Avg100: {avg_score:6.2f} | Length: {length:4d} | Fill: {grid_fill_rate:5.1f}% | "
                       f"Loss: {loss:8.4f} | ε: {epsilon:.3f}")
                       
        # Save metrics periodically
        if episode % 500 == 0:
            self.save_metrics()
            
    def save_metrics(self):
        metrics = {
            'episode_scores': self.episode_scores,
            'episode_lengths': self.episode_lengths,
            'loss_history': self.loss_history,
            'epsilon_history': self.epsilon_history,
            'grid_fill_rates': self.grid_fill_rates,
            'best_score': self.best_score,
            'perfect_games': self.perfect_games,
            'training_duration': time.time() - self.training_start_time
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def plot_progress(self, save_path='training_progress.png'):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Scores over time
        ax1.plot(self.episode_scores, alpha=0.6, linewidth=0.5)
        if len(self.episode_scores) > 100:
            rolling_avg = np.convolve(self.episode_scores, np.ones(100)/100, mode='valid')
            ax1.plot(range(99, len(self.episode_scores)), rolling_avg, 'r-', linewidth=2, label='100-episode avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.set_title('Training Scores')
        ax1.legend()
        ax1.grid(True)
        
        # Loss history
        ax2.plot(self.loss_history)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True)
        
        # Grid fill rates
        ax3.plot(self.grid_fill_rates)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Grid Fill %')
        ax3.set_title('Grid Fill Percentage')
        ax3.grid(True)
        
        # Epsilon decay
        ax4.plot(self.epsilon_history)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_title('Exploration Rate')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

metrics_tracker = MetricsTracker()
logger.info("🚀 Perfect Snake AI Training Started")


class EnhancedSnakeEnv:
    """Enhanced Snake environment with sophisticated state representation for perfect gameplay"""
    
    def __init__(self, grid_size=20, vision_radius=4):
        self.grid_size = grid_size
        self.vision_radius = vision_radius
        self.vision_size = 2 * vision_radius + 1
        self.max_cells = grid_size * grid_size
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        # Start snake in center
        center = self.grid_size // 2
        self.snake = [(center, center)]
        
        # Random initial direction
        self.direction = random.choice([(0, -1), (1, 0), (0, 1), (-1, 0)])
        
        self.food = self._generate_food()
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        
        # For advanced metrics
        self.prev_dist = self._manhattan_distance(self.snake[0], self.food)
        self.max_length_achieved = 1
        
        return self._get_enhanced_state()
    
    def _generate_food(self):
        """Generate food in random empty location"""
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake:
                    empty_cells.append((x, y))
        
        if not empty_cells:
            return None  # Game won - no empty cells
        
        return random.choice(empty_cells)
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        if pos2 is None:
            return 0
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_vision_grid(self, center_pos):
        """Get vision grid around a position"""
        x, y = center_pos
        vision = np.zeros((self.vision_size, self.vision_size, 4))  # 4 channels: empty, snake, food, wall
        
        for i in range(self.vision_size):
            for j in range(self.vision_size):
                world_x = x + i - self.vision_radius
                world_y = y + j - self.vision_radius
                
                # Wall check
                if (world_x < 0 or world_x >= self.grid_size or 
                    world_y < 0 or world_y >= self.grid_size):
                    vision[i, j, 3] = 1  # Wall
                elif (world_x, world_y) in self.snake:
                    vision[i, j, 1] = 1  # Snake body
                elif self.food and (world_x, world_y) == self.food:
                    vision[i, j, 2] = 1  # Food
                else:
                    vision[i, j, 0] = 1  # Empty space
        
        return vision.flatten()
    
    def _get_connectivity_metrics(self):
        """Analyze empty space connectivity to prevent trapping"""
        head = self.snake[0]
        empty_cells = set()
        
        # Find all empty cells
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake:
                    empty_cells.add((x, y))
        
        if not empty_cells:
            return [0, 0, 0, 0, 0]  # No empty cells
        
        # BFS to find connected components
        visited = set()
        components = []
        
        for cell in empty_cells:
            if cell not in visited:
                component = []
                queue = [cell]
                visited.add(cell)
                
                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = curr[0] + dx, curr[1] + dy
                        if (nx, ny) in empty_cells and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                
                components.append(component)
        
        # Calculate metrics
        num_components = len(components)
        largest_component_size = max(len(comp) for comp in components) if components else 0
        
        # Check if head is in largest component
        head_in_largest = head in (max(components, key=len) if components else [])
        
        # Calculate distances to furthest empty cell
        max_dist = 0
        if empty_cells:
            for cell in empty_cells:
                dist = self._manhattan_distance(head, cell)
                max_dist = max(max_dist, dist)
        
        # Check for dead ends (cells with only one neighbor)
        dead_ends = 0
        for cell in empty_cells:
            neighbors = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cell[0] + dx, cell[1] + dy
                if (nx, ny) in empty_cells or (nx, ny) == head:
                    neighbors += 1
            if neighbors == 1:
                dead_ends += 1
        
        return [
            num_components / 10.0,  # Normalized
            largest_component_size / self.max_cells,
            float(head_in_largest),
            max_dist / (self.grid_size * 2),
            dead_ends / 10.0
        ]
    
    def _get_hamiltonian_feasibility(self):
        """Check if Hamiltonian cycle is still possible"""
        # Simplified check: ensure we can still reach all corners
        head = self.snake[0]
        corners = [(0, 0), (0, self.grid_size-1), (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1)]
        
        reachable_corners = 0
        for corner in corners:
            if corner not in self.snake:
                # Simple reachability check
                if self._is_reachable(head, corner):
                    reachable_corners += 1
        
        return reachable_corners / 4.0
    
    def _is_reachable(self, start, target):
        """Check if target is reachable from start"""
        if target in self.snake:
            return False
        
        visited = set()
        queue = [start]
        visited.add(start)
        
        while queue:
            curr = queue.pop(0)
            if curr == target:
                return True
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr[0] + dx, curr[1] + dy
                if (0 <= nx < self.grid_size and 
                    0 <= ny < self.grid_size and 
                    (nx, ny) not in self.snake and 
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return False
    
    def _get_enhanced_state(self):
        """Get comprehensive state representation for perfect gameplay"""
        head = self.snake[0]
        
        # 1. Vision grid around head (9x9x4 = 324 features)
        vision_features = self._get_vision_grid(head)
        
        # 2. Basic directional information
        rel_food_x = 0
        rel_food_y = 0
        if self.food:
            rel_food_x = (self.food[0] - head[0]) / self.grid_size
            rel_food_y = (self.food[1] - head[1]) / self.grid_size
        
        # 3. Snake direction one-hot (4 features)
        dir_map = {
            (0, -1): [1, 0, 0, 0],  # Up
            (1, 0):  [0, 1, 0, 0],  # Right
            (0, 1):  [0, 0, 1, 0],  # Down
            (-1, 0): [0, 0, 0, 1]   # Left
        }
        direction_features = dir_map[self.direction]
        
        # 4. Tail information
        tail = self.snake[-1] if len(self.snake) > 1 else head
        tail_rel_x = (tail[0] - head[0]) / self.grid_size
        tail_rel_y = (tail[1] - head[1]) / self.grid_size
        
        # 5. Snake length information
        length_norm = len(self.snake) / self.max_cells
        progress = len(self.snake) / self.max_cells  # How close to winning
        
        # 6. Wall distances (4 features)
        wall_distances = [
            head[0] / self.grid_size,  # Distance to left wall
            (self.grid_size - 1 - head[0]) / self.grid_size,  # Right wall
            head[1] / self.grid_size,  # Top wall
            (self.grid_size - 1 - head[1]) / self.grid_size  # Bottom wall
        ]
        
        # 7. Connectivity metrics (5 features)
        connectivity_features = self._get_connectivity_metrics()
        
        # 8. Hamiltonian feasibility (1 feature)
        hamiltonian_feasible = self._get_hamiltonian_feasibility()
        
        # 9. Time-based features
        steps_norm = self.steps / 1000.0  # Normalized steps
        steps_since_food_norm = self.steps_since_food / 200.0  # Normalized starvation risk
        
        # Combine all features
        state = (vision_features.tolist() + 
                [rel_food_x, rel_food_y] +
                direction_features +
                [tail_rel_x, tail_rel_y, length_norm, progress] +
                wall_distances +
                connectivity_features +
                [hamiltonian_feasible, steps_norm, steps_since_food_norm])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """Take a step in the environment"""
        # Prevent going backwards into body
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        new_dir = directions[action]
        
        # Check if new direction is opposite to current
        if len(self.snake) > 1:
            if (new_dir[0] == -self.direction[0] and 
                new_dir[1] == -self.direction[1]):
                # Keep current direction (prevent suicide)
                pass
            else:
                self.direction = new_dir
        else:
            self.direction = new_dir
        
        # Move head
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            return self._get_enhanced_state(), -100, True
        
        # Check self collision
        if new_head in self.snake:
            return self._get_enhanced_state(), -100, True
        
        # Add new head
        self.snake.insert(0, new_head)
        self.steps += 1
        self.steps_since_food += 1
        
        # Calculate distance reward
        curr_dist = self._manhattan_distance(new_head, self.food) if self.food else 0
        
        # Check food consumption
        reward = 0
        if self.food and new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._generate_food()
            
            # Check if game is won (all cells filled)
            if self.food is None:
                # PERFECT GAME - ALL 400 CELLS FILLED!
                return self._get_enhanced_state(), 10000, True
            
            self.prev_dist = self._manhattan_distance(new_head, self.food)
            
            # Progressive reward based on length
            base_reward = 100
            length_bonus = len(self.snake) * 2  # Increasing reward for longer snake
            efficiency_bonus = max(0, 50 - self.steps_since_food) if self.steps_since_food < 50 else 0
            reward = base_reward + length_bonus + efficiency_bonus
            
            # Special bonuses for milestones
            if len(self.snake) >= 100:
                reward += 500  # Quarter filled
            if len(self.snake) >= 200:
                reward += 1000  # Half filled
            if len(self.snake) >= 300:
                reward += 2000  # Three quarters filled
            if len(self.snake) >= 350:
                reward += 5000  # Nearly perfect
            
        else:
            self.snake.pop()  # Remove tail if no food eaten
            
            # Distance-based reward
            if curr_dist < self.prev_dist:
                reward = 1.0  # Moving towards food
            else:
                reward = -0.5  # Moving away from food
            
            self.prev_dist = curr_dist
        
        # Connectivity penalty (prevent creating isolated spaces)
        connectivity_metrics = self._get_connectivity_metrics()
        num_components = connectivity_metrics[0] * 10  # Denormalize
        if num_components > 1:
            reward -= 10 * (num_components - 1)  # Penalty for fragmentation
        
        # Starvation check (adaptive based on snake length)
        max_steps_without_food = min(500, self.grid_size * len(self.snake))
        if self.steps_since_food > max_steps_without_food:
            return self._get_enhanced_state(), -50, True
        
        # Update max length achieved
        self.max_length_achieved = max(self.max_length_achieved, len(self.snake))
        
        return self._get_enhanced_state(), reward, False
    
    def get_state_size(self):
        """Get the size of the state representation"""
        return len(self._get_enhanced_state())
    
    def render_ascii(self):
        """Render the game state in ASCII for debugging"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place snake
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                grid[y][x] = 'H'  # Head
            else:
                grid[y][x] = 'S'  # Body
        
        # Place food
        if self.food:
            fx, fy = self.food
            grid[fy][fx] = 'F'
        
        # Print grid
        print(f"Score: {self.score}, Length: {len(self.snake)}, Steps: {self.steps}")
        for row in grid:
            print(''.join(row))
        print()


class SpatialAttentionModule(nn.Module):
    """Spatial attention mechanism for focusing on important regions"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class HybridSnakeDQN(nn.Module):
    """Hybrid CNN-LSTM-Attention network for perfect Snake gameplay"""
    
    def __init__(self, state_size, action_size, vision_size=9):
        super().__init__()
        
        self.vision_size = vision_size
        self.vision_channels = 4  # empty, snake, food, wall
        
        # Calculate feature sizes
        vision_features = vision_size * vision_size * self.vision_channels
        scalar_features = state_size - vision_features
        
        # CNN for spatial processing (vision grid)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.vision_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SpatialAttentionModule(32),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SpatialAttentionModule(128),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            
            nn.Flatten()
        )
        
        # Calculate CNN output size
        cnn_output_size = 256 * 2 * 2  # 1024
        
        # MLP for scalar features
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Feature fusion
        fusion_size = cnn_output_size + 256
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM for temporal dependencies
        self.lstm_hidden_size = 256
        self.lstm = nn.LSTM(512, self.lstm_hidden_size, batch_first=True, num_layers=2, dropout=0.3)
        
        # Attention mechanism for LSTM output
        self.attention = nn.MultiheadAttention(self.lstm_hidden_size, num_heads=8, dropout=0.1)
        
        # Dueling DQN heads
        self.value_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Hidden states for LSTM
        self.hidden_states = None
        
    def _parse_state(self, state):
        """Parse state into vision grid and scalar features"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.size(0)
        
        # Extract vision features
        vision_features = self.vision_size * self.vision_size * self.vision_channels
        vision_flat = state[:, :vision_features]
        vision_grid = vision_flat.view(batch_size, self.vision_channels, self.vision_size, self.vision_size)
        
        # Extract scalar features
        scalar_features = state[:, vision_features:]
        
        return vision_grid, scalar_features
    
    def forward(self, state, hidden_states=None):
        """Forward pass through the network"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.size(0)
        
        # Parse input
        vision_grid, scalar_features = self._parse_state(state)
        
        # CNN processing
        cnn_features = self.cnn(vision_grid)  # (batch_size, 1024)
        
        # Scalar feature processing
        scalar_processed = self.scalar_mlp(scalar_features)  # (batch_size, 256)
        
        # Feature fusion
        fused_features = torch.cat([cnn_features, scalar_processed], dim=1)
        fused_features = self.fusion(fused_features)  # (batch_size, 512)
        
        # LSTM processing (add sequence dimension)
        lstm_input = fused_features.unsqueeze(1)  # (batch_size, 1, 512)
        
        if hidden_states is not None:
            lstm_out, new_hidden = self.lstm(lstm_input, hidden_states)
        else:
            lstm_out, new_hidden = self.lstm(lstm_input)
        
        # Remove sequence dimension
        lstm_out = lstm_out.squeeze(1)  # (batch_size, lstm_hidden_size)
        
        # Self-attention (optional, can be skipped for single timestep)
        attended_features = lstm_out
        
        # Dueling DQN
        value = self.value_head(attended_features)
        advantage = self.advantage_head(attended_features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Store hidden states
        self.hidden_states = new_hidden
        
        return q_values.squeeze(0) if q_values.size(0) == 1 else q_values
    
    def reset_hidden(self):
        """Reset LSTM hidden states"""
        self.hidden_states = None


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer with importance sampling"""
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def beta_by_frame(self, frame_idx):
        """Calculate beta for importance sampling"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch with prioritized sampling"""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        """Update priorities based on TD errors"""
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.buffer)


class PerfectSnakeAgent:
    """Advanced agent for perfect Snake gameplay using hybrid architecture"""
    
    def __init__(self, state_size, action_size, vision_size=9):
        self.state_size = state_size
        self.action_size = action_size
        self.vision_size = vision_size
        
        # Hyperparameters for perfect gameplay
        self.lr = 0.0001
        self.gamma = 0.995  # Higher discount for long-term rewards
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995  # Slower decay for more exploration
        self.batch_size = 64  # Smaller batch for stable learning
        self.memory_size = 100000
        self.update_target_every = 1000  # Less frequent updates for stability
        
        # Networks
        self.q_network = HybridSnakeDQN(state_size, action_size, vision_size).to(device)
        self.target_network = HybridSnakeDQN(state_size, action_size, vision_size).to(device)
        self.update_target_network()
        
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(self.memory_size)
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(self.q_network.parameters(), 
                                    lr=self.lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)
        
        self.steps = 0
        
    def update_target_network(self):
        """Soft update of target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in prioritized replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
        
    def act(self, state, training=True):
        """Choose action using epsilon-greedy with network prediction"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).to(device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        
        return np.argmax(q_values.cpu().numpy())
    
    def replay(self):
        """Training using prioritized experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (Double DQN)
        next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        
        # Compute targets
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # TD errors for priority updates
        td_errors = torch.abs(current_q_values.squeeze() - targets.detach())
        
        # Weighted loss (importance sampling)
        loss = (weights * F.smooth_l1_loss(current_q_values.squeeze(), targets.detach(), reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        priorities = td_errors.detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities)
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()


class HamiltonianPathChecker:
    """Advanced path analysis for ensuring perfect Snake gameplay"""
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.max_cells = grid_size * grid_size
        
    def find_hamiltonian_cycle(self, snake_positions):
        """Find a Hamiltonian cycle that includes all current snake positions"""
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in snake_positions:
                    empty_cells.append((x, y))
        
        # For large grids, use heuristic approach
        if len(empty_cells) > 50:
            return self._heuristic_hamiltonian_check(snake_positions, empty_cells)
        
        # For smaller grids, try exact solution
        return self._exact_hamiltonian_check(snake_positions, empty_cells)
    
    def _heuristic_hamiltonian_check(self, snake_positions, empty_cells):
        """Heuristic check for Hamiltonian cycle feasibility"""
        if not empty_cells:
            return 1.0  # Perfect game achieved
        
        # Check if we can create a simple cycle
        head = snake_positions[0] if snake_positions else (0, 0)
        
        # Calculate minimum spanning tree of empty cells
        mst_score = self._minimum_spanning_tree_score(empty_cells, head)
        
        # Check for bottlenecks (cells that would cut the graph)
        bottleneck_score = self._check_bottlenecks(empty_cells, snake_positions)
        
        # Check parity (Hamiltonian cycles require even number of cells in some configurations)
        parity_score = self._check_parity(empty_cells)
        
        # Combine scores
        feasibility = (mst_score + bottleneck_score + parity_score) / 3.0
        return min(1.0, max(0.0, feasibility))
    
    def _minimum_spanning_tree_score(self, empty_cells, head):
        """Calculate MST score to estimate connectivity"""
        if not empty_cells:
            return 1.0
        
        # Build distance matrix
        all_cells = [head] + empty_cells
        n = len(all_cells)
        
        # Use Prim's algorithm approximation
        visited = set([0])  # Start from head
        total_cost = 0
        
        while len(visited) < n:
            min_edge = float('inf')
            next_node = -1
            
            for v in visited:
                for u in range(n):
                    if u not in visited:
                        cost = abs(all_cells[v][0] - all_cells[u][0]) + abs(all_cells[v][1] - all_cells[u][1])
                        if cost < min_edge:
                            min_edge = cost
                            next_node = u
            
            if next_node != -1:
                visited.add(next_node)
                total_cost += min_edge
        
        # Normalize score (lower cost = higher feasibility)
        max_possible_cost = n * (self.grid_size * 2)
        return 1.0 - (total_cost / max_possible_cost)
    
    def _check_bottlenecks(self, empty_cells, snake_positions):
        """Check for critical cells that could break connectivity"""
        if not empty_cells:
            return 1.0
        
        critical_cells = 0
        total_cells = len(empty_cells)
        
        for cell in empty_cells:
            # Check if removing this cell would break connectivity
            remaining_cells = [c for c in empty_cells if c != cell]
            
            if remaining_cells and not self._is_connected(remaining_cells):
                critical_cells += 1
        
        # Lower ratio of critical cells is better
        return 1.0 - (critical_cells / max(1, total_cells))
    
    def _is_connected(self, cells):
        """Check if a set of cells forms a connected component"""
        if not cells:
            return True
        
        visited = set()
        queue = [cells[0]]
        visited.add(cells[0])
        
        while queue:
            current = queue.pop(0)
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(cells)
    
    def _check_parity(self, empty_cells):
        """Check parity constraints for Hamiltonian cycles"""
        # In a grid, Hamiltonian cycles have specific parity requirements
        # This is a simplified check
        
        if not empty_cells:
            return 1.0
        
        # Check if the shape allows for Hamiltonian cycles
        # Rectangle grids with even dimensions typically work well
        min_x = min(cell[0] for cell in empty_cells)
        max_x = max(cell[0] for cell in empty_cells)
        min_y = min(cell[1] for cell in empty_cells)
        max_y = max(cell[1] for cell in empty_cells)
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        
        # Even dimensions are generally better for Hamiltonian cycles
        even_bonus = 0.2 if (width % 2 == 0 and height % 2 == 0) else 0.0
        
        return 0.6 + even_bonus  # Base score with bonus for even dimensions
    
    def _exact_hamiltonian_check(self, snake_positions, empty_cells):
        """Exact check for small grids (computationally expensive)"""
        # For very small number of empty cells, try to find actual path
        if len(empty_cells) <= 20:
            return self._backtrack_hamiltonian(empty_cells)
        
        # Fall back to heuristic for larger problems
        return self._heuristic_hamiltonian_check(snake_positions, empty_cells)
    
    def _backtrack_hamiltonian(self, cells):
        """Backtracking algorithm for exact Hamiltonian path"""
        if not cells:
            return 1.0
        
        def is_adjacent(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        
        def backtrack(path, remaining):
            if not remaining:
                # Check if we can form a cycle
                if len(path) > 2 and is_adjacent(path[0], path[-1]):
                    return True
                return False
            
            current = path[-1] if path else cells[0]
            
            for cell in remaining:
                if not path or is_adjacent(current, cell):
                    new_path = path + [cell]
                    new_remaining = [c for c in remaining if c != cell]
                    
                    if backtrack(new_path, new_remaining):
                        return True
            
            return False
        
        # Try starting from each cell
        for start_cell in cells[:min(5, len(cells))]:  # Limit attempts for performance
            remaining = [c for c in cells if c != start_cell]
            if backtrack([start_cell], remaining):
                return 1.0
        
        return 0.3  # Partial score if no perfect cycle found


class CurriculumManager:
    """Manages curriculum learning with progressive difficulty"""
    
    def __init__(self, start_size=5, target_size=20, episodes_per_stage=5000):
        self.start_size = start_size
        self.target_size = target_size
        self.episodes_per_stage = episodes_per_stage
        self.current_stage = 0
        
        # Define curriculum stages
        self.stages = []
        size = start_size
        while size <= target_size:
            self.stages.append({
                'grid_size': size,
                'max_episodes': episodes_per_stage,
                'success_threshold': 0.7,  # Need 70% success rate to advance
                'episodes_completed': 0,
                'success_rate': 0.0
            })
            size += 2 if size < 10 else 3  # Smaller increments for small grids
    
    def get_current_stage(self):
        """Get current curriculum stage"""
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return self.stages[-1]  # Stay at final stage
    
    def update_progress(self, success):
        """Update progress and check if we should advance to next stage"""
        if self.current_stage >= len(self.stages):
            return False
        
        stage = self.stages[self.current_stage]
        stage['episodes_completed'] += 1
        
        # Update rolling success rate (simple moving average over last 100 episodes)
        if hasattr(stage, 'recent_successes'):
            stage['recent_successes'].append(success)
            if len(stage['recent_successes']) > 100:
                stage['recent_successes'].pop(0)
        else:
            stage['recent_successes'] = [success]
        
        stage['success_rate'] = sum(stage['recent_successes']) / len(stage['recent_successes'])
        
        # Check if we should advance
        if (stage['episodes_completed'] >= stage['max_episodes'] or 
            (stage['success_rate'] >= stage['success_threshold'] and 
             stage['episodes_completed'] >= 1000)):  # Minimum episodes before advancing
            
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                print(f"Advanced to curriculum stage {self.current_stage + 1}: "
                      f"Grid size {self.stages[self.current_stage]['grid_size']}")
                return True
        
        return False
    
    def is_complete(self):
        """Check if curriculum is complete"""
        return self.current_stage >= len(self.stages) - 1


def train_perfect_snake(episodes=50000, save_interval=2000, test_interval=5000):
    """Train the perfect Snake AI with all advanced features"""
    
    # Initialize curriculum manager
    curriculum = CurriculumManager(start_size=5, target_size=20, episodes_per_stage=5000)
    
    # Setup environment with initial curriculum stage
    current_stage = curriculum.get_current_stage()
    env = EnhancedSnakeEnv(grid_size=current_stage['grid_size'])
    hamiltonian_checker = HamiltonianPathChecker(current_stage['grid_size'])
    
    # Get state size from environment
    test_state = env.reset()
    state_size = len(test_state)
    action_size = 4
    
    # Initialize agent
    agent = PerfectSnakeAgent(state_size, action_size, vision_size=9)
    
    # Metrics tracking
    scores = deque(maxlen=100)
    max_scores = []
    avg_scores = []
    perfect_games = 0
    best_score = 0
    best_avg = 0
    episode_losses = []
    
    # Output directory
    output_dir = Path("perfect_snake_models")
    output_dir.mkdir(exist_ok=True)
    
    print("Starting Perfect Snake AI Training...")
    print(f"Initial curriculum stage: Grid size {current_stage['grid_size']}")
    start_time = time.time()
    
    for episode in range(episodes):
        # Check if we need to update curriculum
        current_stage = curriculum.get_current_stage()
        if env.grid_size != current_stage['grid_size']:
            # Update environment for new grid size
            env = EnhancedSnakeEnv(grid_size=current_stage['grid_size'])
            hamiltonian_checker = HamiltonianPathChecker(current_stage['grid_size'])
            
            # Update agent's state size if needed
            new_state = env.reset()
            if len(new_state) != state_size:
                # Reinitialize agent for new state size
                state_size = len(new_state)
                old_agent = agent
                agent = PerfectSnakeAgent(state_size, action_size, vision_size=9)
                
                # Transfer some knowledge if possible (simplified transfer learning)
                # This would require more sophisticated implementation in practice
                
        # Reset environment and agent state
        state = env.reset()
        agent.q_network.reset_hidden()
        episode_reward = 0
        steps_taken = 0
        max_length_achieved = 1
        
        while True:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            episode_reward += reward
            steps_taken += 1
            max_length_achieved = max(max_length_achieved, len(env.snake))
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.replay()
            if loss > 0:
                episode_losses.append(loss)
            
            state = next_state
            
            if done:
                break
        
        # Record metrics
        final_score = env.score
        scores.append(final_score)
        avg_score = np.mean(scores)
        
        # Calculate additional metrics
        grid_fill_rate = (len(env.snake) / env.max_cells) * 100
        avg_loss = np.mean(episode_losses[-10:]) if episode_losses else 0.0
        
        # Check for perfect game
        is_perfect = final_score >= (env.max_cells - 1)  # All cells filled
        if is_perfect:
            perfect_games += 1
        
        # Update curriculum
        success = final_score >= (current_stage['grid_size'] * current_stage['grid_size'] * 0.3)  # 30% of grid filled
        curriculum.update_progress(success)
        
        # Track best scores
        if final_score > best_score:
            best_score = final_score
        
        # Log comprehensive metrics
        metrics_tracker.log_episode(
            episode=episode,
            score=final_score,
            length=steps_taken,
            loss=avg_loss,
            epsilon=agent.epsilon,
            grid_fill_rate=grid_fill_rate
        )
        
        # Additional detailed logging every 100 episodes
        if episode % 100 == 0:
            elapsed = time.time() - start_time
            current_lr = agent.optimizer.param_groups[0]['lr']
            
            logger.info(f"🎮 Episode {episode:6d} | "
                       f"Grid: {current_stage['grid_size']:2d}x{current_stage['grid_size']:2d} | "
                       f"Score: {final_score:3d} | "
                       f"Best: {best_score:3d} | "
                       f"Avg: {avg_score:5.1f} | "
                       f"Perfect: {perfect_games:3d} | "
                       f"ε: {agent.epsilon:.4f} | "
                       f"LR: {current_lr:.6f} | "
                       f"Loss: {avg_loss:.4f} | "
                       f"Time: {elapsed:.0f}s")
            
            max_scores.append(best_score)
            avg_scores.append(avg_score)
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            checkpoint = {
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'target_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'scheduler_state_dict': agent.scheduler.state_dict(),
                'best_score': best_score,
                'avg_score': avg_score,
                'perfect_games': perfect_games,
                'epsilon': agent.epsilon,
                'state_size': state_size,
                'action_size': action_size,
                'curriculum_stage': curriculum.current_stage,
                'scores': list(scores)
            }
            torch.save(checkpoint, output_dir / f'checkpoint_episode_{episode}.pth')
            
        # Save best model
        if avg_score > best_avg and episode > 1000:
            best_avg = avg_score
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'target_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'best_score': best_score,
                'avg_score': avg_score,
                'perfect_games': perfect_games,
                'epsilon': agent.epsilon,
                'state_size': state_size,
                'action_size': action_size,
                'grid_size': env.grid_size
            }, output_dir / 'best_model.pth')
            print(f"💾 Saved new best model! Avg score: {avg_score:.2f}")
        
        # Test agent periodically
        if episode % test_interval == 0 and episode > 0:
            test_perfect_agent(agent, env.grid_size, num_tests=20)
        
        # Early stopping for perfect performance
        if perfect_games >= 10 and env.grid_size == 20:
            print(f"🎉 Achieved perfect performance! {perfect_games} perfect games on 20x20 grid")
            break
        
        # Stop if curriculum is complete and we have good performance
        if curriculum.is_complete() and avg_score > 100 and episode > 20000:
            print(f"Curriculum complete with strong performance. Avg score: {avg_score:.2f}")
            break
    
    # Save final model
    final_model_path = output_dir / 'final_perfect_model.pth'
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'state_size': state_size,
        'action_size': action_size,
        'grid_size': env.grid_size,
        'final_score': best_score,
        'perfect_games': perfect_games,
        'episodes_trained': episode + 1
    }, final_model_path)
    
    logger.info(f"\n🏆 Training Complete!")
    logger.info(f"Episodes trained: {episode + 1}")
    logger.info(f"Best score achieved: {best_score}")
    logger.info(f"Perfect games (400 cells): {perfect_games}")
    logger.info(f"Final average score: {avg_score:.2f}")
    logger.info(f"Model saved to: {final_model_path}")
    
    # Final metrics save
    metrics_tracker.save_metrics()
    
    return agent, env


def test_perfect_agent(agent, grid_size, num_tests=10):
    """Test the agent's performance"""
    print(f"\n🧪 Testing agent on {grid_size}x{grid_size} grid...")
    
    test_env = EnhancedSnakeEnv(grid_size=grid_size)
    test_scores = []
    perfect_count = 0
    
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during testing
    
    for test_episode in range(num_tests):
        state = test_env.reset()
        agent.q_network.reset_hidden()
        
        while True:
            action = agent.act(state, training=False)
            state, reward, done = test_env.step(action)
            
            if done:
                test_scores.append(test_env.score)
                if test_env.score >= (test_env.max_cells - 1):
                    perfect_count += 1
                break
    
    agent.epsilon = original_epsilon  # Restore epsilon
    
    avg_test_score = np.mean(test_scores)
    max_test_score = np.max(test_scores)
    
    print(f"Test Results: Avg={avg_test_score:.1f}, Max={max_test_score}, Perfect={perfect_count}/{num_tests}")
    
    return avg_test_score, perfect_count


if __name__ == "__main__":
    # Train the perfect Snake AI
    logger.info("🚀 Starting Perfect Snake AI Training...")
    agent, env = train_perfect_snake(episodes=50000)
    
    # Save final metrics and generate plots
    logger.info("📊 Saving final training metrics and generating plots...")
    metrics_tracker.save_metrics()
    metrics_tracker.plot_progress('final_training_progress.png')
    
    # Final comprehensive test
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION")
    logger.info("="*60)
    
    for grid_size in [10, 15, 20]:
        test_perfect_agent(agent, grid_size, num_tests=50)
    
    logger.info("✅ Training completed! Model saved and ready for web deployment.")
    logger.info(f"📈 Training summary:")
    logger.info(f"   • Best Score: {metrics_tracker.best_score}")
    logger.info(f"   • Perfect Games: {metrics_tracker.perfect_games}")
    logger.info(f"   • Training Duration: {(time.time() - metrics_tracker.training_start_time)/3600:.2f} hours")
    logger.info(f"   • Final Avg Score: {np.mean(metrics_tracker.episode_scores[-100:]) if len(metrics_tracker.episode_scores) >= 100 else np.mean(metrics_tracker.episode_scores):.2f}")
    
    # Save final checkpoint with metrics
    final_checkpoint = {
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'training_complete': True,
        'final_metrics': {
            'best_score': metrics_tracker.best_score,
            'perfect_games': metrics_tracker.perfect_games,
            'training_duration': time.time() - metrics_tracker.training_start_time,
            'total_episodes': len(metrics_tracker.episode_scores)
        }
    }
    torch.save(final_checkpoint, 'perfect_snake_models/final_trained_model.pth')
    logger.info("💾 Final model checkpoint saved!")