#!/usr/bin/env python3
"""
Optimized Snake AI Training - Fast convergence to high scores
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
from pathlib import Path
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class SnakeEnv:
    """Snake environment with improved features"""
    
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()
        
    def reset(self):
        # Start snake in center
        center = self.grid_size // 2
        self.snake = [(center, center)]
        
        # Random initial direction
        self.direction = random.choice([(0, -1), (1, 0), (0, 1), (-1, 0)])
        
        self.food = self._generate_food()
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        
        # For reward shaping
        self.prev_dist = self._manhattan_distance(self.snake[0], self.food)
        
        return self._get_state()
    
    def _generate_food(self):
        while True:
            food = (random.randint(0, self.grid_size - 1), 
                   random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food
    
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_state(self):
        """Get comprehensive state representation"""
        head = self.snake[0]
        
        # Get relative food position (normalized)
        rel_food_x = (self.food[0] - head[0]) / self.grid_size
        rel_food_y = (self.food[1] - head[1]) / self.grid_size
        
        # Check dangers in all 8 directions
        dangers = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = head[0] + dx, head[1] + dy
                if (nx < 0 or nx >= self.grid_size or 
                    ny < 0 or ny >= self.grid_size or 
                    (nx, ny) in self.snake):
                    dangers.append(1)
                else:
                    dangers.append(0)
        
        # Snake direction one-hot
        dir_map = {
            (0, -1): [1, 0, 0, 0],  # Up
            (1, 0):  [0, 1, 0, 0],  # Right
            (0, 1):  [0, 0, 1, 0],  # Down
            (-1, 0): [0, 0, 0, 1]   # Left
        }
        direction = dir_map[self.direction]
        
        # Body direction from head (if exists)
        body_dir = [0, 0, 0, 0]
        if len(self.snake) > 1:
            body = self.snake[1]
            if body[0] < head[0]: body_dir[3] = 1  # Body is left
            elif body[0] > head[0]: body_dir[1] = 1  # Body is right
            elif body[1] < head[1]: body_dir[0] = 1  # Body is up
            elif body[1] > head[1]: body_dir[2] = 1  # Body is down
        
        # Wall distances (normalized)
        wall_dist = [
            head[0] / self.grid_size,  # Distance to left wall
            (self.grid_size - 1 - head[0]) / self.grid_size,  # Right wall
            head[1] / self.grid_size,  # Top wall
            (self.grid_size - 1 - head[1]) / self.grid_size  # Bottom wall
        ]
        
        # Snake length (normalized)
        length = len(self.snake) / (self.grid_size * self.grid_size)
        
        # Combine all features
        state = [rel_food_x, rel_food_y] + dangers + direction + body_dir + wall_dist + [length]
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        # Prevent going backwards into body
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        new_dir = directions[action]
        
        # Check if new direction is opposite to current
        if len(self.snake) > 1:
            if (new_dir[0] == -self.direction[0] and 
                new_dir[1] == -self.direction[1]):
                # Keep current direction
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
            return self._get_state(), -10, True
        
        # Check self collision
        if new_head in self.snake:
            return self._get_state(), -10, True
        
        # Add new head
        self.snake.insert(0, new_head)
        self.steps += 1
        self.steps_since_food += 1
        
        # Calculate distance reward
        curr_dist = self._manhattan_distance(new_head, self.food)
        
        # Check food
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._generate_food()
            self.prev_dist = self._manhattan_distance(new_head, self.food)
            
            # Increasing reward based on score
            reward = 10 + self.score * 0.5
        else:
            self.snake.pop()
            
            # Distance-based reward
            if curr_dist < self.prev_dist:
                reward = 0.1
            else:
                reward = -0.1
            self.prev_dist = curr_dist
        
        # Check starvation
        max_steps = self.grid_size * self.grid_size * 0.5
        if self.steps_since_food > max_steps:
            return self._get_state(), -10, True
        
        return self._get_state(), reward, False


class ImprovedDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # Deeper network with batch normalization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        
        # Dueling architecture
        self.value_head = nn.Linear(hidden_size // 2, 1)
        self.advantage_head = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # Handle both single samples and batches
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        
        # Combine value and advantage (dueling DQN)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values.squeeze(0) if q_values.size(0) == 1 else q_values


class OptimizedAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.lr = 0.0005
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.batch_size = 128
        self.memory_size = 50000
        self.update_target_every = 500
        
        # Experience replay
        self.memory = deque(maxlen=self.memory_size)
        
        # Networks
        self.q_network = ImprovedDQN(state_size, 512, action_size).to(device)
        self.target_network = ImprovedDQN(state_size, 512, action_size).to(device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.steps = 0
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).to(device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        
        return np.argmax(q_values.cpu().numpy())
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare tensors
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (double DQN)
        next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        
        # Compute targets
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), targets.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_snake_ai(episodes=10000):
    """Train the snake AI with optimized settings"""
    
    # Setup
    env = SnakeEnv(grid_size=20)
    # Calculate actual state size from environment
    test_state = env.reset()
    state_size = len(test_state)
    action_size = 4
    agent = OptimizedAgent(state_size, action_size)
    
    # Metrics
    scores = deque(maxlen=100)
    max_scores = []
    avg_scores = []
    best_score = 0
    best_avg = 0
    
    # Output directory
    output_dir = Path("snake_ai_models")
    output_dir.mkdir(exist_ok=True)
    
    print("Starting training...")
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Remember
            agent.remember(state, action, reward, next_state, done)
            
            # Learn
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            state = next_state
            
            if done:
                break
        
        # Record metrics
        scores.append(env.score)
        avg_score = np.mean(scores)
        
        # Track best
        if env.score > best_score:
            best_score = env.score
        
        # Log progress
        if episode % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}, Score: {env.score}, Best: {best_score}, "
                  f"Avg: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}, "
                  f"Time: {elapsed:.1f}s")
            
            max_scores.append(best_score)
            avg_scores.append(avg_score)
        
        # Save best average model
        if avg_score > best_avg and episode > 100:
            best_avg = avg_score
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'best_score': best_score,
                'avg_score': avg_score,
                'epsilon': agent.epsilon,
                'state_size': state_size,
                'action_size': action_size
            }, output_dir / 'best_model.pth')
            print(f"-> Saved new best model! Avg score: {avg_score:.2f}")
        
        # Early stopping if we achieve good performance
        if avg_score > 50 and episode > 1000:
            print(f"Achieved excellent performance! Avg score: {avg_score:.2f}")
            break
    
    # Save final model
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'state_size': state_size,
        'action_size': action_size
    }, output_dir / 'final_model.pth')
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(0, len(max_scores) * 100, 100), max_scores)
    plt.title('Maximum Score Progress')
    plt.xlabel('Episode')
    plt.ylabel('Max Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(avg_scores) * 100, 100), avg_scores)
    plt.title('Average Score Progress (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_progress.png')
    
    print(f"\nTraining complete!")
    print(f"Best score: {best_score}")
    print(f"Best average: {best_avg:.2f}")
    print(f"Final average: {avg_score:.2f}")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_scores = []
    agent.epsilon = 0  # No exploration during testing
    
    for _ in range(100):
        state = env.reset()
        while True:
            action = agent.act(state, training=False)
            state, _, done = env.step(action)
            if done:
                test_scores.append(env.score)
                break
    
    print(f"Test results - Avg: {np.mean(test_scores):.2f}, "
          f"Max: {np.max(test_scores)}, Min: {np.min(test_scores)}")


if __name__ == "__main__":
    train_snake_ai(episodes=5000)