#!/usr/bin/env python3
"""
Fast Snake AI Training with export to TensorFlow.js
Optimized for quick convergence and immediate web deployment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import json
import os
from pathlib import Path

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class SnakeEnvSimple:
    """Simple Snake environment matching the JS implementation exactly"""
    
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()
        
    def reset(self):
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = (0, -1)  # Up
        self.food = self._generate_food()
        self.score = 0
        self.steps_since_food = 0
        return self._get_state()
    
    def _generate_food(self):
        while True:
            food = (random.randint(0, self.grid_size - 1), 
                   random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        """Match the JavaScript state representation exactly (12 features)"""
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
        # North
        nx, ny = head[0], head[1] - 1
        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or (nx, ny) in self.snake:
            danger[0] = 1
        # South
        nx, ny = head[0], head[1] + 1
        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or (nx, ny) in self.snake:
            danger[1] = 1
        # East
        nx, ny = head[0] + 1, head[1]
        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or (nx, ny) in self.snake:
            danger[2] = 1
        # West
        nx, ny = head[0] - 1, head[1]
        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or (nx, ny) in self.snake:
            danger[3] = 1
        
        return np.array(snake_dir + food_dir + danger, dtype=np.float32)
    
    def step(self, action):
        # Map action to direction (matching JS exactly)
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        self.direction = directions[action]
        
        # Move head
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check collisions
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            return self._get_state(), -1, True  # Wall collision
        
        if new_head in self.snake:
            return self._get_state(), -1, True  # Self collision
        
        # Move snake
        self.snake.insert(0, new_head)
        self.steps_since_food += 1
        
        # Check food
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._generate_food()
            reward = 10
        else:
            self.snake.pop()
            reward = -0.01
        
        # Check starvation
        if self.steps_since_food >= self.grid_size * 2:  # Match JS
            return self._get_state(), -1, True
        
        return self._get_state(), reward, False


class SimpleDQN(nn.Module):
    """Simple DQN matching the web app architecture"""
    
    def __init__(self, input_size=12, hidden_size=256, output_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FastAgent:
    def __init__(self):
        self.input_size = 12
        self.hidden_size = 256
        self.output_size = 4
        
        # Networks
        self.q_network = SimpleDQN(self.input_size, self.hidden_size, self.output_size).to(device)
        self.target_network = SimpleDQN(self.input_size, self.hidden_size, self.output_size).to(device)
        self.update_target_network()
        
        # Training parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=100000)
        self.batch_size = 1000
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.995
        self.update_target_every = 100
        self.steps = 0
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.output_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.LongTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(device)
        
        # Double DQN
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Use online network to select actions
        next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        # Use target network to evaluate actions
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), targets.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target_network()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def export_to_tfjs_web(model, output_dir="web_model"):
    """Export model in a format compatible with the web app"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the state dict
    state_dict = model.state_dict()
    
    # Convert weights to JSON format
    weights = {}
    for name, tensor in state_dict.items():
        weights[name] = tensor.cpu().numpy().tolist()
    
    # Save as JSON (web app can load this directly)
    with open(f"{output_dir}/model_weights.json", "w") as f:
        json.dump(weights, f)
    
    # Also save model config
    with open(f"{output_dir}/model_config.json", "w") as f:
        json.dump({
            "input_size": 12,
            "hidden_size": 256,
            "output_size": 4,
            "architecture": "simple_dqn"
        }, f)
    
    print(f"Model exported to {output_dir}/")


def train_and_export():
    """Train and immediately export a working model"""
    
    env = SnakeEnvSimple(grid_size=20)
    agent = FastAgent()
    
    # Metrics
    scores = deque(maxlen=100)
    best_avg = 0
    
    print("Training Snake AI...")
    
    # Pre-fill memory with random experiences for better initial learning
    print("Collecting initial experiences...")
    for _ in range(5000):
        state = env.reset()
        for _ in range(random.randint(5, 50)):
            action = random.randrange(4)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    
    print("Starting training...")
    
    for episode in range(2000):  # Fewer episodes for faster training
        state = env.reset()
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Train every step for faster learning
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            if done:
                break
        
        scores.append(env.score)
        avg_score = np.mean(scores)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Score: {env.score}, Avg: {avg_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Save if improved
        if avg_score > best_avg and episode > 100:
            best_avg = avg_score
            export_to_tfjs_web(agent.q_network, "web_model")
            print(f"-> Exported model with avg score: {avg_score:.2f}")
    
    # Final export
    export_to_tfjs_web(agent.q_network, "web_model")
    
    # Test the model
    print("\nTesting trained model...")
    test_scores = []
    agent.epsilon = 0
    
    for _ in range(50):
        state = env.reset()
        steps = 0
        while steps < 1000:  # Prevent infinite loops
            action = agent.act(state, training=False)
            state, _, done = env.step(action)
            steps += 1
            if done:
                test_scores.append(env.score)
                break
    
    print(f"Test results - Avg: {np.mean(test_scores):.2f}, "
          f"Max: {np.max(test_scores)}, Min: {np.min(test_scores)}")
    
    print("\nModel exported to web_model/")
    print("Now updating the web app to load this model...")


if __name__ == "__main__":
    train_and_export()