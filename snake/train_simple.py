#!/usr/bin/env python3
"""
Simplified Snake AI Training - Focused on quick results
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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class SnakeEnv:
    """Simplified Snake environment"""
    
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()
        
    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, -1)
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
        """Enhanced state representation with more features"""
        head = self.snake[0]
        
        # Basic one-hot encoded states (12 features)
        state = []
        
        # Snake direction (4)
        dir_map = {(0, -1): [1,0,0,0], (1, 0): [0,1,0,0], 
                   (0, 1): [0,0,1,0], (-1, 0): [0,0,0,1]}
        state.extend(dir_map.get(self.direction, [0,0,0,0]))
        
        # Food direction (4)
        food_dir = [0, 0, 0, 0]
        if self.food[1] < head[1]: food_dir[0] = 1
        if self.food[0] > head[0]: food_dir[1] = 1
        if self.food[1] > head[1]: food_dir[2] = 1
        if self.food[0] < head[0]: food_dir[3] = 1
        state.extend(food_dir)
        
        # Danger in each direction (4)
        danger = []
        for dx, dy in [(0,-1), (1,0), (0,1), (-1,0)]:
            nx, ny = head[0] + dx, head[1] + dy
            if (nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or
                (nx, ny) in self.snake):
                danger.append(1)
            else:
                danger.append(0)
        state.extend(danger)
        
        # Additional features for better learning
        # Normalized distances
        state.append((self.food[0] - head[0]) / self.grid_size)
        state.append((self.food[1] - head[1]) / self.grid_size)
        
        # Snake length normalized
        state.append(len(self.snake) / (self.grid_size * self.grid_size))
        
        # Distance to walls normalized
        state.append(head[0] / self.grid_size)
        state.append(head[1] / self.grid_size)
        state.append((self.grid_size - 1 - head[0]) / self.grid_size)
        state.append((self.grid_size - 1 - head[1]) / self.grid_size)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        # Map action to direction
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.direction = directions[action]
        
        # Move head
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check collisions
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            return self._get_state(), -10, True
        
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
            # Small reward for surviving and getting closer to food
            dist_to_food = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = -0.01 + 0.1 * (1 / (dist_to_food + 1))
        
        # Check starvation
        if self.steps_since_food > self.grid_size * 5:
            return self._get_state(), -10, True
        
        return self._get_state(), reward, False


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SnakeAgent:
    def __init__(self, state_size=19, action_size=4, lr=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.gamma = 0.95
        self.batch_size = 64
        
        # Neural networks
        self.q_network = DQN(state_size, 512, action_size).to(device)
        self.target_network = DQN(state_size, 512, action_size).to(device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.steps = 0
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.LongTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % 1000 == 0:
            self.update_target_network()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train(episodes=10000):
    env = SnakeEnv()
    agent = SnakeAgent()
    
    scores = deque(maxlen=100)
    best_avg = 0
    
    output_dir = Path("snake_model")
    output_dir.mkdir(exist_ok=True)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        scores.append(env.score)
        avg_score = np.mean(scores)
        
        # Train
        if len(agent.memory) > agent.batch_size:
            agent.replay()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Score: {env.score}, Avg: {avg_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
            
        if episode % 1000 == 0 and avg_score > best_avg:
            best_avg = avg_score
            torch.save({
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'episode': episode,
                'avg_score': avg_score
            }, output_dir / 'best_model.pth')
            print(f"Saved new best model with avg score: {avg_score:.2f}")
    
    # Save final model
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'state_size': agent.state_size,
        'action_size': agent.action_size
    }, output_dir / 'final_model.pth')


def export_simple_model():
    """Export the trained model for TensorFlow.js"""
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    import tensorflowjs as tfjs
    
    # Load model
    checkpoint = torch.load('snake_model/best_model.pth', map_location=device)
    model = DQN(19, 512, 4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 19)
    torch.onnx.export(model, dummy_input, 'snake_model.onnx',
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                  'output': {0: 'batch_size'}})
    
    # Convert to TensorFlow
    onnx_model = onnx.load('snake_model.onnx')
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('snake_tf_model')
    
    # Convert to TensorFlow.js
    tfjs.converters.convert_tf_saved_model('snake_tf_model', 'snake_tfjs_model')
    
    # Save model info
    with open('snake_tfjs_model/model_info.json', 'w') as f:
        json.dump({
            'state_size': 19,
            'action_size': 4,
            'input_size': 12,  # For compatibility with web app
            'enhanced_features': 7
        }, f)
    
    print("Model exported successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        export_simple_model()
    else:
        train(episodes=5000)  # Start with fewer episodes