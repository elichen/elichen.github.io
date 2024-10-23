# breakout_dqn.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
from collections import deque
import time  # Add this import at the top of the file

class Game:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.paddle = {'width': 75, 'height': 10, 'x': width / 2 - 37.5, 'y': height - 20}
        self.ball = {'radius': 5, 'x': width / 2, 'y': height - 30, 'dx': 2, 'dy': -2}
        self.bricks = []
        self.score = 0
        self.game_over = False
        self.colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3']
        self.init_bricks()
        self.last_ball_y = self.ball['y']
        self.penalty_for_losing_ball = -1  # Change from 0 to -1
        self.reward_for_hitting_paddle = 0.1
        self.reward_for_breaking_brick = 1  # Add this line
        self.ball_hit_paddle = False

    def init_bricks(self):
        brick_row_count = 6
        brick_column_count = 13
        brick_width = 61
        brick_height = 20
        brick_padding = 2
        brick_offset_top = 30
        brick_offset_left = 0

        for r in range(brick_row_count):
            for c in range(brick_column_count):
                brick_x = c * (brick_width + brick_padding) + brick_offset_left
                brick_y = r * (brick_height + brick_padding) + brick_offset_top
                self.bricks.append({
                    'x': brick_x,
                    'y': brick_y,
                    'width': brick_width,
                    'height': brick_height,
                    'status': 1,
                    'color': self.colors[r % len(self.colors)]
                })

    def move_paddle(self, direction):
        speed = 15
        if direction == 'left' and self.paddle['x'] > 0:
            self.paddle['x'] -= speed
        elif direction == 'right' and self.paddle['x'] + self.paddle['width'] < self.width:
            self.paddle['x'] += speed
        self.paddle['x'] = max(0, min(self.width - self.paddle['width'], self.paddle['x']))

    def update(self):
        if self.game_over:
            return

        self.last_ball_y = self.ball['y']

        # Move the ball
        self.ball['x'] += self.ball['dx']
        self.ball['y'] += self.ball['dy']

        # Ball collision with walls
        if self.ball['x'] + self.ball['radius'] > self.width or self.ball['x'] - self.ball['radius'] < 0:
            self.ball['dx'] = -self.ball['dx']
        if self.ball['y'] - self.ball['radius'] < 0:
            self.ball['dy'] = -self.ball['dy']

        # Ball collision with paddle
        if (self.ball['y'] + self.ball['radius'] > self.paddle['y'] and
            self.ball['y'] - self.ball['radius'] < self.paddle['y'] + self.paddle['height'] and
            self.ball['x'] > self.paddle['x'] and
            self.ball['x'] < self.paddle['x'] + self.paddle['width']):
            # Only reverse direction if the ball is moving downward
            if self.ball['dy'] > 0:
                self.ball['dy'] = -self.ball['dy']
                
                # Add variation to ball direction based on paddle hit position
                hit_position = (self.ball['x'] - self.paddle['x']) / self.paddle['width']
                max_angle_offset = 1
                self.ball['dx'] = self.ball['dx'] + (hit_position - 0.5) * max_angle_offset
            
            self.ball_hit_paddle = True

        # Ball collision with bricks
        for brick in self.bricks:
            if brick['status'] == 1:
                if (self.ball['x'] > brick['x'] and
                    self.ball['x'] < brick['x'] + brick['width'] and
                    self.ball['y'] > brick['y'] and
                    self.ball['y'] < brick['y'] + brick['height']):
                    self.ball['dy'] = -self.ball['dy']
                    brick['status'] = 0
                    self.score += 1
                    if self.score == len(self.bricks):
                        self.game_over = True

        # Game over if ball touches bottom
        if self.ball['y'] + self.ball['radius'] > self.height:
            self.game_over = True

    def get_reward(self):
        reward = 0

        # Reward for breaking bricks
        if self.score > 0:
            reward += self.score * self.reward_for_breaking_brick
            self.score = 0

        # Reward for hitting the paddle
        if self.ball_hit_paddle:
            reward += self.reward_for_hitting_paddle
            self.ball_hit_paddle = False

        # Penalty for losing the ball
        if self.game_over:
            reward += self.penalty_for_losing_ball

        return reward

    def reset(self):
        self.paddle = {'width': 75, 'height': 10, 'x': self.width / 2 - 37.5, 'y': self.height - 20}
        
        angle = self.get_random_angle()
        speed = 4
        self.ball = {
            'radius': 5,
            'x': self.width / 2,
            'y': self.height / 2,
            'dx': np.cos(angle) * speed,
            'dy': abs(np.sin(angle) * speed)
        }

        self.bricks = []
        self.score = 0
        self.game_over = False
        self.init_bricks()
        self.last_ball_y = self.ball['y']
        self.ball_hit_paddle = False

    def get_random_angle(self):
        return np.random.uniform(210, 330) * np.pi / 180

    def get_state(self):
        state = [
            self.paddle['x'] / self.width,
            self.paddle['y'] / self.height,
            self.ball['x'] / self.width,
            self.ball['y'] / self.height
        ]
        
        for brick in self.bricks:
            if brick['status'] == 1:
                state.extend([brick['x'] / self.width, brick['y'] / self.height])
            else:
                state.extend([-1, -1])
        
        return np.array(state)

class DQNModel:
    def __init__(self, input_size, num_actions):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(num_actions, activation='linear')
        ])
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss='mse')

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def train(self, states, targets):
        return self.model.fit(states, targets, epochs=1, verbose=0)
    
class DQNAgent:
    def __init__(self, input_size, num_actions, batch_size=1000, memory_size=10000, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, fixed_epsilon_episodes=1000,
                 decay_epsilon_episodes=2000, target_update_episodes=10):
        self.input_size = input_size
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.fixed_epsilon_episodes = fixed_epsilon_episodes
        self.decay_epsilon_episodes = decay_epsilon_episodes
        self.epsilon = epsilon_start
        self.episode_count = 0
        self.target_update_episodes = target_update_episodes
        self.episodes_since_update = 0

        self.model = DQNModel(input_size, num_actions)
        self.target_model = DQNModel(input_size, num_actions)
        self.update_target_model()

        # Add this line to use deque for efficient memory management
        self.memory = deque(maxlen=memory_size)

    def act(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_values = self.model.predict(state[np.newaxis, :])[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state), action, reward, np.array(next_state), done))

    def update_target_model(self):
        self.target_model.model.set_weights(self.model.model.get_weights())

    def update_epsilon(self):
        if self.episode_count <= self.fixed_epsilon_episodes:
            self.epsilon = self.epsilon_start
        else:
            decay_progress = min(1, (self.episode_count - self.fixed_epsilon_episodes) / self.decay_epsilon_episodes)
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress
            )

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = self.sample_memory(self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

        current_q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        loss = self.model.train(states, targets).history['loss'][0]
        return loss

    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)

    def increment_episode(self):
        self.episode_count += 1
        self.episodes_since_update += 1
        self.update_epsilon()

        if self.episodes_since_update >= self.target_update_episodes:
            self.update_target_model()
            self.episodes_since_update = 0

def train_dqn(num_episodes, should_plot=False, plot_interval=100):
    game = Game()
    input_size = 2 + 2 + (6 * 13 * 2)
    agent = DQNAgent(input_size, 3)
    
    last_time = time.time()
    rewards_history = []
    loss_history = []
    epsilon_history = []
    
    # Create persistent figure and axes
    if should_plot:
        plt.ioff()  # Turn off interactive mode
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        
        # Offset the right axes
        ax3.spines['right'].set_position(('outward', 60))
        
        # Create lines for each plot
        line_reward, = ax.plot([], [], label='Reward', color='blue')
        line_loss, = ax2.plot([], [], label='Loss', color='red')
        line_epsilon, = ax3.plot([], [], label='Epsilon', color='green')
        
        # Set labels
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward', color='blue')
        ax2.set_ylabel('Loss', color='red')
        ax3.set_ylabel('Epsilon', color='green')
        
        # Add legends
        lines = [line_reward, line_loss, line_epsilon]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
    
    for episode in range(num_episodes):
        game.reset()
        state = game.get_state()
        total_reward = 0
        loss = 0
        
        while not game.game_over:
            action = agent.act(state)
            if action == 0:
                game.move_paddle('left')
            elif action == 1:
                game.move_paddle('right')
            
            game.update()
            next_state = game.get_state()
            reward = game.get_reward()
            total_reward += reward
            agent.remember(state, action, reward, next_state, game.game_over)
            state = next_state
            
        loss = agent.replay()
        agent.increment_episode()
        
        rewards_history.append(total_reward)
        loss_history.append(loss if loss is not None else 0)
        epsilon_history.append(agent.epsilon)
        
        if (episode + 1) % plot_interval == 0:
            current_time = time.time()
            interval_duration = current_time - last_time
            avg_reward = np.mean(rewards_history[-plot_interval:])
            
            if should_plot:
                window_size = 20
                
                if len(loss_history) >= window_size:
                    smoothed_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
                    
                    x_rewards = list(range(len(rewards_history)))
                    x_loss = list(range(window_size-1, len(loss_history)))
                    x_epsilon = list(range(len(epsilon_history)))
                    
                    # Update data
                    line_reward.set_data(x_rewards, rewards_history)
                    line_loss.set_data(x_loss, smoothed_loss)
                    line_epsilon.set_data(x_epsilon, epsilon_history)
                    
                    # Update limits
                    ax.relim()
                    ax2.relim()
                    ax3.relim()
                    ax.autoscale_view()
                    ax2.autoscale_view()
                    ax3.autoscale_view()
                    
                    # Update display
                    fig.canvas.draw()
                    clear_output(wait=True)
                    display(fig)
            
            print(f"Episode: {episode + 1}, Avg Score: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}, Loss: {loss if loss is not None else 'N/A'}, Time: {interval_duration:.2f}s")
            
            last_time = current_time
            tf.keras.backend.clear_session()
    
    if should_plot:
        plt.close(fig)
    
    return agent

# This part is not executed when the file is imported
if __name__ == "__main__":
    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    
    # Add a flag to control profiling
    ENABLE_PROFILING = False
    num_episodes = 1000

    if ENABLE_PROFILING:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        
    agent = train_dqn(num_episodes)
    
    if ENABLE_PROFILING:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(30)

