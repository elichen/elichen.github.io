# breakout_dqn.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
from collections import deque
import time  # Add this import at the top of the file
import psutil
import pandas as pd

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
        self.reward_for_hitting_paddle = 0
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

        # Small time penalty to encourage faster solutions
        reward -= 0.001  # Add small penalty per step

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
        # Include paddle, ball, and brick information
        state = [
            self.paddle['x'] / self.width,    # Normalized paddle x position
            self.ball['x'] / self.width,      # Normalized ball x position
            self.ball['y'] / self.height,     # Normalized ball y position
        ]
        
        # Add normalized brick positions and status
        for brick in self.bricks:
            if brick['status'] == 1:
                state.extend([
                    brick['x'] / self.width,   # Normalized brick x position
                    brick['y'] / self.height,  # Normalized brick y position
                ])
            else:
                state.extend([-1, -1])
        
        return np.array(state)

class DQNModel:
    def __init__(self, input_size, num_actions):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(input_size,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(num_actions, activation='linear')
        ])
        
        # Custom Huber loss that only considers the taken actions
        def masked_huber_loss(y_true, y_pred):
            error = y_true - y_pred
            # Create mask where y_true != y_pred (where we updated values)
            mask = tf.cast(tf.not_equal(y_true, y_pred), tf.float32)
            # Apply Huber loss only to the masked values
            quadratic = tf.minimum(tf.abs(error), 1.0)
            linear = tf.abs(error) - quadratic
            loss = 0.5 * quadratic**2 + linear
            return tf.reduce_mean(loss * mask)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.00025),
            loss=masked_huber_loss
        )

    def predict(self, state):
        return self.model(state)

    def train(self, states, targets):
        return self.model.fit(states, targets, epochs=1, verbose=0)
    
class DQNAgent:
    def __init__(self, 
                 batch_size=32,
                 memory_size=1000000,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 fixed_epsilon_steps=2000,
                 decay_epsilon_steps=4000,
                 target_update_steps=1000):
        # Create a game instance to determine input size
        game = Game()
        self.input_size = len(game.get_state())
        self.num_actions = 3  # Left, Right, or No action
        
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.fixed_epsilon_steps = fixed_epsilon_steps
        self.decay_epsilon_steps = decay_epsilon_steps
        self.epsilon = epsilon_start
        self.steps = 0
        self.target_update_steps = target_update_steps
        self.steps_since_update = 0

        self.model = DQNModel(self.input_size, self.num_actions)
        self.target_model = DQNModel(self.input_size, self.num_actions)
        self.update_target_model()
        self.memory = deque(maxlen=memory_size)

    def act(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_values = self.model.predict(state[np.newaxis, :])[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        # Simple memory addition
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.model.set_weights(self.model.model.get_weights())

    def update_epsilon(self):
        if self.steps <= self.fixed_epsilon_steps:
            self.epsilon = self.epsilon_start
        else:
            decay_progress = min(1, (self.steps - self.fixed_epsilon_steps) / self.decay_epsilon_steps)
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress
            )

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        current_q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)
        
        targets = current_q_values.numpy()
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        loss = self.model.train(states, targets).history['loss'][0]
        
        return loss

    def increment_step(self):
        self.steps += 1
        self.steps_since_update += 1
        self.update_epsilon()

        if self.steps_since_update >= self.target_update_steps:
            self.update_target_model()
            self.steps_since_update = 0

    def train(self, num_episodes=None, should_plot=False, plot_interval=1000):
        game = Game()
        last_time = time.time()
        rewards_history = []
        loss_history = []
        epsilon_history = []
        frame_count = 0
        training_count = 0
        episode = 0
        
        if should_plot:
            plt.ioff()
            fig = plt.figure(figsize=(12, 8))
            
            # Create two subplots
            ax1 = fig.add_subplot(211)  # Episode rewards subplot
            ax2 = fig.add_subplot(212)  # Training metrics subplot
            ax2_twin = ax2.twinx()
            
            # Episode rewards plot
            line_reward, = ax1.plot([], [], label='Episode Reward', color='blue')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward', color='blue')
            ax1.legend(loc='upper left')
            
            # Training metrics plot
            line_loss, = ax2.plot([], [], label='Loss', color='red')
            line_epsilon, = ax2_twin.plot([], [], label='Epsilon', color='green')
            
            ax2.set_xlabel('Training step')
            ax2.set_ylabel('Loss', color='red')
            ax2_twin.set_ylabel('Epsilon', color='green')
            
            # Combine legends for second subplot
            lines2 = [line_loss, line_epsilon]
            labels2 = [l.get_label() for l in lines2]
            ax2.legend(lines2, labels2, loc='upper left')
            
            plt.tight_layout()
        
        while True:  # Run indefinitely if num_episodes is None
            if num_episodes is not None and episode >= num_episodes:
                break
            
            game.reset()
            state = game.get_state()
            total_reward = 0
            
            while not game.game_over:
                action = self.act(state)
                if action == 0:
                    game.move_paddle('left')
                elif action == 1:
                    game.move_paddle('right')
                
                game.update()
                next_state = game.get_state()
                reward = game.get_reward()
                total_reward += reward
                
                self.remember(state, action, reward, next_state, game.game_over)
                
                # Train every 4 frames
                frame_count += 1
                if frame_count % 4 == 0:
                    loss = self.replay()
                    self.increment_step()
                    loss_history.append(loss if loss is not None else 0)
                    epsilon_history.append(self.epsilon)
                    training_count += 1
                    
                    if training_count % plot_interval == 0:
                        plot_window = 50
                        current_time = time.time()
                        interval_duration = current_time - last_time
                        avg_reward = np.mean(rewards_history[-plot_window:]) if rewards_history else 0
                        
                        if should_plot:
                            smoothed_loss = pd.Series(loss_history).rolling(window=plot_window).mean()
                            
                            x_rewards = list(range(len(rewards_history)))
                            x_loss = list(range(len(loss_history)))
                            x_epsilon = list(range(len(epsilon_history)))
                            
                            line_reward.set_data(x_rewards, rewards_history)
                            line_loss.set_data(x_loss, smoothed_loss)
                            line_epsilon.set_data(x_epsilon, epsilon_history)
                            
                            ax1.relim()
                            ax1.autoscale_view()
                            ax2.relim()
                            ax2_twin.relim()
                            ax2.autoscale_view()
                            ax2_twin.autoscale_view()
                            
                            fig.canvas.draw()
                            clear_output(wait=True)
                            display(fig)
                        
                        print(f"Training step: {training_count}, Game steps: {self.steps}, "
                              f"Episode: {episode + 1}, Avg Score: {avg_reward:.2f}, "
                              f"Epsilon: {self.epsilon:.4f}, Loss: {loss if loss is not None else 'N/A'}, "
                              f"Time: {interval_duration:.2f}s")
                        
                        last_time = current_time
                
                state = next_state
                
                # Check if all bricks are destroyed
                if game.score == len(game.bricks):
                    print(f"\nSolved! All bricks destroyed in episode {episode + 1}")
                    print(f"Final training step: {training_count}")
                    print(f"Final epsilon: {self.epsilon:.4f}")
                    rewards_history.append(total_reward)
                    if should_plot:
                        plt.close(fig)
                    return rewards_history, loss_history, epsilon_history
            
            rewards_history.append(total_reward)
            episode += 1  # Increment episode counter at the end of each episode
        
        if should_plot:
            plt.close(fig)
        
        return rewards_history, loss_history, epsilon_history

# This part is not executed when the file is imported
if __name__ == "__main__":
    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    
    ENABLE_PROFILING = True
    num_episodes = 2000
    
    # Create the agent (now without input_size parameter)
    agent = DQNAgent()

    if ENABLE_PROFILING:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        
    # Train the agent
    rewards, losses, epsilons = agent.train(num_episodes)
    
    if ENABLE_PROFILING:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(30)

