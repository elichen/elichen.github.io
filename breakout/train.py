# breakout_dqn.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
from collections import deque
import time
import psutil
import pandas as pd

# tf-agents imports
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.train import learner
from tf_agents.train import triggers

# Set up multi-CPU strategy
strategy = tf.distribute.MirroredStrategy(
    devices=[f"/cpu:{i}" for i in range(8)]
)

class Game:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.paddle = {'width': 75, 'height': 10, 'x': width / 2 - 37.5, 'y': height - 20}
        self.ball = {'radius': 5, 'x': width / 2, 'y': height - 30, 'dx': 2, 'dy': -2}
        self.bricks = []
        self.score = 0
        self.total_score = 0  # Add total score to track overall progress
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
            return 0

        reward = 0
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
            if self.ball['dy'] > 0:  # Only bounce if moving downward
                self.ball['dy'] = -self.ball['dy']
                hit_position = (self.ball['x'] - self.paddle['x']) / self.paddle['width']
                max_angle_offset = 1
                self.ball['dx'] = self.ball['dx'] + (hit_position - 0.5) * max_angle_offset
                reward += 0.1

        # Ball collision with bricks
        for brick in self.bricks:
            if brick['status'] == 1:
                if (self.ball['x'] > brick['x'] and
                    self.ball['x'] < brick['x'] + brick['width'] and
                    self.ball['y'] > brick['y'] and
                    self.ball['y'] < brick['y'] + brick['height']):
                    self.ball['dy'] = -self.ball['dy']
                    brick['status'] = 0
                    self.total_score += 1
                    reward += 1.0  # Reward for breaking brick
                    if self.total_score == len(self.bricks):
                        print(f"\nSolved! All bricks destroyed!")
                        self.game_over = True

        # Game over if ball touches bottom
        if self.ball['y'] + self.ball['radius'] > self.height:
            self.game_over = True
            reward += -1  # Penalty for losing ball

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
        self.total_score = 0  # Reset total score too
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
                state.extend([0.0, 0.0])  # Use 0.0 instead of -1

        return np.array(state, dtype=np.float32)

class BreakoutEnv(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        self._game = Game()
        
        # Define action and observation specs
        # Actions: 0 - stay, 1 - left, 2 - right
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        
        # Observation: A vector of game state as defined in Game.get_state()
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._game.get_state().shape[0],), dtype=np.float32, minimum=0.0, maximum=1.0, name='observation')
        
        self._state = self._game.get_state()
        self._episode_ended = False
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._game.reset()
        self._state = self._game.get_state()
        self._episode_ended = False
        return ts.restart(self._state)
    
    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()
        
        # Map the action to game moves
        if action == 1:
            self._game.move_paddle('left')
        elif action == 2:
            self._game.move_paddle('right')
        # If action == 0, do nothing (stay)
        
        reward = self._game.update()
        self._state = self._game.get_state()
        
        if self._game.game_over:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

def create_replay_buffer(agent, tf_env, max_length=100000):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=max_length
    )

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    
    # Convert to tensor before adding to buffer
    buffer.add_batch(tf.nest.map_structure(tf.convert_to_tensor, traj))

def collect_initial_data(environment, policy, buffer, n_steps):
    for _ in range(n_steps):
        collect_step(environment, policy, buffer)

def train_agent(num_iterations, collect_steps_per_iteration, batch_size, log_interval, tf_env, agent, replay_buffer):
    # Create dataset with parallel processing
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=8,  # Use all cores for dataset processing
        sample_batch_size=batch_size,
        num_steps=2,
        single_deterministic_pass=False
    ).prefetch(tf.data.AUTOTUNE)  # Optimize prefetching

    iterator = iter(dataset)
    agent.train = common.function(agent.train)

    avg_return = []
    rewards_history = []
    episode_reward = 0
    
    for iteration in range(num_iterations):
        # Collect steps using agent's collect policy
        time_step = tf_env.current_time_step()
        action_step = agent.collect_policy.action(time_step)
        next_time_step = tf_env.step(action_step.action)
        
        # Add to episode reward
        episode_reward += next_time_step.reward.numpy()[0]
        
        # If episode ended, record the reward
        if next_time_step.is_last():
            rewards_history.append(episode_reward)
            episode_reward = 0
        
        # Add to replay buffer
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        replay_buffer.add_batch(traj)

        # Sample a batch of data from the buffer and update the agent's network
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        if iteration % log_interval == 0:
            avg_reward = np.mean(rewards_history[-100:]) if rewards_history else 0
            print(f'step = {iteration}: loss = {train_loss.numpy():.6f}, avg_reward = {avg_reward:.2f}')
            avg_return.append(train_loss.numpy())

    return avg_return, rewards_history

class BreakoutTrainer:
    def __init__(self, learning_rate=1e-3):
        # Create the environment
        self.py_env = BreakoutEnv()
        self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)
        
        # Create global step
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
        
        # Define epsilon decay
        epsilon = tf.compat.v1.train.polynomial_decay(
            1.0,  # start epsilon
            self.global_step,
            100000,  # decay steps
            end_learning_rate=0.1,
            power=1.0
        )
        
        # Create agent
        self.agent = self._create_agent(learning_rate, epsilon)
        
        # Create replay buffer with proper dataset options
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=1,
            max_length=100000
        )
        
        # Create random policy for initial collection
        self.random_policy = random_tf_policy.RandomTFPolicy(
            time_step_spec=self.tf_env.time_step_spec(),
            action_spec=self.tf_env.action_spec()
        )

        # Create dataset function that returns the correct format
        def experience_dataset_fn():
            dataset = self.replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=64,
                num_steps=2,
                single_deterministic_pass=False
            )
            return dataset
        
        # Create learner with proper configuration
        self.learner = learner.Learner(
            root_dir='tmp/breakout_training',
            train_step=self.global_step,
            agent=self.agent,
            experience_dataset_fn=experience_dataset_fn,
            checkpoint_interval=10000,
            summary_interval=1000,
            max_checkpoints_to_keep=3,
            use_reverb_v2=False  # Important: tells learner to expect (data, info) tuple
        )
        
        self.rewards_history = []
        self.episode_reward = 0

    def _create_agent(self, learning_rate, epsilon):
        q_net = q_network.QNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=(256, 256, 256, 256)
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        train_step_counter = self.global_step  # Use the same counter
        
        agent = dqn_agent.DqnAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            epsilon_greedy=epsilon,
            target_update_period=1000,
            td_errors_loss_fn=common.element_wise_huber_loss,
            gamma=0.99,
            train_step_counter=train_step_counter,
            gradient_clipping=1.0
        )

        return agent

    def collect_step(self, policy):
        time_step = self.tf_env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = self.tf_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        
        # Track rewards
        self.episode_reward += next_time_step.reward.numpy()[0]
        if next_time_step.is_last():
            self.rewards_history.append(self.episode_reward)
            self.episode_reward = 0
            
        self.replay_buffer.add_batch(traj)

    def collect_data(self, steps, policy):
        for _ in range(steps):
            self.collect_step(policy)

    def train(self, num_iterations=1000000, batch_size=64, log_interval=10000):
        """Train the agent using the learner."""
        try:
            # Collect initial data
            if self.replay_buffer.num_frames() == 0:
                print('Collecting initial data')
                self.collect_data(batch_size * 2, self.random_policy)
                print('Finished collecting initial data')

            print('Starting training...')
            for i in range(0, num_iterations, log_interval):
                # Collect experience
                self.collect_data(log_interval, self.agent.collect_policy)
                
                # Train
                loss_info = self.learner.run(iterations=log_interval)
                
                # Calculate average reward
                avg_reward = np.mean(self.rewards_history[-100:]) if self.rewards_history else 0
                print(f'Iteration {i}, Loss: {loss_info.loss.numpy():.6f}, Avg Reward: {avg_reward:.2f}')

            return loss_info

        except Exception as e:
            print(f"Training error: {str(e)}")
            raise

    def plot_rewards(self):
        """Plot the rewards history."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_history)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()
        
        # Also plot moving average
        window_size = 100
        moving_avg = np.convolve(self.rewards_history, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        plt.figure(figsize=(10, 5))
        plt.plot(moving_avg)
        plt.title(f'Moving Average of Rewards (Window Size: {window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.show()

    def save_model(self, path="."):
        """Save the trained model."""
        import tensorflowjs as tfjs
        import tensorflow as tf
        
        # Get the underlying Keras model
        q_net = self.agent._q_network
        
        # Create a new model that replicates the Q-network structure
        input_shape = self.tf_env.observation_spec().shape
        num_actions = self.tf_env.action_spec().maximum - self.tf_env.action_spec().minimum + 1
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(num_actions)
        ])
        
        # Copy the weights
        time_step = self.tf_env.current_time_step()
        q_values = q_net(time_step.observation, step_type=time_step.step_type)
        model(time_step.observation.numpy())  # Build the model
        
        # Get weights from q_net layers
        q_weights = []
        for layer in q_net.layers:
            if hasattr(layer, 'get_weights'):
                weights = layer.get_weights()
                if weights:  # only add if weights is not empty
                    q_weights.extend(weights)
        
        # Distribute weights to model layers
        weight_index = 0
        for layer in model.layers:
            num_weights = len(layer.get_weights())
            if num_weights > 0:
                layer_weights = q_weights[weight_index:weight_index + num_weights]
                layer.set_weights(layer_weights)
                weight_index += num_weights
        
        # Save the replicated model
        tfjs.converters.save_keras_model(model, path)

