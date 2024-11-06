# breakout_ppo.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
from collections import deque
import time
import psutil
import pandas as pd
from time import time
import datetime
import tensorflowjs as tfjs

# tf-agents imports
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.utils import nest_utils
from tf_agents.specs import tensor_spec
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import py_metrics
from tf_agents.policies import policy_saver
import multiprocessing

# Set up multi-CPU strategy
strategy = tf.distribute.MirroredStrategy(
    devices=[f"/cpu:{i}" for i in range(multiprocessing.cpu_count())]
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
        self.penalty_for_losing_ball = -10
        self.reward_for_hitting_paddle = 0.5
        self.reward_for_breaking_brick = 5
        self.reward_for_survival = 0.01
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

        reward = self.reward_for_survival  # Keep basic survival reward
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
                reward += self.reward_for_hitting_paddle

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
                    reward += self.reward_for_breaking_brick

        # Game over if ball touches bottom
        if self.ball['y'] + self.ball['radius'] > self.height:
            self.game_over = True
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
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        
        # Observation: Same as before
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._game.get_state().shape[0],), 
            dtype=np.float32, 
            minimum=0.0, 
            maximum=1.0, 
            name='observation')
        
        self._state = self._game.get_state()
        self._episode_ended = False
        
        # Add episode step counter for PPO
        self._steps_in_episode = 0
        self._max_steps_per_episode = 1000  # Prevent infinite episodes
    
    def _reset(self):
        self._game.reset()
        self._state = self._game.get_state()
        self._episode_ended = False
        self._steps_in_episode = 0
        return ts.restart(self._state)
    
    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._steps_in_episode += 1
        
        # Map the action to game moves
        if action == 1:
            self._game.move_paddle('left')
        elif action == 2:
            self._game.move_paddle('right')
        
        reward = self._game.update()
        self._state = self._game.get_state()
        
        # End episode if game over or max steps reached
        if self._game.game_over or self._steps_in_episode >= self._max_steps_per_episode:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        
        return ts.transition(self._state, reward=reward, discount=0.99)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

class BreakoutTrainer:
    def __init__(self, learning_rate=3e-4):  # PPO's recommended learning rate
        # Initialize distribution strategy
        self.strategy = tf.distribute.MirroredStrategy(
            devices=[f"/cpu:{i}" for i in range(8)]
        )
        print(f'Number of devices: {self.strategy.num_replicas_in_sync}')
        
        self.py_env = BreakoutEnv()
        self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)
        
        # Create networks
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=(256, 256, 256, 256),
            activation_fn=tf.keras.activations.relu
        )
        
        value_net = value_network.ValueNetwork(
            self.tf_env.observation_spec(),
            fc_layer_params=(256, 256, 256, 256),
            activation_fn=tf.keras.activations.relu
        )
        
        # Create PPO agent with correct import and parameters
        self.agent = ppo_clip_agent.PPOClipAgent(
            time_step_spec=self.tf_env.time_step_spec(),
            action_spec=self.tf_env.action_spec(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            actor_net=actor_net,
            value_net=value_net,
            # PPO specific parameters
            entropy_regularization=0.01,
            importance_ratio_clipping=0.2,  # PPO clip parameter
            discount_factor=0.99,
            lambda_value=0.95,  # Changed from gae_lambda
            num_epochs=10,
            use_gae=True,
            use_td_lambda_return=True,
            normalize_rewards=True,
            reward_norm_clipping=10.0,
            normalize_observations=True,
            policy_l2_reg=0.0,
            value_function_l2_reg=0.0,
            shared_vars_l2_reg=0.0,
            value_pred_loss_coef=0.5,
            debug_summaries=True,
            summarize_grads_and_vars=True,
        )
        
        self.agent.initialize()
        
        # Add train step counter
        self.train_step = tf.Variable(0, dtype=tf.int64)
        
        # Add metrics
        self.train_metrics = [
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]
        
        self.eval_metrics = [
            py_metrics.AverageReturnMetric(),
            py_metrics.AverageEpisodeLengthMetric(),
        ]
        
        # Initialize policy saver
        self.policy_saver = policy_saver.PolicySaver(
            self.agent.policy,
            batch_size=None,  # Allows for variable batch sizes
            train_step=self.train_step  # Add train step counter for versioning
        )
    
    def collect_episode(self, policy, num_episodes=1):
        episodes_per_replica = num_episodes // self.strategy.num_replicas_in_sync
        
        @tf.function
        def collect_replica_episodes():
            replica_episodes = []
            for _ in range(episodes_per_replica):
                time_step = self.tf_env.reset()
                episode_steps = []
                
                while not time_step.is_last():
                    action_step = policy.action(time_step)
                    next_time_step = self.tf_env.step(action_step.action)
                    traj = trajectory.from_transition(time_step, action_step, next_time_step)
                    episode_steps.append(traj)
                    time_step = next_time_step
                
                if episode_steps:
                    replica_episodes.append(
                        tf.nest.map_structure(
                            lambda *arrays: tf.stack(arrays), 
                            *episode_steps
                        )
                    )
            return replica_episodes

        distributed_episodes = self.strategy.run(collect_replica_episodes)
        
        all_episodes = []
        for replica_episodes in distributed_episodes:
            all_episodes.extend(replica_episodes)
        
        return all_episodes

    def train(self, num_iterations=1000, eval_interval=100):
        # Wrap the training loop in the distribution strategy scope
        with self.strategy.scope():
            # Training metrics
            returns = []
            steps = []
            
            # Timing metrics
            start_time = time()
            episode_times = []
            
            # Training loop
            for iteration in range(num_iterations):
                iter_start = time()
                
                # Collect experience and train
                episodes = self.collect_episode(self.agent.collect_policy, num_episodes=1)
                for episode in episodes:
                    processed_episode = tf.nest.map_structure(
                        lambda x: tf.reshape(x, [1, -1] + list(x.shape[2:])) if len(x.shape) > 2 else tf.reshape(x, [1, -1]),
                        episode
                    )
                    self.agent.train(experience=processed_episode)
                
                # Update timing metrics
                iter_time = time() - iter_start
                episode_times.append(iter_time)
                
                # Evaluation and logging
                if iteration % eval_interval == 0:
                    avg_return = self.evaluate()
                    returns.append(avg_return)
                    steps.append(iteration)
                    
                    # Calculate timing estimates
                    avg_episode_time = sum(episode_times[-eval_interval:]) / len(episode_times[-eval_interval:])
                    elapsed_time = time() - start_time
                    estimated_remaining = avg_episode_time * (num_iterations - iteration)
                    
                    # Plot first, then print metrics
                    self.plot_metrics(steps, returns)
                    
                    print(f'\nIteration: {iteration}/{num_iterations}')
                    print(f'Average Return: {float(avg_return):.2f}')
                    print(f'Average episode time: {avg_episode_time:.2f}s')
                    print(f'Elapsed time: {datetime.timedelta(seconds=int(elapsed_time))}')
                    print(f'Estimated remaining: {datetime.timedelta(seconds=int(estimated_remaining))}')
                    print(f'Estimated total: {datetime.timedelta(seconds=int(elapsed_time + estimated_remaining))}')

    def evaluate(self, num_episodes=5):
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = self.tf_env.reset()
            episode_return = 0.0
            
            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                time_step = self.tf_env.step(action_step.action)
                episode_return += time_step.reward
            
            total_return += episode_return

        # Return average over episodes
        return total_return / num_episodes

    def plot_metrics(self, steps, returns):
        clear_output(wait=True)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 1, 1)
        plt.plot(steps, returns)
        plt.xlabel('Iterations')
        plt.ylabel('Average Return')
        plt.show()

    def save_model(self, export_dir='saved_model', tfjs_dir='.'):
        """Save the trained policy for both TF and TFJS formats."""
        from tf_agents.policies import policy_saver
        
        # Create policy saver (using a different variable name)
        saver = policy_saver.PolicySaver(
            self.agent.policy,
            batch_size=None,
            train_step=self.train_step
        )
        
        # Save the policy
        saver.save(export_dir)
        print(f"TF Model saved to: {export_dir}")
        
        # Print model structure information
        print("\nModel structure:")
        print("Input spec:", self.tf_env.time_step_spec())
        print("Action spec:", self.tf_env.action_spec())
        
        # For TFJS, convert using the action signature
        tfjs.converters.convert_tf_saved_model(
            export_dir, 
            tfjs_dir,
            signature_def='action',  # Use the action signature
            control_flow_v2=True     # Enable control flow v2
        )

