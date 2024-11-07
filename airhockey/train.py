import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from collections import deque
import time

class AirHockeyEnv:
    def __init__(self, canvas_width=600, canvas_height=800, player_id=0):
        # Game constants from JS version
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.GOAL_WIDTH = 200
        self.GOAL_POSTS = 20
        self.paddle_radius = 20
        self.puck_radius = 15
        self.friction = 0.99
        self.max_speed = 20
        
        # Track which player's perspective (0 = top, 1 = bottom)
        self.player_id = player_id
        
        # Define action space: 8 directions + stay = 9 actions
        self.action_dim = 9
        
        # Initialize stuck detection variables
        self.last_puck_pos = {'x': 0, 'y': 0}
        self.same_position_time = 0
        
        # Initialize game state
        self.reset()
    
    def _normalize_position(self, x, y):
        """Normalize position coordinates to [-1, 1]"""
        norm_x = (2 * x / self.canvas_width) - 1
        norm_y = (2 * y / self.canvas_height) - 1
        return norm_x, norm_y
    
    def _normalize_velocity(self, dx, dy):
        """Normalize velocity to [-1, 1]"""
        norm_dx = dx / self.max_speed
        norm_dy = dy / self.max_speed
        return norm_dx, norm_dy
    
    def _get_obs(self):
        """Get normalized observations from current player's perspective"""
        puck_x, puck_y = self._normalize_position(self.puck['x'], self.puck['y'])
        puck_dx, puck_dy = self._normalize_velocity(self.puck['dx'], self.puck['dy'])
        
        # Get both paddle positions
        top_x, top_y = self._normalize_position(self.top_paddle['x'], self.top_paddle['y'])
        bottom_x, bottom_y = self._normalize_position(self.bottom_paddle['x'], self.bottom_paddle['y'])
        
        if self.player_id == 0:  # Top player's perspective
            return np.array([
                puck_x, puck_y, puck_dx, puck_dy,
                top_x, top_y,      # Own paddle
                bottom_x, bottom_y  # Opponent paddle
            ], dtype=np.float32)
        else:  # Bottom player's perspective (flip coordinates)
            return np.array([
                puck_x, -puck_y, puck_dx, -puck_dy,  # Flip y coordinates
                bottom_x, -bottom_y,  # Own paddle
                top_x, -top_y        # Opponent paddle
            ], dtype=np.float32)
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset paddles
        self.top_paddle = {
            'x': self.canvas_width / 2,
            'y': 50,
            'speed': 5
        }
        
        self.bottom_paddle = {
            'x': self.canvas_width / 2,
            'y': self.canvas_height - 50,
            'speed': 5
        }
        
        # Reset puck to center with random velocity
        self.puck = {
            'x': self.canvas_width / 2,
            'y': self.canvas_height / 2,
            'dx': np.random.uniform(-2, 2),
            'dy': np.random.uniform(-2, 2)
        }
        
        # Reset stuck detection variables
        self.last_puck_pos = {'x': self.puck['x'], 'y': self.puck['y']}
        self.same_position_time = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute one step from current player's perspective"""
        # Store previous state for reward calculation
        prev_puck_pos = (self.puck['x'], self.puck['y'])
        prev_puck_dx = self.puck['dx']
        prev_puck_dy = self.puck['dy']
        prev_paddle_pos = (self.top_paddle['x'], self.top_paddle['y']) if self.player_id == 0 else (self.bottom_paddle['x'], self.bottom_paddle['y'])
        prev_dist_to_puck = np.sqrt((prev_paddle_pos[0] - prev_puck_pos[0])**2 + (prev_paddle_pos[1] - prev_puck_pos[1])**2)
        
        # Store previous positions
        prev_top_x = self.top_paddle['x']
        prev_top_y = self.top_paddle['y']
        prev_bottom_x = self.bottom_paddle['x']
        prev_bottom_y = self.bottom_paddle['y']
        
        # Update current player's paddle based on action
        if action < 8:  # Movement action
            angle = action * (2 * np.pi / 8)
            dx = np.cos(angle) * self.top_paddle['speed']
            dy = np.sin(angle) * self.top_paddle['speed']
            
            if self.player_id == 0:  # Top player
                paddle = self.top_paddle
                y_min = self.paddle_radius
                y_max = self.canvas_height/2 - self.paddle_radius
            else:  # Bottom player
                paddle = self.bottom_paddle
                y_min = self.canvas_height/2 + self.paddle_radius
                y_max = self.canvas_height - self.paddle_radius
            
            paddle['x'] = np.clip(
                paddle['x'] + dx,
                self.paddle_radius,
                self.canvas_width - self.paddle_radius
            )
            paddle['y'] = np.clip(
                paddle['y'] + dy,
                y_min,
                y_max
            )
        
        # Update puck physics
        self.puck['x'] += self.puck['dx']
        self.puck['y'] += self.puck['dy']
        self.puck['dx'] *= self.friction
        self.puck['dy'] *= self.friction
        
        # Handle collisions
        self._handle_wall_collision()
        self._handle_paddle_collision(self.top_paddle, prev_top_x, prev_top_y)
        self._handle_paddle_collision(self.bottom_paddle, prev_bottom_x, prev_bottom_y)
        
        # Check for stuck puck
        if self._is_puck_stuck():
            self._unstick_puck()
        
        # Calculate shaped rewards
        reward = 0
        
        # 1. Proximity reward
        curr_paddle_pos = (self.top_paddle['x'], self.top_paddle['y']) if self.player_id == 0 else (self.bottom_paddle['x'], self.bottom_paddle['y'])
        curr_dist_to_puck = np.sqrt((curr_paddle_pos[0] - self.puck['x'])**2 + (curr_paddle_pos[1] - self.puck['y'])**2)
        dist_change = prev_dist_to_puck - curr_dist_to_puck
        if abs(dist_change) > 5.0:
            reward += dist_change * 0.01
        
        # 2. Hit reward (increased threshold and reward)
        prev_velocity = np.sqrt(prev_puck_dx**2 + prev_puck_dy**2)
        curr_velocity = np.sqrt(self.puck['dx']**2 + self.puck['dy']**2)
        if curr_velocity > prev_velocity + 2.0:  # More significant hits only
            hit_reward = 0.2  # Increased reward
            reward += hit_reward
        
        # 3. Progress reward (increased threshold)
        if self.player_id == 0:
            puck_progress = prev_puck_pos[1] - self.puck['y']
        else:
            puck_progress = self.puck['y'] - prev_puck_pos[1]
        if abs(puck_progress) > 5.0:  # Only reward significant progress
            progress_reward = puck_progress * 0.02
            reward += progress_reward
        
        # 4. Goal rewards (only print these)
        goal = self._check_goal()
        done = False
        
        if goal == 'top':
            reward += -1.0 if self.player_id == 0 else 1.0
            print(f"Goal scored on top!")
            done = True
        elif goal == 'bottom':
            reward += 1.0 if self.player_id == 0 else -1.0
            print(f"Goal scored on bottom!")
            done = True
        
        # 5. Defense penalty (increased and only when puck is close)
        if self.player_id == 0 and self.puck['y'] < self.canvas_height/3:  # Top third
            dist_from_defense = abs(self.top_paddle['x'] - self.puck['x'])
            if dist_from_defense > 50:  # Only penalize if significantly out of position
                defense_penalty = -0.05
                reward += defense_penalty
        
        return self._get_obs(), reward, done, False, {}
    
    def _handle_wall_collision(self):
        """Handle collisions with walls and apply bounces."""
        # Left and right walls
        if self.puck['x'] - self.puck_radius < 0:
            self.puck['x'] = self.puck_radius
            self.puck['dx'] *= -0.8  # Added dampening factor from JS
        elif self.puck['x'] + self.puck_radius > self.canvas_width:
            self.puck['x'] = self.canvas_width - self.puck_radius
            self.puck['dx'] *= -0.8

        # Top and bottom walls (except for goals)
        goal = self._check_goal()
        if not goal:
            if self.puck['y'] - self.puck_radius < 0:
                self.puck['y'] = self.puck_radius
                self.puck['dy'] *= -0.8
            elif self.puck['y'] + self.puck_radius > self.canvas_height:
                self.puck['y'] = self.canvas_height - self.puck_radius
                self.puck['dy'] *= -0.8

    def _handle_paddle_collision(self, paddle, prev_x=None, prev_y=None):
        """Handle collisions between the puck and a paddle."""
        if prev_x is None:
            prev_x = paddle['x']
        if prev_y is None:
            prev_y = paddle['y']
            
        dx = self.puck['x'] - paddle['x']
        dy = self.puck['y'] - paddle['y']
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance < (self.puck_radius + self.paddle_radius):
            # Calculate collision angle
            angle = np.arctan2(dy, dx)
            min_distance = self.paddle_radius + self.puck_radius
            
            # Move puck outside paddle
            self.puck['x'] = paddle['x'] + np.cos(angle) * min_distance
            self.puck['y'] = paddle['y'] + np.sin(angle) * min_distance
            
            # Calculate paddle momentum
            paddle_speed_x = paddle['x'] - prev_x
            paddle_speed_y = paddle['y'] - prev_y
            
            # Calculate dot product
            dot_product = (self.puck['dx'] * dx + self.puck['dy'] * dy) / distance
            
            # Update puck velocity
            self.puck['dx'] = (np.cos(angle) * abs(dot_product) + paddle_speed_x * 0.9)
            self.puck['dy'] = (np.sin(angle) * abs(dot_product) + paddle_speed_y * 0.9)
            
            # Add minimum speed after collision
            speed = np.sqrt(self.puck['dx']**2 + self.puck['dy']**2)
            if speed < 3:
                self.puck['dx'] *= 3 / speed
                self.puck['dy'] *= 3 / speed
            
            # Enforce speed limit
            if speed > self.max_speed:
                scale = self.max_speed / speed
                self.puck['dx'] *= scale
                self.puck['dy'] *= scale

    def _is_puck_stuck(self):
        """Check if the puck is stuck."""
        speed = np.sqrt(self.puck['dx']**2 + self.puck['dy']**2)
        is_slow_moving = abs(self.puck['dx']) < 0.1 and abs(self.puck['dy']) < 0.1
        
        # Check if near any wall
        near_wall = (
            self.puck['x'] - self.puck_radius < 10 or
            self.puck['x'] + self.puck_radius > self.canvas_width - 10 or
            (self.puck['y'] - self.puck_radius < 10 and not self._check_goal()) or
            (self.puck['y'] + self.puck_radius > self.canvas_height - 10 and not self._check_goal())
        )
        
        if not near_wall:
            return False
            
        dist_from_last = np.sqrt(
            (self.puck['x'] - self.last_puck_pos['x'])**2 + 
            (self.puck['y'] - self.last_puck_pos['y'])**2
        )
        
        if dist_from_last < 1:
            self.same_position_time += 1
        else:
            self.same_position_time = 0
            self.last_puck_pos['x'] = self.puck['x']
            self.last_puck_pos['y'] = self.puck['y']
            
        return is_slow_moving or self.same_position_time > 30

    def _unstick_puck(self):
        """Reset puck to random position when stuck."""
        self.puck['x'] = np.random.uniform(self.puck_radius, self.canvas_width - self.puck_radius)
        self.puck['y'] = np.random.uniform(self.puck_radius, self.canvas_height - self.puck_radius)
        self.puck['dx'] = np.random.uniform(-5, 5)
        self.puck['dy'] = np.random.uniform(-5, 5)
        self.same_position_time = 0

    def _check_goal(self):
        """Check if a goal has been scored. Match JS version exactly."""
        # Check if within goal posts in X direction
        in_x_range = (
            (self.canvas_width - self.GOAL_WIDTH)/2 <= self.puck['x'] <= 
            (self.canvas_width + self.GOAL_WIDTH)/2
        )
        
        # Check Y position and X range
        if self.puck['y'] - self.puck_radius < self.GOAL_POSTS and in_x_range:
            return 'top'
        elif self.puck['y'] + self.puck_radius > self.canvas_height - self.GOAL_POSTS and in_x_range:
            return 'bottom'
        return None

class DistributionalDQN:
    def __init__(self, state_dim=8, action_dim=9, learning_rate=0.00025):
        # Distributional DQN parameters
        self.num_atoms = 51
        self.v_min = -10.0
        self.v_max = 10.0
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = tf.linspace(self.v_min, self.v_max, self.num_atoms)
        
        # Network parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = 0.99
        
        # Create networks
        self.online_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.online_network.get_weights())
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Replay buffer
        self.buffer_size = 100000
        self.batch_size = 32
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # Training parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.target_update_freq = 100
        self.train_step_counter = 0

    def _build_network(self):
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_dim * self.num_atoms)(x)
        outputs = tf.keras.layers.Reshape((self.action_dim, self.num_atoms))(outputs)
        outputs = tf.keras.layers.Softmax(axis=-1)(outputs)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    @tf.function
    def _compute_loss(self, states, actions, rewards, next_states, dones):
        # Get next state distributions
        next_distributions = self.target_network(next_states)
        
        # Compute next state values (expected)
        next_values = tf.reduce_sum(
            next_distributions * self.support, axis=-1
        )
        next_actions = tf.argmax(next_values, axis=-1)
        
        # Get next state distributions for selected actions
        next_dists = tf.gather(next_distributions, next_actions, batch_dims=1)
        
        # Compute target distribution
        target_support = rewards[:, None] + (1.0 - dones[:, None]) * self.gamma * self.support
        target_support = tf.clip_by_value(target_support, self.v_min, self.v_max)
        
        # Project onto support
        target_distribution = self._project_distribution(target_support, next_dists)
        
        # Get current distribution
        current_dist = self.online_network(states)
        actions_one_hot = tf.one_hot(actions, self.action_dim)
        current_dist = tf.reduce_sum(
            current_dist * actions_one_hot[:, :, None], axis=1
        )
        
        # Compute cross-entropy loss
        loss = -tf.reduce_sum(target_distribution * tf.math.log(current_dist + 1e-8), axis=-1)
        return tf.reduce_mean(loss)

    def _project_distribution(self, target_support, next_dists):
        """Project value distribution onto support using vectorized operations."""
        batch_size = tf.shape(next_dists)[0]
        
        # Reshape target support to match batch size
        target_support = tf.reshape(target_support, [batch_size, self.num_atoms])
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        
        # Clip target support to be within bounds
        tz_j = tf.clip_by_value(target_support, self.v_min, self.v_max)
        
        # Get indices of bins
        b_j = (tz_j - self.v_min) / delta_z
        l = tf.math.floor(b_j)
        u = tf.math.ceil(b_j)
        
        # Convert to proper indices
        l_idx = tf.cast(l, tf.int32)
        u_idx = tf.cast(u, tf.int32)
        
        # Create batch indices
        batch_indices = tf.range(batch_size)
        batch_indices = tf.reshape(batch_indices, [-1, 1])
        batch_indices = tf.tile(batch_indices, [1, self.num_atoms])
        
        # Flatten indices for scatter
        batch_indices_flat = tf.reshape(batch_indices, [-1])
        l_idx_flat = tf.reshape(l_idx, [-1])
        u_idx_flat = tf.reshape(u_idx, [-1])
        
        # Create indices for scatter
        l_coords = tf.stack([batch_indices_flat, l_idx_flat], axis=1)
        u_coords = tf.stack([batch_indices_flat, u_idx_flat], axis=1)
        
        # Create target distribution
        target_dist = tf.zeros_like(next_dists)
        
        # Calculate weights
        u_weight = u - b_j
        l_weight = b_j - l
        
        # Reshape next_dists and weights for scatter
        next_dists_flat = tf.reshape(next_dists, [-1])
        u_weight_flat = tf.reshape(u_weight, [-1])
        l_weight_flat = tf.reshape(l_weight, [-1])
        
        # Scatter add weighted distributions
        target_dist = tf.tensor_scatter_nd_add(
            target_dist,
            l_coords,
            next_dists_flat * u_weight_flat
        )
        target_dist = tf.tensor_scatter_nd_add(
            target_dist,
            u_coords,
            next_dists_flat * l_weight_flat
        )
        
        return target_dist

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        distributions = self.online_network(state)
        # Calculate expected values for each action
        values = tf.reduce_sum(distributions * self.support, axis=-1)
        return tf.argmax(values[0]).numpy()

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform one step of training."""
        with tf.GradientTape() as tape:
            loss = self._compute_loss(states, actions, rewards, next_states, dones)
        
        # Get gradients and apply them
        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_network.trainable_variables))
        return loss

    def train(self):
        """Train on a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = tf.convert_to_tensor([t[0] for t in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([t[1] for t in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([t[2] for t in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([t[3] for t in batch], dtype=tf.float32)
        dones = tf.convert_to_tensor([t[4] for t in batch], dtype=tf.float32)
        
        # Perform training step
        loss = self.train_step(states, actions, rewards, next_states, dones)
        
        # Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_network.set_weights(self.online_network.get_weights())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.numpy()

def train_self_play(num_episodes=10000, max_steps=1000, batch_size=32):
    # Create two environments (one for each player's perspective)
    env_p1 = AirHockeyEnv(player_id=0)  # Top player
    env_p2 = AirHockeyEnv(player_id=1)  # Bottom player
    
    # Create two agents
    agent_p1 = DistributionalDQN(state_dim=8)
    agent_p2 = DistributionalDQN(state_dim=8)
    
    print("Starting self-play training...")
    start_time = time.time()
    
    for episode in range(num_episodes):
        state_p1, _ = env_p1.reset(seed=episode)
        state_p2, _ = env_p2.reset(seed=episode)
        
        episode_reward_p1 = 0
        episode_reward_p2 = 0
        
        for step in range(max_steps):
            # Get actions from both agents
            action_p1 = agent_p1.select_action(state_p1)
            action_p2 = agent_p2.select_action(state_p2)
            
            # Step both environments
            next_state_p1, reward_p1, done_p1, _, _ = env_p1.step(action_p1)
            next_state_p2, reward_p2, done_p2, _, _ = env_p2.step(action_p2)
            
            # Store transitions and train
            agent_p1.store_transition(state_p1, action_p1, reward_p1, next_state_p1, done_p1)
            agent_p2.store_transition(state_p2, action_p2, reward_p2, next_state_p2, done_p2)
            
            if len(agent_p1.replay_buffer) >= batch_size:
                agent_p1.train()
                agent_p2.train()
            
            episode_reward_p1 += reward_p1
            episode_reward_p2 += reward_p2
            
            if done_p1 or done_p2:
                break
            
            state_p1 = next_state_p1
            state_p2 = next_state_p2
        
        # Print progress more frequently for first 10 episodes, then every 100
        if episode < 10 or episode % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"\nEpisode {episode}/{num_episodes} ({elapsed_time:.1f}s)")
            print(f"Steps: {step + 1}")
            print(f"P1: reward={episode_reward_p1:.2f}, ε={agent_p1.epsilon:.2f}")
            print(f"P2: reward={episode_reward_p2:.2f}, ε={agent_p2.epsilon:.2f}")
            print(f"Buffer size: {len(agent_p1.replay_buffer)}")
            print("----------------------------------------")
    
    return agent_p1, agent_p2

# Train the agents
agent_p1, agent_p2 = train_self_play()