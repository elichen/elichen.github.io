import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from collections import deque

class AirHockeyEnv:
    def __init__(self, canvas_width=600, canvas_height=800):
        # Game constants from JS version
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.GOAL_WIDTH = 200
        self.GOAL_POSTS = 20
        self.paddle_radius = 20
        self.puck_radius = 15
        self.friction = 0.99  # Matches JS
        self.max_speed = 20   # Matches JS
        
        # Track stuck puck state
        self.stuck_time = 0
        self.last_puck_pos = {'x': 0, 'y': 0}
        self.same_position_time = 0
        
        # Define action space: 8 directions + stay = 9 actions
        self.action_dim = 9
        
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
        # Get normalized observations
        puck_x, puck_y = self._normalize_position(self.puck['x'], self.puck['y'])
        puck_dx, puck_dy = self._normalize_velocity(self.puck['dx'], self.puck['dy'])
        paddle_x, paddle_y = self._normalize_position(
            self.ai_paddle['x'], 
            self.ai_paddle['y']
        )
        
        return np.array([
            puck_x, puck_y,
            puck_dx, puck_dy,
            paddle_x, paddle_y
        ], dtype=np.float32)
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset paddles
        self.ai_paddle = {
            'x': self.canvas_width / 2,
            'y': 50,
            'speed': 5
        }
        
        self.player_paddle = {
            'x': self.canvas_width / 2,
            'y': self.canvas_height - 50,
            'speed': 5
        }
        
        # Reset puck to center
        self.puck = {
            'x': self.canvas_width / 2,
            'y': self.canvas_height / 2,
            'dx': 0,
            'dy': 0
        }
        
        return self._get_obs(), {}
    
    def step(self, action):
        # Store previous positions
        prev_ai_x = self.ai_paddle['x']
        prev_ai_y = self.ai_paddle['y']
        
        # Update AI paddle position based on action
        if action < 8:  # Movement action
            angle = action * (2 * np.pi / 8)
            dx = np.cos(angle) * self.ai_paddle['speed']
            dy = np.sin(angle) * self.ai_paddle['speed']
            
            self.ai_paddle['x'] = np.clip(
                self.ai_paddle['x'] + dx,
                self.paddle_radius,
                self.canvas_width - self.paddle_radius
            )
            self.ai_paddle['y'] = np.clip(
                self.ai_paddle['y'] + dy,
                self.paddle_radius,
                self.canvas_height/2 - self.paddle_radius
            )
        
        # Update puck physics (matches JS)
        self.puck['x'] += self.puck['dx']
        self.puck['y'] += self.puck['dy']
        
        self.puck['dx'] *= self.friction
        self.puck['dy'] *= self.friction
        
        # Handle collisions
        self._handle_wall_collision()
        self._handle_paddle_collision(self.ai_paddle, prev_ai_x, prev_ai_y)
        self._handle_paddle_collision(self.player_paddle)
        
        # Check for stuck puck
        if self._is_puck_stuck():
            self._unstick_puck()
        
        # Calculate reward and check if done
        reward = 0
        done = False
        
        goal = self._check_goal()
        if goal == 'top':
            reward = -1
            done = True
        elif goal == 'bottom':
            reward = 1
            done = True
            
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
        """Check if a goal has been scored."""
        # Check vertical position
        if self.puck['y'] - self.puck_radius <= 0:
            # Check if within goal posts
            if (self.canvas_width - self.GOAL_WIDTH)/2 <= self.puck['x'] <= (self.canvas_width + self.GOAL_WIDTH)/2:
                return 'top'
        elif self.puck['y'] + self.puck_radius >= self.canvas_height:
            # Check if within goal posts
            if (self.canvas_width - self.GOAL_WIDTH)/2 <= self.puck['x'] <= (self.canvas_width + self.GOAL_WIDTH)/2:
                return 'bottom'
        return None

class DistributionalDQN:
    def __init__(self, state_dim=6, action_dim=9, learning_rate=0.00025):
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
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 1000
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

def train_agent(env, plot=False, num_episodes=10000, max_steps=1000, batch_size=32):
    """
    Train the agent and return the trained model along with training metrics.
    
    Args:
        env: The Air Hockey environment
        plot: Whether to plot training metrics (useful in notebooks)
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        batch_size: Batch size for training
    
    Returns:
        agent: Trained DistributionalDQN agent
        metrics: Dictionary containing training metrics
    """
    # Initialize agent
    agent = DistributionalDQN()
    
    # Metrics tracking
    metrics = {
        'episode_rewards': [],
        'episode_losses': [],
        'moving_avg_reward': [],
        'moving_avg_loss': []
    }
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.replay_buffer) >= batch_size:
                loss = agent.train()
                episode_loss += loss
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Track metrics
        metrics['episode_rewards'].append(episode_reward)
        if episode_loss > 0:
            metrics['episode_losses'].append(episode_loss / (step + 1))
        else:
            metrics['episode_losses'].append(0)
        
        # Calculate moving averages
        window = 100
        if episode >= window:
            avg_reward = np.mean(metrics['episode_rewards'][-window:])
            avg_loss = np.mean(metrics['episode_losses'][-window:])
        else:
            avg_reward = np.mean(metrics['episode_rewards'])
            avg_loss = np.mean(metrics['episode_losses'])
            
        metrics['moving_avg_reward'].append(avg_reward)
        metrics['moving_avg_loss'].append(avg_loss)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}")
            print(f"Average Reward (last 100): {avg_reward:.2f}")
            print(f"Average Loss (last 100): {avg_loss:.4f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print("----------------------------------------")
    
    if plot:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot rewards
        ax1.plot(metrics['episode_rewards'], alpha=0.3, label='Episode Reward')
        ax1.plot(metrics['moving_avg_reward'], label='Moving Average')
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)
        
        # Plot losses
        ax2.plot(metrics['episode_losses'], alpha=0.3, label='Episode Loss')
        ax2.plot(metrics['moving_avg_loss'], label='Moving Average')
        ax2.set_title('Training Losses')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return agent, metrics