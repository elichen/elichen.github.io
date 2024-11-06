import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AirHockeyEnv(gym.Env):
    def __init__(self, canvas_width=600, canvas_height=800):
        super().__init__()
        
        # Game constants from JS version
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.GOAL_WIDTH = 200
        self.GOAL_POSTS = 20
        self.paddle_radius = 20
        self.puck_radius = 15
        self.friction = 0.99  # Updated to match JS
        self.max_speed = 20   # Updated to match JS
        
        # Track stuck puck state
        self.stuck_time = 0
        self.last_puck_pos = {'x': 0, 'y': 0}
        self.same_position_time = 0
        
        # Define action space: 8 directions + stay = 9 actions
        self.action_space = spaces.Discrete(9)
        
        # Define observation space (normalized to [-1, 1])
        # [puck_x, puck_y, puck_dx, puck_dy, ai_paddle_x, ai_paddle_y]
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )
        
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
        super().reset(seed=seed)
        
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
        
        # Update AI paddle position (same as before)
        if action < 8:
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
        
        # Update puck physics
        self.puck['x'] += self.puck['dx']
        self.puck['y'] += self.puck['dy']
        
        self.puck['dx'] *= self.friction
        self.puck['dy'] *= self.friction
        
        # Handle collisions with updated physics
        self._handle_wall_collision()
        self._handle_paddle_collision(self.ai_paddle, prev_ai_x, prev_ai_y)
        self._handle_paddle_collision(self.player_paddle)
        
        # Check for stuck puck
        if self._is_puck_stuck():
            self._unstick_puck()
        
        # Rest of the step function remains the same
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
        """Check if a goal has been scored.
        
        Returns:
            str: 'top' if goal in top goal, 'bottom' if goal in bottom goal, None otherwise
        """
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

def create_agent():
    """Create and return a new Rainbow agent."""
    from rainbow.agent import Rainbow

    return Rainbow(
        nb_states=6,  # [puck_x, puck_y, puck_dx, puck_dy, paddle_x, paddle_y]
        nb_actions=9, # 8 directions + stay
        gamma=0.99,   # Discount factor
        replay_capacity=100000,
        learning_rate=0.00025,
        batch_size=32,
        # Start with basic DQN features before adding Rainbow improvements
        distributional=False,
        noisy=False,
        prioritized_replay=False,
        multi_steps=1,
        name="air_hockey_dqn"
    )

def train_agent(episodes=10000, max_steps=1000):
    """Train the Rainbow agent on the Air Hockey environment.
    
    Args:
        episodes (int): Number of episodes to train for
        max_steps (int): Maximum steps per episode
        
    Returns:
        list: Episode rewards history
    """
    import numpy as np
    from tqdm import tqdm
    
    env = AirHockeyEnv()
    agent = create_agent()
    episode_rewards = []

    for episode in tqdm(range(episodes)):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.pick_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            agent.store_replay(obs, action, reward, next_obs, done, truncated)
            agent.train()
            
            episode_reward += reward
            obs = next_obs
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward (last 100): {avg_reward:.2f}")

    # Save the trained model
    agent.save("models/air_hockey_dqn")
    return episode_rewards