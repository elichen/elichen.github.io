# Snake RL Implementation Plan (20×20 Grid, SB3, Apple Silicon)

Owner:  
Target environment: M4 MacBook Pro, macOS, Python 3.11+  
RL framework: Stable-Baselines3 (PPO primary, DQN optional later)  
Game: Snake on an `N×N` grid (final target `20×20`, snake starts length 3)

---

## 0. High-Level Goals

1. Implement a `gymnasium`-compatible Snake environment with:
   - Full-grid observations (channels for head, body, food).
   - Discrete action space `{turn_left, go_straight, turn_right}`.
   - Reward shaping (distance-based potential, anti-stall).
2. Train a PPO agent with a custom CNN feature extractor using SB3 on Apple Silicon (MPS).
3. Use a curriculum over board sizes (`6×6 → 8×8 → 10×10 → 14×14 → 20×20`) to reach strong performance.
4. Provide hooks for:
   - Logging and visualization.
   - Later experiments (DQN, Hamiltonian hybrid, etc.).

---

## 1. Environment Implementation (`SnakeEnv`)

### 1.1. Interface

Create module: `snake_env.py`

Class:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, n=20, max_no_food=None, render_mode=None, seed=None):
        super().__init__()
        # TODO: implement
```

#### 1.1.1. Spaces

- `board_size`: `self.n` (default 20; configurable).
- `action_space`: `spaces.Discrete(3)`
  - 0 = turn left
  - 1 = go straight
  - 2 = turn right
- `observation_space`: `spaces.Box(
    low=0.0,
    high=1.0,
    shape=(C, n, n),
    dtype=np.float32
  )`
  - Required channels:
    - `0`: head (1.0 for head position, 0.0 elsewhere).
    - `1`: body (1.0 for all body segments, including tail).
    - `2`: food (1.0 at food location).
  - Optional future channels:
    - e.g. “danger” masks, occupancy age, etc.

Additionally, expose a method or property to return a small side-vector:

- Head direction one-hot: length 4 (`[up, right, down, left]`).
- Normalized snake length: `len / (n * n)`.
- Normalized steps since last food: `steps_since_food / max_no_food`.

We will feed this side-vector via a custom feature extractor (Section 3) by passing it alongside the grid internally.

### 1.2. Internal Game State

Maintain at least:

- `self.snake`: list of `(row, col)` tuples; head is first/last (pick and be consistent).
- `self.direction`: integer or enum in `{0: up, 1: right, 2: down, 3: left}`.
- `self.food_pos`: `(row, col)`.
- `self.steps_since_food`: integer.
- `self.length`: current snake length.
- `self.rng`: random generator (`np.random.Generator`).
- Reward shaping state:
  - `self.prev_phi`: previous value of potential function Φ (distance-based).

### 1.3. Reset Logic

`reset(seed=None, options=None) -> (obs, info)`

1. Seed RNG if provided.
2. Place snake:
   - Initial length = 3.
   - Place it either horizontally or vertically in the center area.
   - Direction aligned with body (e.g., if body is [ (10,10), (10,9), (10,8) ], direction is east).
3. Place food randomly at an empty cell (not occupied by snake).
4. Initialize:
   - `steps_since_food = 0`
   - `prev_phi = compute_potential(state)`
5. Return:
   - Observation: grid tensor.
   - Info: include e.g. `{"length": length, "food_pos": food_pos}`.

### 1.4. Step Logic

`step(action) -> (obs, reward, terminated, truncated, info)`

1. Map relative action to absolute direction:
   - `new_dir = (current_dir + delta[action]) % 4`
     - `delta = {0: -1, 1: 0, 2: +1}`.
2. Compute new head coordinates based on `new_dir`.
3. Check terminal conditions:
   - Out of bounds → `terminated = True`, base reward = `-1.0`.
   - Collision with body (including tail cell if tail doesn’t move) → `terminated = True`, base reward = `-1.0`.
4. If not terminated:
   - If `new_head == food_pos`:
     - Increase length by 1 (add head, do NOT pop tail).
     - Increment score.
     - Place new food at random empty cell.
     - Set `steps_since_food = 0`.
     - Base reward: `+1.0`.
   - Else:
     - Move snake:
       - Add new head.
       - Pop last tail segment to maintain same length.
     - Increment `steps_since_food`.
     - Base reward: `0.0`.
5. Anti-stall:
   - Determine `max_no_food` (if `None` passed, default to `80 + 4 * length` or `80 + 4 * self.length`).
   - If `steps_since_food > max_no_food`:
     - `truncated = True` (not `terminated`).
     - Add stall penalty, e.g. `base_reward += -0.5`.

6. Reward shaping:
   - Compute current potential `phi = -alpha * normalized_manhattan_distance(head, food)`.
     - `d = |hx - fx| + |hy - fy|`
     - `d_norm = d / (2 * (n - 1))`
     - `phi = -alpha * d_norm`, choose `alpha ~ 0.2`.
   - Shaping reward: `r_shape = gamma * phi - prev_phi` (use the same `gamma` as PPO; pass gamma into env init).
   - Total reward:  
     `reward = base_reward + r_shape + survival_bonus`
     - Survival bonus: `+0.001` each non-terminal step.
   - Update `prev_phi = phi`.

7. Encode new observation:
   - Clear 3-channel grid.
   - Set head channel: 1 at head.
   - Set body channel: 1 at all snake cells (including head).
   - Set food channel: 1 at food_pos.

8. Info dictionary:
   - Include:
     - `{"length": length, "score": eaten_food_count, "reason": "wall"|"self"|"stall"|None}`.

Note: make shaping coefficient and stall parameters configurable via env init, so we can tune without changing code.

### 1.5. Render

Implement minimal `render`:

- Mode `"human"`:
  - Simple `print`-based ASCII visualization or `matplotlib` grid.
- Mode `"rgb_array"`:
  - Return `H×W×3` numpy array for potential video logging.

Do not overinvest here initially; sufficient for debugging.

---

## 2. SB3 Integration

### 2.1. Environment Factory

Create `make_env.py`:

```python
from functools import partial
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from snake_env import SnakeEnv

def make_snake_env(n=20, gamma=0.995, max_no_food=None, seed=0):
    def _init():
        env = SnakeEnv(n=n, max_no_food=max_no_food, gamma=gamma)
        env = Monitor(env)
        return env
    return _init
```

Then for training:

```python
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([make_snake_env(n=board_size, gamma=0.995, seed=seed)])
```

Initially, `DummyVecEnv` is fine. If we need throughput, consider `SubprocVecEnv` later.

### 2.2. Device Selection (Apple Silicon)

Training script should:

```python
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
```

Pass `device=device` to SB3 model constructors.

---

## 3. Model Architecture (Custom Feature Extractor)

### 3.1. Requirements

- Base policy: PPO with `CnnPolicy`.
- Replace default CNN with `SnakeFeatureExtractor`.
- Extract both:
  - CNN features from grid observation.
  - Extra scalar features (direction, length, steps_since_food) passed via `info` or encoded into observation (preferred: encode direction as extra channels; length/time as extra channels or blame to separate vector).

For simplicity in SB3, start by encoding direction/length/time as **additional channels** to the grid:

- Channel 3: direction up (1 at all cells if direction is up, else 0).
- Channel 4: direction right.
- Channel 5: direction down.
- Channel 6: direction left.
- Channel 7: entire board filled with normalized length.
- Channel 8: entire board filled with normalized steps_since_food.

Observation shape becomes `(8, n, n)`.

This avoids custom side-vector wiring and lets us keep a standard `CnnPolicy` with custom extractor.

### 3.2. Feature Extractor Skeleton

Create `snake_features.py`:

```python
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SnakeFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # expect 8

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass with dummy.
        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.cnn(observations)
        return self.linear(x)
```

### 3.3. Policy Configuration

In training script:

```python
from stable_baselines3 import PPO
from snake_features import SnakeFeatureExtractor

policy_kwargs = dict(
    features_extractor_class=SnakeFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
)

model = PPO(
    policy="CnnPolicy",
    env=env,
    device=device,
    n_steps=2048,
    batch_size=256,
    gamma=0.995,
    learning_rate=3e-4,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="logs/snake_ppo",
)
```

Hyperparameters are a starting point; we will tune later.

---

## 4. Reward Shaping Implementation Details

### 4.1. Potential Function Φ

In `SnakeEnv`:

```python
def _manhattan_distance(self, a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _compute_phi(self):
    # Normalize distance in [0, 1]
    d = self._manhattan_distance(self.snake_head, self.food_pos)
    max_d = 2 * (self.n - 1)
    d_norm = d / max_d if max_d > 0 else 0.0
    return -self.alpha * d_norm  # alpha ~ 0.2
```

On `reset`:

```python
self.prev_phi = self._compute_phi()
```

On each `step`:

```python
phi = self._compute_phi()
r_shape = self.gamma * phi - self.prev_phi
self.prev_phi = phi
reward = base_reward + r_shape + survival_bonus
```

Parameters `alpha`, `gamma`, `survival_bonus` should be env attributes with reasonable defaults and configurable via constructor.

---

## 5. Curriculum Learning

### 5.1. Strategy

Board sizes: `[6, 8, 10, 14, 20]`.

For each size:

1. Train PPO until moving average score over last `N_eval` episodes >= threshold.
2. Save model.
3. Recreate env with next board size.
4. Load model weights and continue training.

### 5.2. Implementation Outline

Create `train_curriculum.py`:

```python
BOARD_SIZES = [6, 8, 10, 14, 20]
SCORE_THRESHOLDS = {
    6: 4,
    8: 6,
    10: 8,
    14: 10,
    20: 12,  # just a start; can be adjusted
}

TIMESTEPS_PER_PHASE = {
    6: 1_000_000,
    8: 1_500_000,
    10: 2_000_000,
    14: 3_000_000,
    20: 4_000_000,
}
```

Pseudo-code:

```python
model = None

for n in BOARD_SIZES:
    env = DummyVecEnv([make_snake_env(n=n, gamma=0.995, seed=0)])
    if model is None:
        model = PPO("CnnPolicy", env, device=device, policy_kwargs=policy_kwargs, ...)
    else:
        model.set_env(env)

    timesteps = TIMESTEPS_PER_PHASE[n]
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
    model.save(f"models/snake_ppo_{n}x{n}.zip")

    # Optional: run evaluation episodes and verify average score >= SCORE_THRESHOLDS[n]
```

For evaluation, implement a helper:

```python
def evaluate(model, env, n_episodes=50):
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += info[0].get("score_increment", 0)  # or track via reward-based logic
        scores.append(score)
    return np.mean(scores), np.std(scores)
```

---

## 6. Debugging and Validation

### 6.1. Environment Tests

Implement `tests/test_snake_env.py`:

- Test 1: reset produces valid single head and food; no overlap.
- Test 2: moving straight into wall triggers `terminated=True` and reward ≤ 0.
- Test 3: eating food increases length and places new food on empty cell.
- Test 4: steps_since_food increments and stall condition triggers `truncated=True`.

### 6.2. Sanity Checks with Random Policy

Before full training:

1. Run ~100 episodes with random actions:
   - Collect distribution of episode lengths and scores.
   - Confirm no crashes, no NaN rewards.
2. Plot histogram of random scores to set a baseline (should be very low, e.g., <5).

### 6.3. Logging

- Use TensorBoard logging from SB3:
  - episode reward mean
  - episode length mean
- Optionally log custom scalars:
  - average steps_until_food
  - fraction of deaths by cause.

Hook via `Monitor` and/or custom callbacks.

### 6.4. Visual Rollouts

Implement `rollout_visualization.py`:

- Load trained model.
- Run a few deterministic episodes.
- Render to console or `matplotlib`:
  - Display snake, food, length, score.

Use this to visually confirm non-degenerate strategies.

---

## 7. Extensions / Roadmap (Optional)

Not required for first implementation but keep in mind:

1. **DQN baseline**:
   - Same env, discrete actions.
   - Implement DQN with CNN backbone similar to PPO’s feature extractor.
2. **Tail-aware shaping**:
   - Add a second potential based on distance head→tail to encourage staying near tail, reducing self-traps.
3. **Hybrid Hamiltonian baseline**:
   - Precompute Hamiltonian cycle for 20×20.
   - Implement a safety fallback policy that follows the cycle when RL action is deemed unsafe or uncertain.
   - Meta-policy: RL learns when to deviate from cycle to grab food earlier.

---

## 8. Deliverables

1. `snake_env.py`  
   - `SnakeEnv` implementing Gymnasium API with reward shaping and anti-stall.

2. `snake_features.py`  
   - `SnakeFeatureExtractor` CNN.

3. `make_env.py`  
   - Env factory utilities.

4. `train_curriculum.py`  
   - Curriculum training script, saving models at each board size.

5. `evaluation_utils.py`  
   - `evaluate(model, env, n_episodes)` helper.

6. `rollout_visualization.py`  
   - Script to run and visualize episodes with a trained model.

7. `tests/`  
   - Unit tests validating core env behavior.

Once these are in place and run end-to-end (small board first), we can start tuning hyperparameters and shaping to push average scores on 20×20 as high as possible.
