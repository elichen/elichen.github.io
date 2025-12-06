# Snake AI - Curriculum-Trained PPO

This directory contains a Snake AI agent trained using PPO with curriculum learning.

## Current Model Performance

**Evaluation Results** (100 episodes, 20x20 grid):
- **Mean Score:** 46.19 ± 27.64
- **Max Score:** 79
- **Median:** 54
- **53%** of games score ≥ 50 food

## Training Approach

### Curriculum Learning
The agent was trained progressively on increasing grid sizes:
1. **6×6** → Basic navigation
2. **8×8** → Scaling up
3. **10×10** → Intermediate
4. **14×14** → Near-target complexity
5. **20×20** → Final target

This curriculum approach allows the agent to learn fundamental skills on smaller grids before tackling the full 20×20 challenge.

### Architecture

**8-Channel CNN Observations:**
| Channel | Content |
|---------|---------|
| 0 | Head position (1 at head, 0 elsewhere) |
| 1 | Body positions (1 at all segments) |
| 2 | Food position |
| 3-6 | Direction one-hot (broadcast to all cells) |
| 7 | Normalized length (broadcast) |

**Network:**
- 3 Conv layers: 64→128→256 filters, 3×3, same padding, ReLU
- Global Average Pooling → 256-dim vector
- Dense(256, ReLU) → Dense(256, Tanh) → Dense(256, Tanh) → Dense(3)

**Actions:** 3 relative actions (turn left, go straight, turn right)

### Training Details
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Framework:** Stable-Baselines3
- **Parallel Environments:** 8 (SubprocVecEnv)
- **Hardware:** Apple Silicon (MPS)
- **Total Steps:** ~10M across curriculum

## Files

### Training Code
- `snake_env.py` - Gymnasium-compatible Snake environment
- `snake_features.py` - CNN feature extractor
- `train_curriculum.py` - Curriculum training script
- `make_env.py` - Environment factory utilities

### Web Deployment
- `web_model_new/weights.json` - Exported model weights (~12MB)
- `web_model_new/agent-curriculum.js` - TensorFlow.js inference
- `web_model_new/script-curriculum.js` - Game loop
- `index.html` - Web demo page

### Utilities
- `evaluation_utils.py` - Evaluation helpers
- `rollout_visualization.py` - Visual rollout script
- `export_tfjs_direct.py` - Weight export to JSON

## Running the Demo

### Web (Browser)
Open `index.html` in a browser or serve locally:
```bash
python -m http.server 8000
# Visit http://localhost:8000
```

### Training
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train from scratch
python train_curriculum.py

# Evaluate existing model
python -c "
from stable_baselines3 import PPO
from snake_env import SnakeEnv

model = PPO.load('models/snake_curriculum_20251205_121938/final_model.zip')
env = SnakeEnv(n=20)

for _ in range(5):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(action))
        done = term or trunc
    print(f'Score: {info[\"score\"]}')
"
```

## Export to TensorFlow.js

```bash
python export_tfjs_direct.py models/snake_curriculum_*/final_model.zip web_model_new
```

This exports weights as JSON that can be loaded in the browser with the custom agent class.
