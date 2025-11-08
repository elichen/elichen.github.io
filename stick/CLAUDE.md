# CLAUDE.md - Stick Balancing Project

This file provides guidance for maintaining and improving the inverted pendulum swingup demonstration.

## Project Overview

Browser-based demonstration of a reinforcement learning agent performing the classic inverted pendulum swingup task. The agent learns to swing up a stick from the downward position and balance it upright through coordinated cart movements.

## Current Status (as of Nov 2025)

### Working Solution
- **Algorithm**: SAC (Soft Actor-Critic)
- **Training Steps**: 1,000,000
- **Balance Rate**: 87.7% during training, 76-82% in testing
- **Training Time**: ~100 minutes
- **Model Size**: 263 KB (TensorFlow.js format)

### What Works
1. Swingup behavior is reliably learned
2. Agent maintains balance for extended periods
3. Continuous action control provides smooth movements
4. Shaped rewards enable convergence within 1M steps
5. Browser inference works well with TensorFlow.js

## Architecture

### Web Application
```
stick/
├── index.html           - Main page (distill.pub style)
├── styles.css           - Minimalist styling
├── main.js              - App logic, AI toggle, keyboard controls
├── ai_controller.js     - TensorFlow.js inference
├── environment.js       - Physics simulation (matches Python)
├── visualization.js     - Canvas rendering
└── models/tfjs/         - Trained model files
    ├── model.json       - Network architecture
    ├── group1-shard1of1.bin - Weights
    └── normalization.json    - Observation stats
```

### Training Code
```
stick_python/
├── stick_env.py         - Gymnasium environment
├── train_sac_final.py   - SAC training script
├── export_sac_to_tfjs.py - Model export utility
├── requirements.txt     - Python dependencies
└── README.md            - Training documentation
```

### Environment Specification

**Observation Space** (4D):
- Cart position: [-2.4, 2.4] meters
- Cart velocity: [-6, 6] m/s
- Stick angle: [-π, π] radians (π = downward, 0 = upright)
- Angular velocity: rad/s (unbounded)

**Action Space**:
- Continuous: [-1, 1] → target velocity [-5, 5] m/s
- Cart accelerates toward target velocity with gain of 25.0

**Reward Function** (shaped):
```python
upright_reward = 5.0 * cos(angle)           # Primary goal
energy_reward = 1.0 * energy_term           # Build correct energy
height_reward = 2.0 * height_term           # Get stick high
velocity_penalty = -0.001 * (v² + ω²)      # Smooth control
```

## Key Learnings

### Why SAC Worked (PPO Didn't)
1. **Sample Efficiency**: SAC uses off-policy learning with replay buffer
2. **Exploration**: Automatic entropy tuning maintains good exploration
3. **Continuous Control**: SAC is specifically designed for continuous actions
4. **Stability**: Less sensitive to hyperparameters than PPO

### Critical Success Factors
1. **Shaped Rewards**: Energy and height terms guide swingup learning
2. **Observation Normalization**: VecNormalize crucial for stability
3. **Large Network**: [256, 256] handles complex dynamics
4. **Learning Rate**: 3e-4 was optimal (tested multiple values)

### What Didn't Work
- PPO with default hyperparameters (< 10% balance rate at 1M steps)
- PPO with various hyperparameter tweaks (tested extensively)
- Sparse rewards only (too difficult to learn swingup)
- Small networks [64, 64] (insufficient capacity)

## Next Steps: Achieving 100% Balance Rate

### Goal
Train a policy that achieves 100% balance rate (or close to it) using PPO or improved SAC.

### Potential Approaches

#### Option 1: Extended SAC Training
- Train beyond 1M steps (try 5M or 10M)
- Fine-tune learning rate schedule
- Adjust reward weights for stricter balance requirements

#### Option 2: Optimized PPO
Research suggests PPO can work with proper setup:
- **Hyperparameters to tune**:
  - Learning rate: Try 1e-4 to 5e-4
  - n_steps: 2048-4096 for better value estimates
  - Clip range: 0.1-0.3
  - GAE lambda: 0.95-0.99
- **Training improvements**:
  - Longer training (3M+ steps)
  - Curriculum learning (gradually increase balance requirement)
  - Episode truncation at failures early in training

#### Option 3: Curriculum Learning
1. Phase 1: Learn swingup (current shaped rewards)
2. Phase 2: Refine balance (stricter angle threshold)
3. Phase 3: Perfect control (penalize any deviation)

#### Option 4: Alternative Algorithms
- **TD3**: Twin Delayed DDPG, stable for continuous control
- **TQC**: Truncated Quantile Critics, state-of-the-art sample efficiency
- **DDPG**: If simpler approach needed

### Recommended Next Attempt

Try **PPO with optimized hyperparameters**:
```python
# Key changes from failed attempts:
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
clip_range = 0.2
gae_lambda = 0.95
net_arch = [256, 256]
use_sde = True
target_timesteps = 3_000_000  # Give it more time
```

## Development Workflow

### Training a New Model
```bash
cd stick_python
source venv/bin/activate
python train_sac_final.py --timesteps 1000000
python train_sac_final.py --test  # Verify performance
```

### Exporting to Web
```bash
python export_sac_to_tfjs.py \
  --model models_sac_final/sac_stick_final.zip \
  --vec-norm models_sac_final/vec_normalize.pkl \
  --output ../models/tfjs
```

### Testing Locally
```bash
cd ..  # Back to stick/
python -m http.server 8000
# Open http://localhost:8000/
```

## Physics Matching

The JavaScript environment (environment.js) must exactly match the Python environment (stick_env.py):
- Same gravity, masses, lengths, damping
- Identical control scheme and boundary behavior
- Same reward calculation
- Matching angle normalization [-π, π]

**Important**: Any changes to physics must be synchronized between both implementations.

## Model Export Notes

SAC actor network structure:
1. Features extractor (observation → 256)
2. Latent policy network (256 → 256)
3. Mean network (256 → 1 with tanh)

The exported model outputs continuous actions in [-1, 1].

## Troubleshooting

### Model Not Loading
- Check browser console for CORS errors
- Verify model files in `models/tfjs/`
- Ensure normalization.json is present

### Poor Performance
- Check if observation normalization is applied
- Verify physics constants match between JS and Python
- Test model in Python first before exporting

### Training Instability
- Enable observation normalization (VecNormalize)
- Try lower learning rate
- Check reward shaping isn't too aggressive

## Resources

- **SAC Paper**: https://arxiv.org/abs/1801.01290
- **Stable-Baselines3 Docs**: https://stable-baselines3.readthedocs.io/
- **RL Zoo**: https://github.com/DLR-RM/rl-baselines3-zoo (hyperparameters)

## Maintenance Notes

When updating this demo:
1. Always test in Python before exporting
2. Keep environment.js synchronized with stick_env.py
3. Verify model size (keep under 500 KB for fast loading)
4. Update portfolio index.html description if performance improves
5. Document major changes in commit messages
