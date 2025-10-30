# CLAUDE.md - Racing AI Training Guide

This document provides guidance for training and deploying the racing AI model.

## Project Overview

A reinforcement learning racing game where an AI agent learns to navigate a bean-shaped track with an upward arch. The AI is trained using PPO (Proximal Policy Optimization) via Stable Baselines3 and deployed in the browser using TensorFlow.js.

## Environment Configuration

### Physics Parameters (Synchronized across Python and JavaScript)
- **Files to update**: `racer_env.py` and `car.js`
- **Key parameters**:
  - `max_speed`: 20 (doubled from original 10 for challenging racing dynamics)
  - `acceleration`: 0.4
  - `brake_force`: 0.4
  - `drag_coefficient`: 0.97
  - `turn_speed_base`: 0.02
  - `turn_speed_decrease`: 0.8

### Track Definition
- **Shape**: Bean-shaped oval with upward arch at bottom
- **Files**: `track.js` (JavaScript) and `racer_env.py` (Python)
- **Points**: 15 outer points, 15 inner points
- **Width**: 120px constant
- Both environments must have identical track layouts for consistent training/deployment

## Training Process

### 1. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Standard training (10M timesteps, 10 parallel environments)
python train_sb3.py --train --timesteps 10000000 --envs 10

# Quick test (100k timesteps)
python train_sb3.py --train --timesteps 100000 --envs 4

# Background training
source venv/bin/activate && python train_sb3.py --train --timesteps 10000000 --envs 10 &
```

### 3. Monitor Training
- Models saved to: `models/ppo_racer_YYYYMMDD_HHMMSS/`
- Checkpoints: Every 100k timesteps
- Best model: `best_model/best_model.zip`
- TensorBoard logs: `tensorboard/`

### 4. Training Metrics to Watch
- **Mean reward**: Should increase over time (target: 3000+)
- **Episode length**: Should approach max (3000 steps)
- **Distance traveled**: Should increase significantly
- **Collisions**: Should decrease to near zero
- **Lap completions**: Ultimate goal

## Model Export for Web Deployment

### Convert to TensorFlow.js Format
The web app expects the model at `models/ppo_weights.json`:

```bash
# Export the best model
source venv/bin/activate
python convert_to_tfjs.py models/ppo_racer_YYYYMMDD_HHMMSS/best_model/best_model.zip --output models/ppo_weights.json

# Export a specific checkpoint
python convert_to_tfjs.py models/ppo_racer_YYYYMMDD_HHMMSS/checkpoints/ppo_racer_400000_steps.zip --output models/ppo_weights.json
```

### Validation
The conversion script validates the exported weights. Small errors (<0.1) are acceptable due to precision differences.

## Web Application Files

### Core Files
- `index.html`: Main application entry
- `game.js`: Game loop and initialization
- `car.js`: Car physics (must match Python)
- `track.js`: Track definition (must match Python)
- `racer_agent.js`: AI agent that loads and runs the model
- `models/ppo_weights.json`: Trained model weights

### Testing the Model
1. Ensure model is exported to `models/ppo_weights.json`
2. Open `index.html` in a web browser
3. Press 'A' to enable AI control
4. The AI will load automatically from the JSON file

## Common Tasks

### Update Physics for Higher/Lower Speed
1. Edit `max_speed`, `acceleration`, `brake_force` in both:
   - `racer_env.py` (lines 57-64)
   - `car.js` (lines 15-22)
2. Retrain the model with new physics
3. Export and test

### Check Training Progress
```bash
# List all training sessions
ls -la models/

# Check latest checkpoint
ls -la models/ppo_racer_*/checkpoints/

# View training logs
tensorboard --logdir models/ppo_racer_*/tensorboard/
```

### Resume Training from Checkpoint
```bash
# Edit train_sb3.py to load from checkpoint (not currently implemented)
# Or start fresh training with same parameters
```

## Troubleshooting

### Model Not Loading in Browser
- Check browser console for errors
- Verify `models/ppo_weights.json` exists
- Ensure JSON file is valid (not corrupted)
- Check CORS if loading from different domain

### Poor AI Performance
- Train for more timesteps (10M+ recommended)
- Check physics parameters match between Python and JS
- Verify track definition is identical
- Ensure reward function encourages desired behavior

### Training Not Converging
- Check environment reset works correctly
- Verify observations are normalized properly
- Adjust PPO hyperparameters if needed
- Ensure episode doesn't timeout too early

## Background Training Sessions
Multiple training sessions can run in parallel. Use the BashOutput tool to monitor them:
- Check output: `BashOutput bash_id=<session_id>`
- Kill session: `KillShell shell_id=<session_id>`

## Key Insights
- Higher speeds (20 vs 10) require the AI to learn braking and proper racing lines
- The bean-shaped track with upward arch provides interesting challenges
- Training typically shows good progress after 200-400k timesteps
- Best models achieve 0 collisions and maximize distance traveled