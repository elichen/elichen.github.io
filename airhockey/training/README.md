# Air Hockey PPO Training Pipeline

This directory contains the complete training pipeline for training a PPO (Proximal Policy Optimization) agent to play air hockey, then converting and deploying it to the web application.

## Overview

The pipeline consists of:
1. **Python Training** - Train PPO agent using Stable Baselines 3
2. **Model Export** - Export PyTorch model to ONNX format
3. **Model Conversion** - Convert ONNX to TensorFlow.js format
4. **Web Deployment** - Deploy model to web application

## Quick Start

### Option 1: Run Complete Pipeline

Run the entire pipeline with default settings (500,000 timesteps):

```bash
cd training
./run_training.sh
```

Or specify custom number of timesteps:

```bash
./run_training.sh 1000000  # Train for 1 million timesteps
```

This will:
- Test the environment
- Train the PPO agent
- Evaluate the trained model
- Export to ONNX
- Convert to TensorFlow.js
- Copy model to web directory

### Option 2: Run Steps Manually

#### 1. Test Environment

```bash
python test_env.py
```

This validates that the Python environment matches the JavaScript implementation exactly.

#### 2. Train Model

```bash
# Basic training (500k timesteps, ~2-4 hours)
python train_ppo.py --timesteps 500000

# Extended training (1M timesteps, ~4-8 hours)
python train_ppo.py --timesteps 1000000

# With observation normalization (may improve stability)
python train_ppo.py --timesteps 500000 --normalize

# Continue training from checkpoint
python train_ppo.py --timesteps 500000 --load-model models/checkpoint.zip
```

Training progress can be monitored with TensorBoard:

```bash
tensorboard --logdir models/tensorboard
```

#### 3. Evaluate Model

```bash
# Evaluate for 10 episodes
python evaluate_model.py --model models/ppo_airhockey_YYYYMMDD_HHMMSS_final

# Watch agent play continuously
python evaluate_model.py --model models/ppo_airhockey_YYYYMMDD_HHMMSS_final --watch
```

#### 4. Export to ONNX

```bash
python export_to_onnx.py --model models/ppo_airhockey_YYYYMMDD_HHMMSS_final
```

This creates `models/onnx/ppo_airhockey_YYYYMMDD_HHMMSS_final.onnx`

#### 5. Convert to TensorFlow.js

```bash
python convert_to_tfjs.py --model models/onnx/ppo_airhockey_YYYYMMDD_HHMMSS_final.onnx
```

Or convert and automatically copy to web directory:

```bash
python convert_to_tfjs.py --model models/onnx/ppo_airhockey_YYYYMMDD_HHMMSS_final.onnx --copy-to-web
```

## Using Pre-trained Model in Web App

After conversion, the model is copied to `../model/`. To use it in the web application:

1. Open the browser console on the air hockey page
2. Load the pre-trained model:

```javascript
// Load pre-trained model
await agent.loadPretrainedModel('model/model.json');

// The model is now active! Play against it by switching to Play mode
```

Or add automatic loading to `game.js`:

```javascript
// In game.js initialization
async function init() {
    // ... existing code ...

    // Load pre-trained model if available
    try {
        await agent.loadPretrainedModel('model/model.json');
        console.log('Using pre-trained model');
    } catch (e) {
        console.log('No pre-trained model found, using trainable model');
    }

    // ... rest of initialization ...
}
```

## File Structure

```
training/
├── air_hockey_env.py       # Gymnasium environment (matches JS)
├── test_env.py             # Environment validation tests
├── train_ppo.py            # PPO training script
├── evaluate_model.py       # Model evaluation
├── export_to_onnx.py       # ONNX export
├── convert_to_tfjs.py      # TensorFlow.js conversion
├── run_training.sh         # Complete pipeline script
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── models/                 # Saved models directory
    ├── checkpoints/        # Training checkpoints
    ├── best/               # Best models during training
    ├── logs/               # Evaluation logs
    ├── tensorboard/        # TensorBoard logs
    ├── onnx/              # ONNX exports
    └── tfjs/              # TensorFlow.js models
```

## Environment Details

The Python environment (`air_hockey_env.py`) exactly matches the JavaScript implementation:

- **Canvas**: 600x800 pixels
- **Observation Space**: 14 dimensions (normalized to [-1, 1])
  - Puck position (relative to paddle)
  - Puck velocity
  - Opponent position (relative to paddle)
  - Distance/angle to puck
  - Goal distances
  - Paddle velocity
  - Puck speed
- **Action Space**: 2D continuous ([-1, 1])
  - X-axis movement
  - Y-axis movement
- **Reward Structure**: 3-stage curriculum learning
  1. HIT_PUCK: Learn to hit the puck (500 hits to advance)
  2. SCORE_GOAL: Learn to score goals (50 goals to advance)
  3. STRATEGY: Strategic gameplay

## Training Hyperparameters

Matching the JavaScript implementation:

```python
learning_rate = 0.0005
gamma = 0.99           # Discount factor
gae_lambda = 0.95      # GAE lambda
clip_range = 0.2       # PPO clipping
ent_coef = 0.02        # Entropy coefficient
n_steps = 2048         # Steps per update
batch_size = 64        # Batch size
n_epochs = 10          # Training epochs per update
```

Network architecture:
- **Actor**: 4 layers × 256 units (ReLU activation)
- **Critic**: 4 layers × 256 units (ReLU activation)

## Troubleshooting

### Training is slow
- Use GPU if available: `--device cuda` (NVIDIA) or `--device mps` (Apple Silicon)
- Reduce timesteps for quick testing: `--timesteps 100000`
- The environment runs at simulation speed (no rendering overhead)

### Model conversion fails
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that ONNX export created valid file: `ls -lh models/onnx/`
- Try updating packages: `pip install --upgrade onnx onnx-tf tensorflowjs`

### Web model not loading
- Check browser console for errors
- Verify model files exist: `ls -lh ../model/`
- Ensure `model.json` and `.bin` files are present
- Check that model path is correct: `model/model.json` (relative to HTML file)

### Agent performance is poor
- Train longer: `--timesteps 1000000` or more
- Check evaluation results: did the agent reach STRATEGY stage?
- Review TensorBoard logs for learning curves
- Try with normalization: `--normalize` flag

## Expected Training Timeline

- **20,000 steps** (~30 mins): Basic puck hitting
- **100,000 steps** (~2 hours): Consistent hits, entering SCORE_GOAL stage
- **300,000 steps** (~4 hours): Scoring goals, entering STRATEGY stage
- **500,000+ steps** (~6+ hours): Strategic gameplay, good win rate

## Tips for Best Results

1. **Monitor training**: Use TensorBoard to watch reward curves
2. **Save checkpoints**: Models are saved every 50k steps automatically
3. **Evaluate frequently**: Run evaluation to check progress
4. **Curriculum learning**: Agent must progress through all 3 stages
5. **Self-play**: Agent learns by playing against itself (both paddles use same network)

## Advanced Usage

### Custom Training Schedule

```python
# In train_ppo.py, modify curriculum thresholds:
env.HITS_TO_ADVANCE = 1000  # Spend more time learning to hit
env.GOALS_TO_ADVANCE = 100  # Spend more time learning to score
```

### Fine-tuning Pre-trained Model

```bash
# Continue training from a pre-trained model
python train_ppo.py --load-model models/ppo_airhockey_final --timesteps 100000
```

### Batch Conversion

```bash
# Convert multiple models
for model in models/*.zip; do
    python export_to_onnx.py --model "$model"
done
```

## Performance Benchmarks

On Apple M1/M2:
- Training: ~1000 steps/second
- Full training (500k): 2-4 hours
- ONNX export: <1 minute
- TF.js conversion: 1-2 minutes

On CPU (Intel i7):
- Training: ~200-400 steps/second
- Full training (500k): 8-12 hours

## Next Steps

After successful training and deployment:

1. Test the agent in the web interface
2. Compare against the JavaScript-trained agent
3. Fine-tune if needed
4. Share your results!

## Support

For issues or questions:
- Check the main project README
- Review this documentation
- Check browser/Python console for error messages
