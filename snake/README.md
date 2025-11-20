# Snake AI - Deep Q-Learning Implementation

This project implements an AI that learns to play Snake using Deep Q-Learning (DQN) with TensorFlow.js for web deployment.

## Features

- **Web-based Training**: Train the AI directly in your browser using TensorFlow.js
- **Pre-trained Model**: Load a pre-trained model that can play Snake effectively
- **Real-time Visualization**: Watch the AI learn with live performance charts
- **Customizable Speed**: Adjust game speed for better viewing

## Files

### Web Application
- `index.html` - Original training interface
- `index-pretrained.html` - Demo with pre-trained model
- `snake.js` - Game engine
- `agent.js` / `agent-pretrained.js` - AI agent implementation
- `tf-model.js` / `tf-model-pretrained.js` - Neural network models
- `visualization.js` - Performance charts

### Training Scripts (Python)
- `train_fast.py` - Quick training script with web export
- `train_optimized.py` - Advanced training with better features
- `train_snake.py` - Full Rainbow DQN implementation
- `train_sb3_fullboard.py` - Stable-Baselines3 curriculum trainer with full-board CNN inputs
- `sb3_fullboard_env.py` - Gymnasium environment exposing the entire grid as a 4-channel tensor

### Pre-trained Models
- `web_model/model_weights.json` - Pre-trained weights for web
- `web_model/model_config.json` - Model configuration

## Usage

### Running the Pre-trained Demo

1. Open `index-pretrained.html` in a web browser
2. The AI will automatically load pre-trained weights and start playing
3. Use controls to:
   - Start/stop the demo
   - Continue training to improve performance
   - Adjust game speed

### Training from Scratch

1. Open `index.html` for browser-based training
2. Or use Python scripts for faster training:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python train_fast.py
   ```

### SB3 Full-Board Training (Mac-friendly)

1. Create a fresh environment and install the RL stack:
   ```bash
   python3 -m venv sb3
   source sb3/bin/activate
   pip install -r requirements.txt
   ```
2. Launch the curriculum trainer (defaults: grids 7→10→14→20, 2M PPO steps each, 8 parallel envs on MPS/CPU):
   ```bash
python train_sb3_fullboard.py \
    --grid-sizes 7,10,14,20 \
    --steps-per-stage 2000000 \
    --n-envs 8 \
    --export-obs
```

To continue training from an existing SB3 checkpoint, add `--resume-from sb3_fullmap_model/checkpoints/final_fullboard.zip` (any `.zip` produced by SB3 works, as long as the observation configuration matches).
3. Monitor progress via TensorBoard:
   ```bash
   tensorboard --logdir sb3_fullmap_model/tensorboard
   ```
4. The best checkpoints land under `sb3_fullmap_model/checkpoints/` and the latest deterministic policy is `sb3_fullmap_model/checkpoints/final_fullboard.zip`.

Use `sb3_fullboard_env.py` if you want to plug the full-board observation into other SB3 algorithms or evaluation scripts. The optional `--export-obs` flag drops a `sample_observation.json` you can feed into ONNX/TF.js conversion pipelines for validating preprocessing outside SB3.

> **Note:** The training requirements are Torch/SB3-only. If/when you need to convert a PyTorch checkpoint to TensorFlow.js, install the optional packages listed in `requirements-conversion.txt` inside a separate Python environment (TensorFlow does not yet support Python 3.13, so use Python 3.10–3.11 via `pyenv`/Conda for the conversion step).

### Evaluating a Checkpoint

Sample 10 deterministic games to check how many foods the model eats on average:

```bash
source sb3/bin/activate
python eval_sb3_checkpoint.py \
    --model sb3_fullmap_model/checkpoints/final_fullboard.zip \
    --grid-size 20 \
    --episodes 10
```

If you evaluate an intermediate curriculum stage (e.g., 10×10 grid) but trained with a larger observation size, add `--obs-grid-size 20` so the observation tensor matches what the policy expects. The script prints per-episode foods/steps plus a summary block with averages so you can quickly compare checkpoints without replaying them in the browser.

### Exporting to TensorFlow.js

```bash
source sb3/bin/activate
python convert_to_tfjs.py \
    --checkpoint sb3_fullmap_model/checkpoints/grid20_4000000_steps.zip \
    --sample sb3_fullmap_model/sample_observation.json \
    --output snake_policy.onnx

onnx-tf convert -i snake_policy.onnx -o snake_policy_tf
tensorflowjs_converter --input_format=tf_saved_model snake_policy_tf web_model/snake_tfjs
```

This produces an ONNX graph plus a TF SavedModel that can be converted into a TF.js bundle (stored in `web_model/snake_tfjs`). Update `index.html` to load the new TF.js weights once the front-end agent matches the stacked observation shape (16 board channels and 24 scalar stats).

## Model Architecture

- **Input**: 12 binary features (snake direction, food direction, danger detection)
- **Hidden Layers**: 2 layers with 256 neurons each (ReLU activation)
- **Output**: 4 Q-values (one for each action: up, right, down, left)
- **Algorithm**: Double DQN with experience replay

## Performance

The pre-trained model can achieve:
- Average score: 5-10 (continues to improve with more training)
- Reliable food collection
- Basic obstacle avoidance
- No starvation issues

Further training can improve performance significantly.
