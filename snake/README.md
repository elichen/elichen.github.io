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