# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered Snake game that uses Deep Q-Learning (DQN) with TensorFlow.js to train an autonomous agent. The project demonstrates reinforcement learning concepts through an interactive browser-based game.

## Current Development Strategy (Updated)

### Approach: TensorFlow.js-First Architecture

After discovering compatibility issues between complex PyTorch architectures and TensorFlow.js, we've pivoted to a **TF.js-native approach**:

1. **Simple, Compatible Architecture**: Use only layers directly supported by TensorFlow.js
2. **Quick Validation**: 100-episode test runs to verify learning
3. **Direct Weight Loading**: Ensure weights load correctly in browser
4. **Scale Up After Validation**: Only increase complexity after proving the pipeline works

### Phase 1: Simple Model Validation ✅ (Current)

**Architecture** (`SimpleDQN` in `train_simple_dqn.py`):
```
Input (11 features) → Dense(128) → ReLU → Dense(128) → ReLU → Dense(4 outputs)
```

**State Features** (11 total):
- Direction one-hot (4): Current snake direction
- Food direction (2): Normalized dx, dy to food
- Danger detection (4): Walls/body in each direction
- Snake length (1): Normalized by grid area

**Training**:
```bash
python train_simple_dqn.py  # 100 episodes for validation
```

**Export**: Weights saved as `simple_model/weights.json` for direct TF.js loading

### Phase 2: Web Integration Test

**TensorFlow.js Model**:
```javascript
// Simple model that exactly matches Python architecture
const model = tf.sequential({
  layers: [
    tf.layers.dense({inputShape: [11], units: 128, activation: 'relu'}),
    tf.layers.dense({units: 128, activation: 'relu'}),
    tf.layers.dense({units: 4})
  ]
});

// Load weights directly
await model.loadWeights('./simple_model/weights.json');
```

### Phase 3: Full Training (After Validation)

Once the simple model shows:
- ✅ Successful weight loading in TF.js
- ✅ Intelligent behavior (avoiding walls, seeking food)
- ✅ Score improvement over episodes

Then scale up:
1. **Extended Training**: 10,000+ episodes
2. **Curriculum Learning**: 5x5 → 7x7 → 9x9 → 11x11 → 20x20
3. **Network Scaling**: Gradually increase hidden layer sizes
4. **Advanced Features**: Add more state features if needed

## File Structure

### Training Scripts
- `train_simple_dqn.py` - Simple TF.js-compatible training
- `train_perfect_snake.py` - Complex architecture (experimental, not TF.js compatible)

### Model Directories
- `simple_model/` - Simple DQN weights and metadata
  - `weights.json` - Direct weight values for TF.js
  - `metadata.json` - Architecture info
  - `model.pth` - PyTorch checkpoint
  
### Web Components
- `simple-agent.js` - TF.js agent using simple model
- `index.html` - Main game interface
- `snake.js` - Game engine
- `visualization.js` - Training charts

### Experimental/Advanced
- `model-components/` - Modular architecture components
- `direct-weights-agent.js` - Attempts to load complex PyTorch weights
- `hybrid-dqn-model.js` - Complex architecture implementation

## Key Learnings

### What Works in TF.js ✅
- Dense/Linear layers
- Conv2D, MaxPool, AvgPool
- BatchNormalization
- LSTM, GRU (built-in)
- Standard activations (ReLU, Sigmoid, Tanh)
- Dropout

### What Doesn't Work in TF.js ❌
- Custom PyTorch modules (SpatialAttention)
- Multi-head Attention (needs custom implementation)
- Complex nested architectures
- Direct PyTorch state dict loading

### Architecture Complexity vs Performance

| Architecture | Parameters | TF.js Compatible | Best Score |
|-------------|------------|------------------|------------|
| SimpleDQN | ~20K | ✅ Yes | TBD |
| OriginalDQN | ~150K | ✅ Yes | ~15 |
| HybridDQN | ~2.8M | ❌ No | 37+ |

## Development Commands

### Quick Start
```bash
# Train simple model
python train_simple_dqn.py

# Start web server
python -m http.server 8000

# Open browser to http://localhost:8000
```

### Testing
```bash
# Test model components
open http://localhost:8000/model-components/test-components.html

# Test integrated model
open http://localhost:8000/test-integrated.html
```

## Next Steps

1. **Immediate**: Run `train_simple_dqn.py` for 100 episodes
2. **Validate**: Load weights in TF.js and verify snake behavior
3. **Iterate**: Adjust architecture if needed (keeping TF.js compatibility)
4. **Scale**: Once working, train for thousands of episodes
5. **Optimize**: Fine-tune hyperparameters for perfect game achievement