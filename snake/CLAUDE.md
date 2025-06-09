# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered Snake game that uses Deep Q-Learning (DQN) with TensorFlow.js to train an autonomous agent. The project demonstrates reinforcement learning concepts through an interactive browser-based game.

## Architecture

The codebase follows a modular architecture with clear separation between game logic, AI agent, and visualization:

- **Game Engine** (`snake.js`): Core game state management, collision detection, and rendering. Uses a 20x20 grid with configurable parameters for max moves without food (starvation).

- **AI Agent** (`agent.js`): Implements Double DQN with epsilon-greedy exploration. State representation uses 12 binary inputs encoding snake direction, food direction, and immediate dangers. Maintains experience replay buffer (100k capacity) for batch training.

- **Neural Network** (`tf-model.js`): Policy and target networks with architecture: Input(12) → Dense(256, ReLU) → Dense(256, ReLU) → Output(4). Target network updates every 100 steps for stability.

- **Visualization** (`visualization.js`): Real-time Chart.js graphs tracking score and epsilon decay over episodes. Performance-optimized with decimation and no animations.

- **Main Controller** (`script.js`): Orchestrates training (1000 episodes) and testing (100 episodes) cycles. Configures game parameters and manages the game loop.

## Development

This is a client-side only application:

```bash
# Start local server from snake directory
python -m http.server 8000    # Python 3
# or
npx http-server               # Node.js

# Open http://localhost:8000 in browser
```

No build process, compilation, or server-side components required. All ML computation runs in the browser using TensorFlow.js.

## Key Technical Details

- **RL Algorithm**: Double DQN with experience replay
- **Hyperparameters**: 
  - Learning rate: 0.001 (Adam optimizer)
  - Discount factor (gamma): 0.95
  - Epsilon decay: 0.995 (1.0 → 0.0)
  - Batch size: 1000
- **Reward Structure**: +10 (food), -1 (death), -0.01 (per move)
- **Memory Management**: Uses `tf.tidy()` to prevent tensor memory leaks