# 🏎️ Racing AI with Stable Baselines3 and TensorFlow.js

A browser-based racing game with an AI driver trained using reinforcement learning (PPO algorithm from Stable Baselines3) and deployed with TensorFlow.js for real-time inference.

## Features

- **Ray-based Sensors**: 9 distance sensors for environment perception
- **Continuous Control**: Smooth steering and throttle control
- **PPO Training**: State-of-the-art reinforcement learning algorithm
- **Browser Deployment**: Real-time AI inference using TensorFlow.js
- **Visual Debugging**: Optional ray sensor visualization

## Project Structure

```
racer/
├── index.html          # Main game interface
├── game.js            # Game loop and AI integration
├── car.js             # Car physics and controls
├── track.js           # Track geometry and collision detection
├── lapTimer.js        # Lap timing functionality
├── racer_agent.js     # TensorFlow.js AI inference
├── racer_env.py       # Gymnasium environment for training
├── train_sb3.py       # PPO training script
├── convert_to_tfjs.py # Model conversion utility
└── models/
    ├── ppo_weights.json           # Deployed model (200k steps)
    └── ppo_racer_20251029_150739/ # Training artifacts
```

## How to Use

### Playing the Game

1. Open `index.html` in a modern web browser (or use local server)
2. Use arrow keys for manual control:
   - ⬆️ Accelerate
   - ⬇️ Brake/Reverse
   - ⬅️➡️ Steer

### AI Controls

- Press **A** to toggle AI mode
- Press **R** to toggle ray sensor visualization
- Use the buttons in the UI panel

## Training Your Own Model

### Prerequisites

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch gymnasium numpy stable-baselines3 tensorboard tqdm rich matplotlib
```

### Training

```bash
# Quick test (100k steps)
python train_sb3.py --train --timesteps 100000 --envs 4

# Full training (1M steps with 10 parallel environments)
python train_sb3.py --train --timesteps 1000000 --envs 10

# Test a trained model
python train_sb3.py --test models/ppo_racer_*/best_model/best_model
```

### Converting to TensorFlow.js

```bash
python convert_to_tfjs.py models/ppo_racer_*/checkpoints/ppo_racer_200000_steps.zip --output models/ppo_weights.json
```

## Technical Details

### Observation Space (11 dimensions)
- 9 ray sensors: Normalized distances to track boundaries (-90° to +90°)
- Speed: Normalized current velocity
- Angular velocity: Rate of rotation

### Action Space (2 dimensions)
- Steering: Continuous [-1, 1]
- Throttle/Brake: Continuous [-1, 1]

### Reward Function (Critical for Success!)
- **Large bonus** for lap completion (inversely proportional to time)
- **Strong reward** for forward movement (0.1 × distance)
- **Penalty** for backward movement (-0.5 × distance) - prevents reverse driving
- Progress rewards through checkpoints (0.5 × progress)
- Mild collision penalty (-10) to encourage aggressive driving
- Time penalty (-0.1) to encourage speed

### Network Architecture
- Custom feature extractor: Input (11) → 256 → 256 → 256 (features)
- Policy network (Actor): Features → 256 → 256 → 2 actions
- Value network (Critic): Features → 256 → 256 → 1 value
- Total parameters: ~266,756

## Training Progress

### 🏆 Current Deployed Model (200k timesteps)

**Performance Metrics:**
- Max distance achieved: **15,595 units** (equivalent to 5-6 full laps!)
- Typical distances: 2,000-6,000 units per episode
- Near-zero collision rate
- Smooth, aggressive forward driving
- Proper counter-clockwise navigation

**Training Evolution:**
1. **Phase 1 (0-100k)**: Discovered reverse-driving exploit
   - Problem: Reward function didn't distinguish forward/backward
   - Learned to drive in reverse to go counter-clockwise

2. **Phase 2 (100k-200k)**: Fixed with proper reward shaping
   - Added strong penalty for backward movement
   - 10x reward increase for forward movement
   - Immediate improvement in distances (100→1,000→15,000 units)

**Technical Achievements:**
- Parallelization: 10 environments for 1,171 FPS throughput
- Actor-Critic architecture with separate policy/value networks
- Ray-based perception (9 sensors) for robust navigation
- Continuous control for smooth, realistic driving

## Performance Tips

- Train for at least 200k timesteps for good driving behavior
- Use 8-10 parallel environments for optimal training speed
- The forward/backward penalty is critical - don't skip it!
- Adjust checkpoint detection threshold if lap counting seems off
- Monitor reward trends - positive eval rewards indicate good progress

## Browser Compatibility

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support (may need to enable WebGL)

## Future Improvements

- Multiple track layouts
- Opponent AI for racing
- Curriculum learning (progressive difficulty)
- Vision-based input (camera view)
- Online leaderboard
- Lap completion optimization (currently tracks distance, needs lap counting fix)

## Lessons Learned

1. **Reward shaping is critical**: The AI will exploit any loopholes (e.g., driving in reverse)
2. **Parallelization matters**: 10 envs = 2.5x faster than 4 envs
3. **Direction matters**: Counter-clockwise vs clockwise must be consistent everywhere
4. **Distance ≠ Laps**: High distances don't always mean lap completion (detection logic matters)

## License

MIT

## Acknowledgments

- Stable Baselines3 for the RL algorithms
- TensorFlow.js for browser deployment
- OpenAI Gymnasium for the environment framework