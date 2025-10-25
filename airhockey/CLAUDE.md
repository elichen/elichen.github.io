# CLAUDE.md - Air Hockey AI Project Style Guide

## Core Philosophy
**Sacrifice readability for conciseness.** Write dense, minimal code that works.

## Style Rules

### 1. Minimal Comments
- No docstrings unless absolutely critical
- No inline explanations for obvious code
- Remove "helpful" comments that state the obvious

### 2. No Fallbacks or Error Handling
- Assume everything works
- No try/catch blocks for fallback behavior
- No "graceful degradation" - if it fails, let it fail
- Remove all console.log statements except critical ones

### 3. Single Path Execution
- One model, one strategy, one way
- No difficulty levels or options
- No alternative implementations
- Remove all conditional paths that aren't essential

### 4. Concise Over Clear
- Short variable names where context is obvious
- Chain operations instead of intermediate variables
- Use ternary operators liberally
- Inline simple functions

### 5. No Defensive Programming
- Don't check if files exist before reading
- Don't validate inputs
- Assume correct usage
- Remove all "safety" checks

### 6. Minimal Dependencies
- Use only what's absolutely necessary
- No convenience libraries
- Direct implementations over abstractions

## Project Structure

### Web App (Minimal)
```
index.html         - Basic UI, no frills
game.js           - Core game loop, <100 lines ideal
environment.js    - Game physics only
ppo_agent.js      - ONNX inference only
styles.css        - Minimal styling
```

### Training Scripts
```
air_hockey_env.py    - Gym environment (12 features: puck-focused, dense rewards)
train_selfplay.py    - Training script with self-play + parallel envs
export_to_onnx.py    - ONNX export (auto-detects obs dimension)
evaluate_model.py    - CRITICAL: Validate model actually plays

Shell scripts:
run_training.sh      - Run training with parameters
deploy_model.sh      - Export and deploy model
quick_test.sh        - Quick training test
```

### Models (Single)
```
model/ppo_selfplay_final.onnx  - Current production model
```

## What to Remove
- Difficulty selection
- Multiple AI strategies
- Fallback mechanisms
- Error recovery code
- Verbose documentation
- Unused functions
- Alternative implementations
- Console logging (except critical)
- Input validation
- Helper functions that are used once

## Training Philosophy
- Dense reward shaping REQUIRED to break defensive Nash equilibrium
- Puck-focused observations (12 features, NO opponent tracking)
- Self-play with 20-opponent frozen pool
- Parallel training: batch_size scales with n_envs (160 for 10 envs)

## CRITICAL: Model Validation
**Training metrics LIE. ALWAYS validate with evaluate_model.py**

After training, MUST run:
```bash
python evaluate_model.py --model models/ppo_selfplay_final.zip --episodes 100
```

Success criteria:
- >70% win rate vs random (24% baseline with dense rewards)
- >0.5 goals/game (0.24 baseline)
- <50% timeout rate (75% baseline - still high)

## Experimental Findings

1. **Dense Rewards Break Defensive Deadlock**:
   - Sparse rewards (2/3, -1, -1/3) create defensive Nash equilibrium
   - Dense shaping needed: +0.001*puck_speed, +0.01 offensive positioning, -0.005*dist_to_puck
   - Result: 3x improvement (8% → 24% win rate vs random)

2. **Opponent Observations Harm Offensive Play**:
   - WITH opponent tracking: Agent shadows opponent instead of attacking puck
   - WITHOUT opponent: Forces puck engagement (testing in progress)
   - 12-feature puck-focused > 16-feature with opponent velocity

3. **Parallel Training Optimization**:
   - batch_size MUST scale with n_envs: `int(64 * n_envs / 4)`
   - Without scaling: 10 envs = 1.2x speedup (bottlenecked by training phase)
   - With scaling: 10 envs = 1.95x speedup
   - Optimal: 10 envs on 14-CPU system (~2,900 FPS)

4. **Fictitious Self-Play REQUIRED**:
   - 20-opponent frozen pool (checkpoints every 50k steps)
   - Prevents overfitting to single strategy

## When Asked to Modify
1. First remove before adding
2. Simplify existing code before extending
3. Combine multiple functions into one
4. Remove options, don't add them
5. Make it work with less code, not more

## Example Transformations

### Before (Verbose):
```javascript
/**
 * Load an ONNX model for inference
 * @param {string} modelPath - Path to model
 * @returns {Promise<boolean>} Success status
 */
async function loadONNXModel(modelPath) {
    try {
        console.log(`Loading model from ${modelPath}...`);
        if (!modelPath) {
            console.error('No model path provided');
            return false;
        }
        const session = await ort.InferenceSession.create(modelPath);
        console.log('Model loaded successfully');
        return true;
    } catch (error) {
        console.error('Failed to load model:', error);
        return false;
    }
}
```

### After (Concise):
```javascript
async loadONNXModel(modelPath) {
    this.onnxSession = await ort.InferenceSession.create(modelPath);
    return true;
}
```

## Remember
**Every line of code is a liability.** The best code is no code. The second best is minimal code that just works.