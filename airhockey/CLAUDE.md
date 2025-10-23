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

### Training Scripts (NO DUPLICATES - Check before creating)
```
air_hockey_env_v3.py    - Gym environment, sparse rewards only
train_selfplay_v3.py    - Training script with self-play (NO train_selfplay.py)
export_to_onnx.py       - ONNX export (embeds weights, no external data)
test_observation.py     - Test observation space

Shell scripts:
run_training.sh         - Run training with parameters
deploy_model.sh         - Export and deploy model
quick_test.sh          - Quick training test
```

### Models (Single)
```
model/ppo_selfplay_v3_final.onnx  - One model only (no timestamps)
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
- Sparse rewards only (no reward shaping)
- Single balanced strategy (2/3, -1, -1/3)
- Self-play with fictitious opponents
- No hyperparameter options in UI

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