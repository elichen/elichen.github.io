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
game.js            - Game loop (fixed 60 Hz), human paddle, self-play jitter
environment.js     - Game physics only
ppo_agent.js       - Pure-JS MLP inference + 12-feature observation
model/policy.bin   - float32 [n, sizes..., per layer W (out x in), b]
```

### Training (`training/`, JAX on the GPU)
```
hockey.py       - physics port of environment.js + game.js moveAgentPaddle, scripted players, MLP, match helpers
parity.mjs/.py  - frame-by-frame parity of hockey.step against the browser (must print PASS)
bc.py           - DAgger clone of the scripted expert (the starting policy)
league.py       - PPO league: latest self, PFSP snapshot pool, fixed opponents (--vs), scripted bots
final_eval.py   - continuous browser-style matches, goals per minute
```

## Physics
Puck reflects off paddles as off a hand-held mallet (kinematic, restitution 0.8), walls 0.9, friction 0.997/frame, max 30 px/frame, 4 substeps. Paddles: 10 px/frame, 0.6/0.4 smoothing, own half only. 1200-frame (20 s) point timeout; conceder serves. Any change to environment.js must be mirrored in hockey.py and re-checked with parity.

## Training & Deployment
Runs on Nitro (`~/play/airhockey`, `~/play/cloth-venv`), ~600k steps/s.
```bash
cd training
node parity.mjs > parity.json && python parity.py parity.json      # physics parity, needs PASS
python bc.py --out bc.pkl                                            # ~5 min
python league.py --name L --init bc.pkl --logstd -1.2 --warmup 30 --lr 1e-4 --shape .2 --shape_end 600 --draw -.2
python league.py --name E --init F.pkl --vs F.pkl --vs_blocks 12 --self_blocks 0 --bots 0 ...   # exploiter vs frozen F
python league.py --name H --init F.pkl --vs F.pkl --vs_blocks 8 --delay 12 --motor .15 ...       # human-limited best response
python final_eval.py F.pkl --vs bot:12:6 late:12:.15:H/ckpt.pkl                                   # goals per minute
cp runs/<run>/policy_XXXXX.bin ../model/policy.bin
```

## Experimental Findings
1. **From scratch, sparse reward: the agent hides in a corner.** With real collisions, random touches score more own goals than goals. Cloning the scripted expert first fixes it; PPO then improves fast (beats the expert ~95% of points within ~100M steps).
2. **Critic warm-up** (`--warmup`, actor frozen) before PPO touches a cloned actor.
3. **Strength plateaus around 600M steps, then cycles.** Longer runs (L4, L5) beat the exploiters more but lose to their own starting point; pick the deployed policy by round-robin, not by latest checkpoint.
4. **Same-budget exploiters** reach ~0.58 match score against the deployed policy from random starts. Human-limited best responses (200 ms delay) lose ~90% of points.
5. Deterministic policy vs itself replays one point forever; the page adds small action jitter in self-play mode.

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