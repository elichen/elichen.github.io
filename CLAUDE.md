# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a GitHub Pages website (elichen.github.io) containing interactive browser-based experiments and applications focused on AI/ML demonstrations, games, and visualizations. All applications run client-side without requiring server infrastructure.

## Architecture

The repository follows a simple structure where each project exists in its own directory with:
- `index.html` - Main entry point
- JavaScript files for logic (often `script.js`, `main.js`, or domain-specific names)
- `styles.css` or similar for styling
- Project-specific assets and dependencies

Key project categories:
1. **AI/ML Experiments**: TensorFlow.js-based demos including reinforcement learning agents, neural networks, and language models
2. **Interactive Games**: Browser-based games, many featuring AI opponents or demonstrations
3. **Visualizations**: Data visualizations, simulations, and educational tools
4. **Utilities**: Tools like whiteboard, password cracker, etc.

## Development Commands

### GPT Language Model Project
Located in `/gpt/` directory with Node.js-based training:
```bash
cd gpt
npm install  # Install dependencies
npm run train  # Train model for 100 epochs
npm run train-long  # Train model for 1000 epochs
npm run generate  # Generate text from trained model
npm run interactive  # Interactive text generation
npm run test  # Quick test with 10 epochs
npm run clean  # Clean build artifacts
```

Note: Uses `run_with_node22.sh` wrapper script for Node.js v22 compatibility.

### General Development
Since this is primarily a static site:
1. Projects are self-contained - navigate to any project directory
2. Open `index.html` in a browser or use a local server
3. For local development with hot reload: `python -m http.server 8000` or `npx http-server`

## Key Technical Details

- **TensorFlow.js**: Many AI experiments use TensorFlow.js for browser-based ML
- **WebGL**: Several projects use WebGL for 3D graphics and visualizations
- **Canvas API**: Extensively used for 2D games and visualizations
- **No Build Process**: Most projects don't require compilation or bundling
- **Client-Side Only**: Everything runs in the browser - no backend servers

## Adding New Projects

1. Create a new directory at the root level
2. Add `index.html` as the entry point
3. Include project files (JS, CSS, assets)
4. Add entry to main `index.html` in appropriate section
5. Test locally before committing

## Testing

Since projects are browser-based:
1. Open the project's `index.html` in different browsers
2. Check browser console for errors
3. Test on different screen sizes for responsive design
4. Verify no external dependencies are broken

## Streaming RL Project (`/streaming-rl/`)

CartPole swingup with Stream Q(λ), from "Streaming Deep Reinforcement Learning Finally Works" (arXiv:2410.14606). The pole starts hanging down; a pretrained agent swings it up and keeps learning from every step, and sliders change the pole length and motor force mid-run so viewers can watch it re-adapt.

- `index.html` + `swingup-main.js`: page, controls (physics sliders, learning toggle, reset, speed) and run loop
- `cartpole-swingup.js`: environment (`setPhysics` for the sliders); `return-chart.js`: episode-return chart
- `agent.js`, `network.js`, `optimizer.js`, `normalization.js`: Stream Q(λ), with the per-weight bounded `StreamingOptimizer` from the paper's 2026 revision
- `train_swingup.py`: PyTorch training (2024 ObGD) that produced `trained-weights-swingup.json` / `trained-normalization-swingup.json`

```bash
cd streaming-rl
python3 train_swingup.py --steps 500000
```

**Slider limits are measured:** pole 1.0–1.6 m, motor 8–15 N, gravity 6–12 m/s². Outside them continual learning can collapse (6 N or gravity 13: the agent learns to drive into the wall; 1.8 m and gravity 5 fail in some seeds). Cart friction was tried and left out: below 5 N·s/m the frozen policy doesn't notice, above it learning makes things worse. Re-run a seeded adaptation sweep before widening them or changing optimizer settings. `AddTimeInfo.timeLimit` = 10000 on purpose (see the comment there).

The double pendulum moved to `/double-pendulum/` (an SB3 SAC policy, not streaming).

## Learning on Ice (`/learning-on-ice/`)

Distill-style article plus live demo of continual, streaming RL: a Unitree G1 humanoid (MuJoCo Menagerie) on MuJoCo's official WebAssembly build (`@mujoco/mujoco` 3.14 from jsDelivr), learning with Stream-AC from the 2026 revision of arXiv 2410.14606 while you change its environment.

- `g1-env.js` is shared by the page and the Node tools, so both run identical physics. It is set up like Unitree's own controller so Unitree's walking policy can teach ours: the agent drives only the 12 leg joints on Unitree's PD gains (waist and arms hold the stand pose), and the observation (36) is heading-free with a 0.8 s gait clock. 4 ms physics steps and robot-floor contacts only (~3x faster). `setFriction` changes every geom; `setBackpack` recomputes constants with `mj_setConst` on scratch data (it overwrites the state it is given).
- `robot/` is Menagerie's G1 packed by `tools/pack_g1.py` (decimated visual meshes, convex-hull collision meshes; same masses).
- `worlds.js` defines the environments (normal μ 0.6, ice μ 0.15, right leg at 20% torque, 15 kg backpack); the page, worker and tools all use it. Each changes physics only, never the observation.
- `worker.js` runs physics + learning off the main thread; `main.js` (UI), `render.js` (three.js, z-up; ice mirror, leg tint, backpack), `chart.js`
- `stream-ac.js` is the readable reference learner (also init, save/load); `wasm-learner.js` runs the same math from `stream-ac.wasm`, built from `tools/wasm-learner` (Rust, SIMD): `tools/wasm-learner/build.sh` (needs `rustup target add wasm32-unknown-unknown`)
- **Where the agent comes from:** `tools/distill.py` copies Unitree's policy (unitree_rl_gym `motion.pt`) into the actor with DAgger; its Python `G1` class mirrors `g1-env.js` and must stay in sync. Then the critic is warmed up with the actor fixed (`--lr-policy 0`), then 3M steps of streaming RL on the normal floor give `model/agent.{json,bin}`. RL with the full body (29 joints) or without the tilt penalty drifted into a hunched gait. RL from scratch walks too but with an odd gait, and only some seeds.
- Streaming RL dips before it improves on a freshly copied policy (the optimizer takes full-size steps even when TD errors are just noise); after a few million steps on the normal floor it is stable. `data/tour.json` is the article's recorded run.

```bash
cd learning-on-ice && npm install         # MuJoCo for Node (tools only)
node tools/check-wasm.mjs                 # wasm learner must match stream-ac.js; prints speeds
python3 tools/distill.py <unitree_rl_gym checkout> tools/runs/student.json   # needs mujoco, torch
node tools/pretrain.mjs --wasm --student tools/runs/student.json --lr-policy 0 --steps 300000 --out warm
node tools/pretrain.mjs --wasm --load warm --lr-policy 1e-4 --steps 2000000 --seed 1 --out g1-2m
node tools/pretrain.mjs --wasm --load g1-2m --steps 1000000 --seed 1 --out g1   # shipped as model/agent
node tools/pretrain.mjs --wasm --load g1 --steps 4000000 --worlds ice@0,normal@800000,hurt@1600000,normal@2400000,backpack@3200000 --out tour
node tools/export-run.mjs tour             # CSV -> data/tour.json for the article
```

Testing: a hidden or background Chrome tab runs the worker ~6x slower and pauses requestAnimationFrame, so measure speed in a visible tab.

## Robot Arm (`/robotarm/`)

A Franka Panda (MuJoCo Menagerie, as set up in MuJoCo Playground's PandaPickCube) picks up a box and throws it, simulated by MuJoCo's WebAssembly build (`@mujoco/mujoco` 3.14 from jsDelivr) on the main thread. Two Brax PPO policies trained on Nitro with MJX-Warp (see `../mjx-rl-experiments/NITRO_MJX_TRAINING_PLAYBOOK.md`): pick lifts the box; once it has stayed within 5 cm of the gripper, 10 cm up, for 0.2 s, throw takes over and throws along a random heading (an input).

- `panda-env.js` is shared by the page and `tools/eval.mjs`: env, both observations (pick 66, throw 59), and the `Throws` loop (handover and landing rules). `policy.js` is the MLP (swish, tanh of the mean). `main.js` page + mouse grab (spring via `xfrc_applied`), `render.js` three.js.
- `robot/` is packed by `tools/pack_panda.py` (decimated visual STLs, convex hulls for the never-colliding collision meshes; physics bitwise identical to the original).
- `training/`: `panda_live.py` (pick task: random faces/drops, target jumps, mid-episode respawns), `panda_throw.py` (starts from `collect_grasps.py` handover states; reward = progress along heading − sideways), `train.py`, `export.py` (→ `model/{pick,throw}/policy.{json,bin}` + `test.json`), `eval_mjx.py`, `job.sh`. The handover rule in `collect_grasps.py` must match `HANDOVER` in `panda-env.js`.

```bash
cd robotarm && npm install                  # MuJoCo for Node (tools only)
node tools/eval.mjs --throws 400            # wasm: handovers, throw distances, angle off the arrow
```

Training on Nitro (code in `~/play/robotarm`, venv `.venv-jax011-cu13-braxmain-mj311`, outputs `/mnt/c/w/robotarm`): write the settings to `jobs/current.env` and start `job.sh` with `C:\w\gns\launch.ps1 -Job ../robotarm/job.sh -TaskName RobotArm`. Shipped: pick = `live1` (`TASK=live POLICY=256,256,128`, 131M steps) at 124.5M; grasps from `collect_grasps.py` on it; throw = `throw1` (`TASK=throw DISCOUNTING=0.99`, 123M) → `throw2` (`LATERAL=5 LEARNING_RATE=0.0003`, restored, 66M) → `throw3` (same, 5 cm grasps) at 24.6M. Pick checkpoints by `tools/eval.mjs` on 1,200 rounds, not by the training reward. On wasm the final pair throws 1.90 m on average (MJX: 2.05 m); most of the gap is fumbles right after the handover.

## Air Hockey (`/airhockey/`)

Human vs a policy trained on Nitro with JAX (~600k steps/s): DAgger clone of a scripted expert, then PPO league self-play (PFSP snapshot pool, exploiters, scripted bots with human reaction delays). Details, commands and findings are in `airhockey/CLAUDE.md`.

- Physics in `environment.js` was rewritten 2026-10-08 to be physical (puck reflects off paddles, low friction, substeps, fixed 60 Hz in `game.js`); `training/hockey.py` mirrors it and `training/parity.mjs` + `parity.py` must print PASS after any physics change.
- `ppo_agent.js` is a pure-JS MLP reading `model/policy.bin` (no ONNX runtime).

## Army Ant Bridge (`/ant-bridge/`)

*Eciton hamatum* cross a forked twig and build a living bridge that slides into the gap (Reid et al. 2015 PNAS, Garnier et al. 2013, Lutz et al. 2021). Replaced the old `neural-bridge/` (now a redirect). No bridge logic anywhere: three per-ant rules in `sim.js` (walk the shortest way over footing; hold when footing is poor and lock if walked over, less on sagging footing; leave when traffic over you drops, never while others hang from you).

- `sim.js` is shared by the page and the Node tools: 1.25 mm footing grid, Dijkstra path fields every 0.4 s, bridge ants as two-particle PBD bodies with leg constraints. `render.js` (instanced ants, leg IK, half-res gather DOF + ACES), `ant-model.js` (procedural worker), `main.js`.
- Calibrated against Reid's Fig. 2 (distance moved vs angle and traffic) by sweeps; the article's "Measured" table is `node tools/sweep.mjs --angles 12,20,40,60 --traffics 50,100,200,300 --seeds 4 --minutes 30`. Re-run it after changing any parameter in `PARAMS` and update the table.
- Joining must only come from walking off the bridge's edge (tail on bridge ants, centre ≥ `barkMargin` from bark) or spanning a bark gap; any looser rule grows ribbons of ants along the tines. `tools/snap.mjs` draws top-down PNGs for checking this by eye.
- Testing: a background Chrome tab pauses requestAnimationFrame; `antBridge.advance(seconds)` in the console steps the sim and renders. Python's http.server lets Chrome cache modules, so serve with no-cache headers.

```bash
cd ant-bridge
node tools/run.mjs --angle 20 --traffic 200 --minutes 30 [--stop 10]   # one run; --stop times how long the bridge takes to come apart
node tools/sweep.mjs --angles 12,20,40,60 --traffics 100,200 --seeds 3 --minutes 20 [--params '{"lockChance":0.5}']
node tools/snap.mjs --angle 40 --at 60,300,900 --out /tmp/snap
```

## Path Tracer (`/pathtracer/`)

WebGPU path tracer: one WGSL megakernel (`trace.wgsl`) over per-mesh BVHs built in a worker (`bvh.js`, `mesh.js`), placed by instances (`assemble.js`).

**GPU safety:** a shader loop that never ends hangs the whole Mac, not just the tab. On 2026-10-02 a bad BVH froze WindowServer and caused a kernel panic. Before testing any change to `trace.wgsl`, `bvh.js`, `assemble.js` or scene geometry in a browser:
```bash
node pathtracer/tools/check_scenes.mjs   # builds every scene, validates buffers, measures worst traversal
```
`validateScene` in `assemble.js` must keep rejecting trees where an inner child doesn't come after its parent. Keep every shader loop bounded (traversal has a 2048-step cap; real rays need < 300).

- Meshes: `tools/pack_mesh.py` packs Stanford PLY scans into gzipped `model/*.mesh`
- `tools/compare.html`: benchmark against three-gpu-pathtracer (dev only, loads it from a CDN)
- In the console, `pathtracer.run(n)` renders n frames even when the tab is in the background
