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

- `denoise.wgsl`: edge-aware à-trous filter (normal, depth, material, brightness vs noise) on the first-hit guide that `trace.wgsl` stores next to each pixel's sum; it only shows while there are few samples (dragging) and fades out by 256. The trace pass is at WebGPU's default 8 storage buffers, which is why the guide shares the accumulation buffer.
- Meshes: `tools/pack_mesh.py` packs Stanford PLY scans into gzipped `model/*.mesh`
- `tools/compare.html`: benchmark against three-gpu-pathtracer (dev only, loads it from a CDN)
- In the console, `pathtracer.run(n)` renders n frames even when the tab is in the background

## Drone Racing (`/drone-racing/`)

The pure-RL racing quad from `../mjx-rl-experiments/drone/racing` (RealFlight Tempest v7 plant, checkpoint `gen-all5-seg3` 159M, which flew 20/20 on RealFlight's Flight School) laps five UZH courses on a photographed airfield, with a RealFlight-style radio overlay, telemetry and synthesized motor sound. RealFlight's own assets are proprietary, so the field (Poly Haven `derelict_airfield_01`, CC0), radio and sound are recreations.

- `sim.js` is a JS port of `RingRace` in qualification mode (shared by page and tools): MuJoCo Euler free body at 4 ms, rate loop on a filtered gyro, attitude-priority mixer, motor lag, inflow/airspeed thrust loss, swept-sphere ring strikes, the 204-value observation with the detector dropout model. The MJX body uses the MJCF inertia (0.0025, 0.0021, 0.0043); the config's fitted `nominal_inertia_diag_kg_m2` never reaches the rigid body, so the port doesn't use it either.
- `flight.js`: scripted ferry (takeoff from a pad under the staging point, launch to the trained handoff 2.5 m before gate 0 at 2.7 m/s), then the policy laps until a crash (`course.laps = Infinity`; the policy sees no lap counter). Open Multistory is closed from its finish plane back to gate 0 (extrapolation, 0 crashes in 100 laps). Split-S is Multistory's geometry and isn't exported.
- `render.js`: the panorama is projected from the pilot's eye (`EYE_HEIGHT` in `layout.js`) onto the ground, walls and a sky dome, so the pilot view is exactly the photo and the other cameras get parallax. The walls stand where hedges, reeds and tree lines meet the ground (`PHOTO_WALLS`, distances read off the panorama by column), pushed back behind the course; their sky is cut out by color. The free camera stays inside the walls and below 6 m (above that the photo's grass smears), and far ground is blurred sideways when the camera is away from the pilot; sun light placed at the photo's sun (u 0.6035, 18.6°); a second near-zero-intensity sun carries a tight shadow map that follows the drone. `layout.js` puts the pilot behind the pad.
- The view fills the window; `controls.js` turns drags, pinches, the wheel and WASD/QE into camera moves. Grabbing the view from Pilot or FPV switches to the free camera from the current view; in Chase, dragging circles the drone. The article is in the "How it works" panel.
- `audio.js`: one oscillator per motor at its rotation rate (rpm from thrust, k = 1.682e-8 N/rpm²) with blade-pass/electrical harmonics; band levels were tuned against an onboard recording (Commons "The Sound of Drone Racing", CC BY 3.0) with an OfflineAudioContext.
- `tools/export.py` (needs a JAX/MJX/Brax venv and the mjx-rl-experiments checkout) writes `model/policy.{json,bin}`, `model/track.json` and `model/test.json` (MJX reference episodes).

```bash
cd drone-racing
python tools/export.py ../../mjx-rl-experiments <params_step_000158597120> model   # checkpoint + .contract.json from Nitro
node tools/parity.mjs        # JS vs MJX: teacher-forced ticks + closed loop; must print PASS after any sim.js change
node tools/eval.mjs --flights 10 --laps 10   # takeoff → handoff → laps; numbers in the article's "Measured"
```

Testing: a background Chrome tab pauses requestAnimationFrame; `droneRacing.run(seconds)` in the console steps and draws.

## Concept Dictionary (`/concepts/`)

Distill-style article around an explorer of 6,132 concepts that a TopK sparse autoencoder found in the last block (`layer4.2`) of Lucid's robust ResNet-50 (`../lucid/robust-resnet.js`, `../lucid/robust-resnet50/`). A photo's concepts, their exact contributions to the top class and to its lead over any other top-5 class (layer4.2 feeds average pooling and the classifier, so contribution = mean activation × decoder row · classifier column × scale; for a lead, the difference of two columns), examples, renders, and painting a concept into the photo. The picked concept is shown without darkening the photo: a contour at half its peak, color fading outside.

- `sae.js`: the dictionary. Only the encoder's matrix product runs on the GPU; top-k, contributions, class weights and neighbours are sparse/CPU (`model/`: int8 weights with one scale per concept). `painter.js`: painting = Adam on a Lucid-style Fourier delta added under the brush (raw-pixel steps leave rainbow fringes), ~40 ms/step on WebGPU. `app.js`: the explorer.
- `data/concepts.json` (frequency, best activation, top-20 classes, example pool images at full/half/a fifth of best), `data/names.json` (names written by Claude subagents from contact sheets of render + 6 spotlighted examples, graded clear/mixed/unclear), `data/pool.json` + `pool/` (6,106 ImageNet val crops picked so 98.6% of concepts get 6 examples at ≥ half their best; Eli confirmed ImageNet crops may be hosted), `renders/` (direction × cos³, 128 steps).
- `training/` runs on Nitro (`~/play/sae`, venv `~/play/torch-venv`, `job.sh` via `launch.ps1 -Job ../sae/job.sh -TaskName SAE`): `prep.py` (HF ImageNet parquet → 256 crops), `train.py` (run2: `--data train --layers layer4.1:16384,layer4.2:16384 --epochs 10 --lr 2.8e-4`, k 32), `mine.py`, `render.py run2 all:layer4.2:1e-4 0 dircos`, `codes.py`, `pool_stats.py` + `pool_cover.py`, `export_app.py` (CPU only; writes `model/`, `data/`, `pool/`, `renders/`, `tools/test/`), `pool_codes.py` + `tools/make_sheets.py` for naming sheets.
- Half of run2's 16,384 latents died; keeping those with validation frequency > 1e-4 loses nothing (`subset.py`). layer4.2: FVU 0.252, spliced top-1 61.4% vs 64.1%, int8 changes 1.25% of top-k picks.

```bash
cd concepts/tools && npm install && node parity.js ../.. 2   # tfjs CPU vs PyTorch references; must match top-5 and top-10 concepts
```

Testing: open `tools/parity.html` for the WebGPU check. Keep the tab in front: in a background Chrome tab WebGPU work never runs and pages hang with nothing on the GPU. `window.conceptsApp` exposes state; `analyze()`, `select(j)`, `drawPhoto()` are globals.

## Brownian Motion (`/brownian/`)

Distill-style article around a 2D molecular dynamics liquid on WebGPU: ~14,000 WCA discs (density 0.7, T = 1, dt = 0.005 τ) and five beads (radius 4σ, mass 35 = the liquid they displace). No random numbers after the initial state; the beads' jiggle comes only from collisions. Draggable beads, Perrin-style trails, live mean squared displacement chart, equipartition readouts.

- `sim.js` is shared by the page and the Node tools: two dispatches per step (molecules: cell-grid forces, leapfrog, insert into the next grid; beads: one workgroup each + the thermostat sums). Positions/state double-buffered, grid triple-buffered (count + 7 slots per cell). Weak Berendsen thermostat (τ_T = 20) relative to the liquid's mean flow; any force on a bead (drag spring, `pull`) is pushed back on the liquid so momentum stays zero. Bead records go to a GPU ring buffer that the page reads back each frame.
- `render.js` draws discs straight from the GPU buffers (4 periodic images per disc); `main.js` page, readback, trails, dragging, adaptive steps per frame; `analysis.js` multiple-tau MSD; `chart.js`.
- Bead size matters: radius 10σ beads in 2D keep their velocity for hundreds of τ (2D long-time tail), paths look like smooth drifting. Radius 3–5σ gives step correlation ~0.1 at 300 τ.
- GPU safety: every shader loop has a fixed bound (see `pathtracer` note); run `node tools/check.mjs` after any change to `sim.js`.

```bash
cd brownian && npm install                # Dawn WebGPU for Node (tools only; keep the GPU object referenced or Dawn segfaults)
node tools/check.mjs                       # GPU vs double-precision CPU step, energy, momentum, grid overflow; speed
node tools/measure.mjs --a 4 --L 160,100 --beads 5 --time 4000    # MSD, VACF, step correlation of Perrin paths
node tools/einstein.mjs --free 50000 --pulls 20 --force 0.5        # D from jiggling, γ from pulling: D·γ vs kT (article numbers)
```

Testing: a hidden Chrome tab pauses requestAnimationFrame; `brownian.advance(τ)` steps and draws, `brownian.colors({...})` tries a palette.

## Concept Steering (`/concept-steering/`)

Gemma 3 270M (base) reads a passage; the Gemma Scope 2 layer-12 16k sparse autoencoder lists the concepts it lights up (ranked by mean act/best over the passage, 1e-6 < freq < 1e-2); pinned concepts are added at layer 12 (every position but `<bos>`) to write a new completion, next to the same random draw unsteered. Everything runs in the browser on WebGPU.

- `gemma.js`: own forward pass (int8 weights with one scale per row, fp32 activations: the residual reaches ~15,000, past fp16), steered + unsteered rows batched, sequences capped at 512 tokens so sliding windows never slide. Every shader loop is bounded (see the `pathtracer` note). `tokenizer.js`: Gemma's BPE from `data/vocab.json` + `data/merges.bin`. `worker.js`: loading (Cache Storage), reading, exact top-p sampling on shared uniforms; markup tokens are never sampled; after each write it re-reads both continuations unsteered (`scored`) so the page can color where the pinned concepts fire. `app.js`: the page (Distill-style article around one figure).
- Weights live on Hugging Face (`eli-aifund/gemma-3-270m-scope-steering-int8`, public, pinned to a commit in `worker.js`), not on Pages; `model/` is gitignored. Labels in `data/names.json` are Neuronpedia's automatic explanations (S3 export `v1/gemma-3-270m/12-gemmascope-2-res-16k/explanations/`).
- Strength: measured on Nitro for all 14,983 live concepts at 0.5-4x best on 4 prompts (two seeds pooled). The slider multiplies each concept's own 1x: the smallest strength where it clearly shows up in the output (presence on re-read +0.08) with judge PPL <= 3x unsteered and distinct-3 >= 0.8 (`tools/defaults.py`). Only 1,095 concepts (7%) get there; the rest recognize a topic but steering them never writes about it (tested to 13x: junk repetition), so the page fades them and gives them no pin. `data/stats.bin` = best, freq, strength (x best; 0 = dead), presence lift at 1x (0 = not pinnable). The fluency-limit table (largest strength with PPL <= 1.5x, distinct-3 >= 0.9) is the input; 1x there was too weak (cats appear only above it).
- A PyTorch steering hook must also add to the generated tokens: with the KV cache they arrive one at a time, so `h[:, 1:]` only steers the prompt.

```bash
cd concept-steering
python3 tools/export.py <strength table.json> <texts.json>   # model/, data/, tools/test/ref.json
python3 tools/defaults.py <strength table.json> <calib dirs...>  # 1x per concept into data/stats.bin
cd tools && npm install && node check-tokenizer.mjs && node parity.mjs   # must print PASS (Dawn WebGPU vs PyTorch)
```

Testing: a hidden Chrome tab throttles the worker (~10x slower sampling); time it in a visible tab. `window.steeringApp` exposes state and the worker.

## Grand Prix (`/grand-prix/`)

Race a neural network around 22 TUM circuits (the 22 training tracks of `../mjx-rl-experiments/racecar`), same car and physics for both; ghost cars (the policy has never seen another car), rolling start, 1/3/5 laps; "Watch the AI" runs the race with your car hidden (`Race` option `watch`). Full-window three.js scene. Palette A: sampled from a Commons photo of a Lotus 72 (JPS) beside a McLaren M19. The old `/racer/` demo is separate and stays.

- `sim.js` (shared by page and tools): JS port of the repo's `MultiTrackRaceCar` (track state, the 73-input observation, failure rules) on the **load-sensitive plant** of `training/plant.py`, plus the policy MLP. `race.js` race logic and your input aids, `input.js` keyboard/gamepad/touch, `scene.js` circuit builder (TUM centerline + widths), `render.js` cameras, `car-model.js`, `audio.js`, `main.js` (picker, HUD, palettes `PALETTES` + `grandPrix.colors('B')`).
- **The plant was changed on purpose (2026-10-22).** The repo's tires had grip and cornering stiffness exactly proportional to load, so braking at speed made the car unstable (0.002 rad steer spun it at 300 km/h). `training/plant.py`: load-sensitive grip, Pacejka-saturating stiffness, ~5-6° peak slip, brake force limited by tire capacity, 60% front bias. `plant.use()` patches `racecar.env.integrate` so the repo env runs on it. `python training/check_plant.py` compares old vs new figures.
- **Policy:** DAgger (`training/dagger.py`) from a path-following teacher (`training/teacher.py`: curvature feedforward + heading/Stanley/yaw-damping steering, gains grid-searched; pedal tracking `training/speed_profile.py`'s minimum-time profile at 0.9 corner speed, throttle capped by `plant.throttle_limit`). Started from the repo's `fastest_racer.params`; 16 rounds, ~20 min on Nitro: 22/22, 1,887 s (teacher 1,886 s; repo policy on this plant 2,211 s; full-limit QSS 1,736 s). Then **ES** (`training/es.py`: antithetic, 64 pairs, σ 0.002, Adam 5e-4, centered ranks; each candidate drives 44 laps = rolling + one perturbed start per circuit; saved only when the center finishes all 44 faster) for 300 generations (~30 min): 1,792 s, then 600 more from that with σ shrinking 0.002 → 0.0008 (`--sigma-end`, es2): **1,774 s** at generation 450, 6% faster than the teacher, 660/660 laps from unseen random starts (`node tools/robust.mjs --starts 30`). ES only scores first laps, so check later checkpoints for 3-lap crashes and random starts: es2's best (1,770 s, gen 592) and gens 400/500-575 crashed on 3 laps or random starts (Suzuka ~790 m spin). Shipped: `es2/es_gen_0450` (sha256 ee01ee8b...).
- What failed (don't retry blindly): PPO with progress reward from the repo policy (old and new plant), PPO with reference starts on the repo's CEM laps, BC on CEM laps (never brakes), DAgger around CEM laps (jittery at-the-limit labels), a geometric teacher with untuned gains (oscillates; also NaN from `sqrt` of a fused negative).
- Your car only gets input aids: a full key = 95% of the steering for the most steady cornering at that speed (`CORNERING` in sim.js, from `node tools/corner-table.mjs`; re-run after plant changes), a brake that leaves the grip the steering asks for (`brakeLimit`, friction circle: trail braking; a steering cut under braking blocked it), traction control, counter-steer beyond 0.1 rad of slip (`COUNTER`). `node tools/turning.mjs` compares key vs AI cornering g by speed (keys now match); `node tools/keyboard.mjs` counts spins for a scripted key driver (7 with aids, 861 without).

```bash
cd grand-prix
python tools/export.py ../../mjx-rl-experiments <params> model   # needs a JAX/Brax venv; weights, tracks, reference laps
node tools/parity.mjs        # JS vs JAX on all 22: must print PASS after any sim.js / plant change
node tools/eval.mjs --laps 3 # lap times, top speed, crashes in JS
node tools/robust.mjs        # laps from unseen random starts
node tools/keyboard.mjs      # scripted keyboard driver, aids on/off
```

Training on Nitro (code in `~/play/grand-prix`, outputs `/mnt/c/w/grand_prix/<run>`, `base.params` = repo policy): `python training/speed_profile.py --scale 0.9` (slow, pure Python) → `teacher_job.sh` (gain search) → `dagger_job.sh` → `es_job.sh`.

## Two Hours of Atari (`/atari/`)

BBF (Schwarzer et al. 2023, `configs/BBF.gin`: replay ratio 64 = 2 updates per move) trained on Atari 100k (100,000 moves = 1 h 51 min of play, about the 2 hours of practice behind DQN's human scores) for Space Invaders, Ms. Pac-Man, Q*bert and Kung-Fu Master, one seed each; Ms. Pac-Man trained on to 400k moves (the curve marks where the tester's practice ends). The page is an arcade cabinet: the wasm ALE (`@farama/ale-wasm` 0.12.0 in `vendor/`, with its ROMs) plays each game with the policy at any 10k-move snapshot (practice lever); the article is in the Menu dialog, in Simplified Technical English.

- `game.js` (ALE + Dopamine's preprocessing: frame skip 4, max-pool, INTER_AREA 84×84 in `preprocess.js`, 4-frame stack, life loss clears the stack, 0-29 no-ops), `policy.js` (BBF's network in tfjs from int8 dense / f16 conv weights), `app.js` (page, curve, sound, `atari.evaluate(n, seed)`).
- Weights live on Hugging Face (`eli-aifund/bbf-atari100k-tfjs`, pinned to a commit in `MODEL` in `app.js`; re-pin after each `hf upload`), not on Pages; `model/` is gitignored. int4 lost 1.5-4k points late in training, so it is int8.
- `data/manifest.json`: per game the paper's 14-seed RR2 numbers, human/random, our run's final (BBF's own 100-game evaluation), and each snapshot's score over 20 games (`eval_ref.py` on Nitro's CPU). The page opens each game on its best-scoring snapshot.
- Sound: ALE's SDL audio through WebAudio. SDL asks for a 64-sample ScriptProcessor (browsers need ≥ 256) and ~3% of slices come back as garbage, so `app.js` shims `createScriptProcessor`. SDL opens audio once per wasm module, so each game on screen gets a new module (~6 ms); dispose the old `Game` first. Sound does not change emulation.

```bash
cd atari/tools && npm install
node parity.mjs            # wasm ALE vs native ale-py (parity_ref.py): every frame's screen, RAM, reward, lives
node game_check.mjs        # game.js vs Dopamine's AtariPreprocessing (game_ref.py)
node resize_check.mjs      # preprocess.js vs cv2 INTER_AREA (test data from resize_ref.py, gitignored)
node policy_check.mjs ../model/<game>/<step>   # tfjs vs JAX on the exported weights: must print PASS
```

Training on Nitro (BBF checkout + shims in `~/play/bbf/work`, venv `~/play/bbf/.venv`, outputs `/mnt/c/w/bbf/<run>`): add `<run> <Game> <seed> 100000` to `~/play/bbf/queue.txt` (runs: spaceinvaders, mspacman2, qbert, kungfu; `mspacman400k` = seed 1 for 400k moves) and start `job2.sh` with `C:\w\gns\launch.ps1 -Job ../bbf/job2.sh -TaskName BBF`. One run at a time (two at once run ~4x slower each); ~6 h per game on the RTX 5050. `tools/follow.sh` fetches snapshots as they appear (`fetch.sh` exports on Nitro, then copies), `evals.sh`/`evalq.sh` score them on Nitro. Results: Space Invaders 842 (paper 944, 435-1,497), Ms. Pac-Man: the page shows `mspacman400k` (seed 1 again, 400k moves, replay capacity 400k; 2,000 in 20 games at 100k, 2,527 in 100 games at 400k; snapshots every 10k to 100k, then every 50k). Other 100k runs: seed 1 1,258, seed 2 2,053; the same seed does not repeat on the GPU, Q*bert 4,711 (451-4,719), Kung-Fu Master 14,770 (10,532-26,806).

## BAAIWorm, live (`/worm-sim/`)

BAAIWorm (Zhao et al. 2024, github.com/Jessie940611/BAAIWorm) running in the browser: its 136-neuron multicompartment circuit and its SoftCon FEM body, both re-implemented in Rust (`engine/`, compiled to `engine.wasm`) and checked against the original's recorded data. Rebuilt 2026-10-29; it replaced a replay of recorded WetNet output on a fake body.

- `engine/src/brain.rs`: NEURON's fixed-step backward Euler (secondorder 0) on 24,496 nodes (23,848 segments + zero-area section ends), Hines solve per cell, 5/3 ms. Channels transcribed from `eworm/components/mechanism/modfile` with NEURON's reading of their units (gbar in S/cm², ek = -77 only for kqt3, the rest hard-coded). Nodes are re-ordered by depth so cells interleave (2x faster); zero-gbar channels are skipped.
- `engine/src/body.rs`: SoftCon projective dynamics (`Metaworm/sim/fem`): corotated tets (μ vol, plus ν μ vol volume term), 4,558 fiber constraints (target stretch 1 - a), per-triangle drag + lift, 1/240 s, ≤ 50 iterations, stops at ‖Δx‖/3n < 5e-7 keeping the previous iterate. One scalar 984×984 envelope Cholesky (vertices sorted along the body) solved for x, y, z.
- `engine/src/worm.rs`: the loop of `worm_in_env.py` ("fc"): 60 neural steps, mean of 80 motor somas @ 80×96 readout, muscle = clip((x+80)/100, 0, 0.8), 8 body steps. The page plays the recorded 2 s sensory cycle ("offline", BAAIWorm's own demo), input k = cycle[(k-1) % 20] as in `worm_in_env.py`. No warm-up: like BAAIWorm, it starts at v = -65 and the head swings round in the first half second.
- **The page crawls on agar and seeks food (ours, 2026-10-29), not BAAIWorm's swimming.** `Worm::body_steps` = 24 (100 ms of body per 100 ms of brain; BAAIWorm's 8 runs the body 3x too fast and its 1.5 Hz gait is the 2 s input cycle at that speed), `Body::set_agar` (resistive-force drag inside the PD solve, c_t 0.003 / c_n 0.09, plus a pull onto the plate plane), `Body::balance_muscles` (SoftCon's quadrants differ by up to 20% per segment → circling). The page and `seek` use `model/body_sym.bin` from `tools/make_body.py` (one quadrant of a structured tube mirrored twice: exactly D-V and L-R symmetric, 1,008 vertices, section radii envelope SoftCon's outline +5% so the skin fits; its muscles at ±45°/±135° on BAAIWorm's segment ends); `model/body.bin` stays SoftCon's for the parity tools. Regenerate neurons.bin and skin.bin after changing it (export_neurons on body_sym; export_skin body_sym.bin skin.bin body.bin). `behavior.rs` sits between the brain's muscles and the body: pirouettes at Iino & Yoshida's fitted rate (dC/dt in mM, 2-6 mM Gaussian hill σ 1 cm), reversal = gait mirrored head-to-tail, omega = ventral bias; weathervane gain fit to 12.7 °/mm per mM/mm in a planar gradient (`P_PLANE=±0.2`, turns and wander off, first 60 s, half the difference of the two sides; regressions on hill trials are biased by where the worm goes) and OU wander to SD 32 °/mm (`seek_stats.py`); `straighten` = 0 on body_sym (SoftCon's mesh needed 0.015; check with `tools/turning.py` after any body change). `GAIT_TRIM` must match the brain (`seek` asserts it). The brain worker warms up 30 steps (the start from -65 mV swings the head 130° and makes trials diverge); the body fades muscles in over 1 s. Validation = Pierce-Shimomura 1999's assay: start 22 mm from the peak, 20 min, share reaching the 5 mm zone (`seek` + `seek_stats.py`, runs on the office Mac mini via ssh `macmini`, ~/worm, tmux).
- The page draws `model/skin.bin` (`tools/export_skin.py`: smooth cuticle tube + pharynx, grinder, intestine, eggs; each vertex embedded in a rest tet like the neurons, extrapolated a little), brightfield shading in `main.js` (`SKIN` colors); the Neurons button switches to the old see-through body with the circuit. Re-export it after any change to body_sym.bin. Python's http.server lets Chrome cache model/*.bin: use a fresh headless profile after regenerating them.
- `worker.js` runs the body paced to real time and starts `brain-worker.js`, which runs the brain up to 8 control steps ahead (the recorded input never depends on the body, so results equal the serial loop); `main.js` draws the body, every compartment as a line (embedded in the rest mesh by `tools/export_neurons.py`), muscles, raster; palettes A-D sampled from Commons photos, `wormSim.colors('A')`, `?palette=`.
- **Checked against BAAIWorm's own code** (Eli allowed running it on 2026-10-29): `tools/original/` builds its SoftCon body (`Metaworm/sim`, Eigen only) into a headless pybind11 `interact` module (same functions as `Metaworm/interact/interact.cpp`, plus `tick()` = the MC_C302 branch of `SimulationWindow::Timer`), and `run.py` drives its unmodified `WormInEnv` (NEURON 9, `nrnivmodl` on its modfiles; numpy-1.x aliases shimmed in run.py). Offline, 20 s from the same start, every step: muscles within 6e-5, body within 1.4e-7 (any vertex) when fed its muscles, whole loop within 0.02 units (any vertex; body is 0.63 long) for 10 s, then the brain differences grow. Input cycle = BAAIWorm's float32 values, not rounded.
- Online chemotaxis (head concentration → current clamp, `Worm::set_food`; first smell is taken at the body origin, as BAAIWorm does) matches BAAIWorm's loop for ~0.7 s, then diverges (the 11-step min/max normalization amplifies 1e-6 differences). It is not on the page because it does not seek food, in BAAIWorm's own code either. "Farther in 10/10" was a biased metric (any straight swimmer ends far away); the control is own food vs other runs' food: original 9.43 vs 9.25 (10 runs), ours 9.02 vs 8.66 over 100 runs (slight drift away after 40 s, none before). Root cause: the min/max-of-11 normalization is exactly invariant to the sign and size of the approach rate (checked with their function: toward vs away identical to 3e-14), all 15 sensors get the same current, and both readouts are ridge fits to one forward gait (`pre_interaction.py`). Their teaching start (`step_init` + `output/test.npy`, unused as shipped; `step()` crashes after it, fixed in run.py) for 10 or 33 s changes nothing. A sign-keeping input (`-0.03 + 2.5Δ`) degrades the gait (2-3 units in 60 s) and steers at most weakly. The paper shows one closed-loop run, no trials (Fig. 5, Supp. Videos 1-2); the six-worm script hand-places a target per worm. Lesion sliders scale all synapses / gap junctions (0% = all of that kind gone; the paper's `abstract2detailed_remove` is referenced in a comment but not in the published repo). Over 60 s: intact 0.19 units/s forward; GJ 50% 0.01; SYN 0% drifts backward at 0.03.
- The site's own exports still read BAAIWorm's `.pkl` files with `tools/safe_pickle.py` (classes stubbed). NEURON (pip) computes segment areas/axial resistances in `tools/extract_geom.py` from hoc text parsed there.

```bash
# regenerate model/ from a BAAIWorm checkout (pip install neuron numpy scipy)
python tools/extract_geom.py <BAAIWorm> geom.json && python tools/export_brain.py geom.json model
python tools/export_body.py <BAAIWorm> model/body.bin && python tools/export_neurons.py geom.json model/body.bin model/neurons.bin
python tools/make_body.py model model/body_sym.bin           # the page's symmetric body (+ body_sym_muscles.bin)
python tools/export_neurons.py geom.json model/body_sym.bin model/neurons.bin && python tools/export_skin.py model/body_sym.bin model/skin.bin model/body.bin
python tools/export_refs.py <BAAIWorm> model <ref dir>       # readouts + reference data (ref dir not committed)
cd engine && ./build.sh                                     # engine.wasm (rustup target add wasm32-unknown-unknown)
cargo run --release --bin brain_parity ../model <ref>      # vs NEURON's recorded motor neurons: must print PASS
cargo run --release --bin body_check ../model <ref>        # SoftCon's recorded muscles → speed vs its recorded states
cargo run --release --bin loop_run ../model offline 20     # closed loop; GJ=0.5 / SYN=0 for lesions
node tools/bench.mjs 5 [agar]                               # wasm speed (agar: ~3.7x real time body, 6x brain on M4)
cargo run --release --bin seek ../model 1200 13.86 <angle> <seed> out.csv   # one food-seeking trial on agar (P_* = behavior params, P_C_PEAK=0 control)
python tools/seek_stats.py 13.86 out_*.csv                  # reach %, weathervane slope, curving SD, turn rate by dC/dt
python tools/tracks.py tracks.png out_*.csv --food=13.86,<angle>  # FPS=10 python tools/kinematics.py <CENTER file> for gait
# BAAIWorm's own loop (venv: neuron numpy scipy xlrd==1.2.0 pymp-pypi networkx tqdm torch pybind11 pandas seaborn; brew eigen)
(cd <BAAIWorm>/eworm/components/mechanism && nrnivmodl modfile)
cmake -S tools/original -B <build> -DBAAIWORM=<BAAIWorm> -DPython_EXECUTABLE=<venv python> && cmake --build <build>
BAAIWORM=<BAAIWorm> INTERACT=<build> python tools/original/run.py offline 600 ref.npz   # ~6x slower than real time
DUMP=m.txt DUMPX=x.bin cargo run --release --bin loop_run ../model offline 20 && python tools/original/compare.py ref.npz x.bin m.txt
BAAIWORM=<BAAIWorm> PY=<venv python> python tools/original/sweep.py out && python tools/original/summary.py out   # chemotaxis, both (INIT=300: teaching start; ONLY_REF=1)
```

Testing: a tab under the Chrome extension's debugger runs wasm slowly; measure speed in a normal tab or in headless Chrome (`puppeteer-core`, own profile, `--use-angle=metal`). On 2026-10-29 the automated tab lost its WebGL context twice after many reloads + screenshots and Chrome then blocked WebGL for that origin until restart; the same Restart sequence never lost context in headless Metal or SwiftShader. `?report` makes the page GET `/report?...` with its speed for a local test server.
