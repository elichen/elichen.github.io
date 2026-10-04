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
