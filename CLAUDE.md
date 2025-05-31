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