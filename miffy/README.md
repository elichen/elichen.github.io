# Miffy’s little garden

A small, sunlit Three.js garden. Miffy pulls a carrot and tosses it into a woven basket. One button, draggable camera, no build step.

## Run

From the repository root:

```sh
python3 -m http.server 8765
```

Open http://localhost:8765/miffy/.

- Click **Pull a carrot**, or focus the button and press Enter / Space.
- After each harvest, the next carrot gently sprouts over 1.8 seconds; the button becomes available when it is grown. Reduced motion shortens regrowth to 0.45 seconds.
- Drag to look around. Scroll or pinch to zoom.
- The first harvest plays automatically. Reduced-motion preferences disable the automatic harvest and ambient movement, and shorten the requested harvest.

All models are made from geometry in `main.js`. Three.js and OrbitControls are vendored locally with their MIT license. Google Fonts is optional; local font fallbacks work without it. No images, account, storage, or backend needed.

This is an unofficial fan-made character study. Miffy was created by Dick Bruna.

See [REFERENCES.md](REFERENCES.md) for the image comparison, proportion corrections, and evaluated downloadable models.
