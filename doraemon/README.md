# Pocket Skies

A small Three.js sky adventure featuring a procedurally modeled Doraemon and his bamboo copter. Open `index.html` through a local HTTP server, or visit `/doraemon/` on the site. No build step is required.

```sh
# From the repository root
python3 -m http.server 8765
# http://localhost:8765/doraemon/
```

## Explore

- Drag to orbit the model; scroll or pinch to zoom. Tap Doraemon to say hello.
- Choose Daydream, Golden hour, or Moonlight. The sky choice is remembered locally.
- Select **Let’s fly** to take control. With only a mouse, hold the left button and drag sideways to turn or up/down to climb/descend. Release to hold course. Use the wheel, speed slider, or + / − buttons to change speed; **Hover** brings you to a stop.
- Keyboard controls also work: W/S or up/down changes speed, A/D or left/right turns, Space rises, and Shift descends.
- On touch screens, use the joystick and altitude buttons.
- **Let Doraemon guide** follows the eight-ring course automatically. **Scenic** periodically swings around to show his face; **Follow** stays behind him. Mouse, keyboard, or touch flight input returns control to you and smoothly restores the follow camera.
- Escape returns to orbit. The flight resumes from its previous position when you select Fly again.
- The camera button or P downloads a PNG postcard. Sound is opt-in and synthesized in the browser.

## Implementation

- `model.js`: reference-grounded proportions and face, a neck-pivot flight rig with trailing feet and an upright head, modeled bamboo airfoils, rotor afterimages, blinking, and greetings. See [REFERENCES.md](REFERENCES.md) for the visual sources.
- `world.js`: miniature island, houses, lighthouse, trees, sailboat, hidden Anywhere Door, sky/water shaders, instanced clouds, and birds. Static scenery is batched by material.
- `main.js`: rendering, camera modes, flight physics, ring course, guided flight, audio, touch input, and UI.
- `styles.css`: responsive interface and reduced-motion handling.
- `vendor/`: pinned Three.js 0.180.0 and OrbitControls, with the upstream MIT license. The app’s rendering dependencies are local; Google Fonts is optional and falls back to sans-serif.

WebGL 2 is required. A lost graphics context shows a recovery screen. Animation and audio pause in a hidden tab, flight pauses while a dialog is open, and pixel density is reduced automatically on slower devices.
