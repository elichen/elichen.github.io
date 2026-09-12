# Pocket Skies

A storybook Three.js sky adventure featuring a procedurally modeled Doraemon, his bamboo copter, and a miniature archipelago. Open `index.html` through a local HTTP server, or visit `/doraemon/` on the site. No build step is required.

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
- On touch screens, use the joystick and altitude buttons. On phones (including landscape), **Controls** opens the flight settings, sky moods, camera actions, and chart in a scrollable panel. Flight pauses while it is open; **Back to the sky** resumes it. Only ring progress and steering stay over the scene.
- **Take the scenic route** starts a guided flight directly from the introduction. In flight, **Let Doraemon guide** follows the eight-ring course automatically. **Scenic** periodically swings around to show his face; **Follow** stays behind him. After the course, **Keep wandering** continues a guided circuit. Choose **Take the controls**, or use mouse or keyboard flight input, to fly yourself. Touch controls appear when flying manually.
- The flight chart shows your position, trail, remaining rings, and four places to discover: Himitsu Village, Windmill Cay, Moonflower Garden, and Balloon Crossing. Tap its header to fold it away. The completed course adds time, distance, and discoveries to your flight journal.
- The frame button or **H** hides the interface for an unobstructed view. **Show controls** or Escape restores it. Otherwise, Escape returns to orbit. Your flight resumes from its previous position when you select Fly again.
- The camera button or **P** downloads a framed PNG postcard with the sky mood, location, and a travel stamp. Sound is opt-in and synthesized in the browser.

## Implementation

- `model.js`: reference-grounded proportions and face, a neck-pivot flight rig with trailing feet and an upright head, modeled bamboo airfoils, rotor afterimages, a sculpted smile, curved eyes, one-eye winks, responsive gaze, and greetings. See [REFERENCES.md](REFERENCES.md) for the visual sources.
- `world.js`: an island village, turning windmill, cherry grove and moon gate, balloons, distant islands, sailboat, and hidden Anywhere Door. Procedural sky and water include moonlight, stars, reflections, shallows, and moving surf. Static scenery is batched by material; clouds are instanced.
- `main.js`: rendering, camera modes, flight physics, ring course, guided flight, audio, touch input, and UI.
- `journey.js`: live canvas flight chart, discovery tracking, travel statistics, tapered hand trails, and printable postcard composition.
- `styles.css`: a responsive travel-journal interface and reduced-motion handling.
- `vendor/`: pinned Three.js 0.180.0 and OrbitControls, with the upstream MIT license. The app’s rendering dependencies are local; Google Fonts is optional, with local serif and sans-serif fallbacks.

WebGL 2 is required. A lost graphics context shows a recovery screen. Animation and audio pause in a hidden tab, flight pauses while a dialog is open, and pixel density is reduced automatically on slower devices.

## Mobile interface rationale

The compact layout follows Nielsen Norman Group’s [progressive disclosure](https://www.nngroup.com/articles/progressive-disclosure/) and [deferring secondary mobile content](https://www.nngroup.com/articles/defer-secondary-content-for-mobile/) guidance: prioritize the current task, reveal secondary tools on demand, and preserve space for the scene. Controls use at least 44px-high touch targets, and the panel adapts to the available viewport height.
