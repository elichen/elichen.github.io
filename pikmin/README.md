# A little following

A standalone Three.js Pikmin crowd game. Serve the repository over HTTP and open `/pikmin/`:

```sh
python3 -m http.server 8765
```

Move the mouse to lead Olimar; on a touchscreen, drag. Contact with Olimar or any recruited Pikmin recruits a stray. Space pauses; M toggles the optional synthesized collection sounds. The information button contains restart and credits.

## Implementation

- Original procedural Olimar and red, yellow, and blue Pikmin models. Nintendo's [Olimar artwork](https://play.nintendo.com/themes/friends/captain-olimar/) and [Pikmin site](https://pikmin4.nintendo.com/) were visual references; no ripped models or Nintendo assets are included.
- Three.js 0.180.0 is vendored with its MIT license. Google Fonts is optional; local serif and sans-serif fallbacks keep the game usable without it. No build step.
- Three dynamically growing instanced character batches, merged vertex-colored geometry, shader-animated feet and leaves, and reduced geometry above 4,000 followers. No population cap or automatic deletion of recruited Pikmin.
- Linear-time follower formation and spatially hashed, actual character-to-character recruitment. Wild groups and scenery regenerate around the player; the camera scales with the square root of population.
- Environmental and sparkle pools are bounded independently of the unbounded crowd. Practical population depends on the device's CPU, GPU, and memory.

## Validation

Checked in desktop Chrome and touch emulation at 390 × 844: initial rendering, mouse movement, drag movement, follower-only recruitment, pause, information dialog, restart, camera expansion, and crowd buffer growth through over 50,000 followers. No runtime errors. Read-only `window.__meadow.stats` exposes population, capacity, draw calls, and camera state for debugging. No performance overlay is shown in the game.

Unofficial fan project. Pikmin and Olimar belong to Nintendo.
