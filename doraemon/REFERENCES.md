# Visual and camera grounding

The model is built from original procedural geometry. Reference images were inspected during development; they are not bundled into the app.

## Character

- [Doraemon Channel: official character page](https://www.dora-world.com/character/doraemon), particularly [the character artwork](https://www.dora-world.com/assets/images/characters/doraemon/doraemon/d_001_card_detail.png): round head, compact torso, neighboring oval eyes, small round hands, short legs, broad smile, red collar, bell, and semicircular pocket.
- [Anime episode stills published by Animedia, June 19, 2020](https://cho-animedia.jp/article/2020/06/19/18509.html), including [Doraemon flying beside Nobita](https://cho-animedia.jp/imgs/p/Dybu1wcglzgN6YILHHa9fs3Ml8DBwsPExcbH/118383.jpg?zoom=spacing): Take-copter on the head, spread arms, short feet hanging behind the torso.
- [Stand by Me Doraemon flight still, TMDB image archive](https://image.tmdb.org/t/p/original/1aABIiqBY7yoQESE8qWvR0w9bJZ.jpg): used as a secondary visual reference for a forward glide, trailing legs, a nearly upright face, smaller pupils, and a rounded smile in 3D.

The references show different flight attitudes, rather than one fixed pose. The app interpolates from an upright hover to a forward-leaning body as speed increases. A neck pivot lets the head counter-rotate, keeping the face readable and the copter upright. These are artistic interpretations, not claimed official model measurements.

## Guided-flight camera

- [Pavel Boytchev's original third-person camera example on the Three.js forum](https://discourse.threejs.org/t/third-person-camera/55349/2): a camera positioned using a spherical orbit relative to the character, with smoothed angles and a character-relative look target.
- [Three.js camera documentation](https://threejs.org/docs/pages/Camera.html).

Scenic mode adapts that approach into a 16-second shot cycle: follow, ease into a front three-quarter view, hold, and return. The orbit angle and distance change smoothly instead of moving straight through the character. Follow mode stays behind Doraemon; taking manual control returns to that view automatically.

Nearby clouds fade per instance when they overlap Doraemon or the camera’s line of sight. This keeps the face and copter visible during a pass. Reduced-motion mode holds a front three-quarter angle instead of repeatedly orbiting.
