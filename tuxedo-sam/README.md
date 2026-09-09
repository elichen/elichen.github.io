# The Little Atlas · A World of Small Wonders

A fully procedural Babylon.js travel diorama: Tuxedo Sam waddles around a miniature Earth while the globe rolls beneath his feet. No build step or external model assets required.

Run from the repository root:

```sh
python3 -m http.server 8765
```

Open <http://localhost:8765/tuxedo-sam/>.

## The journey

- A rounded, reference-informed Sam with a blue face, surface-conforming white belly, oval eyes, yellow beak and feet, pink bow tie, tilted sailor cap, ribbon tails, and overnight satchel.
- Alternating foot lift and swing, body sway, flipper motion, blinks, hat bounce, and tap-to-wave greetings.
- A geographically grounded, hand-colored globe with coastal shelves, survey lines, forests, mountains, deserts, sailboats, and miniature landmarks.
- Seven destinations: Tuxedo Island, Cape Town, Paris, London, the Arctic, Tokyo, and Sydney. The closed spherical route uses an arc-length lookup so travel speed stays consistent.
- Moving clouds, a circling propeller plane, fine celestial orbit lines, and a starfield.
- Daydream, Golden hour, and Starlight lighting, with smooth transitions.
- Four quiet controls: pause, lighting, optional music, and downloadable framed PNG postcards.

## Controls

Drag to orbit; scroll or pinch to zoom. Tap Sam to wave. Double-click the scene or press R to reset the camera. Space pauses or resumes walking. The lighting button cycles through day, sunset, and night. The small gesture hint fades away automatically.

Reduced-motion preferences pause the scene initially. Music is opt-in. Hidden tabs suspend animation/audio. The app handles graphics context loss and lowers rendering resolution if frame rate drops. Rendering dependencies and geographic data are local; Google Fonts has local fallbacks.

## Files

- `main.js`: Babylon initialization, lighting, shadows, camera, animation, UI, music, postcards, and adaptive resolution.
- `objects.js`: procedural Sam, miniature landmarks, trees, and plane.
- `world.js`: painted geographic atlas, batched scenery, arc-length route, and globe orientation.
- `style.css`: minimal responsive controls.
- `vendor/babylon.js`: pinned Babylon.js 9.25.0, Apache 2.0; see `vendor/LICENSE.md`.
- `land.geojson`: Natural Earth 1:110m land polygons, public domain.

See [REFERENCES.md](REFERENCES.md) for character and map sources.
