# Luma

A small optical workbench, built with plain JavaScript and Canvas. No packages,
build step, external assets, or network services are required.

From the repository root, run `python3 -m http.server 8000`, then open
`http://localhost:8000/luma/`.

## Play

- Drag a light source, prism, mirror, or lens to move it.
- Drag the circular handle above a selected object to rotate it. Hold Shift to
  snap rotation to 15° increments.
- Select objects with the Objects menu to edit them without a mouse. Arrow keys
  move the selection; Q and E rotate it. Hold Shift for larger steps.
- Use Hide controls (F) for an unobstructed view; Show controls brings the workbench back.
- Keys 1–4 add objects; D duplicates; Backspace/Delete removes; G toggles the grid.
- Command/Ctrl+Z undoes. Command/Ctrl+Shift+Z redoes. Preset changes and resets
  can also be undone.
- Scenes persist in local storage when available. Save image downloads a PNG.

## Model

`physics.mjs` is independent of the interface. Each source emits parallel rays;
white sources sample 41 wavelengths from 390 to 690 nm. Segments are traced to
their closest surface, up to 24 interactions per ray. Mirrors use specular
reflection. Prisms use vector Snell refraction and total internal reflection.
A Cauchy-style model controls dispersion, referenced at n = 1.52 at 550 nm.
It represents an illustrative material rather than a measured glass.

Lenses are ideal thin lenses using a paraxial slope transformation in their local
coordinate system. The focal distance is in workbench units. There is no physical
scale calibration. Colors, glow, and moving glints are illustrative, not a spectral
power or speed-of-light measurement. Diffraction, interference, and partial
Fresnel reflections are omitted. For overlapping prisms, the last prism in the
object list containing a point determines the local medium.

Background: OpenStax's [refraction](https://openstax.org/books/university-physics-volume-3/pages/1-3-refraction)
and [thin lenses](https://openstax.org/books/university-physics-volume-3/pages/2-4-thin-lenses).

The interface fits a fixed 1200 × 760 workbench into its viewport. Light paths
are cached between edits. Glints stop when reduced motion is requested, and
rendering pauses in hidden tabs. The canvas has an equivalent object picker,
keyboard movement, and labeled HTML controls.

## Verify

Run `node --test luma/physics.test.mjs` from the repository root. Tests cover Snell's
law and reversibility, total internal reflection, lens focus from both directions,
rotated lenses, preset routing, and bounded reflection loops.
