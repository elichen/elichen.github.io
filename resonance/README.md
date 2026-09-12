# Resonance

Draw and reshape a membrane, strike it, and hear its calculated vibration modes.
The numerical solver, Canvas renderer, and Web Audio synthesizer run locally in
the browser. No account, audio samples, build step, or API key is required.
Google Fonts is optional; the interface has local font fallbacks.

## Run

From the repository root:

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000/resonance/`. Serve over HTTP rather than opening the
HTML as a file so ES modules and the background worker can load. Sound starts
after a click, tap, or keyboard gesture.

## Play and create

- Choose Round, Square, Petal, or Droplet. In **Play**, tap the membrane or press
  **Space** to strike it. The crosshair button strikes its center.
- **Focus** expands the surface and harmonics into a quiet playing view.
  **Exit focus** or **Escape** restores the studio and your previous scroll position.
  On small screens, swipe the harmonic strip to browse the nine visible modes.
- Choose **Reshape** and drag a rim point. With the canvas focused, use
  **Left/Right** to select one of 16 points and **Up/Down** to move it radially.
  The Width slider changes the horizontal extent. Geometry changes recalculate
  the modes in a worker; restricted worker environments use a main-thread fallback.
- **Draw your own outline** accepts a loop and fits a smooth radial contour to it.
  The result is centered, scaled, and represented by 16 bounded radii, so holes,
  overhangs, disconnected shapes, and exact tracing of arbitrary contours are
  outside this editor's shape representation.
- **Undo** or **Command/Ctrl+Z** restores recent shape edits. **Reset shape**
  restores the current starting shape, or Round for a custom outline.
- **Tension** changes pitch. **Resonance** changes decay time. **Mallet** changes
  the spatial width of the strike and upper-mode damping. Volume sets the level
  for playback and export; mute silences live output without changing that level.
- Click a mode thumbnail to isolate and audition it. The page control switches
  between modes 1–9 and 10–18. **Full sound** recombines the modes.
  A thumbnail auditions its mode directly at a strong vibration point so the
  contact pickup cannot hide the selected mode.
- Save up to four instruments in the rack. Click a filled pad or press **1–4**
  to play it; an empty pad saves the current instrument. The small arrow opens a
  saved instrument in the workbench, and **×** removes it. Editing a loaded pad
  does not overwrite the saved copy. **Play a pattern** sequences the filled pads.
  **Escape** stops the pattern and sounding voices and returns to Play.
- **Export sound .wav** exports one strike of the current instrument or isolated
  mode as 44.1 kHz, 16-bit mono PCM. It is a dry single voice, rather than a
  recording of the rack performance or the live output compressor. Export uses
  the saved volume-slider level even when live output is muted; setting the
  volume slider to zero produces a silent export.
- **Copy instrument link** includes the outline, width, tension, resonance, and
  mallet settings in the URL fragment. It does not include the rack, volume,
  strike position, or selected mode. The current instrument, rack, and volume
  also persist in local storage when available.

## Physical model

The surface is a linear, uniformly tensioned membrane with a fixed rim. On a
41 × 41 Cartesian grid, a five-point finite-difference approximation solves
`−Δφ = λφ` with zero displacement outside the outline. Banded Cholesky solves,
block inverse iteration, and a Rayleigh–Ritz projection recover its lowest
18 eigenmodes. Frequencies are `f = c sqrt(λ) / (2π)`, with
`c = 320 sqrt(tension / 50)` in the app's normalized spatial units.

A strike samples each mode at its center and eight nearby points to approximate
a finite mallet footprint. Modal displacement is proportional to the projected
strike divided by modal mass and angular frequency. Sound is the resulting
displacement at a fixed virtual contact pickup at `(−0.17, 0.19)`. Multiplying
the strike and pickup responses makes the sound invariant to an eigensolver's
arbitrary eigenvector sign. The display uses the same spatial modes and strike
coefficients, with slowed frequencies and exaggerated motion for legibility.
Visual decay is illustrative; it is not an audio-rate waveform display.

`audio.mjs` preserves signed modal ratios, normalizes each strike's total absolute
amplitude for headroom, applies a 3 ms attack and frequency-dependent exponential
decay, and fades the tail to zero. It supports up to 32 modes and eight overlapping
voices. A compressor controls overlapping live peaks; an analyser reports the
actual output waveform to the level indicator. The Resonance setting is the
lowest active mode's approximate time to −60 dB. Higher modes generally decay
faster, depending on Mallet.

This is a coarse membrane model, not a bending metal plate or calibrated model
of a manufactured instrument. The grid approximates the curved boundary, only
18 modes contribute, and the finite strike footprint is sampled sparsely.
Dimensions, tension, gain, and damping are illustrative; the simulation does
not include enclosed air, acoustic radiation, material dispersion, nonlinear
stretching, or collisions. Individual strikes are loudness-normalized, so their
absolute loudness is not a calibrated force measurement. The rendered viewpoint
does not simulate a microphone position. Reduced-motion preferences suppress
surface animation while retaining the controls and sound.

## Sources and possible extensions

- [Dan Russell, Penn State: Vibrational Mode Shapes of a Circular Membrane](https://www.acs.psu.edu/drussell/demos/membranecircle/circle.html)
  explains membrane modes and their nodal patterns.
- [Bruyns and Bindel: Shape-changing objects for sound synthesis (2006)](https://www.cs.cornell.edu/~bindel/papers/2006-sound.pdf)
  connects editable geometry with modal sound synthesis.
- [Doug James: Modal Sound Explorer](https://dougjam.github.io/demos/modal-sound-explorer/)
  is an interaction reference for exploring and striking sounding shapes.
- [Jin et al.: DiffSound (SIGGRAPH 2024)](https://arxiv.org/abs/2409.13486)
  differentiates through geometry, finite-element analysis, and audio synthesis
  to infer physical properties, shape, and impact location. It suggests a future
  “fit a sound” direction; this app does not implement its inverse-rendering pipeline.
- [Diaz and Sandler: Fast Differentiable Modal Simulation of Non-linear Strings,
  Membranes, and Plates (DAFx 2025)](https://arxiv.org/abs/2505.05940)
  uses JAX and GPU computation for differentiable modal simulation and parameter
  fitting. Nonlinear response and inverse design are possible future extensions,
  rather than features of the current linear browser model.

## Verify

```sh
node --test resonance/physics.test.mjs resonance/audio.test.mjs
```

The numerical checks cover membrane eigenvalues, mode orthogonality, shape
changes, sampling, determinism, and invalid geometry. The audio checks cover finite bounded output,
decay, signed coefficients, phase cancellation, frequency filtering, mode limits,
and WAV encoding. Browser QA should also verify gesture-based audio startup,
real analyser activity, keyboard controls, reshape interactions, rack playback,
link round-tripping, downloads, and reduced-motion behavior.
