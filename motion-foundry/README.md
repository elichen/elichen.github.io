# Motion Foundry

Sketch a closed path and search for a planar linkage whose tracing point follows
it. The result is an animated assembly with fixed-length bars and analytically
closed joints. Shape matching, candidate comparison, and tracing-point edits
run in the browser without an account or API key.

## Run

From the repository root:

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000/motion-foundry/`. Use an HTTP server for ES modules
and background workers. There is no package installation or build step.
Google Fonts is optional; the interface includes local font fallbacks.

## Use the workbench

- Drag a continuous loop in **Your path**, or click individual points and choose
  **Close loop**. With the sketch canvas focused, arrow keys move its drawing
  cursor, **Shift** increases the step, **Space** adds a point, and **Enter**
  closes the path. **Backspace** removes the last draft point; **Escape** cancels
  the draft. Clear and Undo drawing let you revise it.
- Choose Walking step, Orbit, Figure eight, or Petal for a starting trajectory.
  **Machine family** limits the search to four-bar linkages, crank-sliders, or
  both. **Find a mechanism** searches and refines candidate designs.
- Compare **Possible machines** and the target and mechanism paths. The error
  readout and search-progress plot report measured fit rather than a claimed
  exact match. A complex drawing can remain a poor fit within these families.
- Pause or play the assembly; **Space** also toggles playback when the machine
  canvas is focused. **Drive speed** sets crank revolutions per second.
  **Target** and **Dimensions** control the drawing overlays.
- Drag the tracing point, or use **Along the coupler** and **Away from the
  coupler** to change its attachment coordinates. Ground-anchor edits adjust
  assembly placement and scale. **Restore best fit** returns to the fitted design.
- Export downloads an SVG derived from sampled mechanism poses: animated while
  playing, or a still drawing when paused or reduced motion is enabled.
  **Copy this study** shares the target and current mechanism in a link.
  Dimensions are in arbitrary design units.

## Kinematic model

Two topology families are available. Each has one rotating input crank and a
point rigidly attached to its coupler:

1. A four-bar linkage has fixed ground pivots `O = (0, 0)` and `G = (1, 0)` in
   canonical coordinates, a crank, a coupler, and a rocker. At crank angle `θ`,
   the moving crank joint is `A = crank × (cos θ, sin θ)`. The other coupler joint
   is the selected intersection of a circle centered at `A` with coupler radius
   and a circle centered at `G` with rocker radius.
2. An offset crank-slider replaces the rocker with a joint constrained to the
   horizontal rail `y = railOffset`. The slider position is the selected
   intersection of that rail and the circle around the crank joint with coupler
   radius. The chosen assembly branch is held throughout the cycle.

If the coupler endpoints are `A` and `B`, the tracing point is
`P = A + traceAlong × (B − A) + traceOffset × R90(B − A)`, where `R90` rotates
a vector by 90 degrees. The tracing point may lie beyond the physical segment;
it represents a rigid extension of the same coupler, not an additional joint.
A similarity transform places, rotates, and uniformly scales the complete
assembly without changing its proportions or joint constraints.

Full-turn geometry constraints reject designs that cannot close across an
entire crank revolution. Dense cycle sampling provides an additional closure
check. All displayed links therefore follow the same geometric assembly rather
than independently interpolated paths. This is kinematic position synthesis:
there is no simulation of force, torque, inertia, joint friction, or impacts.

## Search and fit metric

The search starts by retrieving promising designs from an atlas of valid
geometries. Differential evolution explores the remaining mechanism parameters,
followed by Nelder–Mead local refinement. The default search uses a population of
64 and 150 generations. The optimizer is heuristic and cannot certify that its
best candidate is globally optimal.

Search candidates must also pass a hard size constraint: a conservative bound
on the mechanism-and-tracer extent must be no more than five times the target's
bounding-box diagonal. This excludes oversized assemblies tracing a tiny part
of their possible motion. It is an admissibility filter, not a penalty added to
the reported RMS error. Manual edits and imported designs are measured directly.

Each mechanism is sampled through its closed cycle and resampled by arc length.
The search samples each mechanism at 128 constant-crank-angle positions, then
compares 64 uniformly arc-length-spaced target and mechanism points in their
curve order, while removing cyclic starting phase and traversal direction.
Analytic similarity alignment solves translation, rotation, and scale rather
than spending population dimensions on them. Mirrored candidates are represented
by corresponding physical assembly/tracer parameters.

The displayed percentage is

```text
100 × sqrt(mean(||target[i] − fittedPath[i]||²)) / targetBoundingBoxDiagonal
```

It is an ordered RMS path deviation over those 64 samples, not maximum pointwise
error or a guarantee of an exact match between samples. It is not a percentage
guarantee of manufacturing accuracy. The fit is geometric: it does
not attempt to reproduce the speed of the user's drawing. Playback instead
drives the crank at constant angular speed, so the tracing point naturally moves
faster through some parts of the same curve.

The serializable design contract is:

```js
{
  family: 'fourbar', // or 'slider'
  params: {
    crank, coupler, rocker, // rocker applies to four-bar designs
    railOffset,            // railOffset applies to slider designs
    traceAlong, traceOffset, branch
  },
  transform: { a, b, tx, ty }
}
```

The transform acts as complex multiplication by `a + i b` followed by
translation `(tx, ty)`. The engine provides pose evaluation, cycle sampling,
design validation, and refitting after edits.

## Limits

The engine searches these two planar linkage families, not arbitrary mechanisms,
gears, cams, springs, or motor programs. Links have zero thickness for kinematic
purposes. Crossing lines may require physically separated layers in an actual
assembly; collision, clearance, material strength, bearing size, transmission
angle quality, required motor torque, and manufacturability are not validated.
SVG exports are inspectable geometric studies rather than certified build plans.

Sampling and finite optimization budgets limit the path match. Exact analytic
closure does not imply that the mechanism precisely traces an arbitrary target.
Dimensions have no calibrated real-world unit, and placing the linkage near a
singular configuration can still be undesirable even when the geometric
constraints permit a full turn.

## Research context

- [Carnegie Mellon: Planar Linkages](https://www.cs.cmu.edu/~rapidproto/mechanisms/chpt5.html)
  introduces planar linkage families and four-bar mobility.
- [Storn and Price: Differential Evolution (1997)](https://doi.org/10.1023/A:1008202821328)
  describes the population search method used here.
- [Nelder and Mead: A Simplex Method for Function Minimization (1965)](https://doi.org/10.1093/comjnl/7.4.308)
  describes derivative-free local refinement.
- [Nobari et al.: LInK (2024)](https://arxiv.org/abs/2405.20592)
  combines learned mechanism/trajectory representations, retrieval, and numerical
  refinement for linkage synthesis. Its retrieval-then-refine strategy is an
  inspiration; this app uses its own small geometric atlas and does not load
  LInK's learned models or large mechanism dataset.
- [Jadhav and Farimani: LinkD (2026 preprint)](https://arxiv.org/abs/2601.04054)
  explores autoregressive diffusion for broader linkage topology and geometry
  generation. Learned topology generation is a possible future direction, not
  part of this workbench's two-family numerical search.

## Verify

Run the mechanism tests from the repository root:

```sh
node --test motion-foundry/*.test.mjs
```

Browser QA should cover mouse and keyboard drawing, full-cycle joint closure,
candidate selection, target changes during search, tracing-point edits, playback
controls, reduced-motion preferences, shared-study validation, and SVG playback.
An exported trace should agree with the engine's sampled tracing-point positions.
