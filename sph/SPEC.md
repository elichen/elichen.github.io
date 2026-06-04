# WebGPU SPH Fluid — Implementation Spec

A 2D Smoothed Particle Hydrodynamics fluid that runs entirely on the GPU with
WebGPU compute shaders, rendered as a shaded liquid surface (screen-space
metaballs). The user can stir the fluid with the mouse, tilt the box to slosh it,
and drag obstacles through it. This lives at `/sph/` and follows the same
self-contained, no-build pattern as `/galaxy/`.

Target: ~40,000 particles at 60 fps on a modern laptop GPU. Degrade particle
count on weaker hardware via a dropdown, the way `/galaxy/` does.

## Files

- `sph/index.html` — canvas + control panel, same structure/classes as `galaxy/index.html`.
- `sph/main.js` — WebGPU setup, buffers, pipeline orchestration, input handling, render loop.
- `sph/styles.css` — dark, minimal panel styling consistent with `galaxy/styles.css`.
- `sph/shaders.js` — exports WGSL source strings (keep all WGSL here as template literals so `main.js` stays readable). Alternatively inline in `main.js`; one file of WGSL is fine.

No external libraries. No build step. Pure ES modules loaded with `<script type="module">`.

## WebGPU bring-up

- Request adapter + device. If `navigator.gpu` is missing or adapter request
  fails, replace the canvas with a centered notice: "This demo needs WebGPU —
  try Chrome or Edge on desktop." Match the tone/markup of any existing fallback
  in the repo; keep it simple.
- Configure the canvas context with `navigator.gpu.getPreferredCanvasFormat()`
  (NOT `device.getPreferredCanvasFormat` — that method does not exist; see
  `galaxy/main.js:408`), `alphaMode: 'premultiplied'`.
- Handle DPR: size the canvas backing store to `clientWidth*dpr` etc., and on
  resize rebuild the screen-space render targets (below). Cap dpr at 1.5 (matches
  `galaxy/main.js:240`; the float screen textures make dpr 2 expensive on large
  displays).

## Simulation model — WCSPH (Müller 2003 + Tait pressure)

Weakly compressible SPH in 2D. Symplectic Euler integration. Fixed substep dt.

State per particle. **Ping-pong is required, not optional.** The force/integrate
pass reads neighbor positions and velocities while writing this particle's new
position and velocity — updating in place is a data race across invocations
within one dispatch (unlike `galaxy`, where each invocation only reads and writes
itself, `galaxy/main.js:34`/`:55`). So:
- `particlesIn` and `particlesOut` buffers, each holding `{ pos: vec2<f32>,
  vel: vec2<f32> }` per particle. The force/integrate pass reads `particlesIn`,
  writes `particlesOut`; swap the two after each substep.
- The spatial grid is rebuilt every substep from the current read buffer
  (`particlesIn`).

Derived per substep (separate buffers, recomputed each substep):
- `density: f32`
- `pressure: f32`
These are computed from `particlesIn` before the force pass, so reading them in
the force pass is safe.

### Constants — derive everything from particle spacing (expose starred ones in UI)

Everything keys off the rest spacing `s` so the numbers stay self-consistent.
Concrete coherent set (start here):

- `s = 0.5` — rest spacing between particles (sim units).
- `h = 2 * s = 1.0` — smoothing radius / kernel support. A disc of radius `h`
  then holds ~12–13 neighbors, the right count for stable 2D SPH. Grid cell
  size = `h`.
- `restDensity` (`rho0`) `= 1.0`.
- `mass = rho0 * s^2 = 0.25` — the 2D rest-density estimate. Verify at reset by
  evaluating the discrete poly6 density on the seeded lattice; nudge `mass` so
  the interior settles near `rho0`.
- The **box** is sized to hold the largest preset comfortably: use `200 x 130`
  sim units → grid `200 x 130 = 26,000` cells at cell size 1.0. (Recompute grid
  dims from box/`h` in JS; don't hard-code if you change the box.)
- Particle-count presets fix `s` and grow the **spawn block** (see Initial
  conditions): at `s = 0.5`, area per particle is `0.25`, so 40K needs a
  `10,000`-unit² block (e.g. `100 wide x 100 tall`), 80K needs `~20,000`
  (e.g. `160 x 125`). Both fit the `200 x 130` box. If you change the box,
  re-derive these.

UI-exposed (starred) constants modulate behavior on top of this base:
- `stiffness` (`k`)* — pressure stiffness (Tait/gas). UI slider.
- `viscosity` (`mu`)* — UI slider.
- `gravity`* — UI slider; direction is modulated by box tilt (below).
- `tilt`* — UI slider (see Boundaries/tilt).

Fixed (non-UI) constants:
- `dt` — fixed, e.g. `0.0016`–`0.004`. Run 1–2 substeps per displayed frame;
  make substeps a compile-time constant.
- `boundaryStiffness`, `boundaryDamping` — penalty for walls/obstacles.

### Kernels (2D normalizations)

Use the classic Müller kernel set, 2D-normalized:
- Poly6 for density: `W(r) = (4 / (pi*h^8)) * (h^2 - r^2)^3` for `0 <= r <= h`.
- Spiky gradient for pressure force: `gradW(r) = -(30 / (pi*h^5)) * (h - r)^2 * (rvec/r)`.
- Viscosity Laplacian: `lapW(r) = (40 / (pi*h^5)) * (h - r)`.

(If you prefer cubic-spline kernels, that's acceptable — just keep the 2D
normalization consistent across density and forces. Document whichever you use.)

### Pressure (Tait equation, weakly compressible)

`pressure = k * (density - restDensity)`, clamped at `>= 0` to avoid tensile
instability (or use the Tait form `k*((density/rho0)^7 - 1)` if it stays stable;
start with the linear clamped form — it's robust).

### Forces per particle — work in ACCELERATION units

To avoid unit-mismatch bugs between SPH forces (force *density*) and penalty
forces, accumulate **acceleration** directly and integrate without a final
`/density`. Conventions:
- `r_ij = pos_i - pos_j` (vector from j to i); `r = length(r_ij)`.
- `gradW` is the full kernel gradient **with respect to particle i**, sign
  included. The spiky form above already carries its negative sign, so `gradW`
  has magnitude `(30/(pi*h^5))*(h-r)^2` and points along **`-r_ij/r`** (toward
  the neighbor). Do not re-apply a sign. With the pressure formula below this
  makes positive pressure repulsive (correct); if you instead see particles
  clumping, the sign got flipped. Skip any neighbor with `r < epsilon`
  (e.g. `1e-6`) for pressure/viscosity to avoid div-by-zero; **include self** in
  the density sum.

Accelerations (summed):
- Pressure: `a_pressure = -sum_j mass * (p_i/rho_i^2 + p_j/rho_j^2) * gradW(r_ij)`.
  (Symmetric form using `rho_i^2`/`rho_j^2` — momentum-conserving and stable.)
- Viscosity: `a_visc = (mu/rho_i) * sum_j (mass/rho_j) * (v_j - v_i) * lapW(r_ij)`.
- Gravity: `a_gravity = gravityVec` (rotated by tilt) — added directly, no density factor.
- Boundary/obstacle penalty: added directly as accelerations (below).
- Mouse interaction: added directly as acceleration (below).

Integrate (symplectic Euler): `vel += dt * a_total; pos += dt * vel`. Clamp
`rho_i >= rho0 * 0.1` before dividing. Apply a small velocity clamp (CFL: keep
`maxSpeed` such that `dt * maxSpeed < h`) and optional global damping `*0.999`.

## Neighbor search — GPU spatial hash grid

Rebuild the uniform grid every substep. Cell size = `h`, so each particle only
checks its own cell plus the 8 neighbors (3x3 in 2D).

Use **counting sort** into cells (no bitonic sort needed). Within-cell order is
nondeterministic because of atomic scatter — that's fine for this demo.

1. **Clear** `cellCount` buffer (one u32 per cell) to 0 **and** clear the
   `cellOffset` scratch buffer to 0 (one u32 per cell). `cellOffset` must be
   re-zeroed every substep or scatter writes go out of range after frame 1.
2. **Count** pass: each particle computes its cell index from `pos`, does
   `atomicAdd(cellCount[cell], 1)`.
3. **Prefix sum** over `cellCount` → `cellStart` (exclusive scan). The grid has
   ~26K cells — too many for one workgroup invocation-per-cell (exceeds the
   256-ish max invocations / 16KB workgroup-memory limits). Use **either**:
   (a) a single-invocation serial exclusive scan over all cells in one
   `@workgroup_size(1)` dispatch (simple, correct, ~26K iterations is cheap once
   per substep), **or** (b) a proper two-level block scan. Start with (a) for
   correctness; only move to (b) if it profiles as a bottleneck.
4. **Scatter** pass: each particle computes cell, `idx = cellStart[cell] +
   atomicAdd(cellOffset[cell], 1)`, writes its particle index into
   `sortedIndices[idx]`.

Then density/force passes iterate neighbors by looping the 3x3 cells around a
particle's cell and walking `sortedIndices[cellStart[c] .. cellStart[c]+count]`.

Grid covers the box bounds (plus margin). Clamp particle cell coords into range.
Particles that leave the box are pushed back by boundary forces, so they should
stay in-bounds, but clamp defensively to avoid OOB reads.

### Compute pass ordering per substep

1. Clear grid counts AND clear `cellOffset` scratch to 0.
2. Count particles per cell (reads `particlesIn`).
3. Prefix sum → cellStart.
4. Scatter → sortedIndices.
5. Density pass (reads `particlesIn` + sorted grid) → density, pressure.
6. Force + integrate pass (reads `particlesIn` + density/pressure + sorted grid)
   → writes `particlesOut`. Swap `particlesIn`/`particlesOut` after the substep.

All as separate `dispatchWorkgroups` calls in one `CommandEncoder`. Workgroup
size 64 or 256 for the particle passes. Use `@group(0)` bindings; a uniform
buffer holds all sim constants + frame state (mouse, tilt, gravity, counts).

## Boundaries, tilt, and obstacles

- **Box walls:** penalty acceleration when a particle is within ~`h` of a wall
  (acceleration units, consistent with the force section):
  `a += boundaryStiffness * penetration * inwardNormal - boundaryDamping * vel`.
  Plus a hard clamp of position inside the box after integration as a safety net.
- **Tilt:** the gravity vector is `gravity * (sin(tiltAngle), -cos(tiltAngle))`.
  Control via the UI `tilt` slider (the reliable default). Keep `tiltAngle` in
  the uniform buffer. Optionally show the box frame rotating slightly.
- **Obstacles:** support up to N (e.g. 4) circular obstacles. Pass them in the
  uniform buffer as **`vec4<f32>(x, y, radius, active)`** — use vec4, not vec3,
  so WGSL uniform 16-byte alignment is satisfied (pack the typed array carefully,
  the way `galaxy/main.js:159` packs its uniforms). `active <= 0` means the slot
  is unused. Carry **obstacle velocity** too (a second `vec4` per obstacle, or
  compute it in JS from frame-to-frame position and store it) so a dragged disc
  shoves fluid rather than teleporting through it.
  - Particles within `radius + h` of an obstacle feel an outward penalty
    acceleration plus damping against **relative** velocity:
    `a += boundaryStiffness*penetration*outwardNormal - boundaryDamping*(vel - obsVel)`.
  - Drag: pick the nearest obstacle whose disc contains the cursor on mousedown,
    move it on drag; update its velocity from cursor motion.
  - Render obstacles as solid dark discs on top of the fluid. A dragged obstacle
    should shove fluid aside and leave a visible wake.

## Mouse interaction (stir / push / pull)

While dragging on the fluid (not on an obstacle):
- Default drag = **stir/push**: add a force to particles within a radius of the
  cursor in the direction of cursor velocity (carry `mouseVel` in the uniform).
- Shift-drag = **pull** toward the cursor (like galaxy's attract).
- Pass `mousePos`, `mouseVel`, `mouseRadius`, `mouseStrength`, and a mode flag in
  the uniform buffer; apply in the force pass.

## Rendering — screen-space metaball fluid

Render the particles as a smooth liquid surface, not visible dots. Pipeline:

1. **Thickness/field pass:** draw each particle as an **instanced quad** (6
   vertices per particle; WebGPU point-list has no programmable point size, so
   point sprites are not an option — `galaxy`'s point-list is 1px only,
   `galaxy/main.js:205`). Give the quad a radial falloff in the fragment shader,
   **additively blended** into an offscreen **`rgba16float`** texture (default to
   rgba16float for reliable additive-blend + texture-binding support, mirroring
   `galaxy/main.js:198`/`:249`; only try a narrower format after verifying blend
   support). **Clear this texture every frame** before the splat pass — it is a
   current-frame field, NOT an accumulating trail buffer (`galaxy` deliberately
   fades/accumulates at `:358`; do not copy that here). The accumulated value is
   a density field in screen space. Quad half-size in screen pixels ≈ `h` mapped
   through the world→screen transform.
2. **Surface shading pass:** full-screen pass samples the field. Threshold the
   field to define the fluid surface (anything above `iso` is liquid). Compute a
   screen-space normal from the gradient of the field (finite differences) for
   fake lighting. Shade with:
   - A water-blue albedo with depth-tinting (deeper/thicker = darker, richer blue).
   - A single directional light + a specular highlight from the normal for a wet sheen.
   - Soft anti-aliased edge at the iso threshold (smoothstep over the field).
   - Optional: subtle fresnel rim, faint refraction by offsetting a background
     gradient using the normal. Keep it tasteful, not noisy.
3. Composite obstacles and the box frame on top.

Avoid an explicit blur pass if the quad falloff is wide enough to give a smooth
field; if it looks blobby, add one separable Gaussian blur pass on the field
texture before shading.

**World→screen mapping:** map the fixed SPH box into the visible canvas with
padding (aspect-aware, like `galaxy/main.js:414` but for a bounded box rather
than an infinite field). Keep one transform used by both the splat vertex shader
and JS mouse→sim-space picking so input and rendering agree.

Color direction: deep, slightly cyan water on a near-black background, matching
the repo's dark aesthetic. Make it look like liquid, with highlights that move
as the fluid moves.

## Initial conditions

- Spawn particles on a jittered grid at spacing `s` (= 0.5) in a block sitting
  against the left wall / bottom, sized to the preset count (40K → ~`100 x 100`
  units, 80K → ~`160 x 125`). The tall block collapses and splashes under gravity
  on load — an immediately satisfying opening.
- Add small position jitter (±`0.1*s`) so the lattice doesn't pop into perfectly
  ordered columns. Verify rest density on this lattice when tuning `mass`.
- `Reset` re-seeds this block; changing the Particles preset reallocates buffers
  and re-seeds.

## UI panel (mirror galaxy's panel markup/classes)

- Title: "WEBGPU SPH FLUID".
- `Particles` dropdown: 10K / 20K / 40K / 80K (default 40K; changing it
  reallocates buffers and reseeds).
- Sliders: `Gravity`, `Viscosity`, `Stiffness`, `Tilt`.
- Buttons: `Pause`, `Reset`, `Add obstacle` (spawns a draggable disc, up to max).
- Hint line: "Drag to stir · Shift-drag to pull · Drag the discs · Tilt to slosh".
- Stats line: `<fps> fps · <count> particles`. Compute fps with a rolling average
  like galaxy does.

## Performance & correctness notes

- Run 1–2 substeps per frame; expose substeps as a constant, not UI.
- Keep all sim state on the GPU; never read particle buffers back to JS in the
  hot loop. Only the uniform buffer is written from JS each frame (mouse, tilt,
  gravity, obstacle positions, counts).
- The prefix-sum scan is the trickiest part to get right — verify the grid by
  testing at low particle counts first; an off-by-one in `cellStart` shows up as
  particles ignoring neighbors (no incompressibility, fluid passes through
  itself). Get the simulation correct at 10K before chasing 80K.
- Guard against NaN blow-ups: clamp `density >= restDensity*0.1` before dividing,
  clamp max velocity, clamp positions into the box each step.
- Pause should freeze the sim (skip compute dispatch) but keep rendering and
  input responsive.

## Acceptance criteria

1. Loads at `/sph/index.html`, no console errors, WebGPU fallback notice on
   unsupported browsers.
2. Fluid block collapses and splashes on load, settles into a flat pool with a
   visible, shaded liquid surface — not a cloud of dots.
3. Dragging stirs/pushes the fluid; shift-drag pulls it; the response feels
   fluid and incompressible (no particles tunneling through each other en masse).
4. Tilt sloshes the pool to one side and it settles level when tilt returns to 0.
5. At least one draggable circular obstacle pushes fluid aside and leaves a wake.
6. 40K particles hold ~60 fps on a modern laptop GPU; lower presets for weaker
   hardware. Stats line is accurate.
7. Panel controls all work live without reload (except Particles count, which may
   reallocate).

## Repo integration (after the demo works)

Add a card to the root `index.html` in the **Built with AI** section
(`#ai-apps`), matching the existing `.game-box` markup, e.g.:

> **WebGPU Fluid** — A 2D smoothed-particle-hydrodynamics fluid running entirely
> on the GPU. Stir it, drag obstacles through it, and tilt the box to slosh
> tens of thousands of particles in real time. (link: `sph/index.html`)

Keep the copy plain and specific; no marketing adjectives.
