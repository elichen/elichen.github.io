// Molecular dynamics of a 2D liquid with a few large beads, on WebGPU.
// Shared by the page and the Node tools (tools/gpu.mjs gives them a device).
//
// Units: molecule diameter σ = 1, molecule mass m = 1, ε = kB = 1, time τ = σ√(m/ε).
// Molecules repel each other with the WCA potential (Lennard-Jones cut at its
// minimum), beads repel molecules with the same potential shifted out to their
// radius. Nothing here is random after the initial state: the beads' jiggle comes
// only from collisions.
//
// One step is two dispatches:
//   molecules: forces from a cell grid, leapfrog kick and drift, insert into the
//              next grid, clear the grid after that, partial sums per workgroup
//   beads:     one workgroup per bead sums the force from nearby molecules and
//              moves the bead; one more workgroup sums the partials (temperature,
//              momentum) and sets the thermostat for the next step
// Molecule positions and the state (beads + system) are double-buffered, the grid
// triple-buffered, so no dispatch reads what it writes.
//
// GPU safety: every shader loop has a fixed bound (a runaway kernel can hang the
// whole Mac), and grid cells hold at most SLOTS molecules.

export const RC = 2 ** (1 / 6);
export const SLOTS = 7;
export const MAX_BEADS = 8;
const WG = 256;
const STRIDE = SLOTS + 1;
const PARAM_BYTES = 96;
const SYS_BYTES = 32;
const BEAD_BYTES = 64;
export const STATE_BYTES = SYS_BYTES + MAX_BEADS * BEAD_BYTES;
export const REC_BYTES = 32;

const COMMON = /* wgsl */ `
const RC: f32 = ${RC};
const RC2: f32 = ${RC * RC};
const SLOTS: u32 = ${SLOTS}u;
const STRIDE: u32 = ${STRIDE}u;
const MAX_BEADS: u32 = ${MAX_BEADS}u;

struct Params {
  L: vec2f, invCell: vec2f,
  n: u32, nx: u32, ny: u32, nb: u32,
  dt: f32, T0: f32, tauT: f32, grab: i32,
  anchor: vec2f, kSpring: f32, nPartials: u32,
  recordEvery: u32, ringSize: u32, thermostat: u32, fMax: f32,
  pull: vec2f, pullBead: i32, p0: u32,
}
struct Bead {
  x: vec2f, v: vec2f, wrap: vec2i, a: f32, m: f32,
  ext: vec2f, f: vec2f, step: u32, p0: u32, p1: u32, p2: u32,
}
struct Sys { lambda: f32, T: f32, overflow: u32, steps: u32, px: f32, py: f32, p0: f32, p1: f32 }
struct State { sys: Sys, beads: array<Bead, ${MAX_BEADS}> }
struct Rec { x: vec2f, v: vec2f, wrap: vec2i, f: vec2f }

@group(0) @binding(0) var<uniform> P: Params;

fn minImage(d: vec2f) -> vec2f { return d - P.L * round(d / P.L); }

fn cellCoord(x: vec2f) -> vec2u {
  return min(vec2u(max(x * P.invCell, vec2f(0.0))), vec2u(P.nx - 1u, P.ny - 1u));
}
fn cellIndex(x: vec2f) -> u32 { let c = cellCoord(x); return c.y * P.nx + c.x; }
fn wrapCell(c: i32, n: u32) -> u32 { let m = i32(n); return u32(((c % m) + m) % m); }

// WCA between molecules, d = xi - xj, r2 = |d|^2 < RC2
fn pairForce(d: vec2f, r2: f32) -> vec2f {
  let ir2 = 1.0 / max(r2, 0.36);
  let ir6 = ir2 * ir2 * ir2;
  return d * ((48.0 * ir6 - 24.0) * ir6 * ir2);
}

// WCA with the contact pushed out by delta (surface separation s = r - delta)
fn shiftedForce(d: vec2f, delta: f32) -> vec2f {
  let reach = delta + RC;
  let r2 = dot(d, d);
  if (r2 >= reach * reach) { return vec2f(0.0); }
  let r = sqrt(r2);
  let is = 1.0 / max(r - delta, 0.6);
  let is2 = is * is;
  let is6 = is2 * is2 * is2;
  return d * ((48.0 * is6 - 24.0) * is6 * is / max(r, 1e-6));
}
`;

const MOLECULES = COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> posIn: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> posOut: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> vel: array<vec2f>;
@group(0) @binding(4) var<storage, read> gridCur: array<u32>;
@group(0) @binding(5) var<storage, read_write> gridNext: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> gridClear: array<u32>;
@group(0) @binding(7) var<storage, read> S: State;
@group(0) @binding(8) var<storage, read_write> partials: array<vec4f>;
var<workgroup> red: array<vec4f, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = gid.x;
  let ncell = P.nx * P.ny;
  let stride = nwg.x * ${WG}u;
  for (var c = i; c < ncell; c += stride) { gridClear[c * STRIDE] = 0u; }

  var acc = vec4f(0.0);
  if (i < P.n) {
    let x = posIn[i];
    var f = vec2f(0.0);
    let cc = vec2i(cellCoord(x));
    for (var oy = -1; oy <= 1; oy++) {
      let cy = wrapCell(cc.y + oy, P.ny);
      for (var ox = -1; ox <= 1; ox++) {
        let base = (cy * P.nx + wrapCell(cc.x + ox, P.nx)) * STRIDE;
        let cnt = min(gridCur[base], SLOTS);
        for (var k = 0u; k < cnt; k++) {
          let j = gridCur[base + 1u + k];
          if (j != i) {
            let d = minImage(x - posIn[j]);
            let r2 = dot(d, d);
            if (r2 < RC2) { f += pairForce(d, r2); }
          }
        }
      }
    }
    var ext = vec2f(0.0);
    for (var b = 0u; b < min(P.nb, MAX_BEADS); b++) {
      let B = S.beads[b];
      f += shiftedForce(minImage(x - B.x), B.a - 0.5);
      ext += B.ext;
    }
    // whatever pulls on a bead pushes back on the liquid, so total momentum stays zero
    f -= ext / f32(P.n);

    // the thermostat rescales velocities relative to the liquid's mean flow, keeping its momentum
    let u = vec2f(S.sys.px, S.sys.py) / f32(P.n);
    let v0 = u + (vel[i] - u) * S.sys.lambda;
    let v1 = v0 + f * P.dt;
    var x1 = x + v1 * P.dt;
    x1 -= P.L * floor(x1 / P.L);
    x1 = select(x1, vec2f(0.0), x1 >= P.L);
    vel[i] = v1;
    posOut[i] = x1;
    let c1 = cellIndex(x1) * STRIDE;
    let slot = atomicAdd(&gridNext[c1], 1u);
    var over = 0.0;
    if (slot < SLOTS) { atomicStore(&gridNext[c1 + 1u + slot], i); } else { over = 1.0; }
    let vm = 0.5 * (v0 + v1);
    acc = vec4f(v1, dot(vm, vm), over);
  }
  red[li] = acc;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { partials[wid.x] = red[0]; }
}
`;

const BEADS = COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> posIn: array<vec2f>;
@group(0) @binding(2) var<storage, read> gridCur: array<u32>;
@group(0) @binding(3) var<storage, read> S: State;
@group(0) @binding(4) var<storage, read_write> Snext: State;
@group(0) @binding(5) var<storage, read> partials: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> ring: array<Rec>;
var<workgroup> red: array<vec4f, ${WG}>;

fn reduce(li: u32) {
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
}

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  let b = wid.x;
  if (b < P.nb) {
    let B = S.beads[b];
    let delta = B.a - 0.5;
    let reach = delta + RC;
    let lo = vec2i(floor((B.x - vec2f(reach)) * P.invCell));
    let hi = vec2i(floor((B.x + vec2f(reach)) * P.invCell));
    let w = u32(clamp(hi.x - lo.x + 1, 1, i32(P.nx)));
    let h = u32(clamp(hi.y - lo.y + 1, 1, i32(P.ny)));
    let total = min(w * h, 8192u);
    var f = vec2f(0.0);
    for (var k = li; k < total; k += ${WG}u) {
      let cx = wrapCell(lo.x + i32(k % w), P.nx);
      let cy = wrapCell(lo.y + i32(k / w), P.ny);
      let base = (cy * P.nx + cx) * STRIDE;
      let cnt = min(gridCur[base], SLOTS);
      for (var q = 0u; q < cnt; q++) {
        f -= shiftedForce(minImage(posIn[gridCur[base + 1u + q]] - B.x), delta);
      }
    }
    red[li] = vec4f(f, 0.0, 0.0);
    reduce(li);
    if (li == 0u) {
      let fluid = red[0].xy;
      var F = fluid;
      for (var c = 0u; c < min(P.nb, MAX_BEADS); c++) {
        if (c != b) {
          let O = S.beads[c];
          F += shiftedForce(minImage(B.x - O.x), B.a + O.a - 1.0);
        }
      }
      var ext = vec2f(0.0);
      if (P.grab == i32(b)) {
        ext = P.kSpring * minImage(P.anchor - B.x);
        let e = length(ext);
        if (e > P.fMax) { ext *= P.fMax / e; }
      }
      if (P.pullBead == i32(b)) { ext += P.pull; }
      F += ext;
      let v1 = B.v + F * (P.dt / B.m);
      var x1 = B.x + v1 * P.dt;
      let fl = floor(x1 / P.L);
      x1 -= P.L * fl;
      // x just below 0 can round up to L: call it 0 in the next box over
      let edge = x1 >= P.L;
      x1 = select(x1, vec2f(0.0), edge);
      let wrap = B.wrap + vec2i(fl) + select(vec2i(0), vec2i(1), edge);
      let step = B.step + 1u;
      Snext.beads[b] = Bead(x1, v1, wrap, B.a, B.m, ext, fluid, step, 0u, 0u, 0u);
      if (step % P.recordEvery == 0u) {
        ring[((step / P.recordEvery) % P.ringSize) * P.nb + b] = Rec(x1, v1, wrap, fluid);
      }
    }
  } else if (b == P.nb) {
    var acc = vec4f(0.0);
    for (var k = li; k < P.nPartials; k += ${WG}u) { acc += partials[k]; }
    red[li] = acc;
    reduce(li);
    if (li == 0u) {
      let t = red[0];
      let T = t.z / (2.0 * f32(P.n));
      var lambda = 1.0;
      if (P.thermostat != 0u) {
        lambda = sqrt(clamp(1.0 + P.dt / P.tauT * (P.T0 / max(T, 1e-3) - 1.0), 0.81, 1.21));
      }
      Snext.sys = Sys(lambda, T, S.sys.overflow + u32(t.w), S.sys.steps + 1u, t.x, t.y, 0.0, 0.0);
    }
  }
}
`;

const BUILD = COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> posIn: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> grid: array<atomic<u32>>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let c = cellIndex(posIn[i]) * STRIDE;
  let slot = atomicAdd(&grid[c], 1u);
  if (slot < SLOTS) { atomicStore(&grid[c + 1u + slot], i); }
}
`;

// Small seeded generator so a run can be repeated exactly.
export function rng(seed) {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.normal = () => Math.sqrt(-2 * Math.log(1 - next())) * Math.cos(2 * Math.PI * next());
  return next;
}

// Molecules on a jittered square lattice (spacing > RC, so nothing overlaps),
// none inside a bead; Maxwell velocities with zero total momentum.
export function initialState({ L, density, beads, T = 1, seed = 1 }) {
  const rand = rng(seed);
  const cols = Math.round(L[0] * Math.sqrt(density));
  const rows = Math.round(L[1] * Math.sqrt(density));
  const hx = L[0] / cols, hy = L[1] / rows;
  const pos = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = (c + 0.5) * hx + (rand() - 0.5) * 0.05;
      const y = (r + 0.5) * hy + (rand() - 0.5) * 0.05;
      let free = true;
      for (const b of beads) {
        let dx = x - b.x, dy = y - b.y;
        dx -= L[0] * Math.round(dx / L[0]);
        dy -= L[1] * Math.round(dy / L[1]);
        if (dx * dx + dy * dy < (b.a + 0.6) ** 2) { free = false; break; }
      }
      if (free) pos.push(x, y);
    }
  }
  const n = pos.length / 2;
  const vel = new Float64Array(2 * n);
  const bv = beads.map(b => [rand.normal() * Math.sqrt(T / b.m), rand.normal() * Math.sqrt(T / b.m)]);
  let px = 0, py = 0, mass = n;
  for (let i = 0; i < n; i++) {
    vel[2 * i] = rand.normal() * Math.sqrt(T);
    vel[2 * i + 1] = rand.normal() * Math.sqrt(T);
    px += vel[2 * i]; py += vel[2 * i + 1];
  }
  beads.forEach((b, k) => { px += b.m * bv[k][0]; py += b.m * bv[k][1]; mass += b.m; });
  const ux = px / mass, uy = py / mass;
  for (let i = 0; i < n; i++) { vel[2 * i] -= ux; vel[2 * i + 1] -= uy; }
  return {
    n,
    pos: Float32Array.from(pos),
    vel: Float32Array.from(vel),
    beads: beads.map((b, k) => ({
      x: [b.x, b.y], v: [bv[k][0] - ux, bv[k][1] - uy], wrap: [0, 0], a: b.a, m: b.m, step: 0,
    })),
  };
}

export class Sim {
  // cfg: { L: [Lx, Ly], dt, T, tauT, thermostat, recordEvery, ringSize }
  // init: { n, pos, vel, beads } (see initialState)
  constructor(device, cfg, init) {
    if (init.beads.length > MAX_BEADS) throw new Error(`at most ${MAX_BEADS} beads`);
    this.device = device;
    this.L = cfg.L;
    this.n = init.n;
    this.nb = init.beads.length;
    this.nx = Math.floor(cfg.L[0] / RC);
    this.ny = Math.floor(cfg.L[1] / RC);
    if (this.nx < 3 || this.ny < 3) throw new Error('box too small');
    this.ncell = this.nx * this.ny;
    this.nPartials = Math.ceil(this.n / WG);
    this.dt = cfg.dt ?? 0.005;
    this.T0 = cfg.T ?? 1;
    this.tauT = cfg.tauT ?? 20;
    this.thermostat = cfg.thermostat ?? true;
    this.recordEvery = cfg.recordEvery ?? 10;
    this.ringSize = cfg.ringSize ?? 4096;
    this.grab = -1;
    this.anchor = [0, 0];      // where a grabbed bead is pulled (spring)
    this.kSpring = cfg.kSpring ?? 2;
    this.fMax = cfg.fMax ?? 20;
    this.pullBead = -1;            // a constant force on one bead (for measuring drag)
    this.pull = [0, 0];
    this.s = 0;

    const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
    const buf = (size, usage) => device.createBuffer({ size: Math.max(16, size), usage });
    this.params = buf(PARAM_BYTES, GPUBufferUsage.UNIFORM | CD);
    this.pos = [buf(8 * this.n, S | CD | CS), buf(8 * this.n, S | CD | CS)];
    this.vel = buf(8 * this.n, S | CD | CS);
    this.grid = [0, 1, 2].map(() => buf(4 * STRIDE * this.ncell, S | CD | CS));
    this.state = [buf(STATE_BYTES, S | CD | CS), buf(STATE_BYTES, S | CD | CS)];
    this.partials = buf(16 * this.nPartials, S | CS);
    this.ring = buf(REC_BYTES * this.ringSize * this.nb, S | CS);

    const pipe = code => device.createComputePipeline({
      layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
    this.molPipe = pipe(MOLECULES);
    this.beadPipe = pipe(BEADS);
    this.buildPipe = pipe(BUILD);
    const group = (p, list) => device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: list.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.molGroups = [];
    this.beadGroups = [];
    for (let k = 0; k < 6; k++) {
      const p2 = k % 2, g3 = k % 3;
      this.molGroups.push(group(this.molPipe, [this.params, this.pos[p2], this.pos[1 - p2], this.vel,
        this.grid[g3], this.grid[(g3 + 1) % 3], this.grid[(g3 + 2) % 3], this.state[p2], this.partials]));
      this.beadGroups.push(group(this.beadPipe, [this.params, this.pos[p2], this.grid[g3],
        this.state[p2], this.state[1 - p2], this.partials, this.ring]));
    }
    this.buildGroup = group(this.buildPipe, [this.params, this.pos[0], this.grid[0]]);
    this.load(init);
  }

  writeParams() {
    const a = new ArrayBuffer(PARAM_BYTES), f = new Float32Array(a), u = new Uint32Array(a), i = new Int32Array(a);
    f[0] = this.L[0]; f[1] = this.L[1]; f[2] = this.nx / this.L[0]; f[3] = this.ny / this.L[1];
    u[4] = this.n; u[5] = this.nx; u[6] = this.ny; u[7] = this.nb;
    f[8] = this.dt; f[9] = this.T0; f[10] = this.tauT; i[11] = this.grab;
    f[12] = this.anchor[0]; f[13] = this.anchor[1]; f[14] = this.kSpring; u[15] = this.nPartials;
    u[16] = this.recordEvery; u[17] = this.ringSize; u[18] = this.thermostat ? 1 : 0; f[19] = this.fMax;
    f[20] = this.pull[0]; f[21] = this.pull[1]; i[22] = this.pullBead;
    this.device.queue.writeBuffer(this.params, 0, a);
  }

  // Replace the whole state (molecules, beads); starts again at step 0.
  load({ pos, vel, beads, lambda = 1, T, p = [0, 0] }) {
    const q = this.device.queue;
    this.s = 0;
    this.steps = 0;
    q.writeBuffer(this.pos[0], 0, pos);
    q.writeBuffer(this.vel, 0, vel);
    const st = new ArrayBuffer(STATE_BYTES), f = new Float32Array(st), u = new Uint32Array(st), i = new Int32Array(st);
    f[0] = lambda; f[1] = T ?? this.T0; f[4] = p[0]; f[5] = p[1];
    beads.forEach((b, k) => {
      const o = (SYS_BYTES + k * BEAD_BYTES) / 4;
      f[o] = b.x[0]; f[o + 1] = b.x[1]; f[o + 2] = b.v[0]; f[o + 3] = b.v[1];
      i[o + 4] = b.wrap[0]; i[o + 5] = b.wrap[1]; f[o + 6] = b.a; f[o + 7] = b.m;
      u[o + 12] = 0;
    });
    q.writeBuffer(this.state[0], 0, st);
    q.writeBuffer(this.state[1], 0, st);
    this.beadInfo = beads.map(b => ({ a: b.a, m: b.m }));
    this.writeParams();
    const enc = this.device.createCommandEncoder();
    for (const g of this.grid) enc.clearBuffer(g);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.buildPipe);
    pass.setBindGroup(0, this.buildGroup);
    pass.dispatchWorkgroups(this.nPartials);
    pass.end();
    q.submit([enc.finish()]);
  }

  // Encode `count` steps into `enc` (call writeParams first if settings changed).
  encode(enc, count) {
    const pass = enc.beginComputePass();
    for (let k = 0; k < count; k++) {
      const g = this.s % 6;
      pass.setPipeline(this.molPipe);
      pass.setBindGroup(0, this.molGroups[g]);
      pass.dispatchWorkgroups(this.nPartials);
      pass.setPipeline(this.beadPipe);
      pass.setBindGroup(0, this.beadGroups[g]);
      pass.dispatchWorkgroups(this.nb + 1);
      this.s++;
    }
    pass.end();
    this.steps += count;
  }

  destroy() {
    for (const b of [this.params, ...this.pos, this.vel, ...this.grid, ...this.state, this.partials, this.ring]) b.destroy();
  }

  run(count) {
    const enc = this.device.createCommandEncoder();
    this.encode(enc, count);
    this.device.queue.submit([enc.finish()]);
  }

  // Buffers holding the current molecule positions and state.
  get posBuffer() { return this.pos[this.s % 2]; }
  get stateBuffer() { return this.state[this.s % 2]; }

  async read(buffer, size, offset = 0) {
    const rb = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, offset, rb, 0, size);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = rb.getMappedRange().slice(0);
    rb.unmap();
    rb.destroy();
    return out;
  }

  static parseState(ab, nb) {
    const f = new Float32Array(ab), u = new Uint32Array(ab), i = new Int32Array(ab);
    const beads = [];
    for (let k = 0; k < nb; k++) {
      const o = (SYS_BYTES + k * BEAD_BYTES) / 4;
      beads.push({
        x: [f[o], f[o + 1]], v: [f[o + 2], f[o + 3]], wrap: [i[o + 4], i[o + 5]], a: f[o + 6], m: f[o + 7],
        ext: [f[o + 8], f[o + 9]], f: [f[o + 10], f[o + 11]], step: u[o + 12],
      });
    }
    return { lambda: f[0], T: f[1], overflow: u[2], steps: u[3], p: [f[4], f[5]], beads };
  }

  async readState() { return Sim.parseState(await this.read(this.stateBuffer, STATE_BYTES), this.nb); }

  async readAll() {
    const [pos, vel, st] = await Promise.all([
      this.read(this.posBuffer, 8 * this.n), this.read(this.vel, 8 * this.n), this.read(this.stateBuffer, STATE_BYTES),
    ]);
    return { n: this.n, pos: new Float32Array(pos), vel: new Float32Array(vel), ...Sim.parseState(st, this.nb) };
  }

  // Records k (bead positions every recordEvery steps) for k in [from, to), oldest first.
  async readRecords(from, to) {
    const out = [];
    const nb = this.nb, size = REC_BYTES * nb;
    let k = from;
    while (k < to) {
      const slot = k % this.ringSize;
      const len = Math.min(to - k, this.ringSize - slot);
      const ab = await this.read(this.ring, len * size, slot * size);
      const f = new Float32Array(ab), i = new Int32Array(ab);
      for (let r = 0; r < len; r++) {
        const rec = [];
        for (let b = 0; b < nb; b++) {
          const o = (r * nb + b) * 8;
          rec.push({ x: [f[o], f[o + 1]], v: [f[o + 2], f[o + 3]], wrap: [i[o + 4], i[o + 5]], f: [f[o + 6], f[o + 7]] });
        }
        out.push(rec);
      }
      k += len;
    }
    return out;
  }
}
