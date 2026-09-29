// WebGPU inference for the hierarchical cloth graph network trained in train/mgn.py.
//
// One step() advances the cloth one frame (1/60 s) on the GPU:
//   1. find "world" edges: vertices within WORLD_R in space but far apart on the cloth
//   2. encode vertices (velocity, gripper, table, nearest obstacle, material, wind, and the nearest
//      vertices holding the cloth up: held by a gripper, or touching a ball or rod)
//      and edges on each level (rest offset, current offset, strain)
//   3. run the schedule of message-passing steps, each on one level of the grid
//      hierarchy (fine mesh, every 2nd row/column, every 4th)
//   4. decode accelerations, integrate, push vertices out of obstacles, apply grippers
//
// MLP kernels keep a tile of ROWS rows in workgroup memory and give each of the
// 128 threads one output column. Edge lists are CSR (grouped by receiver).

const ROWS = 16;
const L = 128;
const SLOTS = 3; // position ring: previous, current, next
const MAX_GRIPS = 15; // held vertices the node encoder looks through ([count, ...] fills 4 vec4u)
const MAX_GRIP_K = 4;

// ---------------------------------------------------------------- graph (mirrors mgn.graph_tables)

export function gridMesh(nx, ny) {
  const vid = (i, j) => i * ny + j;
  const tris = [];
  for (let i = 0; i < nx - 1; i++) {
    for (let j = 0; j < ny - 1; j++) {
      const a = vid(i, j), b = vid(i + 1, j), c = vid(i + 1, j + 1), d = vid(i, j + 1);
      if ((i + j) % 2 === 0) tris.push(a, b, c, a, c, d);
      else tris.push(a, b, d, b, c, d);
    }
  }
  const edges = new Set();
  for (let t = 0; t < tris.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const p = tris[t + k], q = tris[t + ((k + 1) % 3)];
      edges.add(p < q ? p * 1e6 + q : q * 1e6 + p);
    }
  }
  const edgeList = [...edges].map((e) => [Math.floor(e / 1e6), e % 1e6]);
  return { nx, ny, n: nx * ny, tris: new Uint32Array(tris), edges: edgeList };
}

function coarseAxis(n, s) {
  const idx = [];
  for (let i = 0; i < n; i += s) idx.push(i);
  if (idx[idx.length - 1] !== n - 1) idx.push(n - 1);
  return idx;
}

// Per level: node vertex ids, and CSR edges grouped by receiver row.
export function buildGraph(mesh, strides) {
  const { nx, ny, n } = mesh;
  const levels = [];
  const nb = Array.from({ length: n }, () => []);
  for (const [a, b] of mesh.edges) {
    nb[a].push(b);
    nb[b].push(a);
  }
  levels.push({ nodes: Array.from({ length: n }, (_, i) => i), nbrs: nb, stride: 1 });
  for (let l = 1; l < strides.length; l++) {
    const s = strides[l];
    const I = coarseAxis(nx, s), J = coarseAxis(ny, s);
    const nodes = [];
    const nbrs = [];
    for (let a = 0; a < I.length; a++) {
      for (let b = 0; b < J.length; b++) {
        nodes.push(I[a] * ny + J[b]);
        const list = [];
        for (let da = -1; da <= 1; da++) {
          for (let db = -1; db <= 1; db++) {
            if ((da || db) && a + da >= 0 && a + da < I.length && b + db >= 0 && b + db < J.length) {
              list.push(I[a + da] * ny + J[b + db]);
            }
          }
        }
        nbrs.push(list);
      }
    }
    levels.push({ nodes, nbrs, stride: s });
  }
  return levels.map((lv) => {
    const rowStart = new Uint32Array(lv.nodes.length + 1);
    const send = [];
    const recv = [];
    lv.nbrs.forEach((list, r) => {
      rowStart[r] = send.length;
      for (const s of list) {
        send.push(s);
        recv.push(r);
      }
    });
    rowStart[lv.nodes.length] = send.length;
    return {
      nodes: new Uint32Array(lv.nodes), rowStart, send: new Uint32Array(send), recv: new Uint32Array(recv),
      count: lv.nodes.length, edges: send.length, stride: lv.stride,
    };
  });
}

// ---------------------------------------------------------------- engine

export class ClothEngine {
  static async create(device, meta, weights) {
    const e = new ClothEngine();
    e.device = device;
    e.meta = meta;
    e.off = {};
    for (const t of meta.tensors) e.off[t.name] = t.offset;
    const cfg = meta.config;
    if (cfg.latent !== L || cfg.hidden !== L || cfg.mlp_layers !== 2) throw new Error('cloth engine expects latent = hidden = 128');
    e.schedule = cfg.schedule;
    if (!(meta.grip_k <= MAX_GRIP_K)) throw new Error('cloth engine expects a model with gripper features');
    e.nodeIn = 21 + 10 * meta.grip_k;
    e.W = device.createBuffer({ size: weights.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(e.W, 0, weights);
    return e;
  }

  // Build buffers and pipelines for a cloth of nx x ny vertices.
  async setCloth(nx, ny) {
    const d = this.device;
    const meta = this.meta;
    this.mesh = gridMesh(nx, ny);
    this.graph = buildGraph(this.mesh, meta.strides);
    const n = (this.n = this.mesh.n);
    this.head = 1;
    const S = GPUBufferUsage.STORAGE;
    const CD = GPUBufferUsage.COPY_DST;
    const CS = GPUBufferUsage.COPY_SRC;
    const mk = (bytes, usage = S) => d.createBuffer({ size: Math.max(16, bytes), usage });
    const upload = (arr, usage = S) => {
      const b = mk(arr.byteLength, usage | CD);
      d.queue.writeBuffer(b, 0, arr);
      return b;
    };
    const rest = new Float32Array(n * 2);
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      rest[2 * (i * ny + j)] = i * meta.spacing;
      rest[2 * (i * ny + j) + 1] = j * meta.spacing;
    }
    this.rest = rest;
    const WK = meta.world_k;
    const maxEw = n * WK;
    const maxE = Math.max(...this.graph.map((g) => g.edges));
    this.buf = {
      pos: mk(SLOTS * n * 16, S | CD | CS | GPUBufferUsage.VERTEX),
      kin: mk(n * 16, S | CD),
      held: mk(n * 4, S | CD),
      grips: mk((1 + MAX_GRIPS) * 4, GPUBufferUsage.UNIFORM | CD),
      flags: mk(n * 4),
      rest: upload(rest),
      scene: mk(64, GPUBufferUsage.UNIFORM | CD),
      obs: mk(96 * 4, S | CD),
      st: mk(16, S | CD | CS),
      wCnt: mk(n * 4),
      wNbr: mk(n * WK * 4),
      wRow: mk((n + 1) * 4),
      wSend: mk(maxEw * 4),
      wRecv: mk(maxEw * 4),
      wIndirect: mk(16, S | GPUBufferUsage.INDIRECT),
      v: mk(n * L * 4),
      ps: mk(n * L * 4),
      pr: mk(n * L * 4),
      psw: mk(n * L * 4),
      prw: mk(n * L * 4),
      enew: mk(maxE * L * 4),
      ew: mk(maxEw * L * 4),
      ewnew: mk(maxEw * L * 4),
      acc: mk(n * 16, S | CS),
      lv: this.graph.map((g) => ({
        nodes: upload(g.nodes), rowStart: upload(g.rowStart), send: upload(g.send), recv: upload(g.recv),
        e: mk(g.edges * L * 4),
      })),
    };
    d.queue.writeBuffer(this.buf.st, 0, new Uint32Array([this.head, 0, 0, 0]));
    this.setScene({ kb: 1e-5, mu: 0.5, wind: [0, 0, 0] });
    this.setObstacles({ spheres: [], capsules: [] });
    await this.buildPipelines();
  }

  // ---------------------------------------------------------------- WGSL helpers

  common() {
    const m = this.meta;
    const s = m.stats;
    const v3 = (a) => `vec3f(${a[0]}, ${a[1]}, ${a[2]})`;
    return /* wgsl */ `
const L = 128u;
const ROWS = ${ROWS}u;
const N = ${this.n}u;
const WK = ${m.world_k}u;
const WORLD_R = ${m.world_r};
const SELF_EXCL = ${m.self_exclude};
const D_CLIP = ${m.d_clip};
const OBS_OFFSET = ${m.obs_offset};
const SPACING = ${m.spacing};
const GRIP_K = ${m.grip_k}u;
const GRIP_SCALE = ${m.grip_scale};
const CONTACT_D = ${m.contact_d};
const FRAME_DT = ${m.frame_dt};
const VEL_MEAN = ${v3(s.vel_mean)};
const VEL_STD = ${v3(s.vel_std)};
const ACC_MEAN = ${v3(s.acc_mean)};
const ACC_STD = ${v3(s.acc_std)};
struct Scene { kbFeat: f32, mu: f32, pad0: f32, pad1: f32, wind: vec4f };
struct St { head: u32, EW: u32, pad0: u32, pad1: u32 };
fn prevSlot(h: u32) -> u32 { return (h + ${SLOTS - 1}u) % ${SLOTS}u; }
fn nextSlot(h: u32) -> u32 { return (h + 1u) % ${SLOTS}u; }
`;
  }

  // Obstacles: 2 spheres [c.xyz, r, v.xyz, on] then 2 capsules [a.xyz, r, b.xyz, on, va.xyz, 0, vb.xyz, 0];
  // the current state at offset 0, the next frame's poses at offset 48.
  obstacleWGSL() {
    return /* wgsl */ `
struct Hit { d: f32, n: vec3f, v: vec3f };
fn nearestObstacle(p: vec3f, base: u32) -> Hit {
  var best = Hit(1e9, vec3f(0.0), vec3f(0.0));
  for (var k = 0u; k < 2u; k++) {
    let o = base + k * 8u;
    if (obs[o + 7u] > 0.0) {
      let c = vec3f(obs[o], obs[o + 1u], obs[o + 2u]);
      let dv = p - c;
      let dist = sqrt(dot(dv, dv) + 1e-20);
      let dd = dist - obs[o + 3u] - OBS_OFFSET;
      if (dd < best.d) { best = Hit(dd, dv / dist, vec3f(obs[o + 4u], obs[o + 5u], obs[o + 6u])); }
    }
  }
  for (var k = 0u; k < 2u; k++) {
    let o = base + 16u + k * 16u;
    if (obs[o + 7u] > 0.0) {
      let a = vec3f(obs[o], obs[o + 1u], obs[o + 2u]);
      let b = vec3f(obs[o + 4u], obs[o + 5u], obs[o + 6u]);
      let ab = b - a;
      let t = clamp(dot(p - a, ab) / (dot(ab, ab) + 1e-12), 0.0, 1.0);
      let dq = p - (a + t * ab);
      let dist = sqrt(dot(dq, dq) + 1e-20);
      let dd = dist - obs[o + 3u] - OBS_OFFSET;
      let va = vec3f(obs[o + 8u], obs[o + 9u], obs[o + 10u]);
      let vb = vec3f(obs[o + 12u], obs[o + 13u], obs[o + 14u]);
      if (dd < best.d) { best = Hit(dd, dq / dist, va + t * (vb - va)); }
    }
  }
  return best;
}
`;
  }

  tiles() {
    return /* wgsl */ `
var<workgroup> ta: array<f32, ${ROWS * L}>;
var<workgroup> tb: array<f32, ${ROWS * L}>;
`;
  }

  // acc[r] += sum_i SRC[r][i] * W[woff + i*L + t] for i < dims (weights [dims, 128] row-major)
  matmul(src, woff, dims = L) {
    return `for (var i = 0u; i < ${dims}u; i++) {
    let w = W[${woff}u + i * L + t];
    for (var r = 0u; r < ROWS; r++) { acc[r] += ${src}[r * L + i] * w; }
  }`;
  }

  initAcc(boff) {
    return `var acc: array<f32, ${ROWS}>;
  { let b = W[${boff}u + t]; for (var r = 0u; r < ROWS; r++) { acc[r] = b; } }`;
  }

  dense(src, dst, prefix, layer, relu = true) {
    const o = this.off;
    return `{
  ${this.initAcc(o[`${prefix}.l${layer}.b`])}
  ${this.matmul(src, o[`${prefix}.l${layer}.w`])}
  workgroupBarrier();
  for (var r = 0u; r < ROWS; r++) { ${dst}[r * L + t] = ${relu ? 'max(acc[r], 0.0)' : 'acc[r]'}; }
  workgroupBarrier();
}`;
  }

  lnStats(src, scratch) {
    return `if (t < ROWS) {
    var m = 0.0;
    for (var i = 0u; i < L; i++) { m += ${src}[t * L + i]; }
    m /= 128.0;
    var v = 0.0;
    for (var i = 0u; i < L; i++) { let d = ${src}[t * L + i] - m; v += d * d; }
    ${scratch}[t * 2u] = m;
    ${scratch}[t * 2u + 1u] = inverseSqrt(v / 128.0 + 1e-5);
  }
  workgroupBarrier();`;
  }

  lnApply(prefix, r, src, scratch) {
    const o = this.off;
    return `((${src}[${r} * L + t] - ${scratch}[${r} * 2u]) * ${scratch}[${r} * 2u + 1u] * W[${o[`${prefix}.ln.g`]}u + t] + W[${o[`${prefix}.ln.b`]}u + t])`;
  }

  // Encoder MLP from a feature tile in `ta` (dims wide) to a LayerNormed latent; leaves result in tb, stats in ta.
  encoder(prefix, dims) {
    const o = this.off;
    return `{
    ${this.initAcc(o[`${prefix}.l0.b`])}
    ${this.matmul('ta', o[`${prefix}.l0.w`], dims)}
    workgroupBarrier();
    for (var r = 0u; r < ROWS; r++) { tb[r * L + t] = max(acc[r], 0.0); }
    workgroupBarrier();
  }
  ${this.dense('tb', 'ta', prefix, 1)}
  ${this.dense('ta', 'tb', prefix, 2, false)}
  ${this.lnStats('tb', 'ta')}`;
  }

  // ---------------------------------------------------------------- kernels

  srcWorldSearch() {
    return `${this.common()}
@group(0) @binding(0) var<storage, read> pos: array<vec4f>;
@group(0) @binding(1) var<storage, read> st: St;
@group(0) @binding(2) var<storage, read> rest: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> wCnt: array<u32>;
@group(0) @binding(4) var<storage, read_write> wNbr: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= N) { return; }
  let base = st.head * N;
  let p = pos[base + i].xyz;
  let ri = rest[i];
  var bd: array<f32, ${this.meta.world_k}>;
  var bj: array<u32, ${this.meta.world_k}>;
  var cnt = 0u;
  for (var j = 0u; j < N; j++) {
    let dv = pos[base + j].xyz - p;
    let d2 = dot(dv, dv);
    let dr = rest[j] - ri;
    if (d2 < WORLD_R * WORLD_R && dot(dr, dr) > SELF_EXCL * SELF_EXCL) {
      // insertion into the sorted list of the WK nearest
      var k = min(cnt, WK - 1u);
      if (cnt < WK || d2 < bd[WK - 1u]) {
        while (k > 0u && bd[k - 1u] > d2) {
          bd[k] = bd[k - 1u];
          bj[k] = bj[k - 1u];
          k--;
        }
        bd[k] = d2;
        bj[k] = j;
        cnt = min(cnt + 1u, WK);
      }
    }
  }
  wCnt[i] = cnt;
  for (var k = 0u; k < cnt; k++) { wNbr[i * WK + k] = bj[k]; }
}`;
  }

  srcWorldScan() {
    return `${this.common()}
@group(0) @binding(0) var<storage, read> wCnt: array<u32>;
@group(0) @binding(1) var<storage, read_write> wRow: array<u32>;
@group(0) @binding(2) var<storage, read_write> st: St;
@group(0) @binding(3) var<storage, read_write> ind: array<u32>;
var<workgroup> part: array<u32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) t: u32) {
  let per = (N + 255u) / 256u;
  let a = min(t * per, N);
  let b = min(a + per, N);
  var s = 0u;
  for (var i = a; i < b; i++) { s += wCnt[i]; }
  part[t] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off <<= 1u) {
    var v = 0u;
    if (t >= off) { v = part[t - off]; }
    workgroupBarrier();
    part[t] += v;
    workgroupBarrier();
  }
  var run = 0u;
  if (t > 0u) { run = part[t - 1u]; }
  for (var i = a; i < b; i++) { wRow[i] = run; run += wCnt[i]; }
  if (t == 255u) {
    wRow[N] = part[255];
    st.EW = part[255];
    ind[0] = max((part[255] + ROWS - 1u) / ROWS, 1u);
    ind[1] = 1u;
    ind[2] = 1u;
  }
}`;
  }

  srcWorldWrite() {
    return `${this.common()}
@group(0) @binding(0) var<storage, read> wCnt: array<u32>;
@group(0) @binding(1) var<storage, read> wNbr: array<u32>;
@group(0) @binding(2) var<storage, read> wRow: array<u32>;
@group(0) @binding(3) var<storage, read_write> wSend: array<u32>;
@group(0) @binding(4) var<storage, read_write> wRecv: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= N) { return; }
  for (var k = 0u; k < wCnt[i]; k++) {
    wSend[wRow[i] + k] = wNbr[i * WK + k];
    wRecv[wRow[i] + k] = i;
  }
}`;
  }

  // Per-vertex flags for the node encoder: held (bit 0), touching a ball or rod and not held (bit 1).
  srcFlags() {
    return `${this.common()}${this.obstacleWGSL()}
@group(0) @binding(0) var<storage, read> pos: array<vec4f>;
@group(0) @binding(1) var<storage, read> st: St;
@group(0) @binding(2) var<storage, read> held: array<u32>;
@group(0) @binding(3) var<storage, read> obs: array<f32>;
@group(0) @binding(4) var<storage, read_write> flags: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= N) { return; }
  let h = held[i] != 0u;
  let touching = !h && nearestObstacle(pos[st.head * N + i].xyz, 0u).d < CONTACT_D;
  flags[i] = select(0u, 1u, h) | select(0u, 2u, touching);
}`;
  }

  srcEncodeNodes() {
    return `${this.common()}${this.obstacleWGSL()}${this.tiles()}
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> st: St;
@group(0) @binding(3) var<storage, read> kin: array<vec4f>;
@group(0) @binding(4) var<storage, read> flags: array<u32>;  // bit 0: held, bit 1: touching a ball or rod
@group(0) @binding(5) var<uniform> scene: Scene;
@group(0) @binding(6) var<storage, read> obs: array<f32>;
@group(0) @binding(7) var<storage, read_write> V: array<f32>;
@group(0) @binding(8) var<storage, read> rest: array<vec2f>;
@group(0) @binding(9) var<uniform> grips: array<vec4u, ${(1 + MAX_GRIPS) / 4}>;  // [count, vertex...]
fn grip(k: u32) -> u32 { return grips[k / 4u][k % 4u]; }
// Nearest vertices in the rest shape, by key = squared grid distance * 1024 + vertex (ties to the lower index).
struct Near { keys: array<u32, ${MAX_GRIP_K}>, ids: array<u32, ${MAX_GRIP_K}> };
fn emptyNear() -> Near {
  var nr: Near;
  for (var s = 0u; s < GRIP_K; s++) { nr.keys[s] = 0xffffffffu; }
  return nr;
}
fn insertNear(nr: ptr<function, Near>, cell: vec2i, v: u32) {
  let dc = vec2i(round(rest[v] / SPACING)) - cell;
  var key = u32(dc.x * dc.x + dc.y * dc.y) * 1024u + v;
  var id = v;
  for (var s = 0u; s < GRIP_K; s++) {
    if (key < (*nr).keys[s]) {
      let tk = (*nr).keys[s]; let ti = (*nr).ids[s];
      (*nr).keys[s] = key; (*nr).ids[s] = id;
      key = tk; id = ti;
    }
  }
}
// flag, world offset / GRIP_SCALE, rest distance / GRIP_SCALE for each slot (zeros when empty)
fn writeNear(q0: u32, nr: Near, x0: vec3f, h: u32) {
  for (var s = 0u; s < GRIP_K; s++) {
    let q = q0 + 5u * s;
    if (nr.keys[s] != 0xffffffffu) {
      let off = (pos[h * N + nr.ids[s]].xyz - x0) / GRIP_SCALE;
      ta[q] = 1.0;
      ta[q + 1u] = off.x; ta[q + 2u] = off.y; ta[q + 3u] = off.z;
      ta[q + 4u] = sqrt(f32(nr.keys[s] / 1024u)) * SPACING / GRIP_SCALE;
    } else {
      for (var c = 0u; c < 5u; c++) { ta[q + c] = 0.0; }
    }
  }
}
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let base = wg.x * ROWS;
  // one thread per row computes the ${this.nodeIn} features
  if (t < ROWS) {
    let row = min(base + t, N - 1u);
    let h = st.head;
    let x0 = pos[h * N + row].xyz;
    let x1 = pos[prevSlot(h) * N + row].xyz;
    let isHeld = (flags[row] & 1u) != 0u;
    let o = t * L;
    let v = (x0 - x1 - VEL_MEAN) / VEL_STD;
    ta[o] = v.x; ta[o + 1u] = v.y; ta[o + 2u] = v.z;
    ta[o + 3u] = select(1.0, 0.0, isHeld);
    ta[o + 4u] = select(0.0, 1.0, isHeld);
    var k = vec3f(0.0);
    if (isHeld) { k = (kin[row].xyz - x0) / VEL_STD; }
    ta[o + 5u] = k.x; ta[o + 6u] = k.y; ta[o + 7u] = k.z;
    ta[o + 8u] = clamp((x0.y - OBS_OFFSET) / D_CLIP, 0.0, 1.0);
    let hit = nearestObstacle(x0, 0u);
    let near = hit.d < D_CLIP;
    ta[o + 9u] = clamp(hit.d / D_CLIP, -1.0, 1.0);
    let nn = select(vec3f(0.0), hit.n, near);
    let nv = select(vec3f(0.0), hit.v * FRAME_DT / VEL_STD, near);
    ta[o + 10u] = nn.x; ta[o + 11u] = nn.y; ta[o + 12u] = nn.z;
    ta[o + 13u] = nv.x; ta[o + 14u] = nv.y; ta[o + 15u] = nv.z;
    ta[o + 16u] = scene.kbFeat;
    ta[o + 17u] = scene.mu;
    ta[o + 18u] = scene.wind.x; ta[o + 19u] = scene.wind.y; ta[o + 20u] = scene.wind.z;
    // the GRIP_K held vertices, then the GRIP_K vertices touching a ball or rod, nearest in the rest shape
    let cell = vec2i(round(rest[row] / SPACING));
    var nh = emptyNear();
    for (var gi = 0u; gi < grip(0u); gi++) { insertNear(&nh, cell, grip(1u + gi)); }
    writeNear(o + 21u, nh, x0, h);
    var nc = emptyNear();
    for (var j = 0u; j < N; j++) { if ((flags[j] & 2u) != 0u) { insertNear(&nc, cell, j); } }
    writeNear(o + 21u + 5u * GRIP_K, nc, x0, h);
  }
  workgroupBarrier();
  ${this.encoder('enc_node', this.nodeIn)}
  for (var r = 0u; r < ROWS; r++) {
    let row = base + r;
    if (row < N) { V[row * L + t] = ${this.lnApply('enc_node', 'r', 'tb', 'ta')}; }
  }
}`;
  }

  // Edge encoder for level l (static CSR) or the world edges (dynamic CSR, indirect dispatch).
  srcEncodeEdges(l) {
    const world = l === 'world';
    const scale = world ? 'WORLD_R' : `${this.graph[l].stride}.0 * SPACING`;
    const count = world ? 'st.EW' : `${this.graph[l].edges}u`;
    return `${this.common()}${this.tiles()}
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> st: St;
@group(0) @binding(3) var<storage, read> send: array<u32>;
@group(0) @binding(4) var<storage, read> recv: array<u32>;
@group(0) @binding(5) var<storage, read_write> E: array<f32>;
${world ? '' : `@group(0) @binding(6) var<storage, read> rest: array<vec2f>;
@group(0) @binding(7) var<storage, read> nodes: array<u32>;`}
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let base = wg.x * ROWS;
  let nE = ${count};
  if (base >= nE) { return; }
  if (t < ROWS) {
    let row = min(base + t, nE - 1u);
    let s = send[row];
    let r = ${world ? 'recv[row]' : 'nodes[recv[row]]'};
    let h = st.head * N;
    let dx = (pos[h + s].xyz - pos[h + r].xyz) / (${scale});
    let lx = sqrt(dot(dx, dx) + 1e-12);
    let o = t * L;
    ${world ? `ta[o] = dx.x; ta[o + 1u] = dx.y; ta[o + 2u] = dx.z; ta[o + 3u] = lx;` : `
    let u = (rest[s] - rest[r]) / (${scale});
    let lu = sqrt(dot(u, u) + 1e-12);
    ta[o] = u.x; ta[o + 1u] = u.y; ta[o + 2u] = lu;
    ta[o + 3u] = dx.x; ta[o + 4u] = dx.y; ta[o + 5u] = dx.z; ta[o + 6u] = lx;
    ta[o + 7u] = (lx / lu - 1.0) * 10.0;`}
  }
  workgroupBarrier();
  ${this.encoder(world ? 'enc_world' : `enc_edge${l}`, world ? 4 : 8)}
  for (var r = 0u; r < ROWS; r++) {
    let row = base + r;
    if (row < nE) { E[row * L + t] = ${this.lnApply(world ? 'enc_world' : `enc_edge${l}`, 'r', 'tb', 'ta')}; }
  }
}`;
  }

  // Sender/receiver projections of the first edge layer for the nodes of step i's level.
  srcProj(i) {
    const lvl = this.schedule[i];
    const count = this.graph[lvl].count;
    const w = this.off[`proc${i}.edge.l0.w`];
    const ww = lvl === 0 ? this.off[`proc${i}.world.l0.w`] : 0;
    const part = (woff, out) => `{
    var acc: array<f32, ${ROWS}>;
    ${this.matmul('ta', woff)}
    for (var r = 0u; r < ROWS; r++) { if (base + r < ${count}u) { ${out}[nodes[base + r] * L + t] = acc[r]; } }
  }`;
    return `${this.common()}${this.tiles()}
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read> nodes: array<u32>;
@group(0) @binding(3) var<storage, read_write> PS: array<f32>;
@group(0) @binding(4) var<storage, read_write> PR: array<f32>;
${lvl === 0 ? `@group(0) @binding(5) var<storage, read_write> PSW: array<f32>;
@group(0) @binding(6) var<storage, read_write> PRW: array<f32>;` : ''}
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let base = wg.x * ROWS;
  for (var r = 0u; r < ROWS; r++) { ta[r * L + t] = V[nodes[min(base + r, ${count - 1}u)] * L + t]; }
  workgroupBarrier();
  ${part(w + L * L, 'PS')}
  ${part(w + 2 * L * L, 'PR')}
  ${lvl === 0 ? part(ww + L * L, 'PSW') : ''}
  ${lvl === 0 ? part(ww + 2 * L * L, 'PRW') : ''}
}`;
  }

  // Edge MLP for step i over one edge set: level edges (static) or world edges (dynamic).
  srcEdge(i, world) {
    const lvl = this.schedule[i];
    const p = `proc${i}.${world ? 'world' : 'edge'}`;
    const o = this.off;
    const count = world ? 'st.EW' : `${this.graph[lvl].edges}u`;
    return `${this.common()}${this.tiles()}
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> st: St;
@group(0) @binding(2) var<storage, read_write> E: array<f32>;
@group(0) @binding(3) var<storage, read_write> ENEW: array<f32>;
@group(0) @binding(4) var<storage, read> PS: array<f32>;
@group(0) @binding(5) var<storage, read> PR: array<f32>;
@group(0) @binding(6) var<storage, read> send: array<u32>;
@group(0) @binding(7) var<storage, read> recv: array<u32>;
${world ? '' : '@group(0) @binding(8) var<storage, read> nodes: array<u32>;'}
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let base = wg.x * ROWS;
  let nE = ${count};
  if (base >= nE) { return; }
  for (var r = 0u; r < ROWS; r++) { ta[r * L + t] = E[min(base + r, nE - 1u) * L + t]; }
  workgroupBarrier();
  {
    var acc: array<f32, ${ROWS}>;
    let b = W[${o[`${p}.l0.b`]}u + t];
    for (var r = 0u; r < ROWS; r++) {
      let row = min(base + r, nE - 1u);
      let rv = ${world ? 'recv[row]' : 'nodes[recv[row]]'};
      acc[r] = b + PS[send[row] * L + t] + PR[rv * L + t];
    }
    ${this.matmul('ta', o[`${p}.l0.w`])}
    workgroupBarrier();
    for (var r = 0u; r < ROWS; r++) { tb[r * L + t] = max(acc[r], 0.0); }
    workgroupBarrier();
  }
  ${this.dense('tb', 'ta', p, 1)}
  ${this.dense('ta', 'tb', p, 2, false)}
  ${this.lnStats('tb', 'ta')}
  for (var r = 0u; r < ROWS; r++) {
    let row = base + r;
    if (row < nE) {
      let val = ${this.lnApply(p, 'r', 'tb', 'ta')};
      ENEW[row * L + t] = val;
      E[row * L + t] += val;
    }
  }
}`;
  }

  srcNode(i) {
    const lvl = this.schedule[i];
    const p = `proc${i}.node`;
    const o = this.off;
    const count = this.graph[lvl].count;
    const w0 = o[`${p}.l0.w`];
    return `${this.common()}${this.tiles()}
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read_write> V: array<f32>;
@group(0) @binding(2) var<storage, read> nodes: array<u32>;
@group(0) @binding(3) var<storage, read> rowStart: array<u32>;
@group(0) @binding(4) var<storage, read> ENEW: array<f32>;
${lvl === 0 ? `@group(0) @binding(5) var<storage, read> wRow: array<u32>;
@group(0) @binding(6) var<storage, read> EWNEW: array<f32>;` : ''}
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let base = wg.x * ROWS;
  ${this.initAcc(o[`${p}.l0.b`])}
  // input = [v, sum of mesh messages${lvl === 0 ? ', sum of world messages' : ''}], one 128-wide chunk at a time
  for (var r = 0u; r < ROWS; r++) { ta[r * L + t] = V[nodes[min(base + r, ${count - 1}u)] * L + t]; }
  workgroupBarrier();
  ${this.matmul('ta', w0)}
  workgroupBarrier();
  for (var r = 0u; r < ROWS; r++) {
    let row = min(base + r, ${count - 1}u);
    var s = 0.0;
    for (var k = rowStart[row]; k < rowStart[row + 1u]; k++) { s += ENEW[k * L + t]; }
    ta[r * L + t] = s;
  }
  workgroupBarrier();
  ${this.matmul('ta', w0 + L * L)}
  workgroupBarrier();
  ${lvl === 0 ? `for (var r = 0u; r < ROWS; r++) {
    let row = min(base + r, ${count - 1}u);
    var s = 0.0;
    for (var k = wRow[row]; k < wRow[row + 1u]; k++) { s += EWNEW[k * L + t]; }
    ta[r * L + t] = s;
  }
  workgroupBarrier();
  ${this.matmul('ta', w0 + 2 * L * L)}
  workgroupBarrier();` : ''}
  for (var r = 0u; r < ROWS; r++) { ta[r * L + t] = max(acc[r], 0.0); }
  workgroupBarrier();
  ${this.dense('ta', 'tb', p, 1)}
  ${this.dense('tb', 'ta', p, 2, false)}
  ${this.lnStats('ta', 'tb')}
  for (var r = 0u; r < ROWS; r++) {
    if (base + r < ${count}u) {
      let node = nodes[base + r];
      V[node * L + t] += ${this.lnApply(p, 'r', 'ta', 'tb')};
    }
  }
}`;
  }

  srcDecode() {
    const o = this.off;
    return `${this.common()}${this.obstacleWGSL()}${this.tiles()}
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos: array<vec4f>;
@group(0) @binding(3) var<storage, read> st: St;
@group(0) @binding(4) var<storage, read> kin: array<vec4f>;
@group(0) @binding(5) var<storage, read> held: array<u32>;
@group(0) @binding(6) var<storage, read> obs: array<f32>;
@group(0) @binding(7) var<storage, read_write> accOut: array<vec4f>;
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let base = wg.x * ROWS;
  for (var r = 0u; r < ROWS; r++) { ta[r * L + t] = V[min(base + r, N - 1u) * L + t]; }
  workgroupBarrier();
  ${this.dense('ta', 'tb', 'dec', 0)}
  ${this.dense('tb', 'ta', 'dec', 1)}
  if (t < 3u) {
    for (var r = 0u; r < ROWS; r++) {
      var a = W[${o['dec.l2.b']}u + t];
      for (var i = 0u; i < L; i++) { a += ta[r * L + i] * W[${o['dec.l2.w']}u + i * 3u + t]; }
      tb[r * 4u + t] = a;
    }
  }
  workgroupBarrier();
  if (t < ROWS) {
    let row = base + t;
    if (row < N) {
      let an = vec3f(tb[t * 4u], tb[t * 4u + 1u], tb[t * 4u + 2u]);
      accOut[row] = vec4f(an, 0.0);
      let h = st.head;
      let x0 = pos[h * N + row].xyz;
      let x1 = pos[prevSlot(h) * N + row].xyz;
      let q = 2.0 * x0 - x1 + an * ACC_STD + ACC_MEAN;
      // push out of the table and of the obstacles' next poses (all pushes measured from q, as in training)
      var p = q;
      let dt = q.y - OBS_OFFSET;
      if (dt < 0.0) { p.y -= dt; }
      for (var k = 0u; k < 4u; k++) {
        let hit = nearestObstacleK(q, 48u, k);
        if (hit.d < 0.0) { p -= hit.d * hit.n; }
      }
      if (held[row] != 0u) { p = kin[row].xyz; }
      pos[nextSlot(h) * N + row] = vec4f(p, 1.0);
    }
  }
}

// The k-th obstacle (0,1 spheres, 2,3 capsules) as a Hit, for summing push-outs like the training code.
fn nearestObstacleK(p: vec3f, base: u32, k: u32) -> Hit {
  if (k < 2u) {
    let o = base + k * 8u;
    if (obs[o + 7u] <= 0.0) { return Hit(1e9, vec3f(0.0), vec3f(0.0)); }
    let c = vec3f(obs[o], obs[o + 1u], obs[o + 2u]);
    let dv = p - c;
    let dist = sqrt(dot(dv, dv) + 1e-20);
    return Hit(dist - obs[o + 3u] - OBS_OFFSET, dv / dist, vec3f(0.0));
  }
  let o = base + 16u + (k - 2u) * 16u;
  if (obs[o + 7u] <= 0.0) { return Hit(1e9, vec3f(0.0), vec3f(0.0)); }
  let a = vec3f(obs[o], obs[o + 1u], obs[o + 2u]);
  let b = vec3f(obs[o + 4u], obs[o + 5u], obs[o + 6u]);
  let ab = b - a;
  let tt = clamp(dot(p - a, ab) / (dot(ab, ab) + 1e-12), 0.0, 1.0);
  let dq = p - (a + tt * ab);
  let dist = sqrt(dot(dq, dq) + 1e-20);
  return Hit(dist - obs[o + 3u] - OBS_OFFSET, dq / dist, vec3f(0.0));
}`;
  }

  srcAdvance() {
    return `${this.common()}
@group(0) @binding(0) var<storage, read_write> st: St;
@compute @workgroup_size(1) fn main() { st.head = nextSlot(st.head); }`;
  }

  // ---------------------------------------------------------------- pipelines

  async buildPipelines() {
    const d = this.device;
    const b = this.buf;
    const make = async (label, code, entries) => {
      const module = d.createShaderModule({ code, label });
      const info = await module.getCompilationInfo();
      const errs = info.messages.filter((x) => x.type === 'error');
      if (errs.length) {
        console.error(label, errs.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join('\n'));
        throw new Error(`WGSL compile failed: ${label}`);
      }
      const pipeline = await d.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
      // 'auto' layouts drop bindings the shader never touches, so bind only the used ones.
      const used = (binding) => {
        const m = code.match(new RegExp(`@binding\\(${binding}\\) var(?:<[^>]*>)? (\\w+)`));
        return m && (code.match(new RegExp(`\\b${m[1]}\\b`, 'g')) || []).length > 1;
      };
      const bind = d.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })).filter((e) => used(e.binding)),
      });
      return { pipeline, bind };
    };
    const lv = b.lv;
    const jobs = [
      ['wsearch', this.srcWorldSearch(), [b.pos, b.st, b.rest, b.wCnt, b.wNbr]],
      ['wscan', this.srcWorldScan(), [b.wCnt, b.wRow, b.st, b.wIndirect]],
      ['wwrite', this.srcWorldWrite(), [b.wCnt, b.wNbr, b.wRow, b.wSend, b.wRecv]],
      ['flags', this.srcFlags(), [b.pos, b.st, b.held, b.obs, b.flags]],
      ['encNodes', this.srcEncodeNodes(), [this.W, b.pos, b.st, b.kin, b.flags, b.scene, b.obs, b.v, b.rest, b.grips]],
      ['encWorld', this.srcEncodeEdges('world'), [this.W, b.pos, b.st, b.wSend, b.wRecv, b.ew]],
      ['decode', this.srcDecode(), [this.W, b.v, b.pos, b.st, b.kin, b.held, b.obs, b.acc]],
      ['advance', this.srcAdvance(), [b.st]],
    ];
    this.graph.forEach((g, l) => {
      jobs.push([`enc${l}`, this.srcEncodeEdges(l), [this.W, b.pos, b.st, lv[l].send, lv[l].recv, lv[l].e, b.rest, lv[l].nodes]]);
    });
    this.schedule.forEach((lvl, i) => {
      const L0 = lvl === 0;
      jobs.push([`proj${i}`, this.srcProj(i), [this.W, b.v, lv[lvl].nodes, b.ps, b.pr, ...(L0 ? [b.psw, b.prw] : [])]]);
      jobs.push([`edge${i}`, this.srcEdge(i, false), [this.W, b.st, lv[lvl].e, b.enew, b.ps, b.pr, lv[lvl].send, lv[lvl].recv, lv[lvl].nodes]]);
      if (L0) jobs.push([`wedge${i}`, this.srcEdge(i, true), [this.W, b.st, b.ew, b.ewnew, b.psw, b.prw, b.wSend, b.wRecv]]);
      jobs.push([`node${i}`, this.srcNode(i), [this.W, b.v, lv[lvl].nodes, lv[lvl].rowStart, b.enew, ...(L0 ? [b.wRow, b.ewnew] : [])]]);
    });
    const built = await Promise.all(jobs.map(([label, code, entries]) => make(label, code, entries)));
    this.P = {};
    jobs.forEach(([label], k) => { this.P[label] = built[k]; });
  }

  // ---------------------------------------------------------------- state

  // x1, x0: Float32Array [n*3] previous/current positions.
  setState(x1, x0) {
    const n = this.n;
    const q = this.device.queue;
    const v4 = (x) => {
      const out = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        out[4 * i] = x[3 * i];
        out[4 * i + 1] = x[3 * i + 1];
        out[4 * i + 2] = x[3 * i + 2];
        out[4 * i + 3] = 1;
      }
      return out;
    };
    this.head = 1;
    q.writeBuffer(this.buf.pos, 0, v4(x1));
    q.writeBuffer(this.buf.pos, n * 16, v4(x0));
    q.writeBuffer(this.buf.st, 0, new Uint32Array([this.head, 0, 0, 0]));
  }

  // Grippers for the next step: list of {vertex, target:[x,y,z]}.
  setGrippers(grips) {
    const n = this.n;
    const kin = new Float32Array(n * 4);
    const held = new Uint32Array(n);
    for (const g of grips) {
      held[g.vertex] = 1;
      kin.set([g.target[0], g.target[1], g.target[2], 1], 4 * g.vertex);
    }
    const list = new Uint32Array(1 + MAX_GRIPS);
    for (const g of grips.slice(0, MAX_GRIPS)) list[1 + list[0]++] = g.vertex;
    this.device.queue.writeBuffer(this.buf.kin, 0, kin);
    this.device.queue.writeBuffer(this.buf.held, 0, held);
    this.device.queue.writeBuffer(this.buf.grips, 0, list);
  }

  setScene({ kb, mu, wind }) {
    const m = this.meta;
    const u = new Float32Array(8);
    u[0] = (Math.log10(kb) - m.kb_log_mid) / m.kb_log_half;
    u[1] = mu;
    u[4] = wind[0] / m.wind_scale;
    u[5] = wind[1] / m.wind_scale;
    u[6] = wind[2] / m.wind_scale;
    this.device.queue.writeBuffer(this.buf.scene, 0, u);
  }

  // cur/next: {spheres:[{c:[x,y,z], r, v:[..]}], capsules:[{a, b, r, va, vb}]}; next only needs poses.
  setObstacles(cur, next = cur) {
    const pack = (o, into, base) => {
      (o.spheres || []).slice(0, 2).forEach((s, k) => {
        into.set([...s.c, s.r, ...(s.v || [0, 0, 0]), 1], base + k * 8);
      });
      (o.capsules || []).slice(0, 2).forEach((c, k) => {
        into.set([...c.a, c.r, ...c.b, 1, ...(c.va || [0, 0, 0]), 0, ...(c.vb || [0, 0, 0]), 0], base + 16 + k * 16);
      });
    };
    const f = new Float32Array(96);
    pack(cur, f, 0);
    pack(next, f, 48);
    this.device.queue.writeBuffer(this.buf.obs, 0, f);
  }

  async readBuffer(buffer, bytes, offset = 0) {
    const d = this.device;
    const staging = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, offset, staging, 0, bytes);
    d.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return out;
  }

  // Current positions as Float32Array [n*4].
  async readPositions() {
    return new Float32Array(await this.readBuffer(this.buf.pos, this.n * 16, this.head * this.n * 16));
  }

  // ---------------------------------------------------------------- step

  encodeStep(enc) {
    const P = this.P;
    const pass = enc.beginComputePass();
    const run = (name, x) => {
      pass.setPipeline(P[name].pipeline);
      pass.setBindGroup(0, P[name].bind);
      pass.dispatchWorkgroups(x);
    };
    const runW = (name) => {
      pass.setPipeline(P[name].pipeline);
      pass.setBindGroup(0, P[name].bind);
      pass.dispatchWorkgroupsIndirect(this.buf.wIndirect, 0);
    };
    const n = this.n;
    const tiles = (rows) => Math.ceil(rows / ROWS);
    run('wsearch', Math.ceil(n / 64));
    run('wscan', 1);
    run('wwrite', Math.ceil(n / 64));
    run('flags', Math.ceil(n / 64));
    run('encNodes', tiles(n));
    runW('encWorld');
    this.graph.forEach((g, l) => run(`enc${l}`, tiles(g.edges)));
    this.schedule.forEach((lvl, i) => {
      const g = this.graph[lvl];
      run(`proj${i}`, tiles(g.count));
      run(`edge${i}`, tiles(g.edges));
      if (lvl === 0) runW(`wedge${i}`);
      run(`node${i}`, tiles(g.count));
    });
    run('decode', tiles(n));
    run('advance', 1);
    pass.end();
    this.head = (this.head + 1) % SLOTS;
  }

  step() {
    const enc = this.device.createCommandEncoder();
    this.encodeStep(enc);
    this.device.queue.submit([enc.finish()]);
  }
}
