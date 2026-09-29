// WebGPU renderer for the cloth sandbox: shaded two-sided cloth read straight from
// the engine's position ring, a gridded table, obstacles, grippers and a shadow map.
//
// Palette (traditional Japanese dye colors): 藍色 indigo cloth, 瓶覗 pale-indigo reverse,
// 生成り undyed-cotton table, 鉛色 lead-gray obstacles, 茜 madder grippers, 藍鉄 ink.

export const PALETTE = {
  cloth: [0x16, 0x5e, 0x83],
  back: [0xa2, 0xd7, 0xdd],
  table: [0xf6, 0xf3, 0xea],
  grid: [0xde, 0xd8, 0xc8],
  obstacle: [0x7b, 0x7c, 0x7d],
  gripper: [0xb7, 0x28, 0x2e],
  ink: [0x39, 0x3f, 0x4c],
};

const rgb = (c) => c.map((v) => v / 255);

// ---------------------------------------------------------------- tiny mat4 (column-major)

export const M4 = {
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  },
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    const o = new Float32Array(16);
    o[0] = f / aspect; o[5] = f; o[10] = far / (near - far); o[11] = -1; o[14] = (near * far) / (near - far);
    return o;
  },
  ortho(l, r, b, t, n, f) {
    const o = new Float32Array(16);
    o[0] = 2 / (r - l); o[5] = 2 / (t - b); o[10] = 1 / (n - f);
    o[12] = (l + r) / (l - r); o[13] = (t + b) / (b - t); o[14] = n / (n - f); o[15] = 1;
    return o;
  },
  lookAt(eye, target, up) {
    const z = norm(sub(eye, target));
    const x = norm(cross(up, z));
    const y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
  },
  invert(m) {
    const inv = new Float32Array(16);
    const a = m;
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
    const det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    for (let i = 0; i < 16; i++) inv[i] /= det;
    return inv;
  },
};
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// ---------------------------------------------------------------- meshes

function sphereMesh(seg = 24, rings = 16) {
  const pos = [];
  const idx = [];
  for (let r = 0; r <= rings; r++) {
    const th = (Math.PI * r) / rings;
    for (let s = 0; s <= seg; s++) {
      const ph = (2 * Math.PI * s) / seg;
      pos.push(Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph));
    }
  }
  for (let r = 0; r < rings; r++) for (let s = 0; s < seg; s++) {
    const a = r * (seg + 1) + s, b = a + seg + 1;
    idx.push(a, b, a + 1, a + 1, b, b + 1);
  }
  return { pos: new Float32Array(pos), idx: new Uint32Array(idx) };
}

// Unit capsule along +y from 0 to 1 with radius 1 caps; the vertex shader stretches the
// cylinder part and keeps the caps round. Vertex w = 0 (bottom half) or 1 (top half).
function capsuleMesh(seg = 20, rings = 8) {
  const pos = [];
  const idx = [];
  const rows = [];
  for (let r = 0; r <= rings; r++) rows.push([Math.PI / 2 + (Math.PI / 2) * (r / rings), 0]);
  for (let r = 0; r <= rings; r++) rows.push([(Math.PI / 2) * (1 - r / rings), 1]);
  rows.forEach(([th, top]) => {
    for (let s = 0; s <= seg; s++) {
      const ph = (2 * Math.PI * s) / seg;
      pos.push(Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph), top);
    }
  });
  for (let r = 0; r < rows.length - 1; r++) for (let s = 0; s < seg; s++) {
    const a = r * (seg + 1) + s, b = a + seg + 1;
    idx.push(a, a + 1, b, a + 1, b + 1, b);
  }
  return { pos: new Float32Array(pos), idx: new Uint32Array(idx) };
}

// ---------------------------------------------------------------- shaders

const COMMON = /* wgsl */ `
struct Cam { viewProj: mat4x4f, lightProj: mat4x4f, eye: vec4f, lightDir: vec4f, n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> cam: Cam;
fn shadowAt(sm: texture_depth_2d, samp: sampler_comparison, wp: vec3f, nrm: vec3f) -> f32 {
  let lp = cam.lightProj * vec4f(wp + nrm * 0.004, 1.0);
  let uv = vec2f(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
  let texel = 1.0 / 2048.0;
  var s = 0.0;
  for (var dx = -1; dx <= 1; dx++) {
    for (var dy = -1; dy <= 1; dy++) {
      s += textureSampleCompareLevel(sm, samp, uv + vec2f(f32(dx), f32(dy)) * texel, lp.z - 0.002);
    }
  }
  let inside = all(uv > vec2f(0.0)) && all(uv < vec2f(1.0));
  return select(1.0, s / 9.0, inside);
}
`;

const CLOTH_WGSL = /* wgsl */ `${COMMON}
struct St { head: u32, EW: u32, pad0: u32, pad1: u32 };
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> st: St;
@group(0) @binding(3) var<storage, read> nrmBuf: array<vec4f>;
@group(0) @binding(4) var<storage, read> uvBuf: array<vec2f>;
@group(1) @binding(0) var shadowMap: texture_depth_2d;
@group(1) @binding(1) var shadowSamp: sampler_comparison;
struct VOut { @builtin(position) p: vec4f, @location(0) wp: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var o: VOut;
  let w = pos[st.head * cam.n + vi].xyz;
  o.wp = w;
  o.n = nrmBuf[vi].xyz;
  o.uv = uvBuf[vi];
  o.p = cam.viewProj * vec4f(w, 1.0);
  return o;
}
@vertex fn vsShadow(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return cam.lightProj * vec4f(pos[st.head * cam.n + vi].xyz, 1.0);
}
@fragment fn fs(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  // grid triangles wind with their normals pointing down, so the side that starts facing up is the back face
  let top = !front;
  var n = normalize(in.n);
  if (!front) { n = -n; }
  var albedo = select(vec3f(${rgb(PALETTE.back).join(', ')}), vec3f(${rgb(PALETTE.cloth).join(', ')}), top);
  // sashiko: white running stitches on a 4 cm grid (front only); also shows how the cloth stretches
  let g = in.uv / 0.04;
  let fw = fwidth(g);
  let lx = 1.0 - smoothstep(0.02, 0.02 + fw.x, abs(fract(g.x + 0.5) - 0.5));
  let lz = 1.0 - smoothstep(0.02, 0.02 + fw.y, abs(fract(g.y + 0.5) - 0.5));
  let dashX = step(0.35, fract(g.y * 4.0));
  let dashZ = step(0.35, fract(g.x * 4.0));
  let stitch = max(lx * dashX, lz * dashZ) * select(0.0, 0.55, top);
  albedo = mix(albedo, vec3f(${rgb(PALETTE.table).join(', ')}), stitch);
  let weave = 1.0;
  let l = -cam.lightDir.xyz;
  let sh = shadowAt(shadowMap, shadowSamp, in.wp, n);
  let wrap = clamp((dot(n, l) + 0.35) / 1.35, 0.0, 1.0);
  let v = normalize(cam.eye.xyz - in.wp);
  let sheen = pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 3.0) * 0.18;
  let hemi = mix(vec3f(0.55, 0.52, 0.48), vec3f(0.78, 0.82, 0.88), n.y * 0.5 + 0.5);
  let col = albedo * weave * (hemi * 0.55 + vec3f(1.0, 0.97, 0.92) * wrap * sh * 0.75) + vec3f(sheen);
  return vec4f(col, 1.0);
}`;

const NORMALS_WGSL = /* wgsl */ `
struct St { head: u32, EW: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> pos: array<vec4f>;
@group(0) @binding(1) var<storage, read> st: St;
@group(0) @binding(2) var<storage, read> tris: array<u32>;
@group(0) @binding(3) var<storage, read> inc: array<i32>;
@group(0) @binding(4) var<storage, read_write> nrmBuf: array<vec4f>;
@group(0) @binding(5) var<uniform> nn: vec4u;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let n = nn.x;
  if (i >= n) { return; }
  let base = st.head * n;
  var acc = vec3f(0.0);
  for (var k = 0u; k < 8u; k++) {
    let t = inc[i * 8u + k];
    if (t < 0) { break; }
    let a = pos[base + tris[u32(t) * 3u]].xyz;
    let b = pos[base + tris[u32(t) * 3u + 1u]].xyz;
    let c = pos[base + tris[u32(t) * 3u + 2u]].xyz;
    acc += cross(b - a, c - a);
  }
  nrmBuf[i] = vec4f(normalize(acc + vec3f(0.0, 1e-9, 0.0)), 0.0);
}`;

const TABLE_WGSL = /* wgsl */ `${COMMON}
@group(1) @binding(0) var shadowMap: texture_depth_2d;
@group(1) @binding(1) var shadowSamp: sampler_comparison;
struct VOut { @builtin(position) p: vec4f, @location(0) wp: vec3f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var c = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));
  let q = c[vi] * 1.6;
  var o: VOut;
  o.wp = vec3f(q.x, 0.0, q.y);
  o.p = cam.viewProj * vec4f(o.wp, 1.0);
  return o;
}
fn gridLine(x: f32, step: f32, width: f32) -> f32 {
  let d = abs(fract(x / step + 0.5) - 0.5) * step;
  let aa = fwidth(x) * 0.8;
  return 1.0 - smoothstep(width, width + aa, d);
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let fine = max(gridLine(in.wp.x, 0.05, 0.0006), gridLine(in.wp.z, 0.05, 0.0006));
  let major = max(gridLine(in.wp.x, 0.25, 0.0012), gridLine(in.wp.z, 0.25, 0.0012));
  var col = mix(vec3f(${rgb(PALETTE.table).join(', ')}), vec3f(${rgb(PALETTE.grid).join(', ')}), max(fine * 0.55, major));
  let sh = shadowAt(shadowMap, shadowSamp, in.wp, vec3f(0.0, 1.0, 0.0));
  col *= 0.72 + 0.28 * sh;
  // fade the table into the page towards its edges
  let r = length(in.wp.xz);
  let fade = smoothstep(1.55, 0.85, r);
  return vec4f(mix(vec3f(${rgb(PALETTE.table).join(', ')}) * 1.01, col, fade), 1.0);
}`;

const SOLID_WGSL = /* wgsl */ `${COMMON}
struct Inst { a: vec4f, b: vec4f, color: vec4f };  // a.xyz, a.w = radius; b.xyz (capsule end), b.w = kind (0 sphere, 1 capsule)
@group(0) @binding(1) var<storage, read> inst: array<Inst>;
@group(1) @binding(0) var shadowMap: texture_depth_2d;
@group(1) @binding(1) var shadowSamp: sampler_comparison;
struct VOut { @builtin(position) p: vec4f, @location(0) wp: vec3f, @location(1) n: vec3f, @location(2) color: vec3f };
fn place(v: vec4f, ii: u32) -> array<vec3f, 2> {
  let I = inst[ii];
  let r = I.a.w;
  if (I.b.w < 0.5) {
    return array<vec3f, 2>(I.a.xyz + v.xyz * r, v.xyz);
  }
  let axis = I.b.xyz - I.a.xyz;
  let len = length(axis);
  let y = axis / max(len, 1e-6);
  var x = normalize(cross(y, vec3f(0.0, 0.0, 1.0)));
  if (abs(y.z) > 0.9) { x = normalize(cross(y, vec3f(1.0, 0.0, 0.0))); }
  let z = cross(x, y);
  let local = x * v.x + y * v.y + z * v.z;
  let base = select(I.a.xyz, I.b.xyz, v.w > 0.5);
  return array<vec3f, 2>(base + local * r, local);
}
@vertex fn vs(@location(0) v: vec4f, @builtin(instance_index) ii: u32) -> VOut {
  let pl = place(v, ii);
  var o: VOut;
  o.wp = pl[0];
  o.n = pl[1];
  o.color = inst[ii].color.rgb;
  o.p = cam.viewProj * vec4f(pl[0], 1.0);
  return o;
}
@vertex fn vsShadow(@location(0) v: vec4f, @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  return cam.lightProj * vec4f(place(v, ii)[0], 1.0);
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let n = normalize(in.n);
  let l = -cam.lightDir.xyz;
  let sh = shadowAt(shadowMap, shadowSamp, in.wp, n);
  let v = normalize(cam.eye.xyz - in.wp);
  let h = normalize(l + v);
  let spec = pow(max(dot(n, h), 0.0), 40.0) * 0.25 * sh;
  let hemi = mix(vec3f(0.5, 0.48, 0.45), vec3f(0.8, 0.83, 0.88), n.y * 0.5 + 0.5);
  let col = in.color * (hemi * 0.5 + vec3f(1.0, 0.97, 0.92) * max(dot(n, l), 0.0) * sh * 0.7) + vec3f(spec);
  return vec4f(col, 1.0);
}`;

// ---------------------------------------------------------------- renderer

export class ClothRenderer {
  constructor(device, canvas) {
    this.device = device;
    this.canvas = canvas;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx = canvas.getContext('webgpu');
    this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.sampleCount = 4;
    this.camBuf = device.createBuffer({ size: 176, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.shadowTex = device.createTexture({ size: [2048, 2048], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.shadowSamp = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    this.instBuf = device.createBuffer({ size: 48 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.instances = 0;
    const sph = sphereMesh();
    const sph4 = new Float32Array((sph.pos.length / 3) * 4);
    for (let i = 0; i < sph.pos.length / 3; i++) sph4.set([sph.pos[3 * i], sph.pos[3 * i + 1], sph.pos[3 * i + 2], 0], 4 * i);
    this.sphere = this.meshBuffers(sph4, sph.idx);
    const cap = capsuleMesh();
    this.capsule = this.meshBuffers(cap.pos, cap.idx);
    this.buildPipelines();
  }

  meshBuffers(pos, idx) {
    const d = this.device;
    const vb = d.createBuffer({ size: pos.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(vb, 0, pos);
    const ib = d.createBuffer({ size: idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(ib, 0, idx);
    return { vb, ib, count: idx.length };
  }

  buildPipelines() {
    const d = this.device;
    const depth = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };
    const ms = { count: this.sampleCount };
    const cloth = d.createShaderModule({ code: CLOTH_WGSL });
    const table = d.createShaderModule({ code: TABLE_WGSL });
    const solid = d.createShaderModule({ code: SOLID_WGSL });
    this.normalsPipe = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: NORMALS_WGSL }), entryPoint: 'main' } });
    const target = [{ format: this.format }];
    this.clothPipe = d.createRenderPipeline({
      layout: 'auto', vertex: { module: cloth, entryPoint: 'vs' }, fragment: { module: cloth, entryPoint: 'fs', targets: target },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample: ms,
    });
    this.clothShadowPipe = d.createRenderPipeline({
      layout: 'auto', vertex: { module: cloth, entryPoint: 'vsShadow' },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.tablePipe = d.createRenderPipeline({
      layout: 'auto', vertex: { module: table, entryPoint: 'vs' }, fragment: { module: table, entryPoint: 'fs', targets: target },
      primitive: { topology: 'triangle-list' }, depthStencil: depth, multisample: ms,
    });
    const vbuf = [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }];
    this.solidPipe = d.createRenderPipeline({
      layout: 'auto', vertex: { module: solid, entryPoint: 'vs', buffers: vbuf }, fragment: { module: solid, entryPoint: 'fs', targets: target },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample: ms,
    });
    this.solidShadowPipe = d.createRenderPipeline({
      layout: 'auto', vertex: { module: solid, entryPoint: 'vsShadow', buffers: vbuf },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    const shadowGroup = (pipe) => d.createBindGroup({
      layout: pipe.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: this.shadowTex.createView() }, { binding: 1, resource: this.shadowSamp }],
    });
    this.tableBind0 = d.createBindGroup({ layout: this.tablePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.camBuf } }] });
    this.tableBind1 = shadowGroup(this.tablePipe);
    this.solidBind0 = d.createBindGroup({ layout: this.solidPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.camBuf } }, { binding: 1, resource: { buffer: this.instBuf } }] });
    this.solidBind1 = shadowGroup(this.solidPipe);
    this.solidShadowBind = d.createBindGroup({ layout: this.solidShadowPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.camBuf } }, { binding: 1, resource: { buffer: this.instBuf } }] });
    this.clothBind1 = shadowGroup(this.clothPipe);
  }

  // Bind to an engine's cloth (call after engine.setCloth).
  setCloth(engine) {
    const d = this.device;
    const { mesh } = engine;
    const n = mesh.n;
    this.n = n;
    this.triCount = mesh.tris.length;
    this.indexBuf = d.createBuffer({ size: mesh.tris.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.indexBuf, 0, mesh.tris);
    const inc = new Int32Array(n * 8).fill(-1);
    const cnt = new Uint8Array(n);
    for (let t = 0; t < mesh.tris.length / 3; t++) for (let k = 0; k < 3; k++) {
      const v = mesh.tris[3 * t + k];
      if (cnt[v] < 8) inc[v * 8 + cnt[v]++] = t;
    }
    const incBuf = d.createBuffer({ size: inc.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(incBuf, 0, inc);
    const uv = new Float32Array(n * 2);
    for (let i = 0; i < mesh.nx; i++) for (let j = 0; j < mesh.ny; j++) {
      uv[2 * (i * mesh.ny + j)] = i / (mesh.nx - 1) * (mesh.nx - 1) * 0.02;
      uv[2 * (i * mesh.ny + j) + 1] = j * 0.02;
    }
    const uvBuf = d.createBuffer({ size: uv.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(uvBuf, 0, uv);
    this.nrmBuf = d.createBuffer({ size: n * 16, usage: GPUBufferUsage.STORAGE });
    const nn = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(nn, 0, new Uint32Array([n, 0, 0, 0]));
    this.normalsBind = d.createBindGroup({ layout: this.normalsPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: engine.buf.pos } }, { binding: 1, resource: { buffer: engine.buf.st } },
      { binding: 2, resource: { buffer: this.indexBuf } }, { binding: 3, resource: { buffer: incBuf } },
      { binding: 4, resource: { buffer: this.nrmBuf } }, { binding: 5, resource: { buffer: nn } }] });
    this.clothBind0 = d.createBindGroup({ layout: this.clothPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.camBuf } }, { binding: 1, resource: { buffer: engine.buf.pos } },
      { binding: 2, resource: { buffer: engine.buf.st } }, { binding: 3, resource: { buffer: this.nrmBuf } },
      { binding: 4, resource: { buffer: uvBuf } }] });
    this.clothShadowBind = d.createBindGroup({ layout: this.clothShadowPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.camBuf } }, { binding: 1, resource: { buffer: engine.buf.pos } },
      { binding: 2, resource: { buffer: engine.buf.st } }] });
  }

  // solids: [{kind:'sphere'|'capsule', a:[..], b:[..], r, color:[r,g,b] 0-255}]
  setSolids(solids) {
    const f = new Float32Array(Math.max(1, solids.length) * 12);
    solids.slice(0, 16).forEach((s, k) => {
      const c = rgb(s.color);
      f.set([...s.a, s.r, ...(s.b || s.a), s.kind === 'capsule' ? 1 : 0, c[0], c[1], c[2], 1], k * 12);
    });
    this.device.queue.writeBuffer(this.instBuf, 0, f);
    this.solids = solids.slice(0, 16);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h || !this.depthTex) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.depthTex?.destroy();
      this.msTex?.destroy();
      this.depthTex = this.device.createTexture({ size: [w, h], format: 'depth24plus', sampleCount: this.sampleCount, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      this.msTex = this.device.createTexture({ size: [w, h], format: this.format, sampleCount: this.sampleCount, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    return w / h;
  }

  // camera: {eye, target, fov}
  render(camera) {
    const aspect = this.resize();
    const view = M4.lookAt(camera.eye, camera.target, [0, 1, 0]);
    const proj = M4.perspective(camera.fov, aspect, 0.02, 20);
    this.viewProj = M4.mul(proj, view);
    const lightDir = norm([-0.35, -1, -0.45]);
    const lview = M4.lookAt(scale(lightDir, -2), [0, 0, 0], [0, 0, 1]);
    const lproj = M4.ortho(-1.1, 1.1, -1.1, 1.1, 0.1, 4);
    const u = new Float32Array(44);
    u.set(this.viewProj, 0);
    u.set(M4.mul(lproj, lview), 16);
    u.set([...camera.eye, 1], 32);
    u.set([...lightDir, 0], 36);
    new Uint32Array(u.buffer)[40] = this.n || 0;
    this.device.queue.writeBuffer(this.camBuf, 0, u);

    const enc = this.device.createCommandEncoder();
    if (this.n) {
      const cp = enc.beginComputePass();
      cp.setPipeline(this.normalsPipe);
      cp.setBindGroup(0, this.normalsBind);
      cp.dispatchWorkgroups(Math.ceil(this.n / 64));
      cp.end();
    }
    const sp = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: {
      view: this.shadowTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    if (this.n) {
      sp.setPipeline(this.clothShadowPipe);
      sp.setBindGroup(0, this.clothShadowBind);
      sp.setIndexBuffer(this.indexBuf, 'uint32');
      sp.drawIndexed(this.triCount);
    }
    this.drawSolids(sp, true);
    sp.end();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.msTex.createView(), resolveTarget: this.ctx.getCurrentTexture().createView(),
        clearValue: { r: PALETTE.table[0] / 255, g: PALETTE.table[1] / 255, b: PALETTE.table[2] / 255, a: 1 }, loadOp: 'clear', storeOp: 'discard' }],
      depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    pass.setPipeline(this.tablePipe);
    pass.setBindGroup(0, this.tableBind0);
    pass.setBindGroup(1, this.tableBind1);
    pass.draw(6);
    if (this.n) {
      pass.setPipeline(this.clothPipe);
      pass.setBindGroup(0, this.clothBind0);
      pass.setBindGroup(1, this.clothBind1);
      pass.setIndexBuffer(this.indexBuf, 'uint32');
      pass.drawIndexed(this.triCount);
    }
    this.drawSolids(pass, false);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  drawSolids(pass, shadow) {
    const solids = this.solids || [];
    if (!solids.length) return;
    pass.setPipeline(shadow ? this.solidShadowPipe : this.solidPipe);
    pass.setBindGroup(0, shadow ? this.solidShadowBind : this.solidBind0);
    if (!shadow) pass.setBindGroup(1, this.solidBind1);
    solids.forEach((s, k) => {
      const m = s.kind === 'capsule' ? this.capsule : this.sphere;
      pass.setVertexBuffer(0, m.vb);
      pass.setIndexBuffer(m.ib, 'uint32');
      pass.drawIndexed(m.count, 1, 0, 0, k);
    });
  }

  // Ray through a canvas pixel (CSS px) -> {origin, dir}
  ray(px, py) {
    const r = this.canvas.getBoundingClientRect();
    const x = ((px - r.left) / r.width) * 2 - 1;
    const y = 1 - ((py - r.top) / r.height) * 2;
    const inv = M4.invert(this.viewProj);
    const unproject = (z) => {
      const v = [x, y, z, 1];
      const o = [0, 0, 0, 0];
      for (let rr = 0; rr < 4; rr++) for (let k = 0; k < 4; k++) o[rr] += inv[k * 4 + rr] * v[k];
      return [o[0] / o[3], o[1] / o[3], o[2] / o[3]];
    };
    const a = unproject(0);
    const b = unproject(1);
    return { origin: a, dir: norm(sub(b, a)) };
  }
}
