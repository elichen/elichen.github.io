// Gemma 3 270M on WebGPU: int8 weights (one fp32 scale per row), fp32 activations (the residual stream reaches
// ~15,000, past what fp16 holds). Rows (sequences) run as one batch so a steered and an unsteered continuation
// cost about as much as one. A sequence is capped at 512 tokens, so the sliding-window layers never slide.
// Every shader loop has a fixed upper bound: a runaway dispatch freezes the whole Mac, not just the tab.

const NMAX = 512;        // tokens per forward, all rows together
const BMAX = 2;          // rows
const WG = 256;

const SHADERS = {
  embed: /* wgsl */ `
@group(0) @binding(0) var<uniform> P: vec4<u32>;            // N, scale bits
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> e0: array<u32>;
@group(0) @binding(3) var<storage, read> e1: array<u32>;
@group(0) @binding(4) var<storage, read> e2: array<u32>;
@group(0) @binding(5) var<storage, read> e3: array<u32>;
@group(0) @binding(6) var<storage, read> sc: array<f32>;
@group(0) @binding(7) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = wid.x;
  if (n >= P.x) { return; }
  let id = min(ids[n], 262143u);
  let c = id >> 16u;
  let r = id & 0xffffu;
  let s = sc[id] * bitcast<f32>(P.y);
  for (var w = li; w < 160u; w += 64u) {
    var q: u32;
    if (c == 0u) { q = e0[r * 160u + w]; } else if (c == 1u) { q = e1[r * 160u + w]; }
    else if (c == 2u) { q = e2[r * 160u + w]; } else { q = e3[r * 160u + w]; }
    for (var b = 0u; b < 4u; b++) {
      y[n * 640u + w * 4u + b] = f32(extractBits(i32(q), b * 8u, 8u)) * s;
    }
  }
}`,

  // y = x W^T * scale, W int8 [O][K] packed 4 per u32; 8 output rows per workgroup, 32 lanes per row,
  // 8 tokens at a time
  matmul: /* wgsl */ `
struct Params { N: u32, K4: u32, O: u32, yStride: u32, yOff: u32, sOff: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> part: array<f32, 2048>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = li / 32u;
  let lane = li % 32u;
  let row = wid.x * 8u + r;
  let valid = row < P.O;
  let N = min(P.N, ${NMAX}u);
  let K4 = min(P.K4, 512u);
  for (var t0 = 0u; t0 < N; t0 += 8u) {
    var acc = array<f32, 8>(0., 0., 0., 0., 0., 0., 0., 0.);
    if (valid) {
      for (var w = lane; w < K4; w += 32u) {
        let q = i32(W[row * K4 + w]);
        let wv = vec4<f32>(f32(extractBits(q, 0u, 8u)), f32(extractBits(q, 8u, 8u)),
                           f32(extractBits(q, 16u, 8u)), f32(extractBits(q, 24u, 8u)));
        for (var t = 0u; t < 8u; t++) {
          if (t0 + t < N) { acc[t] += dot(wv, X[(t0 + t) * K4 + w]); }
        }
      }
    }
    for (var t = 0u; t < 8u; t++) { part[li * 8u + t] = acc[t]; }
    workgroupBarrier();
    if (valid && lane < 8u && t0 + lane < N) {
      var s = 0.0;
      for (var l = 0u; l < 32u; l++) { s += part[(r * 32u + l) * 8u + lane]; }
      Y[P.yOff + (t0 + lane) * P.yStride + row] = s * S[P.sOff + row];
    }
    workgroupBarrier();
  }
}`,

  // per vector: y = x / rms(x) * w, or y += that (add = 1)
  rmsnorm: /* wgsl */ `
@group(0) @binding(0) var<uniform> P: vec4<u32>;            // nvec, dim, add, eps bits
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let v = wid.x;
  let dim = min(P.y, 2048u);
  let base = v * dim;
  var s = 0.0;
  for (var i = li; i < dim; i += 64u) { let a = x[base + i]; s += a * a; }
  red[li] = s;
  workgroupBarrier();
  for (var h = 32u; h > 0u; h >>= 1u) {
    if (li < h) { red[li] += red[li + h]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(dim) + bitcast<f32>(P.w));
  if (v >= P.x) { return; }
  for (var i = li; i < dim; i += 64u) {
    let o = x[base + i] * inv * w[i];
    if (P.z == 1u) { y[base + i] += o; } else { y[base + i] = o; }
  }
}`,

  // q/k norms and rotary embedding; q rotated in place, k and v written into this layer's cache
  qkrope: /* wgsl */ `
@group(0) @binding(0) var<uniform> P: vec4<u32>;            // N, eps bits
@group(0) @binding(1) var<storage, read> tok: array<u32>;  // row, pos per token
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read> kv: array<f32>;    // k [N][256] then v [N][256] (one buffer)
@group(0) @binding(4) var<storage, read> nw: array<f32>;    // q norm [256], k norm [256]
@group(0) @binding(5) var<storage, read> rope: array<f32>;  // [pos][cos 128, sin 128]
@group(0) @binding(6) var<storage, read_write> Kc: array<f32>;
@group(0) @binding(7) var<storage, read_write> Vc: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let n = wid.x;
  let h = wid.y;                                            // 0..3 query heads, 4 = the key head
  let row = min(tok[2u * n], ${BMAX - 1}u);
  let pos = min(tok[2u * n + 1u], 511u);
  var a: f32; var b: f32;
  if (h < 4u) { a = q[n * 1024u + h * 256u + i]; b = q[n * 1024u + h * 256u + i + 128u]; }
  else { a = kv[n * 256u + i]; b = kv[n * 256u + i + 128u]; }
  red[i] = a * a + b * b;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) {
    if (i < s) { red[i] += red[i + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / 256.0 + bitcast<f32>(P.y));
  let wo = select(0u, 256u, h == 4u);
  let x1 = a * inv * nw[wo + i];
  let x2 = b * inv * nw[wo + i + 128u];
  let c = rope[pos * 256u + i];
  let sn = rope[pos * 256u + 128u + i];
  let o1 = x1 * c - x2 * sn;
  let o2 = x2 * c + x1 * sn;
  if (h < 4u) {
    q[n * 1024u + h * 256u + i] = o1;
    q[n * 1024u + h * 256u + i + 128u] = o2;
  } else {
    let base = (row * 512u + pos) * 256u;
    Kc[base + i] = o1;
    Kc[base + i + 128u] = o2;
    Vc[base + i] = kv[P.x * 256u + n * 256u + i];
    Vc[base + i + 128u] = kv[P.x * 256u + n * 256u + i + 128u];
  }
}`,

  // one workgroup per (token, head): scores for keys 0..pos of the token's row, softmax, weighted values
  attention: /* wgsl */ `
@group(0) @binding(0) var<uniform> P: vec4<u32>;            // N
@group(0) @binding(1) var<storage, read> tok: array<u32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> Kc: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> Vc: array<f32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
var<workgroup> qs: array<vec4<f32>, 64>;
var<workgroup> sc: array<f32, 512>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let n = wid.x;
  let h = wid.y;
  if (n >= P.x) { return; }
  let row = min(tok[2u * n], ${BMAX - 1}u);
  let nk = min(tok[2u * n + 1u], 511u) + 1u;
  if (i < 64u) {
    let b = n * 1024u + h * 256u + i * 4u;
    qs[i] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]) * 0.0625;     // query_pre_attn_scalar 256 ** -0.5
  }
  workgroupBarrier();
  var m = -3.0e38;
  for (var j = i; j < nk; j += 256u) {
    let kb = (row * 512u + j) * 64u;
    var s = 0.0;
    for (var d = 0u; d < 64u; d++) { s += dot(qs[d], Kc[kb + d]); }
    sc[j] = s;
    m = max(m, s);
  }
  red[i] = m;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (i < s) { red[i] = max(red[i], red[i + s]); }
    workgroupBarrier();
  }
  let mx = red[0];
  workgroupBarrier();
  var z = 0.0;
  for (var j = i; j < nk; j += 256u) { let e = exp(sc[j] - mx); sc[j] = e; z += e; }
  red[i] = z;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (i < s) { red[i] += red[i + s]; }
    workgroupBarrier();
  }
  let zs = red[0];
  var acc = 0.0;
  for (var j = 0u; j < nk; j++) { acc += sc[j] * Vc[(row * 512u + j) * 256u + i]; }
  o[n * 1024u + h * 256u + i] = acc / zs;
}`,

  // g = gelu_tanh(g) * u
  gelu: /* wgsl */ `
@group(0) @binding(0) var<uniform> P: vec4<u32>;            // count
@group(0) @binding(1) var<storage, read_write> g: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= P.x) { return; }
  let x = g[i];
  let t = tanh(clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -15.0, 15.0));
  g[i] = 0.5 * x * (1.0 + t) * u[i];
}`,

  // x += steering vector of the token's row, except at <bos> (position 0)
  steer: /* wgsl */ `
@group(0) @binding(0) var<uniform> P: vec4<u32>;            // N
@group(0) @binding(1) var<storage, read> tok: array<u32>;
@group(0) @binding(2) var<storage, read> vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = wid.x;
  if (n >= P.x || tok[2u * n + 1u] == 0u) { return; }
  let row = min(tok[2u * n], ${BMAX - 1}u);
  for (var i = li; i < 640u; i += 64u) { x[n * 640u + i] += vec[row * 640u + i]; }
}`,
};

const f32bits = v => new Uint32Array(new Float32Array([v]).buffer)[0];

export class Gemma {
  // manifest: model/manifest.json; files: name -> ArrayBuffer
  static async create(device, manifest, files) {
    const g = new Gemma();
    await g.init(device, manifest, files);
    return g;
  }

  async init(device, man, files) {
    this.device = device;
    this.man = man;
    this.L = man.layer;
    this.bufId = 0;
    const tensor = name => {
      const t = man.tensors[name];
      return new Uint8Array(files[t.file], t.offset, t.bytes);
    };
    this.tensorF32 = name => {
      const t = man.tensors[name];
      return new Float32Array(files[t.file], t.offset, t.bytes / 4);
    };
    const upload = (bytes, label) => {
      const b = this.buffer(bytes.byteLength, label);
      device.queue.writeBuffer(b, 0, bytes);
      return b;
    };
    this.embed = [0, 1, 2, 3].map(c => upload(tensor(`embed.${c}`), `embed${c}`));
    this.embedScales = upload(tensor('embed.scales'), 'embed.scales');
    const LIN = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'];
    this.layers = [];
    for (let i = 0; i < man.layers; i++) {
      const l = { type: man.layerTypes[i] };
      for (const n of LIN) {
        const t = man.tensors[`l${i}.${n}`];
        l[n] = { w: upload(tensor(`l${i}.${n}`), `l${i}.${n}`), s: upload(tensor(`l${i}.${n}.scales`), `l${i}.${n}.s`), O: t.shape[0], K: t.shape[1] };
      }
      for (const n of ['input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm']) {
        l[n] = upload(tensor(`l${i}.${n}`), `l${i}.${n}`);
      }
      const qk = new Float32Array(512);
      qk.set(this.tensorF32(`l${i}.self_attn.q_norm`), 0);
      qk.set(this.tensorF32(`l${i}.self_attn.k_norm`), 256);
      l.qkNorm = upload(qk, `l${i}.qk`);
      this.layers.push(l);
    }
    this.finalNorm = upload(tensor('norm'), 'norm');
    this.rope = { sliding_attention: upload(tensor('rope.sliding_attention'), 'rope.s'), full_attention: upload(tensor('rope.full_attention'), 'rope.f') };
    const enc = man.tensors['sae.enc'];
    this.sae = { w: upload(tensor('sae.enc'), 'sae.enc'), s: upload(tensor('sae.enc.scales'), 'sae.s'), O: enc.shape[0], K: enc.shape[1] };
    // decoder rows, biases and thresholds stay on the CPU (steering vectors are built there)
    this.dec = new Int8Array(files[man.tensors['sae.dec'].file], man.tensors['sae.dec'].offset, man.tensors['sae.dec'].bytes);
    this.decScales = this.tensorF32('sae.dec.scales');
    this.bEnc = this.tensorF32('sae.b_enc');
    this.threshold = this.tensorF32('sae.threshold');

    const A = (n, label) => this.buffer(n * 4, label);
    this.x = A(NMAX * 640, 'x');
    this.xn = A(NMAX * 640, 'xn');
    this.q = A(NMAX * 1024, 'q');
    this.kv = A(NMAX * 512, 'kv');
    this.att = A(NMAX * 1024, 'att');
    this.o = A(NMAX * 640, 'o');
    this.g = A(NMAX * 2048, 'g');
    this.u = A(NMAX * 2048, 'u');
    this.xl = A(BMAX * 640, 'xl');
    this.logits = A(BMAX * man.vocab, 'logits');
    this.logitsRead = device.createBuffer({ size: BMAX * man.vocab * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.ids = A(NMAX, 'ids');
    this.meta = A(NMAX * 2, 'meta');
    this.steerVec = A(BMAX * 640, 'steer');
    this.cache = this.layers.map((_, i) => ({ K: A(BMAX * 512 * 256, `K${i}`), V: A(BMAX * 512 * 256, `V${i}`) }));
    this.pipes = {};
    for (const [name, code] of Object.entries(SHADERS)) {
      const module = device.createShaderModule({ code });
      this.pipes[name] = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    }
    this.uniforms = new Map();
    this.groups = new Map();
    this.steering = false;
  }

  buffer(size, label) {
    const b = this.device.createBuffer({ size: Math.ceil(size / 16) * 16, label,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    b.uid = this.bufId++;
    return b;
  }

  uniform(values) {
    const key = values.join(',');
    let b = this.uniforms.get(key);
    if (!b) {
      b = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      b.uid = this.bufId++;
      const a = new Uint32Array(8);
      a.set(values);
      this.device.queue.writeBuffer(b, 0, a);
      this.uniforms.set(key, b);
    }
    return b;
  }

  run(pass, name, buffers, groups) {
    const key = name + ':' + buffers.map(b => b.uid).join(',');
    let bg = this.groups.get(key);
    if (!bg) {
      bg = this.device.createBindGroup({ layout: this.pipes[name].getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
      this.groups.set(key, bg);
    }
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(...groups);
  }

  matmul(pass, N, x, m, y, { yStride = m.O, yOff = 0, sOff = 0, O = m.O } = {}) {
    this.run(pass, 'matmul', [this.uniform([N, m.K / 4, O, yStride, yOff, sOff]), x, m.w, m.s, y], [Math.ceil(O / 8)]);
  }

  norm(pass, nvec, dim, x, w, y, add) {
    this.run(pass, 'rmsnorm', [this.uniform([nvec, dim, add ? 1 : 0, f32bits(this.man.eps)]), x, w, y], [nvec]);
  }

  // steering vectors per row (Float32Array(640) or null)
  setSteer(vecs) {
    const a = new Float32Array(BMAX * 640);
    vecs.forEach((v, r) => v && a.set(v, r * 640));
    this.steering = vecs.some(Boolean);
    this.device.queue.writeBuffer(this.steerVec, 0, a);
  }

  // decoder row j, dequantized, times k
  conceptVector(j, k) {
    const v = new Float32Array(640);
    const s = this.decScales[j] * k;
    for (let i = 0; i < 640; i++) v[i] = this.dec[j * 640 + i] * s;
    return v;
  }

  // layers 0..last on tokens[n] (row, pos given per token); steering after layer L
  encodeLayers(enc, N, last) {
    const pass = enc.beginComputePass();
    const P = this.uniform([N, f32bits(Math.fround(Math.sqrt(640)))]);
    this.run(pass, 'embed', [P, this.ids, ...this.embed, this.embedScales, this.x], [N]);
    for (let i = 0; i <= last; i++) {
      const l = this.layers[i];
      this.norm(pass, N, 640, this.x, l.input_layernorm, this.xn);
      this.matmul(pass, N, this.xn, l['self_attn.q_proj'], this.q);
      this.matmul(pass, N, this.xn, l['self_attn.k_proj'], this.kv);
      this.matmul(pass, N, this.xn, l['self_attn.v_proj'], this.kv, { yOff: N * 256 });
      this.run(pass, 'qkrope', [this.uniform([N, f32bits(this.man.eps)]), this.meta, this.q, this.kv, l.qkNorm, this.rope[l.type],
        this.cache[i].K, this.cache[i].V], [N, 5]);
      this.run(pass, 'attention', [this.uniform([N]), this.meta, this.q, this.cache[i].K, this.cache[i].V, this.att], [N, 4]);
      this.matmul(pass, N, this.att, l['self_attn.o_proj'], this.o);
      this.norm(pass, N, 640, this.o, l.post_attention_layernorm, this.x, true);
      this.norm(pass, N, 640, this.x, l.pre_feedforward_layernorm, this.xn);
      this.matmul(pass, N, this.xn, l['mlp.gate_proj'], this.g);
      this.matmul(pass, N, this.xn, l['mlp.up_proj'], this.u);
      this.run(pass, 'gelu', [this.uniform([N * 2048]), this.g, this.u], [Math.ceil(N * 2048 / WG)]);
      this.matmul(pass, N, this.g, l['mlp.down_proj'], this.o);
      this.norm(pass, N, 640, this.o, l.post_feedforward_layernorm, this.x, true);
      if (i === this.L && this.steering) this.run(pass, 'steer', [this.uniform([N]), this.meta, this.steerVec, this.x], [N]);
    }
    pass.end();
  }

  writeTokens(rows, start) {
    const T = rows[0].length, B = rows.length, N = B * T;
    if (B > BMAX || N > NMAX || start + T > this.man.maxPos || rows.some(r => r.length !== T)) throw new Error('bad batch');
    const meta = new Uint32Array(2 * N);
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) { meta[2 * (b * T + t)] = b; meta[2 * (b * T + t) + 1] = start + t; }
    this.device.queue.writeBuffer(this.ids, 0, new Uint32Array(rows.flat()));
    this.device.queue.writeBuffer(this.meta, 0, meta);
    return { B, T, N };
  }

  // tokens for each row (same count), at positions start.. ; returns next-token logits per row
  async forward(rows, start) {
    const { B, T, N } = this.writeTokens(rows, start);
    const enc = this.device.createCommandEncoder();
    this.encodeLayers(enc, N, this.layers.length - 1);
    for (let b = 0; b < B; b++) enc.copyBufferToBuffer(this.x, ((b + 1) * T - 1) * 640 * 4, this.xn, b * 640 * 4, 640 * 4);
    const pass = enc.beginComputePass();
    this.norm(pass, B, 640, this.xn, this.finalNorm, this.xl);
    for (let c = 0; c < 4; c++) {
      this.matmul(pass, B, this.xl, { w: this.embed[c], s: this.embedScales, O: 65536, K: 640 }, this.logits,
        { yStride: this.man.vocab, yOff: c * 65536, sOff: c * 65536 });
    }
    pass.end();
    const V = this.man.vocab;
    enc.copyBufferToBuffer(this.logits, 0, this.logitsRead, 0, B * V * 4);
    this.device.queue.submit([enc.finish()]);
    await this.logitsRead.mapAsync(GPUMapMode.READ, 0, B * V * 4);
    const out = new Float32Array(this.logitsRead.getMappedRange(0, B * V * 4).slice(0));
    this.logitsRead.unmap();
    return Array.from({ length: B }, (_, b) => out.subarray(b * V, (b + 1) * V));
  }

  // one text: residual after layer L (T x 640) and SAE pre-activations (T x concepts), unsteered
  async read(ids) {
    const steering = this.steering;
    this.steering = false;
    const { N } = this.writeTokens([ids], 0);
    const F = this.sae.O;
    if (!this.pre) {
      this.pre = this.buffer(NMAX * F * 4, 'pre');
      this.preRead = this.device.createBuffer({ size: NMAX * F * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.residRead = this.device.createBuffer({ size: NMAX * 640 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    const enc = this.device.createCommandEncoder();
    this.encodeLayers(enc, N, this.L);
    const pass = enc.beginComputePass();
    this.matmul(pass, N, this.x, this.sae, this.pre);
    pass.end();
    enc.copyBufferToBuffer(this.pre, 0, this.preRead, 0, N * F * 4);
    enc.copyBufferToBuffer(this.x, 0, this.residRead, 0, N * 640 * 4);
    this.device.queue.submit([enc.finish()]);
    this.steering = steering;
    await Promise.all([this.preRead.mapAsync(GPUMapMode.READ, 0, N * F * 4), this.residRead.mapAsync(GPUMapMode.READ, 0, N * 640 * 4)]);
    const pre = new Float32Array(this.preRead.getMappedRange(0, N * F * 4).slice(0));
    const resid = new Float32Array(this.residRead.getMappedRange(0, N * 640 * 4).slice(0));
    this.preRead.unmap();
    this.residRead.unmap();
    // JumpReLU
    for (let n = 0; n < N; n++) {
      for (let j = 0; j < F; j++) {
        const a = pre[n * F + j] + this.bEnc[j];
        pre[n * F + j] = a > this.threshold[j] ? a : 0;
      }
    }
    return { z: pre, resid, T: N, F };
  }
}
