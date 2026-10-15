// Draws the molecules and beads straight from the simulation's GPU buffers.
// Each disc is a quad; molecules and beads near an edge are drawn again on the
// opposite side, so the periodic box wraps without seams.

const VIEW_BYTES = 128;

const COMMON = /* wgsl */ `
struct View {
  toClip: vec4f,      // clip = x * toClip.xy + toClip.zw
  L: vec2f, px: f32, rMol: f32,
  cool: vec4f, hot: vec4f,
  bead: vec4f, beadHi: vec4f, beadLo: vec4f,
  nb: u32, p0: u32, p1: u32, p2: u32,
}
@group(0) @binding(0) var<uniform> V: View;

const CORNERS = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1));

// offset of periodic image k (0..3) of a disc at x with radius r, or none
fn image(x: vec2f, r: f32, k: u32) -> vec3f {
  var s = vec2f(0.0);
  if (x.x < r) { s.x = V.L.x; } else if (x.x > V.L.x - r) { s.x = -V.L.x; }
  if (x.y < r) { s.y = V.L.y; } else if (x.y > V.L.y - r) { s.y = -V.L.y; }
  let pick = vec2f(f32(k & 1u), f32((k >> 1u) & 1u));
  let off = s * pick;
  let needed = all((pick == vec2f(0.0)) | (s != vec2f(0.0)));
  return vec3f(off, select(0.0, 1.0, needed));
}

struct Out {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) shade: f32,
  @location(2) r: f32,
}
`;

const MOLECULES = COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read> vel: array<vec2f>;

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Out {
  let x = pos[ii / 4u];
  let v = vel[ii / 4u];
  let im = image(x, V.rMol + V.px, ii & 3u);
  let c = CORNERS[vi] * (V.rMol + V.px);
  var o: Out;
  o.pos = vec4f((x + im.xy + c) * V.toClip.xy + V.toClip.zw, 0.0, 1.0);
  if (im.z == 0.0) { o.pos = vec4f(2.0, 2.0, 0.0, 1.0); }
  o.uv = c;
  o.shade = 1.0 - exp(-0.5 * dot(v, v));    // uniform in [0, 1] at T = 1
  o.r = V.rMol;
  return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4f {
  let d = length(i.uv);
  let a = clamp((i.r - d) / V.px + 0.5, 0.0, 1.0);
  let col = mix(V.cool, V.hot, i.shade * i.shade);
  return vec4f(col.rgb * a, a);
}
`;

const BEADS = COMMON + /* wgsl */ `
struct Bead {
  x: vec2f, v: vec2f, wrap: vec2i, a: f32, m: f32,
  ext: vec2f, f: vec2f, step: u32, p0: u32, p1: u32, p2: u32,
}
struct State { sys: array<vec4f, 2>, beads: array<Bead, 8> }
@group(0) @binding(1) var<storage, read> S: State;

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Out {
  let b = S.beads[ii / 4u];
  let im = image(b.x, b.a + V.px, ii & 3u);
  let c = CORNERS[vi] * (b.a + V.px);
  var o: Out;
  o.pos = vec4f((b.x + im.xy + c) * V.toClip.xy + V.toClip.zw, 0.0, 1.0);
  if (im.z == 0.0 || ii / 4u >= V.nb) { o.pos = vec4f(2.0, 2.0, 0.0, 1.0); }
  o.uv = c;
  o.shade = 0.0;
  o.r = b.a;
  return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4f {
  let d = length(i.uv);
  let a = clamp((i.r - d) / V.px + 0.5, 0.0, 1.0);
  // a lit sphere: light from the upper left, darker rim
  let q = i.uv / i.r;
  let z = sqrt(max(0.0, 1.0 - dot(q, q)));
  let n = vec3f(q.x, q.y, z);
  let l = normalize(vec3f(-0.45, -0.55, 0.7));
  let diffuse = max(dot(n, l), 0.0);
  var col = mix(V.beadLo.rgb, V.bead.rgb, smoothstep(0.0, 0.8, diffuse));
  col = mix(col, V.beadHi.rgb, pow(max(dot(reflect(-l, n), vec3f(0.0, 0.0, 1.0)), 0.0), 18.0) * 0.9);
  return vec4f(col * a, a);
}
`;

function hexToRGBA(hex, alpha = 1) {
  const h = hex.trim().replace('#', '');
  return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) / 255).concat(alpha);
}

export class Renderer {
  // Bind groups are cached by their first buffer; call forget() when the simulation is replaced.
  constructor(device, context, format) {
    this.device = device;
    this.context = context;
    this.view = device.createBuffer({ size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const blend = {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    const pipe = (code, buffers) => {
      const module = device.createShaderModule({ code });
      return device.createRenderPipeline({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs', buffers },
        fragment: { module, entryPoint: 'fs', targets: [{ format, blend }] },
        primitive: { topology: 'triangle-list' },
      });
    };
    this.molPipe = pipe(MOLECULES, []);
    this.beadPipe = pipe(BEADS, []);
    this.groups = new Map();
    this.colors = null;
  }

  forget() { this.groups.clear(); }

  setColors(c) {
    this.colors = {
      bg: hexToRGBA(c.bg), cool: hexToRGBA(c.cool), hot: hexToRGBA(c.hot),
      bead: hexToRGBA(c.bead), beadHi: hexToRGBA(c.beadHi), beadLo: hexToRGBA(c.beadLo),
    };
  }

  // transform: { scale (px per σ), ox, oy (px) } for a canvas of w × h device pixels
  draw(enc, sim, transform, w, h) {
    const { scale, ox, oy } = transform;
    const a = new ArrayBuffer(VIEW_BYTES), f = new Float32Array(a), u = new Uint32Array(a);
    f.set([2 * scale / w, -2 * scale / h, 2 * ox / w - 1, 1 - 2 * oy / h], 0);
    f.set([sim.L[0], sim.L[1], 1 / scale, 0.47], 4);
    const c = this.colors;
    f.set(c.cool, 8); f.set(c.hot, 12); f.set(c.bead, 16); f.set(c.beadHi, 20); f.set(c.beadLo, 24);
    u[28] = sim.nb;
    this.device.queue.writeBuffer(this.view, 0, a);

    const group = (pipe, buffers) => {
      if (!this.groups.has(buffers[0])) {
        this.groups.set(buffers[0], this.device.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries: [this.view, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
        }));
      }
      return this.groups.get(buffers[0]);
    };
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: c.bg[0], g: c.bg[1], b: c.bg[2], a: 1 }, loadOp: 'clear', storeOp: 'store',
      }],
    });
    pass.setPipeline(this.molPipe);
    pass.setBindGroup(0, group(this.molPipe, [sim.posBuffer, sim.vel]));
    pass.draw(6, 4 * sim.n);
    pass.setPipeline(this.beadPipe);
    pass.setBindGroup(0, group(this.beadPipe, [sim.stateBuffer]));
    pass.draw(6, 4 * sim.nb);
    pass.end();
  }
}
