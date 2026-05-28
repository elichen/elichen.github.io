// WebGPU Galaxy — a million particles integrated on the GPU.
//
// Pipeline per frame:
//   1. compute pass  — gravitational integration (core + optional mouse well)
//   2. render pass   — points drawn additively into an HDR accumulation texture,
//                      after a translucent black quad fades the previous frame (trails)
//   3. blit pass     — tonemap the accumulation texture onto the swapchain

const WORKGROUP_SIZE = 256;

const computeWGSL = /* wgsl */ `
struct Params {
  dt: f32,
  g: f32,
  soft: f32,
  damping: f32,
  count: u32,
  mouseActive: u32,
  pad0: u32,
  pad1: u32,
  mouse: vec2<f32>,
  mouseMass: f32,
  coreMass: f32,
};

@group(0) @binding(0) var<storage, read_write> parts: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> P: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) { return; }

  let s = parts[i];
  var pos = s.xy;
  var vel = s.zw;
  var acc = vec2<f32>(0.0, 0.0);

  // Central galactic core at the origin.
  let dc = -pos;
  let r2c = dot(dc, dc) + P.soft;
  acc += P.g * P.coreMass * dc / (r2c * sqrt(r2c));

  // Mouse gravity well (mass may be negative to push).
  if (P.mouseActive == 1u) {
    let dm = P.mouse - pos;
    let r2m = dot(dm, dm) + P.soft;
    acc += P.g * P.mouseMass * dm / (r2m * sqrt(r2m));
  }

  vel += acc * P.dt;
  vel *= P.damping;
  pos += vel * P.dt;

  parts[i] = vec4<f32>(pos, vel);
}
`;

const renderWGSL = /* wgsl */ `
struct RParams {
  aspect: f32,
  intensity: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<storage, read> parts: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> R: RParams;

struct VSOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) color: vec3<f32>,
};

// Speed -> color: cool blue at rest, cyan/white mid, warm amber when fast.
fn palette(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);
  let cold = vec3<f32>(0.10, 0.20, 0.75);
  let mid  = vec3<f32>(0.35, 0.85, 0.95);
  let warm = vec3<f32>(1.0, 0.75, 0.30);
  let hot  = vec3<f32>(1.0, 0.95, 0.85);
  if (x < 0.4) { return mix(cold, mid, x / 0.4); }
  if (x < 0.75) { return mix(mid, warm, (x - 0.4) / 0.35); }
  return mix(warm, hot, (x - 0.75) / 0.25);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let s = parts[vi];
  let p = s.xy;
  let v = s.zw;
  var o: VSOut;
  o.clip = vec4<f32>(p.x / R.aspect, p.y, 0.0, 1.0);
  let speed = length(v);
  o.color = palette(speed * 0.8) * R.intensity;
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
`;

// Fullscreen triangle shared by the fade and blit passes.
const fullscreenVS = /* wgsl */ `
struct VSOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) uv: vec2<f32>,
};
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var pts = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  let p = pts[vi];
  var o: VSOut;
  o.clip = vec4<f32>(p, 0.0, 1.0);
  o.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
  return o;
}
`;

const fadeWGSL = fullscreenVS + /* wgsl */ `
struct Fade { amount: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> F: Fade;
@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, F.amount);
}
`;

const blitWGSL = fullscreenVS + /* wgsl */ `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let hdr = textureSample(tex, samp, in.uv).rgb;
  // Filmic-ish tonemap so dense cores bloom without clipping to flat white.
  let mapped = vec3<f32>(1.0) - exp(-hdr * 1.1);
  return vec4<f32>(pow(mapped, vec3<f32>(0.85)), 1.0);
}
`;

class Galaxy {
  constructor(device, context, canvas, format) {
    this.device = device;
    this.context = context;
    this.canvas = canvas;
    this.format = format;

    this.paused = false;
    this.count = 1_000_000;
    this.gravity = 60;
    this.fadeAmount = 0.22; // 1 - trails; lower = longer trails
    this.mouse = { x: 0, y: 0, active: false, mass: 0 };
    this.accumCleared = false;

    this.simParams = new Float32Array(12); // matches Params layout (48 bytes)
    this.simBuf = device.createBuffer({
      size: this.simParams.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.rParams = new Float32Array(4);
    this.rBuf = device.createBuffer({
      size: this.rParams.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.fadeParams = new Float32Array(4);
    this.fadeBuf = device.createBuffer({
      size: this.fadeParams.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.buildPipelines();
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    this.resize();
    this.allocateParticles(this.count);
  }

  buildPipelines() {
    const { device, format } = this;

    this.computePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: computeWGSL }), entryPoint: 'main' },
    });

    const renderModule = device.createShaderModule({ code: renderWGSL });
    this.renderPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: renderModule, entryPoint: 'vs' },
      fragment: {
        module: renderModule,
        entryPoint: 'fs',
        targets: [{
          format: 'rgba16float',
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'point-list' },
    });

    const fadeModule = device.createShaderModule({ code: fadeWGSL });
    this.fadePipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: fadeModule, entryPoint: 'vs' },
      fragment: {
        module: fadeModule,
        entryPoint: 'fs',
        targets: [{
          format: 'rgba16float',
          blend: {
            color: { srcFactor: 'zero', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'zero', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });
    this.fadeBindGroup = device.createBindGroup({
      layout: this.fadePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.fadeBuf } }],
    });

    const blitModule = device.createShaderModule({ code: blitWGSL });
    this.blitPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: blitModule, entryPoint: 'vs' },
      fragment: { module: blitModule, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width === w && this.canvas.height === h && this.accumTex) return;
    this.canvas.width = w;
    this.canvas.height = h;
    this.aspect = w / h;

    if (this.accumTex) this.accumTex.destroy();
    this.accumTex = this.device.createTexture({
      size: [w, h],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.accumView = this.accumTex.createView();
    this.accumCleared = false;

    this.blitBindGroup = this.device.createBindGroup({
      layout: this.blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: this.accumView },
      ],
    });
  }

  allocateParticles(count) {
    this.count = count;
    if (this.partBuf) this.partBuf.destroy();

    const data = new Float32Array(count * 4);
    // Differentially-rotating disk: uniform area density, near-circular orbits.
    // GM must equal g*coreMass at the default slider so v = sqrt(GM/r) is circular.
    // Inner radius stays well outside the softened core (sqrt(soft) ~ 0.024) so no
    // particle dives through the singularity and gets slingshotted out.
    const GM = 0.5;
    for (let i = 0; i < count; i++) {
      const r = 0.09 + 0.86 * Math.sqrt(Math.random());
      const a = Math.random() * Math.PI * 2;
      const px = r * Math.cos(a);
      const py = r * Math.sin(a);
      // tiny symmetric spread around the circular speed -> gentle eccentricity, no net drift
      const v = Math.sqrt(GM / r) * (0.99 + Math.random() * 0.02);
      const jitter = 0.012;
      const vx = -Math.sin(a) * v + (Math.random() - 0.5) * jitter;
      const vy = Math.cos(a) * v + (Math.random() - 0.5) * jitter;
      const o = i * 4;
      data[o] = px; data[o + 1] = py; data[o + 2] = vx; data[o + 3] = vy;
    }

    this.partBuf = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.partBuf.getMappedRange()).set(data);
    this.partBuf.unmap();

    this.computeBindGroup = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.partBuf } },
        { binding: 1, resource: { buffer: this.simBuf } },
      ],
    });
    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.partBuf } },
        { binding: 1, resource: { buffer: this.rBuf } },
      ],
    });
    this.accumCleared = false; // restart trails so old positions don't linger
  }

  frame(dt) {
    const { device } = this;

    // intensity falls off as particle count rises so density stays readable
    const intensity = Math.min(1.0, 9_000 / Math.sqrt(this.count));

    // Uniforms.
    const g = this.gravity / 60; // slider 60 -> 1.0 (matches init GM)
    this.simParams.set([
      this.paused ? 0 : Math.min(dt, 0.033) * 0.95, // dt (clamped)
      g,
      0.0006, // softening^2
      0.9999, // damping (near-conservative; lets mouse-stirred energy relax slowly)
    ], 0);
    new Uint32Array(this.simParams.buffer, 16, 4).set([
      this.count,
      this.mouse.active ? 1 : 0,
      0, 0,
    ]);
    this.simParams.set([
      this.mouse.x, this.mouse.y,
      this.mouse.mass,
      0.5, // coreMass (g * coreMass = GM = 0.5 at default slider)
    ], 8);
    device.queue.writeBuffer(this.simBuf, 0, this.simParams);

    this.rParams.set([this.aspect, intensity, 0, 0]);
    device.queue.writeBuffer(this.rBuf, 0, this.rParams);

    this.fadeParams.set([this.fadeAmount, 0, 0, 0]);
    device.queue.writeBuffer(this.fadeBuf, 0, this.fadeParams);

    const encoder = device.createCommandEncoder();

    // 1. integrate
    if (!this.paused) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.computePipeline);
      pass.setBindGroup(0, this.computeBindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.count / WORKGROUP_SIZE));
      pass.end();
    }

    // 2. fade previous frame + draw points (HDR accumulation)
    const accumPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.accumView,
        loadOp: this.accumCleared ? 'load' : 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: 'store',
      }],
    });
    this.accumCleared = true;
    if (this.fadeAmount < 0.999) {
      accumPass.setPipeline(this.fadePipeline);
      accumPass.setBindGroup(0, this.fadeBindGroup);
      accumPass.draw(3);
    }
    accumPass.setPipeline(this.renderPipeline);
    accumPass.setBindGroup(0, this.renderBindGroup);
    accumPass.draw(this.count);
    accumPass.end();

    // 3. tonemap to screen
    const blitPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: 'store',
      }],
    });
    blitPass.setPipeline(this.blitPipeline);
    blitPass.setBindGroup(0, this.blitBindGroup);
    blitPass.draw(3);
    blitPass.end();

    device.queue.submit([encoder.finish()]);
  }
}

async function main() {
  const canvas = document.getElementById('gpu');
  const unsupported = document.getElementById('unsupported');

  if (!navigator.gpu) { unsupported.hidden = false; return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { unsupported.hidden = false; return; }
  const device = await adapter.requestDevice();

  device.addEventListener?.('uncapturederror', (e) => console.error('WebGPU error:', e.error?.message || e));

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const galaxy = new Galaxy(device, context, canvas, format);

  // --- input ---
  function pointerToWorld(ev) {
    const rect = canvas.getBoundingClientRect();
    const nx = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = 1 - ((ev.clientY - rect.top) / rect.height) * 2;
    return { x: nx * galaxy.aspect, y: ny };
  }
  function setWell(ev, active) {
    const w = pointerToWorld(ev);
    galaxy.mouse.x = w.x;
    galaxy.mouse.y = w.y;
    galaxy.mouse.active = active;
    galaxy.mouse.mass = ev.shiftKey ? -2.2 : 2.2;
  }
  canvas.addEventListener('pointerdown', (e) => { canvas.setPointerCapture(e.pointerId); setWell(e, true); });
  canvas.addEventListener('pointermove', (e) => { if (galaxy.mouse.active) setWell(e, true); });
  const release = () => { galaxy.mouse.active = false; };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  // --- controls ---
  document.getElementById('count').addEventListener('change', (e) => {
    galaxy.allocateParticles(parseInt(e.target.value, 10));
    pcountEl.textContent = formatCount(galaxy.count);
  });
  document.getElementById('trails').addEventListener('input', (e) => {
    // slider 0..100 -> fadeAmount 1..0.02 (more trail = less fade)
    galaxy.fadeAmount = 1 - (e.target.value / 100) * 0.98;
  });
  galaxy.fadeAmount = 1 - (78 / 100) * 0.98;
  document.getElementById('gravity').addEventListener('input', (e) => {
    galaxy.gravity = parseInt(e.target.value, 10);
  });
  const pauseBtn = document.getElementById('pause');
  pauseBtn.addEventListener('click', () => {
    galaxy.paused = !galaxy.paused;
    pauseBtn.textContent = galaxy.paused ? 'Play' : 'Pause';
  });
  document.getElementById('reset').addEventListener('click', () => galaxy.allocateParticles(galaxy.count));

  window.addEventListener('resize', () => galaxy.resize());

  // --- stats ---
  const fpsEl = document.getElementById('fps');
  const pcountEl = document.getElementById('pcount');
  const formatCount = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(n % 1e6 ? 1 : 0)}M` : `${n / 1e3}K`;
  pcountEl.textContent = formatCount(galaxy.count);

  let last = performance.now();
  let acc = 0, frames = 0, fpsTimer = 0;
  function loop(now) {
    const dt = (now - last) / 1000;
    last = now;
    galaxy.resize();
    galaxy.frame(dt);

    frames++; fpsTimer += dt;
    if (fpsTimer >= 0.5) {
      fpsEl.textContent = Math.round(frames / fpsTimer);
      frames = 0; fpsTimer = 0;
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main();
