import {
  WORKGROUP_SIZE,
  MAX_OBSTACLES,
  countWGSL,
  scanWGSL,
  scatterWGSL,
  densityWGSL,
  integrateWGSL,
  splatWGSL,
  surfaceWGSL,
  overlayWGSL,
} from './shaders.js';

const SUBSTEPS = 2;
const PARAM_FLOATS = 64;
const PARAM_BYTES = PARAM_FLOATS * 4;

const BOX_W = 200;
const BOX_H = 130;
const SPACING = 0.5;
const H = SPACING * 2;
const REST_DENSITY = 1.0;
const MASS = 0.25;
const DT = 0.0024;
const GRID_W = Math.ceil(BOX_W / H);
const GRID_H = Math.ceil(BOX_H / H);
const GRID_CELLS = GRID_W * GRID_H;

const DEFAULTS = {
  count: 40_000,
  gravity: 28,
  viscosity: 1.1,
  stiffness: 560,
  tilt: 0,
};

const PRESETS = new Map([
  [10_000, { width: 50, height: 50 }],
  [20_000, { width: 70, height: 72 }],
  [40_000, { width: 100, height: 100 }],
  [80_000, { width: 160, height: 125 }],
]);

const OBSTACLE_SEEDS = [
  { x: 128, y: 11, r: 9.0 },
  { x: 152, y: 55, r: 8.5 },
  { x: 96, y: 42, r: 6.2 },
  { x: 170, y: 24, r: 6.8 },
];

class SPHFluid {
  constructor(device, context, canvas, format) {
    this.device = device;
    this.context = context;
    this.canvas = canvas;
    this.format = format;

    this.count = DEFAULTS.count;
    this.paused = false;
    this.gravity = DEFAULTS.gravity;
    this.viscosity = DEFAULTS.viscosity;
    this.stiffness = DEFAULTS.stiffness;
    this.tilt = DEFAULTS.tilt;
    this.srcIndex = 0;

    this.mouse = {
      pos: { x: 0, y: 0 },
      vel: { x: 0, y: 0 },
      active: false,
      mode: 0,
      lastMove: 0,
    };
    this.draggingObstacle = -1;
    this.obstacles = [{ ...OBSTACLE_SEEDS[0], active: 1, vx: 0, vy: 0 }];

    this.params = new Float32Array(PARAM_FLOATS);
    this.paramsU32 = new Uint32Array(this.params.buffer);
    this.paramsBuf = device.createBuffer({
      size: PARAM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.buildPipelines();
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.createGridBuffers();
    this.resize();
    this.allocateParticles(this.count);
  }

  buildPipelines() {
    const { device, format } = this;

    this.countPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: countWGSL }), entryPoint: 'main' },
    });
    this.scanPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: scanWGSL }), entryPoint: 'main' },
    });
    this.scatterPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: scatterWGSL }), entryPoint: 'main' },
    });
    this.densityPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: densityWGSL }), entryPoint: 'main' },
    });
    this.integratePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: integrateWGSL }), entryPoint: 'main' },
    });

    const splatModule = device.createShaderModule({ code: splatWGSL });
    this.splatPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: splatModule, entryPoint: 'vs' },
      fragment: {
        module: splatModule,
        entryPoint: 'fs',
        targets: [{
          format: 'rgba16float',
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const surfaceModule = device.createShaderModule({ code: surfaceWGSL });
    this.surfacePipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: surfaceModule, entryPoint: 'vs' },
      fragment: { module: surfaceModule, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });

    const overlayModule = device.createShaderModule({ code: overlayWGSL });
    this.overlayPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: overlayModule, entryPoint: 'vs' },
      fragment: {
        module: overlayModule,
        entryPoint: 'fs',
        targets: [{
          format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

  }

  createGridBuffers() {
    const { device } = this;
    const gridBytes = GRID_CELLS * Uint32Array.BYTES_PER_ELEMENT;
    this.cellCountBuf = device.createBuffer({
      size: gridBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.cellOffsetBuf = device.createBuffer({
      size: gridBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.cellStartBuf = device.createBuffer({
      size: gridBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.scanBindGroup = device.createBindGroup({
      layout: this.scanPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.cellCountBuf } },
        { binding: 2, resource: { buffer: this.cellStartBuf } },
      ],
    });
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const maxTextureSize = this.device.limits.maxTextureDimension2D;
    const width = Math.min(maxTextureSize, Math.max(1, Math.floor(this.canvas.clientWidth * dpr)));
    const height = Math.min(maxTextureSize, Math.max(1, Math.floor(this.canvas.clientHeight * dpr)));
    if (this.canvas.width === width && this.canvas.height === height && this.fieldTex) {
      return;
    }

    this.canvas.width = width;
    this.canvas.height = height;
    this.dpr = dpr;

    const padding = Math.max(24 * dpr, Math.min(width, height) * 0.055);
    this.viewScale = Math.max(1, Math.min((width - padding * 2) / BOX_W, (height - padding * 2) / BOX_H));
    this.viewOffset = {
      x: (width - BOX_W * this.viewScale) * 0.5,
      y: (height - BOX_H * this.viewScale) * 0.5,
    };

    if (this.fieldTex) {
      this.fieldTex.destroy();
    }
    this.fieldTex = this.device.createTexture({
      size: [width, height],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fieldView = this.fieldTex.createView();
    this.createSurfaceBindGroup();
  }

  createSurfaceBindGroup() {
    if (!this.fieldView || !this.surfacePipeline) {
      return;
    }

    this.surfaceBindGroup = this.device.createBindGroup({
      layout: this.surfacePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: this.fieldView },
      ],
    });
    this.overlayBindGroup = this.device.createBindGroup({
      layout: this.overlayPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.paramsBuf } }],
    });
  }

  allocateParticles(count) {
    this.count = count;
    this.srcIndex = 0;
    const particleBytes = count * 4 * Float32Array.BYTES_PER_ELEMENT;
    const data = seedParticles(count);

    for (const buffer of this.particleBuffers ?? []) {
      buffer.destroy();
    }
    this.densityBuf?.destroy();
    this.pressureBuf?.destroy();
    this.sortedIndicesBuf?.destroy();

    this.particleBuffers = [
      this.createParticleBuffer(data),
      this.device.createBuffer({
        size: particleBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
    ];
    this.densityBuf = this.device.createBuffer({
      size: count * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    });
    this.pressureBuf = this.device.createBuffer({
      size: count * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    });
    this.sortedIndicesBuf = this.device.createBuffer({
      size: count * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    });

    this.createParticleBindGroups();
  }

  createParticleBuffer(data) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  }

  createParticleBindGroups() {
    const { device } = this;

    this.countBindGroups = this.particleBuffers.map((buffer) => device.createBindGroup({
      layout: this.countPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.paramsBuf } },
        { binding: 2, resource: { buffer: this.cellCountBuf } },
      ],
    }));
    this.scatterBindGroups = this.particleBuffers.map((buffer) => device.createBindGroup({
      layout: this.scatterPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.paramsBuf } },
        { binding: 2, resource: { buffer: this.cellStartBuf } },
        { binding: 3, resource: { buffer: this.cellOffsetBuf } },
        { binding: 4, resource: { buffer: this.sortedIndicesBuf } },
      ],
    }));
    this.densityBindGroups = this.particleBuffers.map((buffer) => device.createBindGroup({
      layout: this.densityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.paramsBuf } },
        { binding: 2, resource: { buffer: this.cellCountBuf } },
        { binding: 3, resource: { buffer: this.cellStartBuf } },
        { binding: 4, resource: { buffer: this.sortedIndicesBuf } },
        { binding: 5, resource: { buffer: this.densityBuf } },
        { binding: 6, resource: { buffer: this.pressureBuf } },
      ],
    }));
    this.integrateBindGroups = [
      this.createIntegrateBindGroup(0, 1),
      this.createIntegrateBindGroup(1, 0),
    ];
    this.splatBindGroups = this.particleBuffers.map((buffer) => device.createBindGroup({
      layout: this.splatPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.paramsBuf } },
      ],
    }));
  }

  createIntegrateBindGroup(src, dst) {
    return this.device.createBindGroup({
      layout: this.integratePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffers[src] } },
        { binding: 1, resource: { buffer: this.particleBuffers[dst] } },
        { binding: 2, resource: { buffer: this.paramsBuf } },
        { binding: 3, resource: { buffer: this.cellCountBuf } },
        { binding: 4, resource: { buffer: this.cellStartBuf } },
        { binding: 5, resource: { buffer: this.sortedIndicesBuf } },
        { binding: 6, resource: { buffer: this.densityBuf } },
        { binding: 7, resource: { buffer: this.pressureBuf } },
      ],
    });
  }

  addObstacle() {
    if (this.obstacles.length >= MAX_OBSTACLES) {
      return;
    }

    const seed = OBSTACLE_SEEDS[this.obstacles.length];
    this.obstacles.push({ ...seed, active: 1, vx: 0, vy: 0 });
  }

  reset() {
    this.allocateParticles(this.count);
  }

  pointerToWorld(ev) {
    const rect = this.canvas.getBoundingClientRect();
    const px = (ev.clientX - rect.left) * this.dpr;
    const py = (ev.clientY - rect.top) * this.dpr;
    return {
      x: (px - this.viewOffset.x) / this.viewScale,
      y: BOX_H - (py - this.viewOffset.y) / this.viewScale,
    };
  }

  pickObstacle(pos) {
    let best = -1;
    let bestDist = Infinity;
    for (let i = 0; i < this.obstacles.length; i++) {
      const obs = this.obstacles[i];
      if (!obs.active) {
        continue;
      }
      const dx = pos.x - obs.x;
      const dy = pos.y - obs.y;
      const dist = Math.hypot(dx, dy);
      if (dist <= obs.r && dist < bestDist) {
        best = i;
        bestDist = dist;
      }
    }
    return best;
  }

  pointerDown(ev) {
    const pos = this.pointerToWorld(ev);
    const obstacle = this.pickObstacle(pos);
    this.mouse.lastMove = performance.now();

    if (obstacle >= 0) {
      this.draggingObstacle = obstacle;
      this.mouse.active = false;
      this.mouse.mode = 0;
      return;
    }

    this.draggingObstacle = -1;
    this.mouse.pos = pos;
    this.mouse.vel = { x: 0, y: 0 };
    this.mouse.active = true;
    this.mouse.mode = ev.shiftKey ? 2 : 1;
  }

  pointerMove(ev) {
    const now = performance.now();
    const pos = this.pointerToWorld(ev);
    const elapsed = Math.max((now - this.mouse.lastMove) / 1000, 1 / 240);
    this.mouse.lastMove = now;

    if (this.draggingObstacle >= 0) {
      const obs = this.obstacles[this.draggingObstacle];
      const nextX = clamp(pos.x, obs.r + H, BOX_W - obs.r - H);
      const nextY = clamp(pos.y, obs.r + H, BOX_H - obs.r - H);
      obs.vx = (nextX - obs.x) / elapsed;
      obs.vy = (nextY - obs.y) / elapsed;
      obs.x = nextX;
      obs.y = nextY;
      return;
    }

    if (!this.mouse.active) {
      return;
    }

    this.mouse.vel = {
      x: (pos.x - this.mouse.pos.x) / elapsed,
      y: (pos.y - this.mouse.pos.y) / elapsed,
    };
    this.mouse.pos = pos;
    this.mouse.mode = ev.shiftKey ? 2 : 1;
  }

  pointerUp() {
    this.mouse.active = false;
    this.mouse.mode = 0;
    this.mouse.vel = { x: 0, y: 0 };
    this.draggingObstacle = -1;
    for (const obs of this.obstacles) {
      obs.vx = 0;
      obs.vy = 0;
    }
  }

  updateParams() {
    const p = this.params;
    const u = this.paramsU32;

    u[0] = this.count;
    u[1] = GRID_W;
    u[2] = GRID_H;
    u[3] = this.obstacles.length;
    p[4] = BOX_W;
    p[5] = BOX_H;
    p[6] = this.canvas.width;
    p[7] = this.canvas.height;
    p[8] = H;
    p[9] = MASS;
    p[10] = REST_DENSITY;
    p[11] = this.stiffness;
    p[12] = this.viscosity;
    p[13] = this.gravity;
    p[14] = this.tilt;
    p[15] = DT;
    p[16] = 8500;
    p[17] = 35;
    p[18] = 220;
    p[19] = 0.999;
    p[20] = this.mouse.pos.x;
    p[21] = this.mouse.pos.y;
    p[22] = this.mouse.vel.x;
    p[23] = this.mouse.vel.y;
    p[24] = 9.0;
    p[25] = 19.0;
    u[26] = this.mouse.active ? this.mouse.mode : 0;
    u[27] = 0;
    p[28] = this.viewScale;
    p[29] = 0.42;
    p[30] = this.viewOffset.x;
    p[31] = this.viewOffset.y;

    for (let i = 0; i < MAX_OBSTACLES; i++) {
      const obs = this.obstacles[i];
      const base = 32 + i * 4;
      const velBase = 48 + i * 4;
      if (obs?.active) {
        p[base] = obs.x;
        p[base + 1] = obs.y;
        p[base + 2] = obs.r;
        p[base + 3] = 1;
        p[velBase] = obs.vx;
        p[velBase + 1] = obs.vy;
        p[velBase + 2] = 0;
        p[velBase + 3] = 0;
      } else {
        p[base] = 0;
        p[base + 1] = 0;
        p[base + 2] = 0;
        p[base + 3] = 0;
        p[velBase] = 0;
        p[velBase + 1] = 0;
        p[velBase + 2] = 0;
        p[velBase + 3] = 0;
      }
    }

    this.device.queue.writeBuffer(this.paramsBuf, 0, p);
  }

  frame() {
    this.resize();
    this.fadeTransientInput();
    this.updateParams();

    const encoder = this.device.createCommandEncoder();
    if (!this.paused) {
      for (let step = 0; step < SUBSTEPS; step++) {
        encoder.clearBuffer(this.cellCountBuf);
        encoder.clearBuffer(this.cellOffsetBuf);

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.countPipeline);
        pass.setBindGroup(0, this.countBindGroups[this.srcIndex]);
        pass.dispatchWorkgroups(Math.ceil(this.count / WORKGROUP_SIZE));

        pass.setPipeline(this.scanPipeline);
        pass.setBindGroup(0, this.scanBindGroup);
        pass.dispatchWorkgroups(1);

        pass.setPipeline(this.scatterPipeline);
        pass.setBindGroup(0, this.scatterBindGroups[this.srcIndex]);
        pass.dispatchWorkgroups(Math.ceil(this.count / WORKGROUP_SIZE));

        pass.setPipeline(this.densityPipeline);
        pass.setBindGroup(0, this.densityBindGroups[this.srcIndex]);
        pass.dispatchWorkgroups(Math.ceil(this.count / WORKGROUP_SIZE));

        pass.setPipeline(this.integratePipeline);
        pass.setBindGroup(0, this.integrateBindGroups[this.srcIndex]);
        pass.dispatchWorkgroups(Math.ceil(this.count / WORKGROUP_SIZE));
        pass.end();

        this.srcIndex = 1 - this.srcIndex;
      }
    }

    const fieldPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.fieldView,
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        storeOp: 'store',
      }],
    });
    fieldPass.setPipeline(this.splatPipeline);
    fieldPass.setBindGroup(0, this.splatBindGroups[this.srcIndex]);
    fieldPass.draw(6, this.count);
    fieldPass.end();

    const surfacePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: 'store',
      }],
    });
    surfacePass.setPipeline(this.surfacePipeline);
    surfacePass.setBindGroup(0, this.surfaceBindGroup);
    surfacePass.draw(3);
    surfacePass.setPipeline(this.overlayPipeline);
    surfacePass.setBindGroup(0, this.overlayBindGroup);
    surfacePass.draw(3);
    surfacePass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  fadeTransientInput() {
    const now = performance.now();
    if (this.mouse.active && now - this.mouse.lastMove > 80) {
      this.mouse.vel.x *= 0.72;
      this.mouse.vel.y *= 0.72;
    }

    for (let i = 0; i < this.obstacles.length; i++) {
      if (i === this.draggingObstacle) {
        this.obstacles[i].vx *= 0.82;
        this.obstacles[i].vy *= 0.82;
      } else {
        this.obstacles[i].vx = 0;
        this.obstacles[i].vy = 0;
      }
    }
  }
}

function seedParticles(count) {
  const block = PRESETS.get(count) ?? PRESETS.get(DEFAULTS.count);
  const cols = Math.floor(block.width / SPACING);
  const x0 = H * 1.8;
  const y0 = H * 1.8;
  const data = new Float32Array(count * 4);

  for (let i = 0; i < count; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const jitterX = (Math.random() - 0.5) * SPACING * 0.2;
    const jitterY = (Math.random() - 0.5) * SPACING * 0.2;
    const o = i * 4;
    data[o] = x0 + col * SPACING + jitterX;
    data[o + 1] = y0 + row * SPACING + jitterY;
    data[o + 2] = 0;
    data[o + 3] = 0;
  }

  return data;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatCount(n) {
  return `${Math.round(n / 1000)}K`;
}

async function main() {
  const canvas = document.getElementById('gpu');
  const panel = document.getElementById('panel');
  const unsupported = document.getElementById('unsupported');
  const showUnsupported = () => {
    canvas.hidden = true;
    panel.hidden = true;
    unsupported.hidden = false;
  };

  if (!navigator.gpu) {
    showUnsupported();
    return;
  }

  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch {
    showUnsupported();
    return;
  }

  if (!adapter) {
    showUnsupported();
    return;
  }

  let device;
  try {
    device = await adapter.requestDevice();
  } catch {
    showUnsupported();
    return;
  }

  device.addEventListener?.('uncapturederror', (e) => console.error('WebGPU error:', e.error?.message || e));

  const context = canvas.getContext('webgpu');
  if (!context) {
    showUnsupported();
    return;
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  const fluid = new SPHFluid(device, context, canvas, format);

  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    fluid.pointerDown(e);
  });
  canvas.addEventListener('pointermove', (e) => fluid.pointerMove(e));
  canvas.addEventListener('pointerup', () => fluid.pointerUp());
  canvas.addEventListener('pointercancel', () => fluid.pointerUp());

  const fpsEl = document.getElementById('fps');
  const pcountEl = document.getElementById('pcount');
  pcountEl.textContent = formatCount(fluid.count);

  document.getElementById('count').addEventListener('change', (e) => {
    fluid.allocateParticles(parseInt(e.target.value, 10));
    pcountEl.textContent = formatCount(fluid.count);
  });
  document.getElementById('gravity').addEventListener('input', (e) => {
    fluid.gravity = parseFloat(e.target.value);
  });
  document.getElementById('viscosity').addEventListener('input', (e) => {
    fluid.viscosity = parseFloat(e.target.value);
  });
  document.getElementById('stiffness').addEventListener('input', (e) => {
    fluid.stiffness = parseFloat(e.target.value);
  });
  document.getElementById('tilt').addEventListener('input', (e) => {
    fluid.tilt = parseFloat(e.target.value) * Math.PI / 180;
  });

  const pauseBtn = document.getElementById('pause');
  pauseBtn.addEventListener('click', () => {
    fluid.paused = !fluid.paused;
    pauseBtn.textContent = fluid.paused ? 'Play' : 'Pause';
  });
  document.getElementById('reset').addEventListener('click', () => fluid.reset());
  document.getElementById('addObstacle').addEventListener('click', () => fluid.addObstacle());
  window.addEventListener('resize', () => fluid.resize());

  let last = performance.now();
  let frames = 0;
  let fpsTimer = 0;
  function loop(now) {
    const dt = (now - last) / 1000;
    last = now;
    fluid.frame();

    frames++;
    fpsTimer += dt;
    if (fpsTimer >= 0.5) {
      fpsEl.textContent = Math.round(frames / fpsTimer);
      frames = 0;
      fpsTimer = 0;
    }

    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main();
