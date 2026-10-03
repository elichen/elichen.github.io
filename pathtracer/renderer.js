// WebGPU plumbing: buffers, the trace compute pass, the display pass and picking.

const FRAME_BYTES = 192;
const VIEW_BYTES = 64;
const NONE = 0xffffffff;

export class Renderer {
  static async create(canvas) {
    if (!navigator.gpu) throw new Error('This browser does not support WebGPU.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter is available.');
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    const [trace, display] = await Promise.all(['trace.wgsl', 'display.wgsl'].map(f => fetch(f).then(r => r.text())));
    return new Renderer(canvas, device, adapter, trace, display);
  }

  constructor(canvas, device, adapter, traceCode, displayCode) {
    this.canvas = canvas;
    this.device = device;
    this.adapterInfo = adapter.info || {};
    this.context = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });

    const traceModule = device.createShaderModule({ code: traceCode, label: 'trace' });
    const displayModule = device.createShaderModule({ code: displayCode, label: 'display' });
    this.compileInfo = Promise.all([traceModule, displayModule].map(m => m.getCompilationInfo()));
    this.tracePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: traceModule, entryPoint: 'trace' } });
    this.pickPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: traceModule, entryPoint: 'pick' } });
    this.displayPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: displayModule, entryPoint: 'vs' },
      fragment: { module: displayModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });

    const U = GPUBufferUsage;
    this.frameBuf = device.createBuffer({ size: FRAME_BYTES, usage: U.UNIFORM | U.COPY_DST });
    this.viewBuf = device.createBuffer({ size: VIEW_BYTES, usage: U.UNIFORM | U.COPY_DST });
    this.statsBuf = device.createBuffer({ size: 32, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this.statsRead = device.createBuffer({ size: 32, usage: U.MAP_READ | U.COPY_DST });
    this.statsBusy = false;
    this.frameData = new ArrayBuffer(FRAME_BYTES);
    this.viewData = new ArrayBuffer(VIEW_BYTES);
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'clamp-to-edge' });
    this.buffers = {};
    this.width = this.height = this.fullWidth = this.fullHeight = 0;
    this.preview = false;
    this.setEnvironment(null);
  }

  storage(name, data, minBytes = 64) {
    const size = Math.max(minBytes, Math.ceil(data.byteLength / 4) * 4);
    let buf = this.buffers[name];
    if (!buf || buf.size < size) {
      buf?.destroy();
      buf = this.buffers[name] = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: name });
      this.bindGroups = null;
    }
    if (data.byteLength) this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  setScene(scene) {
    this.storage('nodes', scene.nodes);
    this.storage('prims', scene.prims);
    this.storage('instances', scene.instances);
    this.storage('materials', scene.materials);
    this.storage('lights', scene.lights);
    this.scene = { numInstances: scene.numInstances, numLights: scene.numLights, lightArea: scene.lightArea };
  }

  updateMaterials(materials) { this.storage('materials', materials); }

  updateInstances(instances) { this.storage('instances', instances); }

  // env: { width, height, texels (rgba16f), cdf, gridW, gridH } or null for black
  setEnvironment(env) {
    const d = this.device;
    if (!env) env = { width: 2, height: 1, texels: new Uint16Array(8), cdf: new Float32Array([0, 1, 0, 0.5, 1]), gridW: 2, gridH: 1, irradiance: [1, 1, 1] };
    this.envTex?.destroy();
    this.envTex = d.createTexture({ size: [env.width, env.height], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    d.queue.writeTexture({ texture: this.envTex }, env.texels, { bytesPerRow: env.width * 8 }, [env.width, env.height]);
    this.storage('envCdf', env.cdf);
    this.env = { gridW: env.gridW, gridH: env.gridH, irradiance: env.irradiance };
    this.bindGroups = null;
  }

  resize(width, height) {
    if (width === this.fullWidth && height === this.fullHeight) return;
    this.fullWidth = width; this.fullHeight = height;
    this.buffers.accum?.destroy();
    this.buffers.accum = this.device.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.bindGroups = null;
    this.setPreview(this.preview);
  }

  // While the camera moves, trace a quarter of the pixels into the same buffer.
  setPreview(on) {
    this.preview = !!on;
    this.width = on ? Math.ceil(this.fullWidth / 2) : this.fullWidth;
    this.height = on ? Math.ceil(this.fullHeight / 2) : this.fullHeight;
  }

  makeBindGroups() {
    const d = this.device, b = this.buffers;
    const entries = [
      { binding: 0, resource: { buffer: this.frameBuf } },
      { binding: 1, resource: { buffer: b.nodes } },
      { binding: 2, resource: { buffer: b.prims } },
      { binding: 3, resource: { buffer: b.instances } },
      { binding: 4, resource: { buffer: b.materials } },
      { binding: 5, resource: { buffer: b.lights } },
      { binding: 6, resource: { buffer: b.envCdf } },
      { binding: 7, resource: this.envTex.createView() },
      { binding: 8, resource: this.sampler },
      { binding: 9, resource: { buffer: b.accum } },
      { binding: 10, resource: { buffer: this.statsBuf } },
    ];
    const pickUses = new Set([0, 1, 2, 3, 10]);
    this.bindGroups = {
      trace: d.createBindGroup({ layout: this.tracePipeline.getBindGroupLayout(0), entries }),
      pick: d.createBindGroup({ layout: this.pickPipeline.getBindGroupLayout(0), entries: entries.filter(e => pickUses.has(e.binding)) }),
      display: d.createBindGroup({
        layout: this.displayPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this.viewBuf } }, { binding: 1, resource: { buffer: b.accum } }],
      }),
    };
  }

  writeFrame(cam, o) {
    const f = new Float32Array(this.frameData), u = new Uint32Array(this.frameData);
    f.set(cam.pos, 0); f[3] = cam.lensRadius;
    f.set(cam.right, 4); f[7] = cam.focusDist;
    f.set(cam.up, 8); f[11] = Math.tan(cam.fov / 2);
    f.set(cam.forward, 12); f[15] = this.width / this.height;
    u[16] = this.width; u[17] = this.height; u[18] = o.seed; u[19] = o.spp;
    u[20] = this.scene.numInstances; u[21] = this.scene.numLights; u[22] = o.maxBounces; u[23] = o.mode;
    u[24] = this.env.gridW; u[25] = this.env.gridH; f[26] = o.envIntensity; f[27] = o.envRotation;
    f[28] = this.scene.lightArea; f[29] = o.envVisible ? 1 : 0; f[30] = o.clampMax; f[31] = o.pickX ?? 0;
    f.set(o.bgColor, 32); f[35] = o.pickY ?? 0;
    u[36] = o.accumulate ? 1 : 0;
    if (o.ground) { f.set(o.ground, 40); f[43] = 1; } else f[43] = 0;
    f.set(this.env.irradiance, 44);
    this.device.queue.writeBuffer(this.frameBuf, 0, this.frameData);
  }

  // Trace o.spp more samples per pixel (unless o.spp is 0), then show the average.
  render(cam, o) {
    if (!this.bindGroups) this.makeBindGroups();
    const d = this.device;
    const enc = d.createCommandEncoder();
    if (o.spp > 0) {
      this.writeFrame(cam, o);
      enc.clearBuffer(this.statsBuf, 0, 4);
      const pass = enc.beginComputePass();
      pass.setPipeline(this.tracePipeline);
      pass.setBindGroup(0, this.bindGroups.trace);
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 4));
      pass.end();
    }
    const measure = o.spp > 0 && !this.statsBusy && o.measure;
    if (measure) enc.copyBufferToBuffer(this.statsBuf, 0, this.statsRead, 0, 32);

    const v = new Uint32Array(this.viewData), vf = new Float32Array(this.viewData);
    v[0] = this.width; v[1] = this.height; v[2] = this.canvas.width; v[3] = this.canvas.height;
    vf[4] = o.samples; vf[5] = o.exposure; v[6] = o.mode; v[7] = o.selected ?? NONE;
    vf.set(o.accent, 8);
    vf[12] = o.look?.[0] ?? 1; vf[13] = o.look?.[1] ?? 1;
    d.queue.writeBuffer(this.viewBuf, 0, this.viewData);
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    rp.setPipeline(this.displayPipeline);
    rp.setBindGroup(0, this.bindGroups.display);
    rp.draw(3);
    rp.end();

    const t0 = performance.now();
    d.queue.submit([enc.finish()]);
    const done = d.queue.onSubmittedWorkDone().then(() => performance.now() - t0);
    let rays = null;
    if (measure) {
      this.statsBusy = true;
      rays = this.statsRead.mapAsync(GPUMapMode.READ).then(() => {
        const n = new Uint32Array(this.statsRead.getMappedRange())[0];
        this.statsRead.unmap();
        this.statsBusy = false;
        return n;
      });
    }
    return { done, rays };
  }

  // For debugging from the console: the raw accumulation buffer (rgb sums, material id bits in w).
  async readAccum() {
    const size = this.width * this.height * 16;
    const read = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.buffers.accum, 0, read, 0, size);
    this.device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(read.getMappedRange().slice(0));
    read.destroy();
    return data;
  }

  async pick(cam, x, y, o) {
    if (!this.bindGroups) this.makeBindGroups();
    while (this.statsBusy) await new Promise(r => setTimeout(r, 4));
    this.statsBusy = true;
    this.writeFrame(cam, { ...o, pickX: x, pickY: y, spp: 0 });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pickPipeline);
    pass.setBindGroup(0, this.bindGroups.pick);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(this.statsBuf, 0, this.statsRead, 0, 32);
    this.device.queue.submit([enc.finish()]);
    await this.statsRead.mapAsync(GPUMapMode.READ);
    const r = new Uint32Array(this.statsRead.getMappedRange().slice(0));
    this.statsRead.unmap();
    this.statsBusy = false;
    if (r[4] === NONE) return null;
    return { instance: r[4], material: r[5], depth: new Float32Array(r.buffer)[6] };
  }
}
