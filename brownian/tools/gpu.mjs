// A WebGPU device in Node (Dawn), so the tools run the page's shaders.
// Keep the GPU object alive: if it is garbage-collected, Dawn segfaults on the next event.
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);

const keep = [];

export async function getDevice() {
  const gpu = create([]);
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  keep.push(gpu, adapter);
  if (!adapter) throw new Error('no WebGPU adapter');
  return adapter.requestDevice();
}

// Runs fn and throws if it raised a WebGPU validation or out-of-memory error.
export async function checkErrors(device, fn) {
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  const out = await fn();
  const errors = [await device.popErrorScope(), await device.popErrorScope()].filter(Boolean);
  if (errors.length) throw new Error(errors.map(e => e.message).join('\n'));
  return out;
}
