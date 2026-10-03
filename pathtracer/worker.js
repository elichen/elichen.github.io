// Heavy lifting off the main thread: mesh decoding, BVH builds and HDR decoding.
import { decodeMesh, packTriangles, packSpheres } from './mesh.js';
import { parseHDR, buildEnvironment } from './hdr.js';

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.arrayBuffer();
}

self.onmessage = async e => {
  const { id, type } = e.data;
  try {
    let result;
    if (type === 'mesh') result = packTriangles(await decodeMesh(await fetchBytes(e.data.url)));
    else if (type === 'triangles') result = packTriangles(e.data.geometry);
    else if (type === 'spheres') result = packSpheres(e.data.spheres, e.data.mat);
    else if (type === 'hdr') result = buildEnvironment(parseHDR(await fetchBytes(e.data.url)));
    else throw new Error(`unknown job ${type}`);
    const transfer = Object.values(result).filter(v => ArrayBuffer.isView(v)).map(v => v.buffer);
    self.postMessage({ id, result }, transfer);
  } catch (err) {
    self.postMessage({ id, error: err.message || String(err) });
  }
};
