// Mesh decoding and packing into GPU-ready bottom-level acceleration structures.
//
// Every primitive is four vec4s (64 bytes):
//   triangle: [v0.xyz, mat] [e1.xyz, 0] [e2.xyz, 0] [n0, n1, n2 (octahedral snorm16x2), 0]
//   sphere:   [centre.xyz, radius] [mat, 0, 0, 0] [0] [0]
// stored in BVH leaf order so a leaf's primitives are contiguous.

import { buildBVH } from './bvh.js';

export const KIND_TRIANGLES = 0;
export const KIND_SPHERES = 1;

// The .mesh format written by train/pack_mesh.py (gzipped).
export async function decodeMesh(buf) {
  let bytes = new Uint8Array(buf);
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== 0x3148534d) throw new Error('not a MSH1 mesh');
  const nv = dv.getUint32(4, true), nt = dv.getUint32(8, true);
  const lo = [0, 1, 2].map(k => dv.getFloat32(12 + 4 * k, true));
  const hi = [0, 1, 2].map(k => dv.getFloat32(24 + 4 * k, true));
  let off = 40;
  const positions = new Float32Array(nv * 3);
  const acc = [0, 0, 0];
  for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) {
    const z = dv.getUint16(off, true); off += 2;
    acc[k] = (acc[k] + ((z >>> 1) ^ -(z & 1))) & 0xffff;
    positions[3 * i + k] = lo[k] + (acc[k] / 65535) * (hi[k] - lo[k]);
  }
  const indices = new Uint32Array(nt * 3);
  let next = 0;
  for (let i = 0; i < nt * 3; i++) {
    let code = 0, shift = 0, b;
    do { b = bytes[off++]; code |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80);
    const idx = next - code;
    indices[i] = idx;
    if (idx === next) next++;
  }
  return { positions, indices };
}

export function smoothNormals(positions, indices) {
  const n = new Float32Array(positions.length);
  for (let t = 0; t < indices.length; t += 3) {
    const a = 3 * indices[t], b = 3 * indices[t + 1], c = 3 * indices[t + 2];
    const e1x = positions[b] - positions[a], e1y = positions[b + 1] - positions[a + 1], e1z = positions[b + 2] - positions[a + 2];
    const e2x = positions[c] - positions[a], e2y = positions[c + 1] - positions[a + 1], e2z = positions[c + 2] - positions[a + 2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;  // area weighted
    for (const v of [a, b, c]) { n[v] += nx; n[v + 1] += ny; n[v + 2] += nz; }
  }
  for (let i = 0; i < n.length; i += 3) {
    const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1;
    n[i] /= l; n[i + 1] /= l; n[i + 2] /= l;
  }
  return n;
}

function octPack(x, y, z) {
  const s = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
  x /= s; y /= s;
  if (z < 0) {
    const ox = (1 - Math.abs(y)) * (x >= 0 ? 1 : -1);
    y = (1 - Math.abs(x)) * (y >= 0 ? 1 : -1);
    x = ox;
  }
  const q = v => Math.round(Math.max(-1, Math.min(1, v)) * 32767) & 0xffff;
  return (q(x) | (q(y) << 16)) >>> 0;
}

// positions/indices: indexed triangles. normals: per-vertex (optional; smooth if omitted, face if flat).
// mat: per-triangle local material index (Uint32Array) or a single number.
export function packTriangles({ positions, indices, normals, flat = false, mat = 0 }) {
  const t0 = performance.now();
  const nt = indices.length / 3;
  if (!flat && !normals) normals = smoothNormals(positions, indices);
  const bmin = new Float32Array(nt * 3), bmax = new Float32Array(nt * 3), cent = new Float32Array(nt * 3);
  for (let t = 0; t < nt; t++) for (let k = 0; k < 3; k++) {
    const a = positions[3 * indices[3 * t] + k], b = positions[3 * indices[3 * t + 1] + k], c = positions[3 * indices[3 * t + 2] + k];
    bmin[3 * t + k] = Math.min(a, b, c);
    bmax[3 * t + k] = Math.max(a, b, c);
    cent[3 * t + k] = (a + b + c) / 3;
  }
  const bvh = buildBVH(bmin, bmax, cent, nt);
  const prims = new Float32Array(nt * 16), pu = new Uint32Array(prims.buffer);
  for (let i = 0; i < nt; i++) {
    const t = bvh.order[i], o = 16 * i;
    const a = 3 * indices[3 * t], b = 3 * indices[3 * t + 1], c = 3 * indices[3 * t + 2];
    for (let k = 0; k < 3; k++) {
      prims[o + k] = positions[a + k];
      prims[o + 4 + k] = positions[b + k] - positions[a + k];
      prims[o + 8 + k] = positions[c + k] - positions[a + k];
    }
    pu[o + 3] = typeof mat === 'number' ? mat : mat[t];
    if (flat) {
      const e1 = [prims[o + 4], prims[o + 5], prims[o + 6]], e2 = [prims[o + 8], prims[o + 9], prims[o + 10]];
      const nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
      pu[o + 12] = pu[o + 13] = pu[o + 14] = octPack(nx, ny, nz);
    } else {
      pu[o + 12] = octPack(normals[a], normals[a + 1], normals[a + 2]);
      pu[o + 13] = octPack(normals[b], normals[b + 1], normals[b + 2]);
      pu[o + 14] = octPack(normals[c], normals[c + 1], normals[c + 2]);
    }
  }
  return {
    kind: KIND_TRIANGLES, prims, primCount: nt, nodes: bvh.nodes, nodeCount: bvh.nodeCount,
    bounds: bvh.bounds, maxDepth: bvh.maxDepth, buildMs: performance.now() - t0,
  };
}

// spheres: Float32Array(4n) of centre.xyz, radius; mat: Uint32Array(n) local material per sphere
export function packSpheres(spheres, mat) {
  const t0 = performance.now();
  const n = spheres.length / 4;
  const bmin = new Float32Array(n * 3), bmax = new Float32Array(n * 3), cent = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) {
    const c = spheres[4 * i + k], r = spheres[4 * i + 3];
    bmin[3 * i + k] = c - r; bmax[3 * i + k] = c + r; cent[3 * i + k] = c;
  }
  const bvh = buildBVH(bmin, bmax, cent, n);
  const prims = new Float32Array(n * 16), pu = new Uint32Array(prims.buffer);
  for (let i = 0; i < n; i++) {
    const s = bvh.order[i];
    prims.set(spheres.subarray(4 * s, 4 * s + 4), 16 * i);
    pu[16 * i + 4] = mat[s];
  }
  return {
    kind: KIND_SPHERES, prims, primCount: n, nodes: bvh.nodes, nodeCount: bvh.nodeCount,
    bounds: bvh.bounds, maxDepth: bvh.maxDepth, buildMs: performance.now() - t0,
  };
}
