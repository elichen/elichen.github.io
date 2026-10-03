// Turns a scene description plus built geometry into the flat buffers the GPU reads,
// and checks them first: a malformed tree can trap the shader in a loop and hang the GPU.

import { KIND_TRIANGLES, KIND_SPHERES } from './mesh.js';
import { material } from './scenes.js';

export const FLAG_LIGHT = 1;
export const MAX_LEAF_PRIMS = 64;

export function transform({ position = [0, 0, 0], yaw = 0, scale = 1 }) {
  const a = (yaw * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  const o2w = [[c * scale, 0, s * scale, position[0]], [0, scale, 0, position[1]], [-s * scale, 0, c * scale, position[2]]];
  const L = [[c / scale, 0, -s / scale], [0, 1 / scale, 0], [s / scale, 0, c / scale]];
  const w2o = L.map(r => [...r, -(r[0] * position[0] + r[1] * position[1] + r[2] * position[2])]);
  return { o2w, w2o };
}

const apply = (m, p) => m.map(r => r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3]);
const applyDir = (m, d) => m.map(r => r[0] * d[0] + r[1] * d[1] + r[2] * d[2]);

// For a scattering medium, `color` is the colour the material should look overall. Light inside
// scatters many times, so each event needs an albedo much closer to 1; this inverts that
// relationship (Chiang, Kutz & Burley, "Practical and Controllable Subsurface Scattering", 2016).
const singleScatterAlbedo = A => 1 - (4.09712 + 4.20863 * A - Math.sqrt(9.59217 + 41.6808 * A + 17.7126 * A * A)) ** 2;

export function packMaterials(list) {
  const f = new Float32Array(Math.max(1, list.length) * 16);
  list.forEach((m, i) => {
    const o = i * 16;
    f.set(m.scatter ? m.color.map(singleScatterAlbedo) : m.color, o); f[o + 3] = m.roughness;
    f.set(m.emission, o + 4); f[o + 7] = m.metallic;
    f[o + 8] = m.transmission; f[o + 9] = m.ior; f[o + 10] = m.density; f[o + 11] = m.scatter;
    f[o + 12] = m.anisotropy; f[o + 13] = m.projected ? 1 : 0;
  });
  return f;
}

export function assemble(def, geoms) {
  const used = [...new Set(def.instances.map(i => i.geometry))];
  const base = {};
  let nodeCount = 0, primCount = 0, tris = 0;
  for (const key of used) {
    base[key] = { node: nodeCount, prim: primCount };
    nodeCount += geoms[key].nodeCount;
    primCount += geoms[key].primCount;
    if (geoms[key].kind !== KIND_SPHERES) tris += geoms[key].primCount;
  }
  const nodes = new Float32Array(nodeCount * 16), nu = new Uint32Array(nodes.buffer);
  const prims = new Float32Array(primCount * 16);
  for (const key of used) {
    const g = geoms[key], b = base[key];
    nodes.set(g.nodes, b.node * 16);
    prims.set(g.prims, b.prim * 16);
    for (let n = b.node; n < b.node + g.nodeCount; n++) for (const o of [n * 16, n * 16 + 8]) {
      nu[o + 3] += nu[o + 7] > 0 ? b.prim : b.node;
    }
  }

  const materials = [], owners = [];
  const inst = new Float32Array(def.instances.length * 24), iu = new Uint32Array(inst.buffer);
  const lights = [];
  def.instances.forEach((d, i) => {
    const g = geoms[d.geometry];
    const { o2w, w2o } = transform(d);
    const matBase = materials.length;
    const list = d.materials === 'fromGeometry' ? g.materials : d.materials;
    list.forEach((m, k) => {
      materials.push(material(m));
      owners.push({
        instance: i, name: m.name || (list.length > 1 ? `${d.name} ${k + 1}` : d.name),
        pickable: d.pickable !== false && m.pickable !== false,
      });
    });
    const o = i * 24;
    for (let r = 0; r < 3; r++) inst.set(w2o[r], o + 4 * r);
    // world bounds from the eight corners of the local box
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let c = 0; c < 8; c++) {
      const p = apply(o2w, [0, 1, 2].map(k => g.bounds[(c >> k) & 1 ? k + 3 : k]));
      for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
    }
    inst.set(lo, o + 12); iu[o + 15] = base[d.geometry].node;
    inst.set(hi, o + 16); iu[o + 19] = g.kind;
    iu[o + 20] = matBase; iu[o + 21] = d.light ? FLAG_LIGHT : 0;
    iu[o + 22] = base[d.geometry].prim; iu[o + 23] = base[d.geometry].prim + g.primCount;  // for validation only

    if (d.light) {
      const pu = new Uint32Array(g.prims.buffer, g.prims.byteOffset, g.prims.length);
      for (let p = 0; p < g.primCount; p++) {
        const m = matBase + pu[16 * p + 3];
        if (!materials[m].emission.some(x => x > 0)) continue;
        const q = 16 * p, P = g.prims;
        const v0 = apply(o2w, [P[q], P[q + 1], P[q + 2]]);
        const e1 = applyDir(o2w, [P[q + 4], P[q + 5], P[q + 6]]), e2 = applyDir(o2w, [P[q + 8], P[q + 9], P[q + 10]]);
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const len = Math.hypot(...n);
        lights.push({ v0, e1, e2, n: n.map(x => x / len), area: len / 2, mat: m });
      }
    }
  });

  const lightArea = lights.reduce((s, l) => s + l.area, 0);
  const lf = new Float32Array(Math.max(1, lights.length) * 16), lu = new Uint32Array(lf.buffer);
  let cum = 0;
  lights.forEach((l, i) => {
    cum += l.area;
    lf.set(l.v0, 16 * i); lu[16 * i + 3] = l.mat;
    lf.set(l.e1, 16 * i + 4); lf[16 * i + 7] = i === lights.length - 1 ? 1 : cum / lightArea;
    lf.set(l.e2, 16 * i + 8); lf[16 * i + 11] = l.area;
    lf.set(l.n, 16 * i + 12);
  });

  const gpu = {
    nodes, prims, instances: inst, materials: packMaterials(materials), lights: lf,
    numInstances: def.instances.length, numLights: lights.length, lightArea: lightArea || 1,
    numMaterials: materials.length,
  };
  validateScene(gpu);
  return { gpu, materials, owners, tris, spheres: primCount - tris, nodeCount };
}

// Throws unless the shader is guaranteed to terminate on these buffers. The key rule:
// every inner child sits at a higher index than its parent, so traversal can never cycle.
export function validateScene({ nodes, prims, instances, numInstances, lights, numLights, numMaterials }) {
  const fail = msg => { throw new Error(`scene failed validation: ${msg}`); };
  const nu = new Uint32Array(nodes.buffer, nodes.byteOffset, nodes.length);
  const nodeCount = nodes.length / 16, primCount = prims.length / 16;
  if (!Number.isInteger(nodeCount) || !Number.isInteger(primCount)) fail('buffer sizes are not whole records');
  for (let n = 0; n < nodeCount; n++) for (const o of [n * 16, n * 16 + 8]) {
    for (let k = 0; k < 3; k++) {
      const lo = nodes[o + k], hi = nodes[o + 4 + k];
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo > hi) fail(`node ${n} has a bad box`);
    }
    const a = nu[o + 3], b = nu[o + 7];
    if (b > 0) {
      if (b > MAX_LEAF_PRIMS || a + b > primCount) fail(`node ${n} has a bad leaf [${a}, +${b})`);
    } else if (a <= n || a >= nodeCount) {
      fail(`node ${n} points to node ${a}; children must come after their parent`);
    }
  }
  const iu = new Uint32Array(instances.buffer, instances.byteOffset, instances.length);
  const pu = new Uint32Array(prims.buffer, prims.byteOffset, prims.length);
  for (let i = 0; i < numInstances; i++) {
    const o = i * 24, root = iu[o + 15], kind = iu[o + 19], matBase = iu[o + 20];
    if (root >= nodeCount) fail(`instance ${i} has root ${root} of ${nodeCount}`);
    if (kind !== KIND_TRIANGLES && kind !== KIND_SPHERES) fail(`instance ${i} has kind ${kind}`);
    for (let k = 0; k < 12; k++) if (!Number.isFinite(instances[o + k])) fail(`instance ${i} has a bad transform`);
    for (let p = iu[o + 22]; p < iu[o + 23]; p++) {
      const local = kind === KIND_SPHERES ? pu[16 * p + 4] : pu[16 * p + 3];
      if (matBase + local >= numMaterials) fail(`instance ${i} primitive ${p} uses missing material ${matBase + local}`);
    }
  }
  const lu = new Uint32Array(lights.buffer, lights.byteOffset, lights.length);
  for (let l = 0; l < numLights; l++) if (lu[16 * l + 3] >= numMaterials) fail(`light ${l} uses a missing material`);
}
