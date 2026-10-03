// Builds every scene in Node, validates the GPU buffers, and walks rays through a JS copy of
// the shader's traversal to find the most loop iterations any ray needs.
// Run before testing a new scene on the GPU:  node pathtracer/tools/check_scenes.mjs
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { SCENES, frameCamera } from '../scenes.js';
import { decodeMesh, packTriangles, packSpheres, KIND_SPHERES } from '../mesh.js';
import { assemble } from '../assemble.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const STACK = 40;

async function build(g) {
  if (g.mesh) return packTriangles(await decodeMesh(readFileSync(join(root, g.mesh))));
  if (g.triangles) return packTriangles(g.triangles());
  const s = g.spheres();
  return { ...packSpheres(s.spheres, s.mat), materials: s.materials };
}

function makeTracer({ nodes, prims, instances, numInstances }) {
  const nu = new Uint32Array(nodes.buffer), iu = new Uint32Array(instances.buffer);
  const P = prims;
  const tri = (p, o, d, best) => {
    const q = 16 * p;
    const e1 = [P[q + 4], P[q + 5], P[q + 6]], e2 = [P[q + 8], P[q + 9], P[q + 10]];
    const pv = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
    const det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
    if (det === 0) return best;
    const inv = 1 / det, tv = [o[0] - P[q], o[1] - P[q + 1], o[2] - P[q + 2]];
    const u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
    if (u < 0 || u > 1) return best;
    const qv = [tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]];
    const v = (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]) * inv;
    if (v < 0 || u + v > 1) return best;
    const t = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
    return t > 0 && t < best ? t : best;
  };
  const sphere = (p, o, d, best) => {
    const q = 16 * p, f = [o[0] - P[q], o[1] - P[q + 1], o[2] - P[q + 2]];
    const a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    const tc = -(f[0] * d[0] + f[1] * d[1] + f[2] * d[2]) / a;
    const l = [f[0] + tc * d[0], f[1] + tc * d[1], f[2] + tc * d[2]];
    const h2 = P[q + 3] * P[q + 3] - (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
    if (h2 < 0) return best;
    const s = Math.sqrt(h2 / a);
    let t = tc - s; if (t <= 0) t = tc + s;
    return t > 0 && t < best ? t : best;
  };
  const slab = (o, inv, lo, hi, tmax) => {
    let t0 = 0, t1 = tmax;
    for (let k = 0; k < 3; k++) {
      const a = (lo[k] - o[k]) * inv[k], b = (hi[k] - o[k]) * inv[k];
      t0 = Math.max(t0, Math.min(a, b)); t1 = Math.min(t1, Math.max(a, b));
    }
    return t0 <= t1 ? t0 : Infinity;
  };
  const safeInv = d => d.map(x => 1 / (Math.abs(x) < 1e-12 ? (x >= 0 ? 1e-12 : -1e-12) : x));

  // returns { t, iterations (max over instances, as the shader's per-traversal guard counts) }
  return (ro, rd) => {
    let best = 1e20, maxIter = 0;
    const invw = safeInv(rd);
    for (let i = 0; i < numInstances; i++) {
      const b = i * 24;
      const lo = [instances[b + 12], instances[b + 13], instances[b + 14]], hi = [instances[b + 16], instances[b + 17], instances[b + 18]];
      if (slab(ro, invw, lo, hi, best) === Infinity) continue;
      const r = k => [instances[b + 4 * k], instances[b + 4 * k + 1], instances[b + 4 * k + 2], instances[b + 4 * k + 3]];
      const rows = [r(0), r(1), r(2)];
      const o = rows.map(m => m[0] * ro[0] + m[1] * ro[1] + m[2] * ro[2] + m[3]);
      const d = rows.map(m => m[0] * rd[0] + m[1] * rd[1] + m[2] * rd[2]);
      const inv = safeInv(d);
      const isect = iu[b + 19] === KIND_SPHERES ? sphere : tri;
      const stack = [];
      let node = iu[b + 15], iter = 0;
      for (;;) {
        iter++;
        if (iter > 1e6) throw new Error('traversal did not terminate');
        const q = node * 16;
        const tl = slab(o, inv, [nodes[q], nodes[q + 1], nodes[q + 2]], [nodes[q + 4], nodes[q + 5], nodes[q + 6]], best);
        const tr = slab(o, inv, [nodes[q + 8], nodes[q + 9], nodes[q + 10]], [nodes[q + 12], nodes[q + 13], nodes[q + 14]], best);
        let goL = tl !== Infinity, goR = tr !== Infinity;
        if (goL && nu[q + 7]) { for (let p = nu[q + 3]; p < nu[q + 3] + nu[q + 7]; p++) best = isect(p, o, d, best); goL = false; }
        if (goR && nu[q + 15]) { for (let p = nu[q + 11]; p < nu[q + 11] + nu[q + 15]; p++) best = isect(p, o, d, best); goR = false; }
        if (goL && goR) {
          const [near, far] = tr < tl ? [nu[q + 11], nu[q + 3]] : [nu[q + 3], nu[q + 11]];
          if (stack.length < STACK) stack.push(far);
          node = near;
        } else if (goL) node = nu[q + 3];
        else if (goR) node = nu[q + 11];
        else if (stack.length) node = stack.pop();
        else break;
      }
      maxIter = Math.max(maxIter, iter);
    }
    return { t: best, iter: maxIter };
  };
}

function cameraRays(cam, n) {
  const yaw = cam.yaw * Math.PI / 180, pitch = cam.pitch * Math.PI / 180;
  const dir = [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
  const pos = cam.target.map((t, k) => t + dir[k] * cam.distance);
  const f = dir.map(x => -x);
  const rl = Math.hypot(f[2], f[0]), right = [-f[2] / rl, 0, f[0] / rl];
  const up = [right[1] * f[2] - right[2] * f[1], right[2] * f[0] - right[0] * f[2], right[0] * f[1] - right[1] * f[0]];
  const th = Math.tan(cam.fov * Math.PI / 360), aspect = 16 / 10;
  const rays = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() * 2 - 1) * th * aspect, y = (Math.random() * 2 - 1) * th;
    const d = f.map((v, k) => v + right[k] * x + up[k] * y), l = Math.hypot(...d);
    rays.push([pos, d.map(v => v / l)]);
  }
  return rays;
}

function randomDir() {
  const z = Math.random() * 2 - 1, a = Math.random() * 2 * Math.PI, r = Math.sqrt(1 - z * z);
  return [r * Math.cos(a), r * Math.sin(a), z];
}

let worst = 0;
for (const [name, def] of Object.entries(SCENES)) {
  const geoms = {};
  for (const [key, g] of Object.entries(def.geometry)) geoms[key] = await build(g);
  const { gpu } = assemble(def, geoms);           // throws if the buffers are unsafe
  const trace = makeTracer(gpu);
  const i = gpu.instances;
  const camera = def.object ? frameCamera(def.camera, [i[12], i[13], i[14], i[16], i[17], i[18]], 16 / 10) : def.camera;
  const iters = [];
  let hits = 0;
  for (const [o, d] of cameraRays(camera, 20000)) {
    const h = trace(o, d);
    iters.push(h.iter);
    if (h.t < 1e20) {
      hits++;
      const p = o.map((v, k) => v + d[k] * h.t * 0.999);   // bounce from just before the hit
      iters.push(trace(p, randomDir()).iter);
    }
  }
  iters.sort((a, b) => a - b);
  const max = iters[iters.length - 1];
  worst = Math.max(worst, max);
  console.log(`${name.padEnd(8)} ok: ${gpu.nodes.length / 16} nodes, ${gpu.prims.length / 16} prims, ` +
    `${hits}/20000 camera hits, iterations per traversal median ${iters[iters.length >> 1]}, ` +
    `p99.9 ${iters[Math.floor(iters.length * 0.999)]}, max ${max}`);
}
console.log(`worst traversal: ${worst} iterations`);
