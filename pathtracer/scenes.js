// Scene descriptions: geometry, placements, materials, lighting and a starting camera.

// Measured or commonly used reflectances (linear RGB).
export const PRESETS = {
  gold:    { label: 'Gold', color: [1.0, 0.766, 0.336], roughness: 0.18, metallic: 1 },
  copper:  { label: 'Copper', color: [0.955, 0.638, 0.538], roughness: 0.28, metallic: 1 },
  chrome:  { label: 'Chrome', color: [0.55, 0.556, 0.554], roughness: 0.04, metallic: 1 },
  glass:   { label: 'Glass', color: [1, 1, 1], roughness: 0, transmission: 1, ior: 1.5 },
  frosted: { label: 'Frosted glass', color: [1, 1, 1], roughness: 0.35, transmission: 1, ior: 1.5 },
  amber:   { label: 'Amber', color: [0.95, 0.5, 0.12], roughness: 0.02, transmission: 1, ior: 1.55, density: 2.5 },
  marble:  { label: 'Marble', color: [0.9, 0.88, 0.84], roughness: 0.3, transmission: 1, ior: 1.49, density: 60, scatter: 1 },
  jade:    { label: 'Jade', color: [0.1, 0.42, 0.22], roughness: 0.12, transmission: 1, ior: 1.66, density: 20, scatter: 1 },
  ceramic: { label: 'Glazed ceramic', color: [0.75, 0.08, 0.05], roughness: 0.06 },
  clay:    { label: 'Matte clay', color: [0.8, 0.78, 0.74], roughness: 1 },
};

export function material(m) {
  return {
    color: [0.8, 0.8, 0.8], roughness: 0.5, metallic: 0, transmission: 0, ior: 1.5,
    density: 0, scatter: 0, anisotropy: 0, emission: [0, 0, 0], ...structuredClone(m),
  };
}

// ---------------------------------------------------------------- procedural geometry

class Builder {
  constructor() { this.positions = []; this.normals = []; this.indices = []; this.mat = []; }
  vertex(p, n) { this.positions.push(...p); this.normals.push(...n); return this.positions.length / 3 - 1; }
  // a planar polygon facing `facing` (a point the front side looks towards, or a direction if dir=true)
  polygon(pts, mat, toward, dir = false) {
    const e1 = sub(pts[1], pts[0]), e2 = sub(pts[2], pts[0]);
    let n = norm(cross(e1, e2));
    const want = dir ? toward : sub(toward, pts[0]);
    if (dot(n, want) < 0) { n = n.map(x => -x); pts = [...pts].reverse(); }
    const ids = pts.map(p => this.vertex(p, n));
    for (let i = 1; i + 1 < ids.length; i++) { this.indices.push(ids[0], ids[i], ids[i + 1]); this.mat.push(mat); }
  }
  geometry(flat = false) {
    return {
      positions: new Float32Array(this.positions), indices: new Uint32Array(this.indices),
      normals: flat ? undefined : new Float32Array(this.normals), flat, mat: new Uint32Array(this.mat),
    };
  }
}

const sub = (a, b) => a.map((x, i) => x - b[i]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const l = Math.hypot(...a); return a.map(x => x / l); };

// A photo-studio sweep: floor curving up into a back wall.
export function sweep({ width = 40, front = 16, back = 1.6, radius = 2.2, height = 9, segments = 32 } = {}) {
  const prof = [[front, 0, 0, 1]];               // z, y, normal z, normal y
  for (let i = 0; i <= segments; i++) {
    const t = (i / segments) * Math.PI / 2;
    prof.push([-back - radius * Math.sin(t), radius - radius * Math.cos(t), Math.sin(t), Math.cos(t)]);
  }
  prof.push([-back - radius, height, 1, 0]);
  const b = new Builder();
  const x0 = -width / 2, x1 = width / 2;
  for (let i = 0; i + 1 < prof.length; i++) {
    const [za, ya, nza, nya] = prof[i], [zb, yb, nzb, nyb] = prof[i + 1];
    const a0 = b.vertex([x0, ya, za], [0, nya, nza]), a1 = b.vertex([x1, ya, za], [0, nya, nza]);
    const b0 = b.vertex([x0, yb, zb], [0, nyb, nzb]), b1 = b.vertex([x1, yb, zb], [0, nyb, nzb]);
    b.indices.push(a0, a1, b1, a0, b1, b0); b.mat.push(0, 0);
  }
  const g = b.geometry();
  // make the winding agree with the normals
  for (let t = 0; t < g.indices.length; t += 3) {
    const [i, j, k] = [g.indices[t], g.indices[t + 1], g.indices[t + 2]];
    const p = q => [g.positions[3 * q], g.positions[3 * q + 1], g.positions[3 * q + 2]];
    const n = cross(sub(p(j), p(i)), sub(p(k), p(i)));
    if (dot(n, [0, g.normals[3 * i + 1], g.normals[3 * i + 2]]) < 0) { g.indices[t + 1] = k; g.indices[t + 2] = j; }
  }
  return g;
}

function quad(size, y = 0) {
  const b = new Builder(), s = size / 2;
  b.polygon([[-s, y, -s], [s, y, -s], [s, y, s], [-s, y, s]], 0, [0, 1, 0], true);
  return b.geometry(true);
}

// The Cornell box as measured (graphics.cornell.edu/online/box/data.html), millimetres -> metres.
function cornell() {
  const mm = pts => pts.map(p => p.map(x => x / 1000));
  const b = new Builder();
  const inside = [0.278, 0.274, 0.28];
  b.polygon(mm([[552.8, 0, 0], [0, 0, 0], [0, 0, 559.2], [549.6, 0, 559.2]]), 0, inside);            // floor
  b.polygon(mm([[556, 548.8, 0], [556, 548.8, 559.2], [0, 548.8, 559.2], [0, 548.8, 0]]), 1, inside);  // ceiling
  b.polygon(mm([[549.6, 0, 559.2], [0, 0, 559.2], [0, 548.8, 559.2], [556, 548.8, 559.2]]), 2, inside); // back
  b.polygon(mm([[552.8, 0, 0], [549.6, 0, 559.2], [556, 548.8, 559.2], [556, 548.8, 0]]), 3, inside);  // left, red
  b.polygon(mm([[0, 0, 559.2], [0, 0, 0], [0, 548.8, 0], [0, 548.8, 559.2]]), 4, inside);              // right, green
  b.polygon(mm([[343, 548.7, 227], [343, 548.7, 332], [213, 548.7, 332], [213, 548.7, 227]]), 5, [0, -1, 0], true);
  const block = (top, h, mat) => {
    const c = top.reduce((s, p) => s.map((x, i) => x + p[i] / 4), [0, 0, 0]);
    const centre = [c[0] / 1000, h / 2000, c[2] / 1000];
    const away = p => sub(p, centre).map((x, i) => p[i] + x);   // a point outside, past this face
    const faces = [top.map(p => [p[0], h, p[2]])];
    for (let i = 0; i < 4; i++) {
      const p = top[i], q = top[(i + 1) % 4];
      faces.push([[p[0], 0, p[2]], [q[0], 0, q[2]], [q[0], h, q[2]], [p[0], h, p[2]]]);
    }
    for (const f of faces) { const m = mm(f); const mid = m.reduce((s, p) => s.map((x, i) => x + p[i] / 4), [0, 0, 0]); b.polygon(m, mat, away(mid)); }
  };
  block([[130, 165, 65], [82, 165, 225], [240, 165, 272], [290, 165, 114]], 165, 6);
  block([[423, 330, 247], [265, 330, 296], [314, 330, 456], [472, 330, 406]], 330, 7);
  return b.geometry(true);
}

function mulberry(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The cover of Peter Shirley's "Ray Tracing in One Weekend".
function weekendSpheres() {
  const rnd = mulberry(7);
  const spheres = [], mats = [];
  const add = (c, r, m) => { spheres.push(...c, r); mats.push(material(m)); };
  for (let a = -11; a < 11; a++) for (let b = -11; b < 11; b++) {
    const choose = rnd();
    const c = [a + 0.9 * rnd(), 0.2, b + 0.9 * rnd()];
    if (Math.hypot(c[0] - 4, c[1] - 0.2, c[2]) <= 0.9) continue;
    if (choose < 0.8) add(c, 0.2, { color: [rnd() * rnd(), rnd() * rnd(), rnd() * rnd()], roughness: 1 });
    else if (choose < 0.95) add(c, 0.2, { color: [0.5 + rnd() / 2, 0.5 + rnd() / 2, 0.5 + rnd() / 2], roughness: rnd() / 2, metallic: 1 });
    else add(c, 0.2, PRESETS.glass);
  }
  add([0, 1, 0], 1, PRESETS.glass);
  add([-4, 1, 0], 1, { color: [0.4, 0.2, 0.1], roughness: 1 });
  add([4, 1, 0], 1, { color: [0.7, 0.6, 0.5], roughness: 0, metallic: 1 });
  return { spheres: new Float32Array(spheres), mat: new Uint32Array(mats.map((_, i) => i)), materials: mats };
}

// ---------------------------------------------------------------- objects

// Ten objects to stand in any place. Meshes are normalized to one unit tall; `height` is metres.
export const OBJECTS = {
  dragon: { label: 'Stanford dragon', height: 0.9, yaw: -28, material: 'jade',
    note: 'Laser scanned at Stanford in 1996: 870,616 triangles.' },
  buddha: { label: 'Happy Buddha', height: 1.0, yaw: 20, material: 'gold',
    note: 'Stanford, 1996: 1,086,299 triangles, the most detailed object here.' },
  lucy: { label: 'Lucy', height: 1.1, yaw: 200, material: 'marble',
    note: 'A Stanford scan of an angel statue, 28 million triangles in full; one million here.' },
  armadillo: { label: 'Armadillo', height: 0.9, yaw: 200, material: 'copper',
    note: 'A toy armadillo scanned at Stanford in 1996: 345,944 triangles.' },
  bunny: { label: 'Stanford bunny', height: 0.55, yaw: 10, material: 'glass',
    note: 'The first of the Stanford scans, from 1994: 69,674 triangles.' },
  teapot: { label: 'Utah teapot', height: 0.42, yaw: 30, material: 'ceramic',
    note: 'Martin Newell’s 1975 teapot, built from 32 Bézier patches; 36,672 triangles here.' },
  spot: { label: 'Spot the cow', height: 0.6, yaw: 70, material: 'amber',
    note: 'Keenan Crane’s cow, a newer favourite among graphics test models.' },
  bust: { label: 'Marble bust', height: 0.7, yaw: 10, material: 'marble',
    note: 'A photogrammetry scan of a classical bust, from Poly Haven.' },
  horse: { label: 'Horse statue', height: 0.8, yaw: 60, material: 'copper',
    note: 'A photogrammetry scan of a horse statuette, from Poly Haven.' },
  duck: { label: 'Rubber duck', height: 0.32, yaw: 50, material: 'chrome',
    note: 'Every renderer needs a rubber duck. From Poly Haven.' },
};

// ---------------------------------------------------------------- places

// HDR photographs from Poly Haven. Each lights the scene and is its backdrop; the floor is the
// photo's own ground, projected from the height the camera stood at (see groundAlbedo in trace.wgsl).
export const PLACES = {
  venice: { label: 'Venice at sunset', url: 'env/venice_sunset_2k.hdr', rotation: 0.71, intensity: 1, height: 1.7 },
  shanghai: { label: 'Shanghai at night', url: 'env/shanghai_bund_2k.hdr', rotation: 0.66, intensity: 1, height: 1.7 },
  studio: { label: 'Photo studio', url: 'env/brown_photostudio_02_2k.hdr', rotation: 0, intensity: 1, height: 1.5 },
};

const projectedGround = {
  geometry: 'ground', name: 'Ground', pickable: false,
  materials: [{ color: [0.5, 0.5, 0.5], roughness: 1, projected: true }],
};

function objectScene(key, o) {
  return {
    label: o.label, note: o.note, places: true, object: true,
    geometry: { [key]: { mesh: `model/${key}.mesh` }, ground: { triangles: () => quad(400) } },
    instances: [
      { geometry: key, yaw: o.yaw, scale: o.height, materials: [PRESETS[o.material]], name: o.label },
      projectedGround,
    ],
    camera: { target: [0, o.height / 2, 0], yaw: 18, pitch: 7, distance: 2.5, fov: 36, aperture: 0.03, focus: 2.5 },
  };
}

// Point the camera at the object from its scene's angle, far enough back to fit it in view.
export function frameCamera(cam, bounds, aspect) {
  const [x0, y0, z0, x1, y1, z1] = bounds;
  const r = 0.5 * Math.hypot(x1 - x0, y1 - y0, z1 - z0);
  const half = (cam.fov * Math.PI) / 360;
  const fit = Math.min(half, Math.atan(Math.tan(half) * aspect));
  const distance = (r / Math.sin(fit)) * 0.92;
  return { ...cam, target: [(x0 + x1) / 2, (y0 + y1) / 2 * 0.95, (z0 + z1) / 2], distance, focus: distance, aperture: 0.012 * distance };
}

// ---------------------------------------------------------------- scenes

export const SCENES = {
  ...Object.fromEntries(Object.entries(OBJECTS).map(([k, o]) => [k, objectScene(k, o)])),
  cornell: {
    label: 'Cornell box',
    note: 'Built to the published measurements of the real Cornell box, the scene renderers have been checked against since 1984. All of its light comes from the small panel in the ceiling.',
    geometry: { cornell: { triangles: () => cornell() } },
    instances: [{
      geometry: 'cornell', light: true, name: 'Cornell box',
      materials: [
        { color: [0.725, 0.71, 0.68], roughness: 1, name: 'Floor' },
        { color: [0.725, 0.71, 0.68], roughness: 1, name: 'Ceiling' },
        { color: [0.725, 0.71, 0.68], roughness: 1, name: 'Back wall' },
        { color: [0.63, 0.065, 0.05], roughness: 1, name: 'Red wall' },
        { color: [0.14, 0.45, 0.091], roughness: 1, name: 'Green wall' },
        { color: [0.78, 0.78, 0.78], roughness: 1, emission: [17, 12, 4], name: 'Light', pickable: false },
        { color: [0.725, 0.71, 0.68], roughness: 1, name: 'Short block' },
        { color: [0.725, 0.71, 0.68], roughness: 1, name: 'Tall block' },
      ],
    }],
    env: null,
    exposure: 0.7,
    camera: { target: [0.278, 0.273, 0.28], yaw: 180, pitch: 0, distance: 1.08, fov: 39.3, aperture: 0.012, focus: 1.0 },
  },
  // Not in the menu: the scene tools/compare.html benchmarks against three-gpu-pathtracer.
  bench: {
    label: 'Benchmark', hidden: true,
    geometry: { dragon: { mesh: 'model/dragon.mesh' }, sweep: { triangles: () => sweep() } },
    instances: [
      { geometry: 'dragon', yaw: -28, scale: 1.25, materials: [PRESETS.jade], name: 'Dragon' },
      { geometry: 'sweep', materials: [{ color: [0.62, 0.62, 0.6], roughness: 0.9 }], name: 'Backdrop' },
    ],
    env: { url: 'env/studio_small_09_1k.hdr', intensity: 1.0, rotation: 0.62, visible: false, background: [0.4, 0.4, 0.39] },
    camera: { target: [0, 0.55, 0], yaw: 18, pitch: 9, distance: 3.3, fov: 38, aperture: 0, focus: 3.3 },
  },
  spheres: {
    label: 'One Weekend',
    note: 'The cover of Peter Shirley’s Ray Tracing in One Weekend: 484 spheres under a real sky, with a shallow depth of field.',
    geometry: {
      spheres: { spheres: () => weekendSpheres() },
      ground: { triangles: () => quad(400) },
    },
    instances: [
      { geometry: 'spheres', name: 'Spheres', materials: 'fromGeometry' },
      { geometry: 'ground', materials: [{ color: [0.5, 0.5, 0.5], roughness: 1 }], name: 'Ground', pickable: false },
    ],
    env: { url: 'env/kloofendal_puresky_1k.hdr', intensity: 1.0, rotation: 0.2, visible: true },
    camera: { target: [0, 0, 0], yaw: 77, pitch: 8.6, distance: 13.5, fov: 20, aperture: 0.1, focus: 10.4 },
  },
};
