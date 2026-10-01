// Three.js view of one or more drums. The membrane is the finite-element mesh,
// displaced by its modes (slowed down so the eye can follow them). With a
// single mode isolated, sand drifts off the moving parts of the drumhead and
// collects on the nodal lines, like Chladni's plates.
//
// Drum coordinates (x, y) sit at world (cx + x s, height, -y s).

import * as THREE from 'three';

export const PALETTE = {
  plate: 0x232524,     // darkest bare plate
  steel: 0x373f56,     // blue steel
  rim: 0x8c8279,       // vibration generator housing
  head: 0x47505c,      // median plate
  sand: 0xe0e0e0,      // sand
  green: 0x37885b,     // the green sand bin
};

const VIS_F = 0.8;     // visual frequency of the lowest mode, Hz

function polyArea(p) {
  let a = 0;
  for (let i = 0, n = p.length / 2; i < n; i++) {
    const j = (i + 1) % n;
    a += p[2 * i] * p[2 * j + 1] - p[2 * j] * p[2 * i + 1];
  }
  return a / 2;
}

function centroid(p) {
  let a = 0, cx = 0, cy = 0;
  for (let i = 0, n = p.length / 2; i < n; i++) {
    const j = (i + 1) % n, cr = p[2 * i] * p[2 * j + 1] - p[2 * j] * p[2 * i + 1];
    a += cr; cx += (p[2 * i] + p[2 * j]) * cr; cy += (p[2 * i + 1] + p[2 * j + 1]) * cr;
  }
  return [cx / (3 * a), cy / (3 * a)];
}

// Uniform grid over a mesh for locating the triangle under a point.
class Locator {
  constructor(nodes, tri) {
    this.nodes = nodes; this.tri = tri;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < nodes.length; i += 2) {
      x0 = Math.min(x0, nodes[i]); x1 = Math.max(x1, nodes[i]); y0 = Math.min(y0, nodes[i + 1]); y1 = Math.max(y1, nodes[i + 1]);
    }
    const G = (this.G = 48);
    Object.assign(this, { x0, y0, dx: (x1 - x0) / G || 1, dy: (y1 - y0) / G || 1 });
    this.cells = Array.from({ length: G * G }, () => []);
    for (let t = 0; t < tri.length / 3; t++) {
      let a = Infinity, b = -Infinity, c = Infinity, d = -Infinity;
      for (let k = 0; k < 3; k++) {
        const v = tri[3 * t + k];
        a = Math.min(a, nodes[2 * v]); b = Math.max(b, nodes[2 * v]); c = Math.min(c, nodes[2 * v + 1]); d = Math.max(d, nodes[2 * v + 1]);
      }
      for (let i = this.cell(a, 0); i <= this.cell(b, 0); i++)
        for (let j = this.cell(c, 1); j <= this.cell(d, 1); j++) this.cells[j * G + i].push(t);
    }
  }

  cell(v, axis) {
    const c = Math.floor(axis ? (v - this.y0) / this.dy : (v - this.x0) / this.dx);
    return Math.min(this.G - 1, Math.max(0, c));
  }

  // [triangle, b0, b1, b2] barycentric, or null outside the mesh.
  find(x, y) {
    const { nodes, tri } = this;
    for (const t of this.cells[this.cell(y, 1) * this.G + this.cell(x, 0)]) {
      const a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
      const ax = nodes[2 * a], ay = nodes[2 * a + 1];
      const v0x = nodes[2 * b] - ax, v0y = nodes[2 * b + 1] - ay, v1x = nodes[2 * c] - ax, v1y = nodes[2 * c + 1] - ay;
      const px = x - ax, py = y - ay, den = v0x * v1y - v1x * v0y;
      const l1 = (px * v1y - v1x * py) / den, l2 = (v0x * py - px * v0y) / den, l0 = 1 - l1 - l2;
      if (l0 >= -1e-9 && l1 >= -1e-9 && l2 >= -1e-9) return [t, l0, l1, l2];
    }
    return null;
  }
}

export class DrumView {
  constructor(canvas, { elevation = 58, spacing = 1.25 } = {}) {
    this.canvas = canvas;
    this.elevation = elevation;
    this.spacing = spacing;
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true }));
    r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(PALETTE.plate);
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.05, 200);
    this.scene.add(new THREE.HemisphereLight(0xdfe4ee, 0x1b1d1c, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(-2, 6, 3);
    this.scene.add(sun);
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.drums = [];
    this.strokeLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: PALETTE.sand }));
    this.strokeLine.frustumCulled = false;
    this.scene.add(this.strokeLine);
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.frame3();
  }

  // Place the camera so every drum fits, with a margin.
  frame3() {
    if (!this.drums.length) {
      this.camera.position.set(0, 4, 3);
      this.camera.lookAt(0, 0, 0);
      return;
    }
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    for (const D of this.drums) {
      x0 = Math.min(x0, D.box[0]); x1 = Math.max(x1, D.box[1]); z0 = Math.min(z0, D.box[2]); z1 = Math.max(z1, D.box[3]);
    }
    const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
    const corners = [];
    for (const x of [x0, x1]) for (const z of [z0, z1]) for (const y of [0.05, -0.45]) corners.push(new THREE.Vector3(x, y, z));
    const el = (this.elevation * Math.PI) / 180, dir = new THREE.Vector3(0, Math.sin(el), Math.cos(el));
    const fits = (dist) => {
      this.camera.position.set(cx, 0, cz).addScaledVector(dir, dist);
      this.camera.lookAt(cx, -0.15, cz);
      this.camera.updateMatrixWorld();
      return corners.every((c) => {
        const p = c.clone().project(this.camera);
        return Math.abs(p.x) < 0.9 && Math.abs(p.y) < 0.86;
      });
    };
    let lo = 0.5, hi = 80;
    for (let i = 0; i < 40; i++) {
      const mid = (lo + hi) / 2;
      if (fits(mid)) hi = mid; else lo = mid;
    }
    fits(hi);
  }

  clear() {
    for (const o of [...this.group.children]) { this.group.remove(o); o.geometry?.dispose(); }
    this.drums = [];
  }

  // drums: [{ outline, nodes, tri, nRim, lam, shapes }]
  setDrums(list) {
    this.clear();
    // side by side, each scaled to the same area, with a gap between
    const placed = list.map((d) => {
      const s = 1.9 / Math.sqrt(Math.abs(polyArea(d.outline)));
      const [ox, oy] = centroid(d.outline);
      let a = Infinity, b = -Infinity;
      for (let i = 0; i < d.outline.length; i += 2) { a = Math.min(a, (d.outline[i] - ox) * s); b = Math.max(b, (d.outline[i] - ox) * s); }
      return { d, s, ox, oy, a, b };
    });
    const gap = 0.6 * this.spacing;
    const total = placed.reduce((t, p) => t + (p.b - p.a), 0) + gap * (placed.length - 1);
    let x = -total / 2;
    for (const p of placed) {
      const cx = x - p.a;
      x += p.b - p.a + gap;
      this.drums.push(this.build(p.d, { s: p.s, ox: p.ox, oy: p.oy, cx }));
    }
    this.frame3();
  }

  build(d, { s, ox, oy, cx }) {
    const N = d.nodes.length / 2;
    const pos = new Float32Array(N * 3), col = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      pos[3 * i] = cx + (d.nodes[2 * i] - ox) * s;
      pos[3 * i + 2] = -(d.nodes[2 * i + 1] - oy) * s;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(Array.from(d.tri, (v) => v));
    // the mesh triangles are counter-clockwise in (x, y), i.e. facing +world-y after the flip
    const head = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.72, metalness: 0.2, side: THREE.DoubleSide }));
    this.group.add(head);
    // rim: a tube round the outline; shell: a wall below it
    const pts = [];
    for (let i = 0; i < d.outline.length; i += 2) pts.push(new THREE.Vector3(cx + (d.outline[i] - ox) * s, 0.005, -(d.outline[i + 1] - oy) * s));
    const curve = new THREE.CatmullRomCurve3(pts, true, 'catmullrom', 0.1);
    const rim = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(64, pts.length * 3), 0.045, 10, true),
      new THREE.MeshStandardMaterial({ color: PALETTE.rim, roughness: 0.42, metalness: 0.65 }));
    this.group.add(rim);
    const wallPos = [], wallIdx = [], m = pts.length, depth = 0.42;
    pts.forEach((p) => wallPos.push(p.x, -0.02, p.z, p.x, -depth, p.z));
    for (let i = 0; i < m; i++) {
      const a = 2 * i, b = 2 * ((i + 1) % m);
      wallIdx.push(a, b, a + 1, a + 1, b, b + 1);
    }
    const wg = new THREE.BufferGeometry();
    wg.setAttribute('position', new THREE.Float32BufferAttribute(wallPos, 3));
    wg.setIndex(wallIdx);
    wg.computeVertexNormals();
    this.group.add(new THREE.Mesh(wg, new THREE.MeshStandardMaterial({ color: PALETTE.steel, roughness: 0.55, metalness: 0.45, side: THREE.DoubleSide })));
    // sand
    const SAND = 3500;
    const sandGeo = new THREE.BufferGeometry();
    sandGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(SAND * 3), 3));
    const sand = new THREE.Points(sandGeo, new THREE.PointsMaterial({ color: PALETTE.sand, size: 0.022, sizeAttenuation: true }));
    sand.frustumCulled = false;
    this.group.add(sand);
    // strike marker
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.075, 32), new THREE.MeshBasicMaterial({ color: PALETTE.sand, transparent: true, opacity: 0, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    this.group.add(ring);
    // displacement scale: the lowest mode peaks at about 0.18 units
    let peak = 0;
    for (const v of d.shapes[0]) peak = Math.max(peak, Math.abs(v));
    let bx0 = Infinity, bx1 = -Infinity, bz0 = Infinity, bz1 = -Infinity;
    for (const p of pts) { bx0 = Math.min(bx0, p.x); bx1 = Math.max(bx1, p.x); bz0 = Math.min(bz0, p.z); bz1 = Math.max(bz1, p.z); }
    const D = {
      d, s, ox, oy, cx, head, geo, pos, col, sand, ring, N, box: [bx0 - 0.06, bx1 + 0.06, bz0 - 0.06, bz1 + 0.06],
      amp: 0.18 / (peak || 1), locator: new Locator(d.nodes, d.tri),
      voices: [], mode: null, grains: new Float64Array(2 * SAND),
    };
    for (let g = 0; g < SAND; g++) this.sprinkle(D, g);
    return D;
  }

  // Drop grain g somewhere on the drumhead, uniformly.
  sprinkle(D, g) {
    const o = D.d.outline;
    if (!D.bbox) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let i = 0; i < o.length; i += 2) {
        x0 = Math.min(x0, o[i]); x1 = Math.max(x1, o[i]); y0 = Math.min(y0, o[i + 1]); y1 = Math.max(y1, o[i + 1]);
      }
      D.bbox = [x0, y0, x1, y1];
    }
    const [x0, y0, x1, y1] = D.bbox;
    for (;;) {
      const x = x0 + Math.random() * (x1 - x0), y = y0 + Math.random() * (y1 - y0);
      if (D.locator.find(x, y)) { D.grains[2 * g] = x; D.grains[2 * g + 1] = y; return; }
    }
  }

  // Which drum is under a client point, and where on it (drum coordinates).
  pick(clientX, clientY) {
    const hit = this.pickPlane(clientX, clientY);
    if (!hit) return null;
    for (let i = 0; i < this.drums.length; i++) {
      const D = this.drums[i];
      const x = (hit[0] - D.cx) / D.s + D.ox, y = -hit[2] / D.s + D.oy;
      const f = D.locator.find(x, y);
      if (f) return { drum: i, x, y, tri: f };
    }
    return null;
  }

  // World point on the plane of the drumheads.
  pickPlane(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const p = new THREE.Vector3();
    return ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p) ? [p.x, p.y, p.z] : null;
  }

  setStroke(world) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(world.flatMap(([x, , z]) => [x, 0.01, z]), 3));
    this.strokeLine.geometry.dispose();
    this.strokeLine.geometry = g;
  }

  // Each mode's displacement at a picked point (barycentric interpolation).
  weightsAt(drum, f) {
    const D = this.drums[drum], [t, l0, l1, l2] = f;
    const tri = D.d.tri, a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
    return D.d.shapes.map((sh) => l0 * sh[a] + l1 * sh[b] + l2 * sh[c]);
  }

  // Start a visual strike at drum coordinates (x, y).
  strike(drum, weights, x, y) {
    const D = this.drums[drum];
    D.mode = null;
    D.strikePeak = 0;
    const lam = D.d.lam;
    D.voices = weights.map((w, k) => {
      const r = lam[k] / lam[0];
      return { k, a: w * Math.exp(-r / 12) / Math.sqrt(r), f: VIS_F * Math.sqrt(r), tau: 2.6 * Math.pow(r, -0.45) };
    });
    D.t0 = performance.now() / 1000;
    D.ring.position.set(D.cx + (x - D.ox) * D.s, 0.02, -(y - D.oy) * D.s);
    D.ringT = D.t0;
  }

  // Show one mode on its own, as a Chladni figure (null to stop). The sand
  // already on the head stays and migrates to the new figure; grains that
  // have drifted to the rim (still for every mode) are sprinkled afresh.
  showMode(drum, k) {
    const D = this.drums[drum];
    D.voices = [];
    D.mode = k;
    if (k === null) return;
    const o = D.d.outline, n = o.length / 2, G = D.grains.length / 2;
    for (let g = 0; g < G; g++) {
      const x = D.grains[2 * g], y = D.grains[2 * g + 1];
      let near = Infinity;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n, ax = o[2 * i], ay = o[2 * i + 1], bx = o[2 * j] - ax, by = o[2 * j + 1] - ay;
        const t = Math.min(1, Math.max(0, ((x - ax) * bx + (y - ay) * by) / (bx * bx + by * by || 1)));
        near = Math.min(near, Math.hypot(x - ax - t * bx, y - ay - t * by));
      }
      if (near < 0.04) this.sprinkle(D, g);
    }
    let peak = 0;
    for (const v of D.d.shapes[k]) peak = Math.max(peak, Math.abs(v));
    D.modePeak = peak || 1;
    D.t0 = performance.now() / 1000;
  }

  frame() {
    const now = performance.now() / 1000;
    for (const D of this.drums) this.animate(D, now);
    this.renderer.render(this.scene, this.camera);
  }

  animate(D, now) {
    const { d, pos, col, N } = D;
    const t = now - (D.t0 || 0);
    const disp = new Float64Array(N);
    if (D.mode !== null) {
      const sh = d.shapes[D.mode], c = Math.cos(2 * Math.PI * VIS_F * 1.5 * t) * 0.6 / D.modePeak;
      for (let i = 0; i < N; i++) disp[i] = sh[i] * c;
      // sand feels how much each spot moves over a cycle, not the instant
      const drive = Float64Array.from(sh, (v) => Math.abs(v) / D.modePeak);
      for (let k = 0; k < 4; k++) this.moveSand(D, drive);
    } else if (D.voices.length) {
      for (const v of D.voices) {
        const e = v.a * Math.exp(-t / v.tau) * Math.cos(2 * Math.PI * v.f * t);
        if (Math.abs(e) < 1e-6) continue;
        const sh = d.shapes[v.k];
        for (let i = 0; i < N; i++) disp[i] += e * sh[i];
      }
      // a strike throws the sand about, less and less as the ring dies
      let mxd = 0;
      for (let i = 0; i < N; i++) mxd = Math.max(mxd, Math.abs(disp[i]));
      D.strikePeak = Math.max(D.strikePeak || 0, mxd);
      if (D.strikePeak > 0) {
        const drive = Float64Array.from(disp, (v) => (1.5 * Math.abs(v)) / D.strikePeak);
        for (let k = 0; k < 2; k++) this.moveSand(D, drive);
      }
    }
    D.disp = disp;
    const head = new THREE.Color(PALETTE.head), sand = new THREE.Color(PALETTE.sand), green = new THREE.Color(PALETTE.green);
    let mx = 1e-9;
    for (let i = 0; i < N; i++) mx = Math.max(mx, Math.abs(disp[i]));
    const scale = D.mode !== null ? 1 : Math.min(1, mx * D.amp / 0.18);
    for (let i = 0; i < N; i++) {
      pos[3 * i + 1] = disp[i] * D.amp;
      const u = (disp[i] / mx) * scale;
      const c = u >= 0 ? sand : green, w = Math.min(1, Math.abs(u)) * (D.mode !== null ? 0.35 : 0.85);
      col[3 * i] = head.r + (c.r - head.r) * w;
      col[3 * i + 1] = head.g + (c.g - head.g) * w;
      col[3 * i + 2] = head.b + (c.b - head.b) * w;
    }
    D.geo.attributes.position.needsUpdate = true;
    D.geo.attributes.color.needsUpdate = true;
    D.geo.computeVertexNormals();
    const rt = now - (D.ringT || -10);
    D.ring.material.opacity = Math.max(0, 0.9 - rt * 0.9);
    D.ring.scale.setScalar(1 + rt * 3);
    this.placeSand(D);
  }

  // Grains sit on the drumhead, riding up and down with it.
  placeSand(D) {
    const { d, locator, grains, disp } = D, tri = d.tri;
    const p = D.sand.geometry.attributes.position.array;
    for (let g = 0; g < grains.length / 2; g++) {
      const x = grains[2 * g], y = grains[2 * g + 1];
      const f = locator.find(x, y);
      let h = 0;
      if (f) {
        const [t, l0, l1, l2] = f;
        h = (l0 * disp[tri[3 * t]] + l1 * disp[tri[3 * t + 1]] + l2 * disp[tri[3 * t + 2]]) * D.amp;
      }
      p[3 * g] = D.cx + (x - D.ox) * D.s;
      p[3 * g + 1] = h + 0.012;
      p[3 * g + 2] = -(y - D.oy) * D.s;
    }
    D.sand.geometry.attributes.position.needsUpdate = true;
  }

  // Sand hops away from where the head moves (it is thrown up by the motion)
  // and comes to rest on the nodal lines, where the head stays still.
  // drive: per node, how hard the head shakes there (0 still, 1 the most).
  moveSand(D, drive) {
    const { d, locator, grains } = D, tri = d.tri;
    for (let g = 0; g < grains.length / 2; g++) {
      const x = grains[2 * g], y = grains[2 * g + 1];
      const f = locator.find(x, y);
      if (!f) { this.sprinkle(D, g); continue; }
      const [t, l0, l1, l2] = f;
      const u = l0 * drive[tri[3 * t]] + l1 * drive[tri[3 * t + 1]] + l2 * drive[tri[3 * t + 2]];
      const hop = (0.06 * Math.min(1.5, u)) / D.s;
      const nx = x + hop * (Math.random() * 2 - 1), ny = y + hop * (Math.random() * 2 - 1);
      if (locator.find(nx, ny)) { grains[2 * g] = nx; grains[2 * g + 1] = ny; }
    }
  }
}
