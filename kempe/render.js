// Three.js view of the plate, the drawing and the machine.
//
// Drawing coordinates (x, y) live on the plate at world (x, 0, -y). Bars that
// share a pin or cross during the cycle get different levels; geared cranks are
// driven from the motor shaft by timing belts (crossed for reverse ratios).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Machine, thetaGrid, toWorld } from './mech.js';

export const PALETTE = {
  plate: 0x2e2e2c,      // Braun anthracite
  panel: 0x1a1a18,      // warm black
  bar: 0x8a8a87,        // Braun grey
  barHi: 0xc5c3be,      // pebble
  pin: 0xf0ede5,        // Braun white
  anchor: 0x1a1a18,
  gear: 0xaab7bf,       // DR01 blue
  ink: 0xd4b018,        // function yellow
  ghost: 0xbfb1a8,      // DR01 grey
};

const W = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

function dotTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = '#2e2e2c';
  g.fillRect(0, 0, 64, 64);
  g.fillStyle = '#383835';
  g.beginPath();
  g.arc(32, 32, 5, 0, Math.PI * 2);
  g.fill();
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

// A flat ribbon along a 2D polyline, lying on the plate at height h. Optional
// per-point alpha makes a fading ink tail.
class Ribbon {
  constructor(maxPoints, color, opacity, h, withAlpha = false) {
    this.max = maxPoints;
    this.h = h;
    this.geo = new THREE.BufferGeometry();
    this.pos = new Float32Array(maxPoints * 2 * 3);
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    if (withAlpha) {
      this.col = new Float32Array(maxPoints * 2 * 4);
      this.geo.setAttribute('color', new THREE.BufferAttribute(this.col, 4));
    }
    const idx = [];
    for (let i = 0; i < maxPoints - 1; i++) {
      const a = 2 * i, b = a + 1, c = a + 2, d = a + 3;
      idx.push(a, c, b, b, c, d);
    }
    this.geo.setIndex(idx);
    this.mat = new THREE.MeshBasicMaterial({
      color: withAlpha ? 0xffffff : color, transparent: true, opacity, depthWrite: false, vertexColors: withAlpha,
      side: THREE.DoubleSide,
    });
    this.color = new THREE.Color(color);
    this.mesh = new THREE.Mesh(this.geo, this.mat);
    this.mesh.renderOrder = 2;
    this.mesh.frustumCulled = false;
    this.set([], 0);
  }

  // pts: flat [x0, y0, x1, y1, ...] in drawing coordinates.
  set(pts, width, closed = false, alpha = null) {
    let n = Math.min(pts.length / 2, this.max - (closed ? 1 : 0));
    const P = this.pos;
    const count = closed && n > 2 ? n + 1 : n;
    for (let k = 0; k < count; k++) {
      const i = k % n;
      const ip = closed ? (i - 1 + n) % n : Math.max(0, i - 1);
      const inx = closed ? (i + 1) % n : Math.min(n - 1, i + 1);
      let tx = pts[2 * inx] - pts[2 * ip], ty = pts[2 * inx + 1] - pts[2 * ip + 1];
      const tl = Math.hypot(tx, ty) || 1;
      tx /= tl; ty /= tl;
      const nx = -ty * width / 2, ny = tx * width / 2;
      const x = pts[2 * i], y = pts[2 * i + 1];
      P.set([x + nx, this.h, -(y + ny), x - nx, this.h, -(y - ny)], 6 * k);
      if (this.col) {
        const a = alpha ? alpha[i] : 1;
        this.col.set([this.color.r, this.color.g, this.color.b, a, this.color.r, this.color.g, this.color.b, a], 8 * k);
      }
    }
    this.geo.setDrawRange(0, Math.max(0, count - 1) * 6);
    this.geo.attributes.position.needsUpdate = true;
    if (this.col) this.geo.attributes.color.needsUpdate = true;
  }
}

// Convex hull of points (for an open belt around two pulleys).
function hull(pts) {
  pts = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const p of pts) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (const p of pts.slice().reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}

// Closed path of a belt around pulleys c1 (r1) and c2 (r2); crossed for
// pulleys turning in opposite directions.
function beltPath(c1, r1, c2, r2, crossed) {
  const N = 48;
  if (!crossed) {
    const pts = [];
    for (let i = 0; i < N; i++) {
      const a = (2 * Math.PI * i) / N;
      pts.push([c1[0] + r1 * Math.cos(a), c1[1] + r1 * Math.sin(a)]);
      pts.push([c2[0] + r2 * Math.cos(a), c2[1] + r2 * Math.sin(a)]);
    }
    return hull(pts);
  }
  const dx = c2[0] - c1[0], dy = c2[1] - c1[1], d = Math.hypot(dx, dy);
  const base = Math.atan2(dy, dx), th = Math.acos(Math.min(1, (r1 + r2) / d));
  const a1p = base + th, a1m = base - th;           // tangent normals at pulley 1
  const out = [];
  const arc = (c, r, from, to, dir) => {             // dir +1 ccw, -1 cw
    let span = dir > 0 ? to - from : from - to;
    while (span < 0) span += 2 * Math.PI;
    const k = Math.max(2, Math.round((span / (2 * Math.PI)) * N));
    for (let i = 0; i <= k; i++) {
      const a = from + (dir * span * i) / k;
      out.push([c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)]);
    }
  };
  arc(c1, r1, a1p, a1m, 1);                          // round the far side of pulley 1
  arc(c2, r2, a1m + Math.PI, a1p + Math.PI, 1);      // cross over, round pulley 2
  return out;
}

export class MachineView {
  constructor(canvas) {
    this.canvas = canvas;
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true }));
    r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.0;
    const scene = (this.scene = new THREE.Scene());
    scene.background = new THREE.Color(PALETTE.plate);
    scene.fog = new THREE.Fog(PALETTE.plate, 30, 90);
    const pmrem = new THREE.PMREMGenerator(r);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.55;

    this.camera = new THREE.PerspectiveCamera(30, 1, 0.05, 2000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
    this.controls.touches = { ONE: null, TWO: THREE.TOUCH.DOLLY_ROTATE };
    this.controls.maxPolarAngle = Math.PI * 0.46;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 400;

    scene.add(new THREE.HemisphereLight(0xf0ede5, 0x1a1a18, 0.9));
    const sun = (this.sun = new THREE.DirectionalLight(0xfff8ee, 2.2));
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    scene.add(sun, sun.target);

    const tex = dotTexture();
    this.plateTex = tex;
    const plate = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      // pushed back in depth so the ink and the drawing never z-fight with it
      new THREE.MeshStandardMaterial({ color: 0xffffff, map: tex, roughness: 0.92, metalness: 0,
        polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 8 }),
    );
    plate.rotation.x = -Math.PI / 2;
    plate.scale.set(600, 600, 1);           // dots every 0.3 plate units
    tex.repeat.set(2000, 2000);
    plate.receiveShadow = true;
    this.plate = plate;
    scene.add(plate);

    this.ghost = new Ribbon(4096, PALETTE.ghost, 0.55, 0.002);
    this.inkFull = new Ribbon(801, PALETTE.ink, 0.5, 0.003);
    this.ink = new Ribbon(801, PALETTE.ink, 1.0, 0.004, true);
    scene.add(this.ghost.mesh, this.inkFull.mesh, this.ink.mesh);

    this.mats = {
      bar: new THREE.MeshStandardMaterial({ color: PALETTE.bar, metalness: 0.7, roughness: 0.38 }),
      barHi: new THREE.MeshStandardMaterial({ color: PALETTE.barHi, metalness: 0.6, roughness: 0.32 }),
      pin: new THREE.MeshStandardMaterial({ color: PALETTE.pin, metalness: 0.1, roughness: 0.5 }),
      anchor: new THREE.MeshStandardMaterial({ color: PALETTE.anchor, metalness: 0.2, roughness: 0.7 }),
      gear: new THREE.MeshStandardMaterial({ color: PALETTE.gear, metalness: 0.6, roughness: 0.35 }),
      belt: new THREE.MeshStandardMaterial({ color: PALETTE.anchor, metalness: 0, roughness: 0.85, side: THREE.DoubleSide }),
      ink: new THREE.MeshBasicMaterial({ color: PALETTE.ink }),
    };
    this.geo = {
      box: new THREE.BoxGeometry(1, 1, 1),
      cyl: new THREE.CylinderGeometry(0.5, 0.5, 1, 28),
      cone: new THREE.ConeGeometry(0.5, 1, 20),
    };
    this.machineGroup = new THREE.Group();
    scene.add(this.machineGroup);
    this.view = 'angle';
    this.setHome(8);
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // Camera framing of a circle (cx, cy, radius) in drawing coordinates.
  frameCircle(cx, cy, rad, view = this.view, instant = false) {
    this.view = view;
    const fov = (this.camera.fov * Math.PI) / 180;
    const aspect = Math.max(0.5, this.camera.aspect);
    const fit = rad / Math.sin(Math.min(fov / 2, Math.atan(Math.tan(fov / 2) * aspect)));
    const dist = fit * 1.08;
    const elev = view === 'plan' ? Math.PI / 2 - 1e-3 : (62 * Math.PI) / 180;
    const az = view === 'plan' ? 0 : (-18 * Math.PI) / 180;
    const target = W(cx, cy);
    const pos = new THREE.Vector3(
      target.x + dist * Math.cos(elev) * Math.sin(az),
      dist * Math.sin(elev),
      target.z + dist * Math.cos(elev) * Math.cos(az),
    );
    this.anim = { from: this.camera.position.clone(), fromT: this.controls.target.clone(), to: pos, toT: target, t: instant ? 1 : 0 };
    this.scene.fog.near = dist * 1.1;
    this.scene.fog.far = dist * 2.6;
    this.camera.near = Math.max(0.02, dist * 0.02);
    this.camera.far = dist * 30;
    this.camera.updateProjectionMatrix();
    if (instant) { this.camera.position.copy(pos); this.controls.target.copy(target); }
    // the shadow camera follows the framing
    const sh = this.sun.shadow.camera;
    sh.left = sh.bottom = -rad * 1.4;
    sh.right = sh.top = rad * 1.4;
    sh.near = 0.1;
    sh.far = rad * 8;
    sh.updateProjectionMatrix();
    this.sun.position.set(target.x - rad * 1.2, rad * 3.5, target.z - rad * 0.8);
    this.sun.target.position.copy(target);
    this.focus = { cx, cy, rad };
  }

  setHome(rad) { this.frameCircle(0, 0, rad, this.view, true); }

  setView(view) {
    const f = this.focus || { cx: 0, cy: 0, rad: 8 };
    this.frameCircle(f.cx, f.cy, f.rad, view);
  }

  // Drawing coordinates under a client point, or null.
  pick(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const hit = new THREE.Vector3();
    if (!ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit)) return null;
    return [hit.x, -hit.z];
  }

  setGhost(pts, closed) {
    const f = this.focus || { rad: 8 };
    this.ghostPts = pts;
    this.ghost.set(pts, Math.max(0.02, f.rad * 0.006), closed);
  }

  clearMachine() {
    for (const o of [...this.machineGroup.children]) this.machineGroup.remove(o);
    this.m = null;
    this.inkFull.set([], 0);
    this.ink.set([], 0);
  }

  // Build meshes for a machine. spec/P/placement come from the worker; target
  // holds the drawing's mean and scale.
  setMachine(spec, P, placement, target, drop = false) {
    this.clearMachine();
    this.dropT = drop ? 0 : 1;
    const m = new Machine({ ...spec, pos: Float64Array.from(spec.pos) });
    this.m = m;
    this.P = Float64Array.from(P);
    this.placement = placement;
    this.target = target;
    this.buf = new Float64Array(m.n * 2);
    this.theta1 = new Float64Array(1);
    const s = target.scale;
    const w = (this.barW = 0.075 * s), th = (this.barT = 0.05 * s), gap = 0.03 * s;
    this.layers = this.assignLevels();
    const nCranks = m.cranks.length;
    const pulleyH = 0.06 * s, housingH = 0.07 * s;
    this.housingH = housingH;
    this.base = housingH + (nCranks - 1) * pulleyH * 1.4 + 0.04 * s;
    const levelY = (L) => this.base + L * (th + gap) + th / 2;
    this.levelY = levelY;

    // bars: body box + rounded ends
    this.barMeshes = m.bars.map(([j1, j2], k) => {
      const crank = m.kind[j2] === 1;
      const mat = crank ? this.mats.barHi : this.mats.bar;
      const body = new THREE.Mesh(this.geo.box, mat);
      const e1 = new THREE.Mesh(this.geo.cyl, mat), e2 = new THREE.Mesh(this.geo.cyl, mat);
      for (const o of [body, e1, e2]) { o.castShadow = true; o.receiveShadow = true; this.machineGroup.add(o); }
      e1.scale.set(w, th, w);
      e2.scale.set(w, th, w);
      return { j1, j2, body, e1, e2, y: levelY(this.layers[k]) };
    });

    // pins at every joint spanning its bars; posts under fixed pivots
    const lo = new Array(m.n).fill(Infinity), hi = new Array(m.n).fill(-Infinity);
    m.bars.forEach(([j1, j2], k) => {
      for (const j of [j1, j2]) { lo[j] = Math.min(lo[j], this.layers[k]); hi[j] = Math.max(hi[j], this.layers[k]); }
    });
    this.pins = [];
    for (let j = 0; j < m.n; j++) {
      if (hi[j] < 0) continue;
      const ground = m.kind[j] === 0;
      const y0 = ground ? 0 : levelY(lo[j]) - th / 2;
      const y1 = levelY(hi[j]) + th / 2 + 0.01 * s;
      const pin = new THREE.Mesh(this.geo.cyl, ground ? this.mats.anchor : this.mats.pin);
      pin.scale.set(ground ? w * 0.55 : w * 0.32, y1 - y0, ground ? w * 0.55 : w * 0.32);
      const cap = new THREE.Mesh(this.geo.cyl, this.mats.pin);
      cap.scale.set(w * 0.46, 0.012 * s, w * 0.46);
      pin.castShadow = cap.castShadow = true;
      this.machineGroup.add(pin, cap);
      const item = { j, pin, cap, y0, y1 };
      if (ground) {
        const flange = new THREE.Mesh(this.geo.cyl, this.mats.anchor);
        flange.scale.set(w * 1.6, 0.03 * s, w * 1.6);
        flange.receiveShadow = true;
        this.machineGroup.add(flange);
        item.flange = flange;
      }
      this.pins.push(item);
    }

    // the pen: a stylus from the last joint down to the plate
    const pen = m.tracer;
    const stylus = new THREE.Mesh(this.geo.cyl, this.mats.barHi);
    const penY = levelY(lo[pen]) - th / 2;
    stylus.scale.set(w * 0.22, penY, w * 0.22);
    const tip = new THREE.Mesh(this.geo.cone, this.mats.ink);
    tip.scale.set(w * 0.5, w * 0.9, w * 0.5);
    tip.rotation.x = Math.PI;
    stylus.castShadow = true;
    this.machineGroup.add(stylus, tip);
    this.pen = { stylus, tip, penY };

    // motor housing and belt drives for geared cranks
    const motorPiv = m.a[m.cranks[0]];
    const housing = new THREE.Mesh(this.geo.cyl, this.mats.anchor);
    housing.scale.set(w * 3.2, housingH, w * 3.2);
    housing.castShadow = true;
    this.machineGroup.add(housing);
    this.motor = { housing, piv: motorPiv };
    this.drives = m.cranks.slice(1).map((c, k) => {
      const g = m.gear[c];
      const rBig = 0.14 * s;
      const r1 = Math.abs(g) >= 1 ? rBig : rBig * Math.abs(g);
      const r2 = r1 / Math.abs(g);
      const y = housingH + pulleyH * (1.4 * k + 0.7);
      const p1 = new THREE.Mesh(this.geo.cyl, this.mats.gear);
      const p2 = new THREE.Mesh(this.geo.cyl, this.mats.gear);
      p1.scale.set(2 * r1, pulleyH, 2 * r1);
      p2.scale.set(2 * r2, pulleyH, 2 * r2);
      const mk1 = new THREE.Mesh(this.geo.box, this.mats.anchor);
      const mk2 = new THREE.Mesh(this.geo.box, this.mats.anchor);
      mk1.scale.set(r1 * 1.6, pulleyH * 1.05, r1 * 0.18);
      mk2.scale.set(r2 * 1.6, pulleyH * 1.05, r2 * 0.25);
      for (const o of [p1, p2, mk1, mk2]) { o.castShadow = true; this.machineGroup.add(o); }
      const belt = new THREE.Mesh(new THREE.BufferGeometry(), this.mats.belt);
      belt.castShadow = true;
      this.machineGroup.add(belt);
      return { c, g, r1, r2, y, p1, p2, mk1, mk2, belt, h: pulleyH * 0.7 };
    });
    this.updateCurve();
  }

  // Levels for bars: bars that share a joint, or cross at some crank angle,
  // must sit at different heights. Greedy colouring, cranks first.
  assignLevels() {
    const m = this.m, T = 48, th = thetaGrid(T);
    const buf = new Float64Array(m.n * T * 2);
    m.simulate(this.P, th, buf);
    const at = (j, t) => [buf[2 * (j * T + t)], buf[2 * (j * T + t) + 1]];
    const inter = (p1, p2, p3, p4) => {
      const d = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
      return d(p1, p2, p3) * d(p1, p2, p4) < 0 && d(p3, p4, p1) * d(p3, p4, p2) < 0;
    };
    const B = m.bars, conflict = B.map(() => new Set());
    for (let i = 0; i < B.length; i++) {
      for (let k = i + 1; k < B.length; k++) {
        const [a1, a2] = B[i], [b1, b2] = B[k];
        let c = a1 === b1 || a1 === b2 || a2 === b1 || a2 === b2;
        for (let t = 0; t < T && !c; t++) c = inter(at(a1, t), at(a2, t), at(b1, t), at(b2, t));
        if (c) { conflict[i].add(k); conflict[k].add(i); }
      }
    }
    const level = new Array(B.length).fill(-1);
    for (let i = 0; i < B.length; i++) {
      const used = new Set([...conflict[i]].map((k) => level[k]));
      let L = 0;
      while (used.has(L)) L++;
      level[i] = L;
    }
    return level;
  }

  // Refinement moved the design: keep the meshes, change the numbers.
  updateDesign(P, placement) {
    if (!this.m) return;
    this.P = Float64Array.from(P);
    this.placement = placement;
    this.updateCurve();
  }

  updateCurve() {
    const m = this.m, T = 400, th = thetaGrid(T);
    const buf = new Float64Array(m.n * T * 2);
    if (m.simulate(this.P, th, buf) < 0) return;
    const pts = new Float64Array(2 * T), base = 2 * m.tracer * T;
    for (let t = 0; t < T; t++) {
      const [x, y] = toWorld(this.placement, this.target, buf[base + 2 * t], buf[base + 2 * t + 1]);
      pts[2 * t] = x; pts[2 * t + 1] = y;
    }
    this.curve = pts;
    this.inkFull.set(pts, this.target.scale * 0.022, true);
  }

  // Pose the machine at crank angle theta.
  frame(theta) {
    const m = this.m;
    if (this.dropT < 1) {   // a new machine settles onto the plate
      this.dropT = Math.min(1, this.dropT + 0.05);
      const e = 1 - Math.pow(1 - this.dropT, 3);
      this.machineGroup.position.y = (1 - e) * (this.target ? this.target.scale * 1.5 : 1);
    } else this.machineGroup.position.y = 0;
    if (m) {
      this.theta1[0] = theta;
      if (m.simulate(this.P, this.theta1, this.buf) >= 0) this.pose(theta);
    }
    if (this.anim && this.anim.t < 1) {
      const a = this.anim;
      a.t = Math.min(1, a.t + 0.045);
      const e = 1 - Math.pow(1 - a.t, 3);
      this.camera.position.lerpVectors(a.from, a.to, e);
      this.controls.target.lerpVectors(a.fromT, a.toT, e);
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  pose(theta) {
    const m = this.m, s = this.target.scale;
    const J = new Array(m.n);
    for (let j = 0; j < m.n; j++) J[j] = toWorld(this.placement, this.target, this.buf[2 * j], this.buf[2 * j + 1]);
    for (const b of this.barMeshes) {
      const [x1, y1] = J[b.j1], [x2, y2] = J[b.j2];
      const L = Math.hypot(x2 - x1, y2 - y1);
      b.body.position.copy(W((x1 + x2) / 2, (y1 + y2) / 2, b.y));
      b.body.rotation.set(0, Math.atan2(y2 - y1, x2 - x1), 0);
      b.body.scale.set(Math.max(1e-4, L), this.barT, this.barW);
      b.e1.position.copy(W(x1, y1, b.y));
      b.e2.position.copy(W(x2, y2, b.y));
    }
    for (const p of this.pins) {
      const [x, y] = J[p.j];
      p.pin.position.copy(W(x, y, (p.y0 + p.y1) / 2));
      p.cap.position.copy(W(x, y, p.y1));
      if (p.flange) p.flange.position.copy(W(x, y, 0.015 * s));
    }
    const [px, py] = J[m.tracer];
    this.pen.stylus.position.copy(W(px, py, this.pen.penY / 2));
    this.pen.tip.position.copy(W(px, py, this.barW * 0.45));
    const [mx, my] = J[this.motor.piv];
    this.motor.housing.position.copy(W(mx, my, this.housingH / 2));
    // the motor crank's angle in world, for pulley spokes
    const c0 = m.cranks[0];
    const motorAng = Math.atan2(J[c0][1] - my, J[c0][0] - mx);
    for (const d of this.drives) {
      const [qx, qy] = J[m.a[d.c]];
      const crankAng = Math.atan2(J[d.c][1] - qy, J[d.c][0] - qx);
      d.p1.position.copy(W(mx, my, d.y));
      d.p2.position.copy(W(qx, qy, d.y));
      d.mk1.position.copy(W(mx, my, d.y));
      d.mk2.position.copy(W(qx, qy, d.y));
      d.mk1.rotation.set(0, motorAng, 0);
      d.mk2.rotation.set(0, crankAng, 0);
      const key = `${mx.toFixed(4)},${my.toFixed(4)},${qx.toFixed(4)},${qy.toFixed(4)}`;
      if (key !== d.key) {   // pivots only move while the design is refined
        d.key = key;
        this.buildBelt(d, [mx, my], [qx, qy]);
      }
    }
    // fresh ink: the last revolution behind the pen, fading with age
    if (this.curve) {
      const T = this.curve.length / 2;
      const tNow = ((theta % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      const k = Math.floor((tNow / (2 * Math.PI)) * T);
      const n = Math.floor(T * 0.92);
      const pts = new Float64Array(2 * (n + 1)), alpha = new Float32Array(n + 1);
      for (let i = 0; i <= n; i++) {
        const src = (k - n + i + 2 * T) % T;
        pts[2 * i] = this.curve[2 * src];
        pts[2 * i + 1] = this.curve[2 * src + 1];
        alpha[i] = Math.pow(i / n, 1.6);
      }
      pts[2 * n] = px; pts[2 * n + 1] = py;
      this.ink.set(pts, this.target.scale * 0.03, false, alpha);
    }
  }

  buildBelt(d, c1, c2) {
    const path = beltPath(c1, d.r1 * 1.02, c2, d.r2 * 1.02, d.g < 0);
    const n = path.length, pos = new Float32Array(n * 2 * 3), idx = [];
    path.forEach(([x, y], i) => {
      pos.set([x, d.y - d.h / 2, -y, x, d.y + d.h / 2, -y], 6 * i);
      const a = 2 * i, b = 2 * ((i + 1) % n);
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    });
    d.belt.geometry.dispose();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    d.belt.geometry = g;
  }

  // World-space bounding circle of the machine over a whole cycle.
  bounds() {
    if (!this.m) return null;
    const m = this.m, T = 64, th = thetaGrid(T), buf = new Float64Array(m.n * T * 2);
    if (m.simulate(this.P, th, buf) < 0) return null;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < buf.length; i += 2) {
      const [x, y] = toWorld(this.placement, this.target, buf[i], buf[i + 1]);
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, rad: Math.hypot(x1 - x0, y1 - y0) / 2 };
  }
}
