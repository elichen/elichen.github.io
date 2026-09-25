/*
 * Clawdbeast design engine.
 *
 * A Theo Jansen linkage walker with a Clawd-shaped chassis, sized for laser
 * cutting at SendCutSend. Everything here is plain math with no DOM access, so
 * the same file runs in the browser (window.Clawdbeast) and in Node (require).
 *
 * Coordinates are side-view, y up, millimetres, origin at the crank axle.
 * The "rear" leg hangs from the pivot at x = -a; the "front" leg is its mirror.
 * z runs outward from the body; each side of the walker is a stack of layers.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.Clawdbeast = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // Linkage

  // Theo Jansen's "holy numbers" (unitless; multiplied by the scale in mm).
  const HOLY = { a: 38, b: 41.5, c: 39.3, d: 40.1, e: 55.8, f: 39.4, g: 36.7, h: 65.7, i: 49, j: 50, k: 61.9, l: 7.8, m: 15 };
  // Which of the two circle intersections each joint takes (+1 left of P->Q).
  const BRANCH = { C: -1, E: 1, D: 1, F: -1, G: -1 };

  const TAU = Math.PI * 2;
  const deg = (r) => (r * 180) / Math.PI;
  const rad = (d) => (d * Math.PI) / 180;
  const hyp = (x, y) => Math.sqrt(x * x + y * y);
  const dist = (p, q) => hyp(q[0] - p[0], q[1] - p[1]);

  function circleIntersect(P, r1, Q, r2, side) {
    const dx = Q[0] - P[0], dy = Q[1] - P[1], d = hyp(dx, dy);
    const a = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
    const h2 = r1 * r1 - a * a;
    if (h2 < 0) return null;
    const h = Math.sqrt(h2), mx = P[0] + (a * dx) / d, my = P[1] + (a * dy) / d;
    return [mx - (side * h * dy) / d, my + (side * h * dx) / d];
  }

  // Joint positions of the rear leg in Jansen units for crank angle theta.
  function rearLegUnits(theta) {
    const H = HOLY;
    const O = [0, 0], A = [H.m * Math.cos(theta), H.m * Math.sin(theta)], B = [-H.a, -H.l];
    const C = circleIntersect(A, H.j, B, H.b, BRANCH.C);
    const E = circleIntersect(A, H.k, B, H.c, BRANCH.E);
    const D = circleIntersect(B, H.d, C, H.e, BRANCH.D);
    const F = circleIntersect(D, H.f, E, H.g, BRANCH.F);
    const G = circleIntersect(F, H.h, E, H.i, BRANCH.G);
    return { O, A, B, C, D, E, F, G };
  }

  // Leg pose in mm. The front leg at crank angle theta is the mirror image of
  // the rear leg at (pi - theta), which puts both crank pins at the same spot.
  function legPose(theta, s, front) {
    const u = rearLegUnits(front ? Math.PI - theta : theta);
    const out = {};
    for (const k in u) out[k] = [(front ? -u[k][0] : u[k][0]) * s, u[k][1] * s];
    return out;
  }

  // Parts of one leg: which joints they span, and which of those get holes.
  const LEG_PARTS = [
    { key: 'j', pts: ['A', 'C'], holes: ['A', 'C'], label: 'Link j' },
    { key: 'k', pts: ['A', 'E'], holes: ['A', 'E'], label: 'Link k' },
    { key: 'c', pts: ['B', 'E'], holes: ['B', 'E'], label: 'Link c' },
    { key: 'f', pts: ['D', 'F'], holes: ['D', 'F'], label: 'Link f' },
    { key: 'bde', pts: ['B', 'C', 'D'], holes: ['B', 'C', 'D'], label: 'Hip triangle' },
    { key: 'ghi', pts: ['E', 'F', 'G'], holes: ['E', 'F'], label: 'Foot triangle' },
  ];

  // ---------------------------------------------------------------------------
  // Presets

  const IN = 25.4;
  const MATERIALS = {
    '5052-090': { name: '5052 H32 aluminum', gauge: '0.090"', t: 0.09 * IN, minPartIn: [0.25, 0.375], coat: true },
    '5052-125': { name: '5052 H32 aluminum', gauge: '0.125"', t: 0.125 * IN, minPartIn: [0.25, 0.375], coat: true },
    '6061-125': { name: '6061 T6 aluminum', gauge: '0.125"', t: 0.125 * IN, minPartIn: [0.25, 0.375], coat: true },
    'acrylic-118': { name: 'Acrylic', gauge: '0.118"', t: 0.118 * IN, minPartIn: [0.187, 0.375], coat: false },
  };

  // Bolt-as-pivot hardware. Heights include one washer; the nut side also
  // allows for the bolt tail sticking out past a nylon-insert lock nut.
  const HARDWARE = {
    M3: {
      name: 'M3', hole: 3.4, rodD: 3, washer: 0.5, spacerOD: 6,
      headH: 1.65 + 0.5, nutH: 0.5 + 4.0 + 2.0, keepR: 3.5, nutLen: 4.0,
      lengths: [6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50],
      bolt: 'M3 button-head screw', nut: 'M3 nylon-insert lock nut', rod: 'M3 threaded rod',
    },
    M4: {
      name: 'M4', hole: 4.4, rodD: 4, washer: 0.8, spacerOD: 8,
      headH: 2.2 + 0.8, nutH: 0.8 + 5.0 + 2.5, keepR: 4.5, nutLen: 5.0,
      lengths: [8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50],
      bolt: 'M4 button-head screw', nut: 'M4 nylon-insert lock nut', rod: 'M4 threaded rod',
    },
  };

  // 6 mm D-profile shaft (0.5 mm deep flat). D-holes leave 0.1 mm of air.
  // The crankshaft is built up from crank arms keyed onto short pieces of
  // this shaft: links j and k ride on the crank pins through a round hole.
  const SHAFT = { d: 6, holeR: 3.1, holeFlat: 2.6, plateHole: 6.5, linkHole: 6.3, spacerOD: 10, collarW: 8 };

  const DEFAULTS = {
    scale: 1.7, // mm per Jansen unit
    material: '5052-090',
    hardware: 'M3',
    slabs: 3, // crank phases per side; legs = 4 * slabs
    linkWidth: 0, // 0 = auto from the hole-to-edge rule
    bodyGap: 40, // mm between the two inner plates
    clearance: 1.0, // mm of air required between moving things
    samples: 180,
  };

  // ---------------------------------------------------------------------------
  // 2D geometry. A contour is a closed list of {x, y, b} vertices where b is
  // the DXF bulge of the segment that starts at that vertex.

  // Convex hull of circles given in counter-clockwise order (all on the hull).
  function hullOfCircles(circles) {
    const n = circles.length;
    const normals = [];
    for (let i = 0; i < n; i++) {
      const p = circles[i], q = circles[(i + 1) % n];
      const dx = q.x - p.x, dy = q.y - p.y, d = hyp(dx, dy);
      const ux = dx / d, uy = dy / d;
      const g = Math.asin((p.r - q.r) / d);
      // Outward normal of the tangent line, right of the travel direction.
      normals.push([Math.cos(g) * uy + Math.sin(g) * ux, Math.cos(g) * -ux + Math.sin(g) * uy]);
    }
    const out = [];
    for (let i = 0; i < n; i++) {
      const c = circles[i], nIn = normals[(i - 1 + n) % n], nOut = normals[i];
      let sweep = Math.atan2(nOut[1], nOut[0]) - Math.atan2(nIn[1], nIn[0]);
      while (sweep < 0) sweep += TAU;
      while (sweep >= TAU) sweep -= TAU;
      out.push({ x: c.x + c.r * nIn[0], y: c.y + c.r * nIn[1], b: Math.tan(sweep / 4) });
      out.push({ x: c.x + c.r * nOut[0], y: c.y + c.r * nOut[1], b: 0 });
    }
    return out;
  }

  function isCCW(pts) {
    const [a, b, c] = pts;
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0;
  }

  function polygon(pts) {
    return pts.map(([x, y]) => ({ x, y, b: 0 }));
  }

  // D-shaped hole: circle radius r with a flat at distance f from the centre,
  // whose outward normal points at angle psi.
  function dHole(cx, cy, r, f, psi) {
    const dlt = Math.acos(f / r);
    const a1 = psi + dlt, a2 = psi - dlt + TAU;
    return [
      { x: cx + r * Math.cos(a1), y: cy + r * Math.sin(a1), b: Math.tan((a2 - a1) / 4) },
      { x: cx + r * Math.cos(a2), y: cy + r * Math.sin(a2), b: 0 },
    ];
  }

  function circleContour([x, y, r]) {
    return [{ x: x + r, y, b: 1 }, { x: x - r, y, b: 1 }];
  }

  // Inward offset of a triangle's edges by distance w; null if it vanishes.
  function shrinkTriangle(pts, w) {
    const [A, B, C] = pts;
    const la = dist(B, C), lb = dist(A, C), lc = dist(A, B), per = la + lb + lc;
    const I = [(la * A[0] + lb * B[0] + lc * C[0]) / per, (la * A[1] + lb * B[1] + lc * C[1]) / per];
    const area = Math.abs((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])) / 2;
    const r = (2 * area) / per;
    if (w >= r) return null;
    const k = (r - w) / r;
    return { pts: pts.map((p) => [I[0] + (p[0] - I[0]) * k, I[1] + (p[1] - I[1]) * k]), inradius: r - w };
  }

  function arcCenter(p, q, bulge) {
    const th = 4 * Math.atan(bulge);
    const cx = (p.x + q.x) / 2, cy = (p.y + q.y) / 2;
    const dx = q.x - p.x, dy = q.y - p.y, chord = hyp(dx, dy);
    const r = chord / (2 * Math.sin(Math.abs(th) / 2));
    const hOff = chord / 2 / Math.tan(th / 2); // signed
    return { x: cx - (dy / chord) * hOff, y: cy + (dx / chord) * hOff, r: Math.abs(r), th };
  }

  // Sample a contour as a polyline (for bounding boxes and plotting).
  function flatten(contour, step = 1) {
    const out = [];
    for (let i = 0; i < contour.length; i++) {
      const p = contour[i], q = contour[(i + 1) % contour.length];
      out.push([p.x, p.y]);
      if (Math.abs(p.b) > 1e-9) {
        const c = arcCenter(p, q, p.b);
        const a0 = Math.atan2(p.y - c.y, p.x - c.x);
        const n = Math.max(4, Math.ceil((Math.abs(c.th) * c.r) / step));
        for (let k = 1; k < n; k++) {
          const a = a0 + (c.th * k) / n;
          out.push([c.x + c.r * Math.cos(a), c.y + c.r * Math.sin(a)]);
        }
      }
    }
    return out;
  }

  function bbox(contours) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const c of contours) for (const [x, y] of flatten(c)) {
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
    return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 };
  }

  // Distances for the collision checks. A part's footprint is the convex hull
  // of its circles: a "core" point / segment / triangle swollen by a radius.
  function segPointDist(a, b, p) {
    const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy;
    let t = L2 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2 : 0;
    t = Math.max(0, Math.min(1, t));
    return hyp(a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
  }
  function segsCross(a, b, c, d) {
    const o = (p, q, r) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    return o(a, b, c) * o(a, b, d) < 0 && o(c, d, a) * o(c, d, b) < 0;
  }
  function inTriangle(p, t) {
    const o = (a, b) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    const s1 = o(t[0], t[1]), s2 = o(t[1], t[2]), s3 = o(t[2], t[0]);
    return (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
  }
  function edges(poly) {
    if (poly.length === 1) return [[poly[0], poly[0]]];
    if (poly.length === 2) return [[poly[0], poly[1]]];
    return poly.map((p, i) => [p, poly[(i + 1) % poly.length]]);
  }
  function coreDist(P, Q) {
    if (P.length === 3 && Q.some((q) => inTriangle(q, P))) return 0;
    if (Q.length === 3 && P.some((p) => inTriangle(p, Q))) return 0;
    let best = Infinity;
    for (const [a, b] of edges(P)) for (const [c, d] of edges(Q)) {
      if (segsCross(a, b, c, d)) return 0;
      best = Math.min(best, segPointDist(a, b, c), segPointDist(a, b, d), segPointDist(c, d, a), segPointDist(c, d, b));
    }
    return best;
  }
  // Exact signed distance from q to the hull of two circles (a tapered link).
  function taperedDist(c1, r1, c2, r2, q) {
    const dx = c2[0] - c1[0], dy = c2[1] - c1[1], h = hyp(dx, dy);
    const ux = dx / h, uy = dy / h;
    const px = Math.abs(-(q[0] - c1[0]) * uy + (q[1] - c1[1]) * ux); // across
    const py = (q[0] - c1[0]) * ux + (q[1] - c1[1]) * uy; // along
    const b = (r1 - r2) / h, a = Math.sqrt(1 - b * b);
    const k = -b * px + a * py;
    if (k < 0) return hyp(px, py) - r1;
    if (k > a * h) return hyp(px, py - h) - r2;
    return a * px + b * py - r1;
  }

  // Gap between a part's footprint and a point, at one pose.
  function partPointGap(P, pose, q) {
    const c = P.circles;
    if (c.length === 2) return taperedDist(pose[c[0].k], c[0].r, pose[c[1].k], c[1].r, q);
    return coreDist(c.map((e) => pose[e.k]), [q]) - c[0].r;
  }
  // Conservative gap between two parts (each swollen by its largest radius).
  function partPairGap(P, Q, pose) {
    return coreDist(P.circles.map((e) => pose[e.k]), Q.circles.map((e) => pose[e.k])) - P.rMax - Q.rMax;
  }

  // ---------------------------------------------------------------------------
  // Design

  function resolve(params) {
    const p = Object.assign({}, DEFAULTS, params || {});
    const mat = MATERIALS[p.material] || MATERIALS[DEFAULTS.material];
    const hw = HARDWARE[p.hardware] || HARDWARE[DEFAULTS.hardware];
    const t = mat.t;
    const edgeRule = 2 * t; // SendCutSend's 5052 page: hole to edge >= 2x thickness
    const autoW = Math.ceil((hw.hole + 2 * edgeRule) * 2) / 2;
    const W = p.linkWidth > 0 ? p.linkWidth : autoW;
    const R = W / 2;
    const pinR = Math.max(R, SHAFT.linkHole / 2 + edgeRule); // link end on a crank pin
    const hubR = SHAFT.holeR + edgeRule; // crank arm ends (D-holes)
    return { p, mat, hw, t, edgeRule, autoW, W, R, pinR, hubR, s: p.scale, g: hw.washer, pitch: t + hw.washer };
  }

  // One "slab" is one crank phase: a pair of crank arms plus a rear and a
  // front leg. Pose keys: O, A (crank pin), then B..G with an R/F suffix.
  function slabParts(cfg) {
    const { R, pinR, hubR } = cfg;
    const parts = [
      { id: 'armI', kind: 'arm', circles: [{ k: 'O', r: hubR }, { k: 'A', r: hubR }], joints: ['O', 'A'] },
      { id: 'armO', kind: 'arm', circles: [{ k: 'O', r: hubR }, { k: 'A', r: hubR }], joints: ['O', 'A'] },
    ];
    for (const leg of ['R', 'F']) {
      for (const lp of LEG_PARTS) {
        const key = (k) => (k === 'A' ? 'A' : k + leg);
        parts.push({
          id: lp.key + leg, kind: lp.key, leg,
          circles: lp.pts.map((k) => ({ k: key(k), r: k === 'A' ? pinR : R })),
          joints: lp.holes.map(key),
        });
      }
    }
    for (const P of parts) P.rMax = Math.max(...P.circles.map((c) => c.r));
    return parts;
  }

  function slabPose(theta, s) {
    const r = legPose(theta, s, false), f = legPose(theta, s, true);
    const pose = { O: r.O, A: r.A };
    for (const k of ['B', 'C', 'D', 'E', 'F', 'G']) { pose[k + 'R'] = r[k]; pose[k + 'F'] = f[k]; }
    return pose;
  }

  // Sweep the mechanism through a full crank turn and record how close every
  // pair of parts, and every part and joint, ever get.
  function sweep(cfg, parts) {
    const n = cfg.p.samples;
    const poses = [];
    for (let i = 0; i < n; i++) poses.push(slabPose((i / n) * TAU, cfg.s));
    const jointNames = Object.keys(poses[0]);
    const pairGap = {}, jointGap = {};
    for (const P of parts) { pairGap[P.id] = {}; jointGap[P.id] = {}; }
    for (let a = 0; a < parts.length; a++) {
      const P = parts[a];
      for (let b = a + 1; b < parts.length; b++) {
        const Q = parts[b];
        let m = Infinity;
        for (const pose of poses) { m = Math.min(m, partPairGap(P, Q, pose)); if (m < -50) break; }
        pairGap[P.id][Q.id] = pairGap[Q.id][P.id] = m;
      }
      for (const J of jointNames) {
        let m = Infinity;
        for (const pose of poses) m = Math.min(m, partPointGap(P, pose, pose[J]));
        jointGap[P.id][J] = m;
      }
    }
    return { poses, pairGap, jointGap, jointNames };
  }

  // Put every part of a slab on a layer so nothing collides:
  //  - parts that share a pin sit on different layers;
  //  - parts on the same layer never touch;
  //  - a pin never passes through a part that isn't on it;
  //  - bolt heads and nuts never hit a part on a neighbouring layer;
  //  - j and k sweep straight across the crank axle, so they must sit between
  //    the two crank arms, where the built-up crankshaft has no axle.
  function solveLayers(cfg, parts, sw) {
    const { hw, t, pitch } = cfg, clr = cfg.p.clearance;
    const byId = Object.fromEntries(parts.map((P) => [P.id, P]));
    const pins = {}; // moving joint -> part ids on it
    for (const P of parts) for (const J of P.joints) {
      if (J === 'O' || J === 'BR' || J === 'BF') continue;
      (pins[J] = pins[J] || []).push(P.id);
    }
    const pinList = Object.keys(pins);
    const shares = (P, Q) => P.joints.some((J) => Q.joints.includes(J));
    const passKeep = (J) => (J === 'A' ? SHAFT.d / 2 : hw.rodD / 2) + clr;
    const axleKeep = SHAFT.d / 2 + clr;
    const hwKeep = hw.keepR + clr;
    const order = ['armI', 'armO', 'jR', 'kR', 'jF', 'kF', 'bdeR', 'cR', 'bdeF', 'cF', 'ghiR', 'fR', 'ghiF', 'fF'];

    function span(J, L) {
      let lo = Infinity, hi = -Infinity;
      for (const id of pins[J]) { if (L[id] === undefined) return null; lo = Math.min(lo, L[id]); hi = Math.max(hi, L[id]); }
      return [lo, hi];
    }

    // Bolt direction for one pin: head low or head high, whichever keeps the
    // head and nut clear of the parts they could hit.
    function orient(J, L, N) {
      const [lo, hi] = span(J, L);
      const out = [];
      for (const headLow of [true, false]) {
        const lowH = headLow ? hw.headH : hw.nutH, highH = headLow ? hw.nutH : hw.headH;
        const z0 = lo * pitch - lowH, z1 = hi * pitch + t + highH;
        let ok = true;
        for (const P of parts) {
          if (pins[J].includes(P.id)) continue;
          const pz0 = L[P.id] * pitch, pz1 = pz0 + t;
          const hit = (pz1 > z0 && pz0 < lo * pitch) || (pz0 < z1 && pz1 > hi * pitch + t);
          if (hit && sw.jointGap[P.id][J] < hwKeep) { ok = false; break; }
        }
        if (ok) out.push({ headLow, below: Math.max(0, -z0), above: Math.max(0, z1 - ((N - 1) * pitch + t)) });
      }
      return out;
    }

    function allowed(id, layer, L) {
      const P = byId[id];
      for (const qid in L) {
        if (L[qid] === layer && (shares(P, byId[qid]) || sw.pairGap[id][qid] < clr)) return false;
      }
      L[id] = layer;
      let ok = true;
      // The axle only exists outside the crank arms.
      if (L.armI !== undefined && L.armO !== undefined) {
        for (const qid in L) {
          const Q = byId[qid];
          if (Q.kind === 'arm') continue;
          if ((L[qid] < L.armI || L[qid] > L.armO) && sw.jointGap[qid].O < axleKeep) { ok = false; break; }
        }
      }
      // Pins can't pass through parts that aren't on them.
      for (const J of pinList) {
        if (!ok) break;
        const sp = span(J, L);
        if (!sp) continue;
        for (const qid in L) {
          if (pins[J].includes(qid)) continue;
          if (L[qid] > sp[0] && L[qid] < sp[1] && sw.jointGap[qid][J] < passKeep(J)) { ok = false; break; }
        }
      }
      delete L[id];
      return ok;
    }

    function search(N, limit) {
      const L = {}, found = [];
      (function rec(i) {
        if (found.length >= limit) return;
        if (i === order.length) {
          const opts = {};
          for (const J of pinList) {
            if (J === 'A') continue; // crank pin: a length of D-shaft, no bolt
            opts[J] = orient(J, L, N);
            if (!opts[J].length) return;
          }
          found.push({ layers: Object.assign({}, L), opts });
          return;
        }
        const id = order[i];
        for (let layer = 0; layer < N; layer++) {
          if (id === 'armO' && layer <= L.armI) continue; // mirror symmetry
          if (allowed(id, layer, L)) { L[id] = layer; rec(i + 1); delete L[id]; }
        }
      })(0);
      return found;
    }

    for (let N = 6; N <= 12; N++) {
      const sols = search(N, 600);
      if (!sols.length) continue;
      // Least hardware sticking out of the slab first, then shortest bolts.
      let best = null;
      for (const sol of sols) {
        let below = 0, above = 0, spans = 0;
        const headLow = {};
        for (const J in sol.opts) {
          const o = sol.opts[J].slice().sort((x, y) => x.below + x.above - (y.below + y.above))[0];
          headLow[J] = o.headLow;
          below = Math.max(below, o.below); above = Math.max(above, o.above);
          const [lo, hi] = span(J, sol.layers);
          spans += hi - lo;
        }
        const score = (below + above) * 10 + spans;
        if (!best || score < best.score) best = { N, layers: sol.layers, headLow, below, above, score, pins };
      }
      return best;
    }
    return null;
  }

  // Clawd, from the Claude Code welcome screen ("▐▛███▜▌" over "▝▜█████▛▘").
  // Pixel grid units, x right and y down; a pixel is twice as tall as wide,
  // like a terminal quadrant block.
  const CLAWD_OUTLINE = [[3, 0], [15, 0], [15, 2], [17, 2], [17, 3], [15, 3], [15, 4], [3, 4], [3, 3], [1, 3], [1, 2], [3, 2]];
  const CLAWD_EYES = [[5, 1], [12, 1]];

  function design(params) {
    const cfg = resolve(params);
    const { p, hw, t, s, R, pitch } = cfg;
    const H = HOLY, clr = p.clearance;
    const issues = [];

    const parts = slabParts(cfg);
    const sw = sweep(cfg, parts);

    // The fixed hip pivots are threaded rods through every layer of a side.
    const rodKeep = hw.rodD / 2 + clr;
    for (const P of parts) for (const J of ['BR', 'BF']) {
      if (P.joints.includes(J)) continue;
      const gap = sw.jointGap[P.id][J];
      if (gap < rodKeep) issues.push({ level: 'fail', text: `${P.id} sweeps within ${Math.max(0, gap).toFixed(1)} mm of the hip pivot rod (needs ${rodKeep.toFixed(1)} mm). Increase the size or narrow the links.` });
    }

    const layout = solveLayers(cfg, parts, sw);
    if (!layout) issues.push({ level: 'fail', text: 'No collision-free layer order exists at this size and link width. Increase the size or narrow the links.' });

    // Z stack for one side, measured outward from the inner plate's inner face.
    const n = p.slabs;
    let stack = null;
    if (layout) {
      const N = layout.N, slabT = (N - 1) * pitch + t;
      const gapIn = Math.max(cfg.g, layout.below + clr);
      const gapOut = Math.max(cfg.g, layout.above + clr);
      const gapMid = Math.max(cfg.g, layout.above + layout.below + clr);
      const slabZ = [];
      let z = t + gapIn; // inner plate occupies [0, t]
      for (let k = 0; k < n; k++) { slabZ.push(z); z += slabT + (k < n - 1 ? gapMid : 0); }
      const outerZ = z + gapOut;
      stack = { N, slabT, gapIn, gapOut, gapMid, slabZ, outerZ, sideWidth: outerZ + t };
    }

    // Crank phases: left side 0, 360/n, ...; right side offset by 180/n.
    const phases = { L: [], R: [] };
    for (let k = 0; k < n; k++) { phases.L.push((360 * k) / n); phases.R.push((360 * k) / n + 180 / n); }

    // The foot slides backward while it's on the ground.
    const ground = footGround();
    const dir = ground.footDx > 0 ? -1 : 1; // crank direction that walks toward +x

    // Clawd plate: wide enough to hold the hip pivots, tall enough that the tie
    // rods pass above everything that moves.
    const holeR = hw.hole / 2;
    const edge = cfg.edgeRule;
    let legTop = -Infinity;
    for (const pose of sw.poses) for (const P of parts) for (const c of P.circles) legTop = Math.max(legTop, pose[c.k][1] + c.r);
    const bottom = -H.l * s - holeR - edge;
    const tieY = legTop + hw.spacerOD / 2 + clr;
    const topNeed = tieY + holeR + edge;
    const pxW = Math.max((H.a * s + holeR + edge) / 6, (topNeed - bottom) / 8);
    const pxH = pxW * 2;
    const top = bottom + 4 * pxH;
    const px = (x, y) => [(x - 9) * pxW, top - y * pxH];
    const tieX = 5 * pxW;
    const plate = {
      pxW, pxH, top, bottom, halfW: 6 * pxW,
      outline: CLAWD_OUTLINE.map(([x, y]) => px(x, y)),
      eyes: CLAWD_EYES.map(([x, y]) => [px(x, y), px(x + 1, y + 1)]),
      pivots: [[-H.a * s, -H.l * s], [H.a * s, -H.l * s]],
      ties: [[-tieX, tieY], [tieX, tieY]],
    };

    const cut = buildCutParts(cfg, plate, phases);
    checkParts(cfg, cut, issues);

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity;
    for (const pose of sw.poses) for (const P of parts) for (const c of P.circles) {
      xMin = Math.min(xMin, pose[c.k][0] - c.r); xMax = Math.max(xMax, pose[c.k][0] + c.r); yMin = Math.min(yMin, pose[c.k][1] - c.r);
    }
    for (const [x] of plate.outline) { xMin = Math.min(xMin, x); xMax = Math.max(xMax, x); }
    const width = stack ? 2 * stack.sideWidth + p.bodyGap : NaN;
    const size = { length: xMax - xMin, height: plate.top - yMin, width, footY: ground.footMin * s - R };

    const stability = stack ? stabilityOf(cfg, phases, stack, layout) : null;
    if (stability && stability.stableFrac < 0.999) {
      issues.push({ level: 'warn', text: `With ${4 * n} legs the body's centre falls outside the feet on the ground for about ${Math.round((1 - stability.stableFrac) * 360)}° of each crank turn, so it will rock. 12 legs keep it planted.` });
    }

    const assembly = stack ? buildAssembly(cfg, parts, sw, layout, stack, phases, size) : null;
    return { cfg, params: p, parts, sweep: sw, layout, stack, phases, dir, plate, cut, issues, size, stability, assembly, ground };
  }

  function footGround() {
    let yMin = Infinity, th0 = 0;
    const n = 720, ys = [];
    for (let i = 0; i < n; i++) {
      const th = (i / n) * TAU, G = rearLegUnits(th).G;
      ys.push(G[1]);
      if (G[1] < yMin) { yMin = G[1]; th0 = th; }
    }
    const a = rearLegUnits(th0 - 0.05).G, b = rearLegUnits(th0 + 0.05).G;
    return { footMin: yMin, footDx: b[0] - a[0], contactFrac: ys.filter((y) => y < yMin + 1).length / n };
  }

  // Static stability: at each crank angle, is the body's centre inside the
  // polygon of feet touching the ground?
  function stabilityOf(cfg, phases, stack, layout) {
    const { s, R, pitch } = cfg;
    const footZ = (side, k, front) => {
      const z = cfg.p.bodyGap / 2 + stack.slabZ[k] + layout.layers[front ? 'ghiF' : 'ghiR'] * pitch;
      return side === 'L' ? z : -z;
    };
    let stable = 0, minFeet = Infinity;
    for (let i = 0; i < 360; i++) {
      const feet = [];
      for (const side of ['L', 'R']) phases[side].forEach((ph, k) => {
        for (const front of [false, true]) {
          const G = legPose(rad(i + ph), s, front).G;
          feet.push({ x: G[0], y: G[1] - R, z: footZ(side, k, front) });
        }
      });
      const low = Math.min(...feet.map((f) => f.y));
      const down = feet.filter((f) => f.y < low + 0.5 * s);
      minFeet = Math.min(minFeet, down.length);
      if (pointInHull([0, 0], down.map((f) => [f.x, f.z]))) stable++;
    }
    return { stableFrac: stable / 360, minFeet };
  }

  function pointInHull(p, pts) {
    if (pts.length < 3) return false;
    const P = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [], upper = [];
    for (const q of P) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
    for (const q of P.reverse()) { while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
    const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
    if (hull.length < 3) return false;
    for (let i = 0; i < hull.length; i++) if (cross(hull[i], hull[(i + 1) % hull.length], p) < 0) return false;
    return true;
  }

  // ---------------------------------------------------------------------------
  // Flat parts

  // Rigid copy of some points with the first at the origin, second on +x.
  function localFrame(pts) {
    const [p0, p1] = pts;
    const ang = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);
    const c = Math.cos(-ang), sn = Math.sin(-ang);
    return pts.map(([x, y]) => [(x - p0[0]) * c - (y - p0[1]) * sn, (x - p0[0]) * sn + (y - p0[1]) * c]);
  }

  function buildCutParts(cfg, plate, phases) {
    const { s, R, pinR, hubR, hw, p } = cfg;
    const holeR = hw.hole / 2;
    const pose = legPose(0.3, s, false);
    const out = [];

    for (const lp of LEG_PARTS) {
      const local = localFrame(lp.pts.map((k) => pose[k]));
      const circles = local.map(([x, y], i) => ({ x, y, r: lp.pts[i] === 'A' ? pinR : R }));
      const outer = hullOfCircles(local.length === 3 && !isCCW(local) ? [circles[0], circles[2], circles[1]] : circles);
      const holes = lp.holes.map((k) => {
        const [x, y] = local[lp.pts.indexOf(k)];
        return { circle: [x, y, k === 'A' ? SHAFT.linkHole / 2 : holeR] };
      });
      const cutouts = [];
      if (local.length === 3) { const pocket = trianglePocket(local, cfg); if (pocket) cutouts.push(pocket); }
      const d = (a, b) => dist(pose[a], pose[b]).toFixed(2);
      const centers = lp.pts.length === 2
        ? `${d(lp.pts[0], lp.pts[1])} mm between centres`
        : `${lp.pts[0]}–${lp.pts[1]} ${d(lp.pts[0], lp.pts[1])}, ${lp.pts[1]}–${lp.pts[2]} ${d(lp.pts[1], lp.pts[2])}, ${lp.pts[0]}–${lp.pts[2]} ${d(lp.pts[0], lp.pts[2])} mm`;
      out.push({ id: lp.key, name: lp.label, qty: 4 * p.slabs, file: `clawdbeast-${lp.key}`, outer, holes, cutouts, centers, anchors: local });
    }

    // Crank arms. Both ends are D-holes: the pin's flat points along the arm,
    // and the axle's flat sits 90° - phase from it. Flipped over, an arm for
    // phase φ serves phase 180° - φ, so there's one file per pair.
    const L = HOLY.m * s;
    const counts = {};
    for (const side of ['L', 'R']) for (const ph of phases[side]) {
      const a = ((ph % 360) + 360) % 360, b = (((180 - a) % 360) + 360) % 360;
      const key = Math.min(a, b);
      counts[key] = (counts[key] || 0) + 2;
    }
    for (const key of Object.keys(counts).map(Number).sort((a, b) => a - b)) {
      out.push({
        id: `arm${key}`, name: `Crank arm ${key}°`, qty: counts[key], file: `clawdbeast-crank-arm-${String(key).padStart(3, '0')}`,
        outer: hullOfCircles([{ x: 0, y: 0, r: hubR }, { x: L, y: 0, r: hubR }]),
        holes: [{ poly: dHole(0, 0, SHAFT.holeR, SHAFT.holeFlat, rad(90 - key)) }, { poly: dHole(L, 0, SHAFT.holeR, SHAFT.holeFlat, 0) }],
        cutouts: [], centers: `${L.toFixed(2)} mm throw`, phase: key,
      });
    }

    out.push({
      id: 'plate', name: 'Clawd plate', qty: 4, file: 'clawdbeast-clawd-plate',
      outer: polygon(plate.outline),
      holes: [
        { circle: [0, 0, SHAFT.plateHole / 2] },
        ...plate.pivots.map(([x, y]) => ({ circle: [x, y, holeR] })),
        ...plate.ties.map(([x, y]) => ({ circle: [x, y, holeR] })),
      ],
      cutouts: plate.eyes.map(([a, b]) => polygon([[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]])),
      centers: 'Chassis side; the eyes are windows',
      coat: true,
    });

    const hl = 40;
    out.push({
      id: 'handle', name: 'Hand crank', qty: 1, file: 'clawdbeast-hand-crank',
      outer: hullOfCircles([{ x: 0, y: 0, r: hubR }, { x: hl, y: 0, r: Math.max(R, holeR + cfg.edgeRule) }]),
      holes: [{ poly: dHole(0, 0, SHAFT.holeR, SHAFT.holeFlat, Math.PI / 2) }, { circle: [hl, 0, holeR] }],
      cutouts: [], centers: `${hl} mm throw; a screw and spacer make the knob`,
    });

    for (const part of out) {
      part.bbox = bbox([part.outer]);
      part.contours = [part.outer, ...part.cutouts, ...part.holes.map((h) => h.poly || circleContour(h.circle))];
    }
    return out;
  }

  // Lightening pocket that keeps a bar of link width along every edge.
  function trianglePocket(pts, cfg) {
    const rp = Math.max(2, cfg.t);
    const shrunk = shrinkTriangle(pts, cfg.R + rp);
    if (!shrunk || shrunk.inradius < 1.5) return null;
    const need = Math.max(cfg.pinR, cfg.hw.hole / 2 + cfg.edgeRule);
    for (const v of pts) for (const c of shrunk.pts) if (dist(v, c) - rp < need) return null;
    const sp = isCCW(shrunk.pts) ? shrunk.pts : [shrunk.pts[0], shrunk.pts[2], shrunk.pts[1]];
    return hullOfCircles(sp.map(([x, y]) => ({ x, y, r: rp })));
  }

  // ---------------------------------------------------------------------------
  // Checks against SendCutSend's published rules

  function checkParts(cfg, cut, issues) {
    const { t, mat, hw } = cfg;
    const minPart = mat.minPartIn.map((v) => v * IN);
    if (hw.hole < t) issues.push({ level: 'fail', text: `Pivot holes (${hw.hole} mm) are smaller than the material thickness (${t.toFixed(2)} mm), SendCutSend's minimum.` });
    for (const part of cut) {
      const dims = [part.bbox.w, part.bbox.h].sort((a, b) => a - b);
      if (dims[0] < minPart[0] || dims[1] < minPart[1]) issues.push({ level: 'fail', text: `${part.name} is under the ${mat.minPartIn[0]}" × ${mat.minPartIn[1]}" minimum part size.` });
      if (dims[1] > 44 * IN || dims[0] > 30 * IN) issues.push({ level: 'fail', text: `${part.name} is over the 30" × 44" instant-quote maximum.` });
    }
    const web = cfg.R - hw.hole / 2;
    if (web < 0.5 * t) issues.push({ level: 'fail', text: `Links leave ${web.toFixed(2)} mm around each hole, under the laser minimum of half the thickness (${(0.5 * t).toFixed(2)} mm).` });
    else if (web < cfg.edgeRule - 1e-6) issues.push({ level: 'warn', text: `Links leave ${web.toFixed(2)} mm between hole and edge. SendCutSend recommends 2× thickness (${cfg.edgeRule.toFixed(2)} mm).` });
  }

  // ---------------------------------------------------------------------------
  // Assembly: what goes where along each pin, and the hardware to buy

  function pickLength(need, list) {
    for (const L of list) if (L >= need - 1e-6) return L;
    return null;
  }

  function buildAssembly(cfg, parts, sw, layout, stack, phases, size) {
    const { hw, t, pitch, p } = cfg;
    const clr = p.clearance;
    const n = p.slabs;
    const lay = layout.layers;
    const zOf = (k, L) => stack.slabZ[k] + L * pitch; // lower face of a layer

    // Layers of one slab, for the stack chart.
    const layers = [];
    for (let L = 0; L < layout.N; L++) layers.push(parts.filter((P) => lay[P.id] === L).map((P) => P.id));

    // Where a spacer would hit something sweeping past, leave the pin bare.
    const sweepsNear = (J, L, keep) => parts.some((P) => lay[P.id] === L && !P.joints.includes(J) && sw.jointGap[P.id][J] < keep);

    const bolts = {}, spacers = {}, shaftSpacers = {};
    const add = (m, len, qty) => {
      const k = (Math.round(len * 2) / 2).toFixed(1);
      if (+k <= hw.washer + 0.05) return;
      m[k] = (m[k] || 0) + qty;
    };
    let washers = 0, nuts = 0;
    const pinStacks = [];
    for (const J in layout.pins) {
      if (J === 'A') continue;
      const ids = layout.pins[J].slice().sort((a, b) => lay[a] - lay[b]);
      const lo = lay[ids[0]], hi = lay[ids[ids.length - 1]];
      const grip = (hi - lo) * pitch + t;
      const len = pickLength(grip + 2 * hw.washer + hw.nutLen + 0.5, hw.lengths);
      const qty = 2 * n;
      bolts[len] = (bolts[len] || 0) + qty;
      nuts += qty; washers += 2 * qty;
      const items = [];
      for (let i = 0; i < ids.length; i++) {
        items.push({ part: ids[i] });
        if (i < ids.length - 1) {
          const gapL = lay[ids[i + 1]] - lay[ids[i]];
          if (gapL === 1) { items.push({ washer: true }); washers += qty; }
          else {
            let bare = false;
            for (let L = lay[ids[i]] + 1; L < lay[ids[i + 1]]; L++) if (sweepsNear(J, L, hw.spacerOD / 2 + clr)) bare = true;
            const len2 = gapL * pitch - t;
            items.push(bare ? { bare: len2 } : { spacer: len2 });
            if (!bare) add(spacers, len2, qty);
          }
        }
      }
      pinStacks.push({ joint: J, items, bolt: len, headLow: layout.headLow[J] });
    }

    // Hip pivots: threaded rod through the whole side, spacers between parts.
    const hipStacks = [];
    for (const leg of ['R', 'F']) {
      const J = 'B' + leg;
      const onPin = ['bde' + leg, 'c' + leg].sort((a, b) => lay[a] - lay[b]);
      const seq = [{ z0: 0, z1: t, part: 'plate' }];
      for (let k = 0; k < n; k++) for (const id of onPin) seq.push({ z0: zOf(k, lay[id]), z1: zOf(k, lay[id]) + t, part: id, slab: k });
      seq.push({ z0: stack.outerZ, z1: stack.outerZ + t, part: 'plate' });
      const items = [];
      for (let i = 0; i < seq.length; i++) {
        items.push(seq[i]);
        if (i === seq.length - 1) break;
        const gap = seq[i + 1].z0 - seq[i].z1;
        // Is the gap crossed by anything that sweeps near the rod?
        let bare = false;
        for (let k = 0; k < n; k++) for (let L = 0; L < layout.N; L++) {
          const z0 = zOf(k, L), z1 = z0 + t;
          if (z1 > seq[i].z1 + 1e-6 && z0 < seq[i + 1].z0 - 1e-6 && sweepsNear(J, L, hw.spacerOD / 2 + clr)) bare = true;
        }
        if (gap <= hw.washer + 0.05) { items.push({ washer: true }); washers += 2; }
        else if (bare) items.push({ bare: gap });
        else { items.push({ spacer: gap }); add(spacers, gap, 2); }
      }
      hipStacks.push({ joint: J, items });
    }
    const hipRod = stack.sideWidth + 2 * (hw.washer + hw.nutLen) + 3;
    nuts += 4 * 2; washers += 4 * 2;

    // Crankshaft: short D-shaft pieces keyed into the crank arms.
    const aI = Math.min(lay.armI, lay.armO), aO = Math.max(lay.armI, lay.armO);
    const armZ = (k, which) => zOf(k, which === 'I' ? aI : aO);
    const cuts = [];
    const main = p.bodyGap + 2 * (armZ(0, 'I') + t);
    cuts.push({ what: 'Main axle (through the body, into both first crank arms)', len: main, qty: 1 });
    cuts.push({ what: 'Crank pin (between a pair of arms)', len: (aO - aI) * pitch + t, qty: 2 * n });
    if (n > 1) cuts.push({ what: 'Link stub (joins one phase to the next)', len: armZ(1, 'I') + t - armZ(0, 'O'), qty: 2 * (n - 1) });
    const endBase = stack.outerZ + t - armZ(n - 1, 'O');
    cuts.push({ what: 'End stub (through the outer plate, plus collar)', len: endBase + SHAFT.collarW + 1, qty: 1 });
    cuts.push({ what: 'End stub, hand-crank side', len: endBase + SHAFT.collarW + t + 4, qty: 1 });
    const shaftTotal = cuts.reduce((a, c) => a + c.len * c.qty, 0) + cuts.reduce((a, c) => a + c.qty, 0) * 1.5;
    // Spacers on the axle, stubs and plates keep the arms in place.
    add(shaftSpacers, armZ(0, 'I') - t, 2);
    for (let k = 0; k < n - 1; k++) add(shaftSpacers, armZ(k + 1, 'I') - (armZ(k, 'O') + t), 2);
    add(shaftSpacers, stack.outerZ - (armZ(n - 1, 'O') + t), 2);

    const tieRod = size.width + 2 * (hw.nutLen + 2);
    nuts += 2 * 4 * 2; // two tie rods, a nut each side of all four plates

    const items = [];
    items.push({ what: `6 mm D-shaft, ${Math.ceil(shaftTotal / 10) * 10} mm total, cut into the pieces below`, qty: 1 });
    items.push({ what: '6 mm shaft collar', qty: 2 });
    for (const L of Object.keys(bolts).sort((a, b) => a - b)) items.push({ what: `${hw.bolt}, ${L} mm`, qty: bolts[L] });
    items.push({ what: `${hw.bolt}, 16 mm (hand-crank knob)`, qty: 1 });
    items.push({ what: `${hw.rod}, ${Math.ceil(hipRod)} mm (hip pivots)`, qty: 4 });
    items.push({ what: `${hw.rod}, ${Math.ceil(tieRod)} mm (tie rods)`, qty: 2 });
    items.push({ what: hw.nut, qty: nuts + 1 });
    items.push({ what: `${hw.name} nylon washer, ${hw.washer} mm`, qty: washers });
    for (const L of Object.keys(spacers).sort((a, b) => a - b)) items.push({ what: `${hw.name} round spacer, ${L} mm long, ${hw.spacerOD} mm OD max`, qty: spacers[L] });
    for (const L of Object.keys(shaftSpacers).sort((a, b) => a - b)) items.push({ what: `6 mm bore shaft spacer, ${L} mm long, ${SHAFT.spacerOD} mm OD max`, qty: shaftSpacers[L] });
    items.push({ what: 'Retaining compound for the D-shaft pieces (e.g. Loctite 609)', qty: 1 });

    return { layers, pinStacks, hipStacks, cuts, items, aI, aO };
  }

  // ---------------------------------------------------------------------------
  // Export

  function fmt(v) { return (Math.abs(v) < 1e-9 ? 0 : v).toFixed(6); }

  // DXF R12 using only LINE, ARC and CIRCLE: every closed contour is made of
  // segments whose endpoints meet exactly. One part per file, 1:1 scale.
  function toDXF(part, units) {
    const k = units === 'mm' ? 1 : 1 / IN;
    const L = [];
    const w = (...a) => { for (const x of a) L.push(String(x)); };
    w(0, 'SECTION', 2, 'HEADER', 9, '$ACADVER', 1, 'AC1009', 9, '$INSUNITS', 70, units === 'mm' ? 4 : 1, 0, 'ENDSEC');
    w(0, 'SECTION', 2, 'ENTITIES');
    for (const h of part.holes) if (h.circle) {
      const [x, y, r] = h.circle;
      w(0, 'CIRCLE', 8, 0, 10, fmt(x * k), 20, fmt(y * k), 30, fmt(0), 40, fmt(r * k));
    }
    const contours = [part.outer, ...part.cutouts, ...part.holes.filter((h) => h.poly).map((h) => h.poly)];
    for (const c of contours) {
      for (let i = 0; i < c.length; i++) {
        const P = c[i], Q = c[(i + 1) % c.length];
        if (Math.abs(P.b) < 1e-12) {
          if (hyp(Q.x - P.x, Q.y - P.y) < 1e-9) continue;
          w(0, 'LINE', 8, 0, 10, fmt(P.x * k), 20, fmt(P.y * k), 30, fmt(0), 11, fmt(Q.x * k), 21, fmt(Q.y * k), 31, fmt(0));
        } else {
          const a = arcCenter(P, Q, P.b);
          let s0 = deg(Math.atan2(P.y - a.y, P.x - a.x)), s1 = deg(Math.atan2(Q.y - a.y, Q.x - a.x));
          if (a.th < 0) [s0, s1] = [s1, s0];
          w(0, 'ARC', 8, 0, 10, fmt(a.x * k), 20, fmt(a.y * k), 30, fmt(0), 40, fmt(a.r * k), 50, fmt((s0 + 360) % 360), 51, fmt((s1 + 360) % 360));
        }
      }
    }
    w(0, 'ENDSEC', 0, 'EOF');
    return L.join('\n') + '\n';
  }

  // SVG path data in the part's own (y-up) coordinates.
  function toSVGPath(contour) {
    let d = '';
    contour.forEach((P, i) => {
      const Q = contour[(i + 1) % contour.length];
      if (i === 0) d += `M${P.x.toFixed(3)} ${P.y.toFixed(3)}`;
      if (Math.abs(P.b) < 1e-12) d += `L${Q.x.toFixed(3)} ${Q.y.toFixed(3)}`;
      else {
        const a = arcCenter(P, Q, P.b);
        d += `A${a.r.toFixed(3)} ${a.r.toFixed(3)} 0 ${Math.abs(a.th) > Math.PI ? 1 : 0} ${a.th > 0 ? 1 : 0} ${Q.x.toFixed(3)} ${Q.y.toFixed(3)}`;
      }
    });
    return d + 'Z';
  }

  return {
    HOLY, MATERIALS, HARDWARE, SHAFT, DEFAULTS, LEG_PARTS, IN,
    design, legPose, slabPose, rearLegUnits, hullOfCircles, dHole, toDXF, toSVGPath, flatten, arcCenter,
    partPointGap, partPairGap, coreDist, isCCW,
  };
});
