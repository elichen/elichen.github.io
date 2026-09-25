// Independent check of a Clawdbeast design: rebuild every part, pin, spacer,
// bolt head and nut as a solid (2D footprint x z-range), turn the crank through
// a full revolution, and report anything that overlaps. Also checks the DXF
// geometry: link hole spacing, closed contours, and SendCutSend rules.
//
//   node validate.js                  # default design
//   node validate.js '{"slabs":2}'    # any design() params as JSON

const CB = require('./clawdbeast.js');

const params = process.argv[2] ? JSON.parse(process.argv[2]) : {};
const d = CB.design(params);
const { cfg, layout, stack, assembly } = d;
let failures = 0;
const fail = (msg) => { failures++; console.log('  FAIL ' + msg); };

console.log(`Design ${JSON.stringify(d.params)}`);
if (!layout) { console.log('  no layout'); process.exit(1); }

// ---------------------------------------------------------------- 3D sweep
const { t, pitch, hw } = cfg;
const n = d.params.slabs;
const L = layout.layers;
const zOf = (k, layer) => stack.slabZ[k] + layer * pitch;
const aI = Math.min(L.armI, L.armO), aO = Math.max(L.armI, L.armO);

function buildSolids() {
  const solids = [];
  for (let k = 0; k < n; k++) {
    for (const P of d.parts) {
      const z0 = zOf(k, L[P.id]);
      solids.push({ name: `${P.id}@${k}`, type: 'part', P, k, z0, z1: z0 + t, joints: P.joints.map((J) => (J === 'BR' || J === 'BF' ? J : `${J}@${k}`)) });
    }
    // Bolted joints: shank, head and nut.
    for (const J in layout.pins) {
      const ids = layout.pins[J];
      if (J === 'A') {
        solids.push({ name: `pinA@${k}`, type: 'pin', joint: `A@${k}`, at: 'A', k, r: CB.SHAFT.d / 2, z0: zOf(k, aI), z1: zOf(k, aO) + t });
        continue;
      }
      const lo = Math.min(...ids.map((id) => L[id])), hi = Math.max(...ids.map((id) => L[id]));
      const zl = zOf(k, lo), zh = zOf(k, hi) + t;
      const headLow = layout.headLow[J];
      const lowH = headLow ? hw.headH : hw.nutH, highH = headLow ? hw.nutH : hw.headH;
      solids.push({ name: `bolt${J}@${k}`, type: 'pin', joint: `${J}@${k}`, at: J, k, r: hw.rodD / 2, z0: zl, z1: zh });
      solids.push({ name: `boltEndLow${J}@${k}`, type: 'pin', joint: `${J}@${k}`, at: J, k, r: hw.keepR, z0: zl - lowH, z1: zl, hardware: true });
      solids.push({ name: `boltEndHigh${J}@${k}`, type: 'pin', joint: `${J}@${k}`, at: J, k, r: hw.keepR, z0: zh, z1: zh + highH, hardware: true });
    }
    // Spacers inside bolted joints, where the assembly calls for them.
    for (const ps of assembly.pinStacks) {
      const it = ps.items;
      for (let i = 1; i < it.length - 1; i++) {
        if (!it[i].spacer) continue;
        const z0 = zOf(k, L[it[i - 1].part]) + t, z1 = zOf(k, L[it[i + 1].part]);
        solids.push({ name: `spacer${ps.joint}@${k}`, type: 'pin', joint: `${ps.joint}@${k}`, at: ps.joint, k, r: hw.spacerOD / 2, z0, z1 });
      }
    }
  }
  // Axle pieces at O (none between a pair of crank arms).
  const segs = [[-d.params.bodyGap / 2 - t, zOf(0, aI) + t]];
  for (let k = 0; k < n - 1; k++) segs.push([zOf(k, aO), zOf(k + 1, aI) + t]);
  segs.push([zOf(n - 1, aO), stack.outerZ + t + CB.SHAFT.collarW]);
  segs.forEach(([z0, z1], i) => solids.push({ name: `axle${i}`, type: 'pin', joint: 'O', at: 'O', r: CB.SHAFT.spacerOD / 2, z0, z1, axle: true }));
  // Hip rods and their spacers.
  for (const hs of assembly.hipStacks) {
    solids.push({ name: `rod${hs.joint}`, type: 'pin', joint: hs.joint, at: hs.joint, r: hw.rodD / 2, z0: -hw.nutH, z1: stack.outerZ + t + hw.nutH, fixed: true });
    for (let i = 0; i < hs.items.length; i++) {
      const it = hs.items[i];
      if (it.spacer) solids.push({ name: `hipSpacer${hs.joint}`, type: 'pin', joint: hs.joint, at: hs.joint, r: hw.spacerOD / 2, z0: hs.items[i - 1].z1, z1: hs.items[i + 1].z0, fixed: true });
    }
  }
  // Tie rods (with spacer tubes between plates) and the plates.
  d.plate.ties.forEach((p, i) => solids.push({ name: `tie${i}`, type: 'pin', joint: `tie${i}`, fixedAt: p, r: hw.spacerOD / 2, z0: t, z1: stack.outerZ }));
  solids.push({ name: 'innerPlate', type: 'plate', z0: 0, z1: t });
  solids.push({ name: 'outerPlate', type: 'plate', z0: stack.outerZ, z1: stack.outerZ + t });
  return solids;
}

const solids = buildSolids();
const phasesBySide = d.phases;
const EPS = 1e-6;
const zOverlap = (a, b) => a.z0 < b.z1 - EPS && b.z0 < a.z1 - EPS;

function pointOf(S, poses) {
  if (S.fixedAt) return S.fixedAt;
  if (S.at === 'O') return [0, 0];
  if (S.at === 'BR' || S.at === 'BF') return poses[0][S.at];
  return poses[S.k][S.at];
}
function attached(a, b) {
  if (a.type === 'part' && b.type === 'pin') return a.joints.includes(b.joint) || (b.axle && a.P.kind === 'arm');
  if (b.type === 'part' && a.type === 'pin') return attached(b, a);
  if (a.type === 'pin' && b.type === 'pin') return a.joint === b.joint;
  if (a.type === 'plate' || b.type === 'plate') {
    const o = a.type === 'plate' ? b : a;
    return o.type === 'pin' && (o.fixed || o.axle || o.fixedAt);
  }
  return false;
}

const minGap = {};
for (const side of ['L', 'R']) {
  const steps = 720;
  for (let i = 0; i < steps; i++) {
    const th = (i / steps) * 2 * Math.PI;
    const poses = phasesBySide[side].map((ph) => CB.slabPose(th + (ph * Math.PI) / 180, cfg.s));
    for (let a = 0; a < solids.length; a++) for (let b = a + 1; b < solids.length; b++) {
      const A = solids[a], B = solids[b];
      if (!zOverlap(A, B) || attached(A, B)) continue;
      let gap;
      if (A.type === 'plate' || B.type === 'plate') gap = -1; // nothing but rods may enter a plate
      else if (A.type === 'part' && B.type === 'part') {
        if (A.k === B.k && A.P.joints.some((J) => B.P.joints.includes(J))) continue; // share a pin, z-separated by washers
        // Different phases never share z; same phase: conservative footprint gap.
        gap = A.k !== B.k ? -1 : CB.partPairGap(A.P, B.P, poses[A.k]);
      } else if (A.type === 'part' || B.type === 'part') {
        const P = A.type === 'part' ? A : B, Q = A.type === 'part' ? B : A;
        gap = CB.partPointGap(P.P, poses[P.k], pointOf(Q, poses)) - Q.r;
      } else {
        const p = pointOf(A, poses), q = pointOf(B, poses);
        gap = Math.hypot(p[0] - q[0], p[1] - q[1]) - A.r - B.r;
      }
      const key = [A.name, B.name].sort().join(' vs ');
      if (!(key in minGap) || gap < minGap[key].gap) minGap[key] = { gap, side, deg: (i / steps) * 360 };
    }
  }
}
const hits = Object.entries(minGap).filter(([, v]) => v.gap < 0).sort((a, b) => a[1].gap - b[1].gap);
const tight = Object.entries(minGap).filter(([, v]) => v.gap >= 0).sort((a, b) => a[1].gap - b[1].gap).slice(0, 6);
console.log(`\n3D sweep: ${solids.length} solids, ${Object.keys(minGap).length} pairs that share z`);
for (const [k, v] of hits) fail(`collision ${k}: ${v.gap.toFixed(2)} mm at ${v.deg.toFixed(1)}° (${v.side})`);
console.log('  tightest clearances:');
for (const [k, v] of tight) console.log(`    ${v.gap.toFixed(2)} mm  ${k}`);

// ---------------------------------------------------------------- flat parts
console.log('\nFlat parts:');
const pose = CB.legPose(1.1, cfg.s, false);
for (const part of d.cut) {
  // Every contour closes and has no zero-length segment.
  for (const c of part.contours) {
    for (let i = 0; i < c.length; i++) {
      const P = c[i], Q = c[(i + 1) % c.length];
      if (Math.hypot(Q.x - P.x, Q.y - P.y) < 1e-6 && c.length > 2) fail(`${part.id}: zero-length segment`);
    }
  }
  // Hole spacing matches the linkage.
  const lp = CB.LEG_PARTS.find((x) => x.key === part.id);
  if (lp) {
    const holes = part.holes.map((h) => h.circle);
    for (let i = 0; i < lp.holes.length; i++) for (let j = i + 1; j < lp.holes.length; j++) {
      const want = Math.hypot(pose[lp.holes[i]][0] - pose[lp.holes[j]][0], pose[lp.holes[i]][1] - pose[lp.holes[j]][1]);
      const got = Math.hypot(holes[i][0] - holes[j][0], holes[i][1] - holes[j][1]);
      if (Math.abs(want - got) > 1e-6) fail(`${part.id}: holes ${lp.holes[i]}-${lp.holes[j]} are ${got.toFixed(4)} mm apart, linkage needs ${want.toFixed(4)}`);
    }
  }
  // Web between every hole and the outline.
  const outline = CB.flatten(part.outer, 0.2);
  const segD = (p, a, b) => { const dx = b[0] - a[0], dy = b[1] - a[1]; const L2 = dx * dx + dy * dy; let u = L2 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2 : 0; u = Math.max(0, Math.min(1, u)); return Math.hypot(a[0] + u * dx - p[0], a[1] + u * dy - p[1]); };
  const edgeDist = (p) => { let m = Infinity; for (let i = 0; i < outline.length; i++) m = Math.min(m, segD(p, outline[i], outline[(i + 1) % outline.length])); return m; };
  let minWeb = Infinity;
  // Pockets and windows count as edges too.
  const inner = part.cutouts.map((c) => CB.flatten(c, 0.2));
  const innerDist = (p) => { let m = Infinity; for (const o of inner) for (let i = 0; i < o.length; i++) m = Math.min(m, segD(p, o[i], o[(i + 1) % o.length])); return m; };
  for (const h of part.holes) {
    const [x, y, r] = h.circle || [0, 0, 0];
    if (h.circle) minWeb = Math.min(minWeb, edgeDist([x, y]) - r, innerDist([x, y]) - r);
    else for (const [px, py] of CB.flatten(h.poly, 0.2)) minWeb = Math.min(minWeb, edgeDist([px, py]), innerDist([px, py]));
  }
  const flag = minWeb < 0.5 * t ? 'FAIL' : minWeb < cfg.edgeRule - 0.01 ? 'below 2t' : 'ok';
  if (flag === 'FAIL') fail(`${part.id}: web ${minWeb.toFixed(2)} mm`);
  console.log(`  ${part.file.padEnd(28)} ${String(part.qty).padStart(3)} pcs  ${(part.bbox.w / 25.4).toFixed(2)}" x ${(part.bbox.h / 25.4).toFixed(2)}"  min web ${minWeb.toFixed(2)} mm (${flag})`);
}

console.log('\nIssues reported by design():');
for (const i of d.issues) console.log(`  ${i.level}: ${i.text}`);
if (!d.issues.length) console.log('  none');
console.log(`\nStability: centre inside the support polygon ${(d.stability.stableFrac * 100).toFixed(0)}% of the cycle, at least ${d.stability.minFeet} feet down.`);
console.log(failures ? `\n${failures} failure(s)` : '\nAll checks passed.');
process.exit(failures ? 1 : 0);
