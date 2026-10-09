// Procedural Eciton hamatum worker, after AntWeb specimen photos: rounded head with a
// single-facet eye, elbowed antennae, a long mesosoma, two waist nodes, an oval gaster
// and long legs. Built in "rig units" (a media worker is 0.67 long from gaster tip to
// the front of the head); the renderer scales each ant to its own size.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export const RIG_LENGTH = 0.67;

// a body part as a loft of rounded rings along +x
// rings: [x, yCentre, halfWidth, halfHeight]; n > 2 makes the cross-section boxier
function loft(rings, { seg = 18, n = 2.2, color = null } = {}) {
    const pos = [], col = [], idx = [];
    const R = rings.length;
    for (let r = 0; r < R; r++) {
        const [x, yc, hw, hh] = rings[r];
        for (let s = 0; s <= seg; s++) {
            const a = s / seg * Math.PI * 2, c = Math.cos(a), sn = Math.sin(a);
            const e = 2 / n;
            const y = yc + hh * Math.sign(sn) * Math.abs(sn) ** e;
            const z = hw * Math.sign(c) * Math.abs(c) ** e;
            pos.push(x, y, z);
            const k = color ? color(x, y, z, r / (R - 1)) : 1;
            col.push(k, k, k);
        }
    }
    for (let r = 0; r < R - 1; r++) for (let s = 0; s < seg; s++) {
        const a = r * (seg + 1) + s, b = a + seg + 1;
        idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
}

// smooth profile from control points [t, value], sampled at t
function profile(points, t) {
    for (let i = 0; i < points.length - 1; i++) {
        const [t0, v0] = points[i], [t1, v1] = points[i + 1];
        if (t <= t1) { const u = (t - t0) / (t1 - t0), s = u * u * (3 - 2 * u); return v0 + (v1 - v0) * s; }
    }
    return points[points.length - 1][1];
}

function sampled(x0, x1, count, fy, fw, fh) {
    const rings = [];
    for (let i = 0; i <= count; i++) {
        const t = i / count, x = x0 + (x1 - x0) * t;
        rings.push([x, fy(t), Math.max(1e-4, fw(t)), Math.max(1e-4, fh(t))]);
    }
    return rings;
}

// a tube with varying radius along a polyline (for mandibles and the funiculus)
function tube(points, radii, { seg = 8, color = null } = {}) {
    const pos = [], col = [], idx = [];
    const up = new THREE.Vector3(0, 1, 0), t = new THREE.Vector3(), nrm = new THREE.Vector3(), bin = new THREE.Vector3();
    for (let i = 0; i < points.length; i++) {
        const p = points[i];
        const a = points[Math.max(0, i - 1)], b = points[Math.min(points.length - 1, i + 1)];
        t.subVectors(b, a).normalize();
        nrm.crossVectors(t, Math.abs(t.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : up).normalize();
        bin.crossVectors(t, nrm);
        for (let s = 0; s <= seg; s++) {
            const ang = s / seg * Math.PI * 2;
            const r = radii[i];
            pos.push(p.x + (nrm.x * Math.cos(ang) + bin.x * Math.sin(ang)) * r,
                p.y + (nrm.y * Math.cos(ang) + bin.y * Math.sin(ang)) * r,
                p.z + (nrm.z * Math.cos(ang) + bin.z * Math.sin(ang)) * r);
            const k = color ? color(i / (points.length - 1)) : 1;
            col.push(k, k, k);
        }
    }
    for (let i = 0; i < points.length - 1; i++) for (let s = 0; s < seg; s++) {
        const a = i * (seg + 1) + s, b = a + seg + 1;
        idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
}

function sphere(r, x, y, z, k, seg = 8) {
    const g = new THREE.SphereGeometry(r, seg, seg - 2);
    g.deleteAttribute('uv');
    g.translate(x, y, z);
    const n = g.attributes.position.count;
    g.setAttribute('color', new THREE.Float32BufferAttribute(new Array(n * 3).fill(k), 3));
    return g;
}

// Sparse erect hairs (setae) on a body part: thin cones on the upper surface, leaning back.
function withHairs(g, count, len, seed) {
    let a = seed;
    const rnd = () => (a = (a * 16807) % 2147483647) / 2147483647;
    const pos = g.attributes.position, nrm = g.attributes.normal;
    const parts = [g];
    const n = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), q = new THREE.Quaternion();
    for (let k = 0, tries = 0; k < count && tries < count * 20; tries++) {
        const i = Math.floor(rnd() * pos.count);
        n.fromBufferAttribute(nrm, i);
        if (n.y < 0.15) continue;
        const d = n.clone().addScaledVector(new THREE.Vector3(-1, 0, 0), 0.35).normalize();
        const l = len * (0.6 + rnd() * 0.8);
        const h = new THREE.CylinderGeometry(0.0006, 0.0022, l, 3, 1, true);
        h.deleteAttribute('uv');
        h.translate(0, l / 2, 0);
        h.applyQuaternion(q.setFromUnitVectors(up, d));
        h.translate(pos.getX(i), pos.getY(i), pos.getZ(i));
        h.setAttribute('color', new THREE.Float32BufferAttribute(new Array(h.attributes.position.count * 3).fill(1.15), 3));
        parts.push(h);
        k++;
    }
    return mergeGeometries(parts);
}

function headGeometry(major) {
    const L = major ? 0.2 : 0.125, W = major ? 0.115 : 0.066, H = major ? 0.085 : 0.05;
    const x0 = 0.145;
    const parts = [];
    // shiny, rounded head; slightly darker toward the mouth
    parts.push(loft(sampled(x0, x0 + L, 14,
        t => 0.005 + profile([[0, 0.0], [0.5, 0.012], [1, 0.0]], t),
        t => W * profile([[0, 0.3], [0.12, 0.8], [0.45, 1], [0.8, 0.95], [1, 0.25]], t),
        t => H * profile([[0, 0.35], [0.15, 0.85], [0.45, 1], [0.8, 0.8], [1, 0.2]], t)),
        { n: 2.6, color: (x) => 1 - 0.25 * Math.max(0, (x - x0) / L - 0.7) / 0.3 }));
    // single-facet eyes
    for (const s of [-1, 1]) parts.push(sphere(major ? 0.011 : 0.008, x0 + L * 0.42, 0.02, s * W * 0.93, 0.06, 7));
    // mandibles
    for (const s of [-1, 1]) {
        const pts = [], rad = [];
        if (major) {
            // long sickles, curving inward and up at the tips
            for (let i = 0; i <= 14; i++) {
                const u = i / 14, ang = u * 1.6;
                pts.push(new THREE.Vector3(x0 + L * 0.88 + Math.sin(ang) * 0.17, -0.02 - 0.02 * u + Math.sin(u * 3) * 0.01, s * (W * 0.55 - (1 - Math.cos(ang)) * 0.11)));
                rad.push(0.016 * (1 - u * 0.85));
            }
        } else {
            for (let i = 0; i <= 6; i++) {
                const u = i / 6;
                pts.push(new THREE.Vector3(x0 + L * 0.9 + u * 0.06, -0.01 - 0.008 * u, s * (W * 0.55 - u * 0.045)));
                rad.push(0.014 * (1 - u * 0.7));
            }
        }
        parts.push(tube(pts, rad, { seg: 7, color: (u) => 0.75 - 0.55 * u }));
    }
    return mergeGeometries(parts);
}

function mesosomaGeometry() {
    // pronotum high and rounded at the front, mesonotum lower, propodeum at the back
    return loft(sampled(-0.1, 0.145, 20,
        t => profile([[0, 0.0], [0.25, 0.008], [0.55, 0.012], [0.8, 0.018], [1, 0.012]], t),
        t => profile([[0, 0.022], [0.12, 0.034], [0.35, 0.032], [0.55, 0.03], [0.78, 0.042], [0.95, 0.03], [1, 0.016]], t),
        t => profile([[0, 0.02], [0.1, 0.034], [0.3, 0.03], [0.5, 0.03], [0.75, 0.04], [0.93, 0.032], [1, 0.016]], t)),
        { n: 2.4 });
}

function nodeGeometry(x0, x1, w, h) {
    return loft(sampled(x0, x1, 10,
        t => profile([[0, -0.006], [0.5, 0.008], [1, -0.002]], t),
        t => w * profile([[0, 0.35], [0.3, 1], [0.7, 1], [1, 0.45]], t),
        t => h * profile([[0, 0.35], [0.35, 1], [0.7, 0.95], [1, 0.45]], t)), { n: 2.2, seg: 14 });
}

function gasterGeometry() {
    // oval and glossy, with pale hind margins on the segments
    return loft(sampled(-0.4, -0.208, 22,
        t => profile([[0, 0.0], [0.5, 0.004], [1, -0.004]], t),
        t => profile([[0, 0.006], [0.12, 0.05], [0.4, 0.07], [0.75, 0.064], [1, 0.022]], t),
        t => profile([[0, 0.004], [0.12, 0.044], [0.4, 0.062], [0.75, 0.056], [1, 0.02]], t)),
        { n: 2.1, seg: 22, color: (x) => { const b = (x + 0.4) / 0.192; return 0.92 + 0.12 * Math.max(0, Math.sin(b * Math.PI * 4.2 + 0.4)) ** 6; } });
}

// unit-length leg segment along +y, base at the origin, tapering from r0 to r1
function segment(r0, r1, { seg = 7, bumps = 0, dark = 1 } = {}) {
    const g = new THREE.CylinderGeometry(r1, r0, 1, seg, bumps ? bumps * 3 : 1, false);
    g.translate(0, 0.5, 0);
    const p = g.attributes.position;
    if (bumps) {
        for (let i = 0; i < p.count; i++) {
            const y = p.getY(i), k = 1 + 0.25 * Math.abs(Math.sin(y * bumps * Math.PI));
            p.setX(i, p.getX(i) * k); p.setZ(i, p.getZ(i) * k);
        }
        g.computeVertexNormals();
    }
    const n = p.count, col = [];
    for (let i = 0; i < n; i++) { const k = dark * (1 - 0.15 * p.getY(i)); col.push(k, k, k); }
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    return g;
}

function broodGeometry() {
    // a wasp or ant pupa: pale, plump, faintly segmented
    return loft(sampled(-0.24, 0.24, 24,
        () => 0,
        t => 0.085 * profile([[0, 0.15], [0.12, 0.8], [0.4, 1], [0.8, 0.85], [1, 0.2]], t) * (1 + 0.04 * Math.sin(t * 40)),
        t => 0.075 * profile([[0, 0.15], [0.12, 0.8], [0.4, 1], [0.8, 0.85], [1, 0.2]], t) * (1 + 0.04 * Math.sin(t * 40))),
        { n: 2, seg: 16, color: (x) => 0.94 + 0.06 * Math.sin(x * 60) });
}

export function buildAntGeometries() {
    return {
        head: withHairs(headGeometry(false), 22, 0.03, 3),
        headMajor: withHairs(headGeometry(true), 30, 0.035, 5),
        mesosoma: withHairs(mesosomaGeometry(), 18, 0.035, 7),
        petiole: nodeGeometry(-0.152, -0.1, 0.022, 0.026),
        postpetiole: nodeGeometry(-0.21, -0.152, 0.03, 0.032),
        gaster: withHairs(gasterGeometry(), 34, 0.04, 11),
        coxa: segment(0.02, 0.017),
        femur: segment(0.016, 0.012),
        tibia: segment(0.011, 0.009, { dark: 0.92 }),
        tarsus: segment(0.008, 0.005, { bumps: 5, dark: 0.85 }),
        scape: segment(0.007, 0.009),
        funiculus: segment(0.009, 0.007, { bumps: 11, dark: 0.7 }),
        brood: broodGeometry(),
    };
}

// legs: hip on the mesosoma, segment lengths, and where the foot rests when standing
const LEG_ROWS = [
    { hip: [0.1, -0.03, 0.028], coxa: 0.045, femur: 0.17, tibia: 0.15, tarsus: 0.14, rest: [0.33, 0.27] },
    { hip: [0.035, -0.03, 0.03], coxa: 0.04, femur: 0.2, tibia: 0.18, tarsus: 0.15, rest: [0.06, 0.4] },
    { hip: [-0.035, -0.03, 0.03], coxa: 0.045, femur: 0.25, tibia: 0.23, tarsus: 0.18, rest: [-0.36, 0.38] },
];

export const LEGS = [];
for (let row = 0; row < 3; row++) for (const side of [-1, 1]) {
    const r = LEG_ROWS[row];
    // tripod gait: front and hind on one side move with the middle leg of the other side
    const group = (row === 1 ? side > 0 : side < 0) ? 0 : 1;
    LEGS.push({ row, side, group, hip: [r.hip[0], r.hip[1], r.hip[2] * side], coxa: r.coxa, femur: r.femur, tibia: r.tibia, tarsus: r.tarsus, rest: [r.rest[0], r.rest[1] * side] });
}

export const ANTENNA = { base: [0.205, 0.035, 0.03], scape: 0.13, funiculus: 0.17 };
export const BODY_HEIGHT = 0.13;   // mesosoma above the ground when walking, rig units
