// THE BACKROOMS — Level 0
// A found-footage recreation of the endless yellow rooms.
// Procedural infinite world, synthesized audio, scripted entity scenes.

import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

// ---------------------------------------------------------------- constants

const CELL = 4;            // meters per grid cell
const WALL_H = 3.2;        // ceiling height
const WALL_T = 0.42;       // wall thickness
const CHUNK = 8;           // cells per chunk side
const VIEW_CHUNKS = 2;     // chunk load radius
const PLAYER_R = 0.35;
const EYE = 1.62;

const WALL_P = 0.22;       // base wall probability
const PILLAR_P = 0.05;
const PANEL_P = 0.30;      // ceiling light probability

// ---------------------------------------------------------------- hashing

function hash2(i, j, salt) {
    let h = (i * 374761393 + j * 668265263 + salt * 2654435761) | 0;
    h = (h ^ (h >> 13)) | 0;
    h = Math.imul(h, 1274126177);
    h = (h ^ (h >> 16)) >>> 0;
    return h / 4294967296;
}

// low-frequency "zone" noise: open halls vs denser office clusters
function zone(i, j) {
    return hash2(Math.floor(i / 7), Math.floor(j / 7), 909);
}

function wallProb(i, j) {
    const z = zone(i, j);
    if (z < 0.30) return 0.06;   // open hall
    if (z > 0.78) return 0.34;   // dense cluster
    return WALL_P;
}

function wallEast(i, j)  { return hash2(i, j, 11) < wallProb(i, j); }
function wallSouth(i, j) { return hash2(i, j, 23) < wallProb(i, j); }
function pillarAt(i, j) {
    return zone(i, j) < 0.30 && hash2(i, j, 37) < PILLAR_P;
}
function panelAt(i, j)   { return hash2(i, j, 53) < PANEL_P; }

// ---------------------------------------------------------------- textures

function makeCanvas(w, h, draw) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    draw(c.getContext('2d'), w, h);
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    return tex;
}

function grain(ctx, w, h, n, alpha, dark) {
    for (let k = 0; k < n; k++) {
        const v = Math.random() * alpha;
        ctx.fillStyle = dark ? `rgba(0,0,0,${v})` : `rgba(255,255,255,${v})`;
        ctx.fillRect(Math.random() * w, Math.random() * h, 1 + Math.random() * 2, 1 + Math.random() * 2);
    }
}

// mono-yellow wallpaper with faint stripes, stains, and a baseboard strip
const wallTex = makeCanvas(256, 384, (ctx, w, h) => {
    ctx.fillStyle = '#b0a058';
    ctx.fillRect(0, 0, w, h);
    for (let x = 0; x < w; x += 16) {
        ctx.fillStyle = (x / 16) % 2 ? 'rgba(0,0,0,0.045)' : 'rgba(255,255,230,0.05)';
        ctx.fillRect(x, 0, 8, h);
    }
    // water stains
    for (let k = 0; k < 9; k++) {
        const x = Math.random() * w, y = Math.random() * h * 0.6, r = 12 + Math.random() * 44;
        const g = ctx.createRadialGradient(x, y, 2, x, y, r);
        g.addColorStop(0, 'rgba(84,70,30,0.16)');
        g.addColorStop(1, 'rgba(84,70,30,0)');
        ctx.fillStyle = g;
        ctx.fillRect(x - r, y - r, r * 2, r * 2);
    }
    grain(ctx, w, h, 1600, 0.07, true);
    // grime gradient near floor
    const g2 = ctx.createLinearGradient(0, h - 110, 0, h);
    g2.addColorStop(0, 'rgba(40,32,10,0)');
    g2.addColorStop(1, 'rgba(40,32,10,0.30)');
    ctx.fillStyle = g2;
    ctx.fillRect(0, h - 110, w, 110);
    // baseboard
    ctx.fillStyle = '#6e6134';
    ctx.fillRect(0, h - 26, w, 26);
    ctx.fillStyle = 'rgba(255,255,220,0.18)';
    ctx.fillRect(0, h - 26, w, 3);
});

const carpetTex = makeCanvas(256, 256, (ctx, w, h) => {
    ctx.fillStyle = '#7a6c3a';
    ctx.fillRect(0, 0, w, h);
    for (let k = 0; k < 5200; k++) {
        const v = 0.05 + Math.random() * 0.12;
        ctx.fillStyle = Math.random() < 0.5 ? `rgba(0,0,0,${v})` : `rgba(210,195,130,${v * 0.7})`;
        ctx.fillRect(Math.random() * w, Math.random() * h, 2, 2);
    }
    // damp patches
    for (let k = 0; k < 5; k++) {
        const x = Math.random() * w, y = Math.random() * h, r = 20 + Math.random() * 50;
        const g = ctx.createRadialGradient(x, y, 2, x, y, r);
        g.addColorStop(0, 'rgba(30,26,8,0.22)');
        g.addColorStop(1, 'rgba(30,26,8,0)');
        ctx.fillStyle = g;
        ctx.fillRect(x - r, y - r, r * 2, r * 2);
    }
});

const ceilTex = makeCanvas(256, 256, (ctx, w, h) => {
    ctx.fillStyle = '#a89a63';
    ctx.fillRect(0, 0, w, h);
    grain(ctx, w, h, 900, 0.05, true);
    ctx.strokeStyle = 'rgba(60,52,20,0.55)';
    ctx.lineWidth = 3;
    for (let x = 0; x <= w; x += 128) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y <= h; y += 128) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    // stains
    for (let k = 0; k < 4; k++) {
        const x = Math.random() * w, y = Math.random() * h, r = 16 + Math.random() * 34;
        const g = ctx.createRadialGradient(x, y, 2, x, y, r);
        g.addColorStop(0, 'rgba(96,74,26,0.28)');
        g.addColorStop(1, 'rgba(96,74,26,0)');
        ctx.fillStyle = g;
        ctx.fillRect(x - r, y - r, r * 2, r * 2);
    }
});

// ---------------------------------------------------------------- materials

const wallMat = new THREE.MeshLambertMaterial({ map: wallTex });
const carpetMat = new THREE.MeshLambertMaterial({ map: carpetTex });
const ceilMat = new THREE.MeshLambertMaterial({ map: ceilTex });
const panelMat = new THREE.MeshLambertMaterial({
    color: 0xfff6cf, emissive: 0xfff2b8, emissiveIntensity: 1.0,
});

// ---------------------------------------------------------------- scene

const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0b03);
scene.fog = new THREE.Fog(0x0d0b03, 7, 34);

const camera = new THREE.PerspectiveCamera(72, innerWidth / innerHeight, 0.1, 80);

const ambient = new THREE.AmbientLight(0xfff0b8, 0.52);
scene.add(ambient);
const hemi = new THREE.HemisphereLight(0xfff4c0, 0x4a4020, 0.35);
scene.add(hemi);

// pool of point lights parked at the nearest ceiling panels
const LIGHT_POOL = 5;
const pointLights = [];
for (let k = 0; k < LIGHT_POOL; k++) {
    const l = new THREE.PointLight(0xffeeaa, 0, 16, 1.8);
    scene.add(l);
    pointLights.push(l);
}

// camcorder light (F)
const camLight = new THREE.SpotLight(0xfff4d0, 0, 22, 0.5, 0.45, 1.2);
camLight.visible = false;
scene.add(camLight);
scene.add(camLight.target);

addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------- geometry builder

// hand-built quads so texture density stays constant regardless of wall length
class MeshBuilder {
    constructor() { this.pos = []; this.nor = []; this.uv = []; this.idx = []; }
    quad(a, b, c, d, n, uw, vh) {
        const s = this.pos.length / 3;
        this.pos.push(...a, ...b, ...c, ...d);
        for (let k = 0; k < 4; k++) this.nor.push(...n);
        this.uv.push(0, 0, uw, 0, uw, vh, 0, vh);
        this.idx.push(s, s + 1, s + 2, s, s + 2, s + 3);
    }
    // axis-aligned box from min/max corner, sides only (no top/bottom)
    boxSides(x0, y0, z0, x1, y1, z1, texW, texH) {
        const uw = lx => lx / texW, vh = (y1 - y0) / texH;
        this.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1], uw(x1 - x0), vh);   // +z
        this.quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1], uw(x1 - x0), vh);  // -z
        this.quad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [1, 0, 0], uw(z1 - z0), vh);   // +x
        this.quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1, 0, 0], uw(z1 - z0), vh);  // -x
    }
    // downward-facing rectangle (light panels)
    quadDown(x0, z0, x1, z1, y) {
        this.quad([x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1], [0, -1, 0], 1, 1);
    }
    build(material) {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
        g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nor, 3));
        g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
        g.setIndex(this.idx);
        return new THREE.Mesh(g, material);
    }
}

// ---------------------------------------------------------------- chunks

const chunks = new Map();   // "ci,cj" -> {group, panels: [Vector3]}

function buildChunk(ci, cj) {
    const group = new THREE.Group();
    const panels = [];
    const walls = new MeshBuilder();
    const lights = new MeshBuilder();
    const x0 = ci * CHUNK * CELL, z0 = cj * CHUNK * CELL;
    const size = CHUNK * CELL;

    // floor
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(size, size), carpetMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(x0 + size / 2, 0, z0 + size / 2);
    floor.geometry.attributes.uv.array.forEach((v, k, arr) => arr[k] = v * (size / 2.2));
    group.add(floor);

    // ceiling
    const ceil = new THREE.Mesh(new THREE.PlaneGeometry(size, size), ceilMat);
    ceil.rotation.x = Math.PI / 2;
    ceil.position.set(x0 + size / 2, WALL_H, z0 + size / 2);
    ceil.geometry.attributes.uv.array.forEach((v, k, arr) => arr[k] = v * (size / 2.4));
    group.add(ceil);

    const T = WALL_T / 2;
    for (let i = ci * CHUNK; i < (ci + 1) * CHUNK; i++) {
        for (let j = cj * CHUNK; j < (cj + 1) * CHUNK; j++) {
            const cx = i * CELL, cz = j * CELL;   // cell min corner
            if (wallEast(i, j))
                walls.boxSides(cx + CELL - T, 0, cz - T, cx + CELL + T, WALL_H, cz + CELL + T, 2.2, WALL_H);
            if (wallSouth(i, j))
                walls.boxSides(cx - T, 0, cz + CELL - T, cx + CELL + T, WALL_H, cz + CELL + T, 2.2, WALL_H);
            if (pillarAt(i, j))
                walls.boxSides(cx + CELL / 2 - 0.35, 0, cz + CELL / 2 - 0.35,
                               cx + CELL / 2 + 0.35, WALL_H, cz + CELL / 2 + 0.35, 2.2, WALL_H);
            if (panelAt(i, j)) {
                lights.quadDown(cx + CELL / 2 - 0.95, cz + CELL / 2 - 0.5,
                                cx + CELL / 2 + 0.95, cz + CELL / 2 + 0.5, WALL_H - 0.02);
                panels.push(new THREE.Vector3(cx + CELL / 2, WALL_H - 0.1, cz + CELL / 2));
            }
        }
    }
    group.add(walls.build(wallMat));
    group.add(lights.build(panelMat));
    scene.add(group);
    return { group, panels };
}

function updateChunks(px, pz) {
    const pci = Math.floor(px / (CHUNK * CELL));
    const pcj = Math.floor(pz / (CHUNK * CELL));
    for (let ci = pci - VIEW_CHUNKS; ci <= pci + VIEW_CHUNKS; ci++)
        for (let cj = pcj - VIEW_CHUNKS; cj <= pcj + VIEW_CHUNKS; cj++) {
            const key = ci + ',' + cj;
            if (!chunks.has(key)) chunks.set(key, buildChunk(ci, cj));
        }
    for (const [key, ch] of chunks) {
        const [ci, cj] = key.split(',').map(Number);
        if (Math.abs(ci - pci) > VIEW_CHUNKS + 1 || Math.abs(cj - pcj) > VIEW_CHUNKS + 1) {
            scene.remove(ch.group);
            ch.group.traverse(o => { if (o.geometry) o.geometry.dispose(); });
            chunks.delete(key);
        }
    }
}

// ---------------------------------------------------------------- collision

// collect wall AABBs near a point
function nearbyBoxes(px, pz) {
    const boxes = [];
    const i0 = Math.floor(px / CELL), j0 = Math.floor(pz / CELL);
    const T = WALL_T / 2 + 0.02;
    for (let i = i0 - 1; i <= i0 + 1; i++)
        for (let j = j0 - 1; j <= j0 + 1; j++) {
            const cx = i * CELL, cz = j * CELL;
            if (wallEast(i, j)) boxes.push([cx + CELL - T, cz - T, cx + CELL + T, cz + CELL + T]);
            if (wallSouth(i, j)) boxes.push([cx - T, cz + CELL - T, cx + CELL + T, cz + CELL + T]);
            if (pillarAt(i, j)) boxes.push([cx + CELL / 2 - 0.37, cz + CELL / 2 - 0.37,
                                            cx + CELL / 2 + 0.37, cz + CELL / 2 + 0.37]);
        }
    return boxes;
}

function collide(px, pz) {
    for (let pass = 0; pass < 2; pass++) {
        for (const [x0, z0, x1, z1] of nearbyBoxes(px, pz)) {
            const nx = Math.max(x0, Math.min(px, x1));
            const nz = Math.max(z0, Math.min(pz, z1));
            const dx = px - nx, dz = pz - nz;
            const d2 = dx * dx + dz * dz;
            if (d2 < PLAYER_R * PLAYER_R) {
                if (d2 > 1e-9) {
                    const d = Math.sqrt(d2);
                    px = nx + (dx / d) * PLAYER_R;
                    pz = nz + (dz / d) * PLAYER_R;
                } else {
                    px = nx + PLAYER_R; // degenerate: push +x
                }
            }
        }
    }
    return [px, pz];
}

// line of sight between two points (checks wall boxes along the segment)
function lineOfSight(ax, az, bx, bz) {
    const steps = Math.ceil(Math.hypot(bx - ax, bz - az) / 0.35);
    for (let s = 1; s < steps; s++) {
        const t = s / steps;
        const x = ax + (bx - ax) * t, z = az + (bz - az) * t;
        for (const [x0, z0, x1, z1] of nearbyBoxes(x, z))
            if (x > x0 - 0.1 && x < x1 + 0.1 && z > z0 - 0.1 && z < z1 + 0.1) return false;
    }
    return true;
}

// ---------------------------------------------------------------- audio

let AC = null, humGain = null, masterGain = null;

function initAudio() {
    if (AC) return;
    AC = new (window.AudioContext || window.webkitAudioContext)();
    masterGain = AC.createGain();
    masterGain.gain.value = 0.9;
    masterGain.connect(AC.destination);

    // fluorescent hum: 120 Hz saw + 240 Hz sine + filtered noise bed
    humGain = AC.createGain();
    humGain.gain.value = 0.0;
    humGain.connect(masterGain);

    const o1 = AC.createOscillator();
    o1.type = 'sawtooth'; o1.frequency.value = 120;
    const f1 = AC.createBiquadFilter();
    f1.type = 'lowpass'; f1.frequency.value = 320;
    const g1 = AC.createGain(); g1.gain.value = 0.06;
    o1.connect(f1); f1.connect(g1); g1.connect(humGain); o1.start();

    const o2 = AC.createOscillator();
    o2.type = 'sine'; o2.frequency.value = 240;
    const g2 = AC.createGain(); g2.gain.value = 0.018;
    o2.connect(g2); g2.connect(humGain); o2.start();

    const nb = noiseBuffer(4);
    const ns = AC.createBufferSource();
    ns.buffer = nb; ns.loop = true;
    const nf = AC.createBiquadFilter();
    nf.type = 'bandpass'; nf.frequency.value = 1150; nf.Q.value = 2.5;
    const ng = AC.createGain(); ng.gain.value = 0.014;
    ns.connect(nf); nf.connect(ng); ng.connect(humGain); ns.start();

    humGain.gain.setTargetAtTime(1.0, AC.currentTime, 2.0);
}

function noiseBuffer(seconds) {
    const b = AC.createBuffer(1, AC.sampleRate * seconds, AC.sampleRate);
    const d = b.getChannelData(0);
    for (let k = 0; k < d.length; k++) d[k] = Math.random() * 2 - 1;
    return b;
}

function playNoise({ dur = 0.2, freq = 500, q = 1, gain = 0.2, type = 'bandpass', attack = 0.005, glideTo = null }) {
    if (!AC) return;
    const s = AC.createBufferSource();
    s.buffer = noiseBuffer(dur + 0.1);
    const f = AC.createBiquadFilter();
    f.type = type; f.frequency.value = freq; f.Q.value = q;
    if (glideTo) f.frequency.linearRampToValueAtTime(glideTo, AC.currentTime + dur);
    const g = AC.createGain();
    g.gain.setValueAtTime(0, AC.currentTime);
    g.gain.linearRampToValueAtTime(gain, AC.currentTime + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, AC.currentTime + dur);
    s.connect(f); f.connect(g); g.connect(masterGain);
    s.start(); s.stop(AC.currentTime + dur + 0.1);
}

function playFootstep(running) {
    playNoise({ dur: 0.09, freq: 260 + Math.random() * 160, q: 1.4, gain: running ? 0.09 : 0.05 });
}

function playRumble() {
    playNoise({ dur: 3.5, freq: 55, q: 0.8, gain: 0.5, type: 'lowpass', attack: 1.2 });
}

function playKnocks() {
    if (!AC) return;
    let t = 0;
    for (let k = 0; k < 3 + Math.floor(Math.random() * 3); k++) {
        setTimeout(() => playNoise({ dur: 0.12, freq: 140, q: 3, gain: 0.12 }), t);
        t += 260 + Math.random() * 240;
    }
}

function playScreech() {
    if (!AC) return;
    const o = AC.createOscillator();
    o.type = 'sawtooth';
    o.frequency.setValueAtTime(920, AC.currentTime);
    o.frequency.exponentialRampToValueAtTime(140, AC.currentTime + 1.3);
    const ws = AC.createWaveShaper();
    const curve = new Float32Array(256);
    for (let k = 0; k < 256; k++) { const x = k / 128 - 1; curve[k] = Math.tanh(x * 4); }
    ws.curve = curve;
    const g = AC.createGain();
    g.gain.setValueAtTime(0.0001, AC.currentTime);
    g.gain.exponentialRampToValueAtTime(0.34, AC.currentTime + 0.12);
    g.gain.exponentialRampToValueAtTime(0.0001, AC.currentTime + 1.4);
    o.connect(ws); ws.connect(g); g.connect(masterGain);
    o.start(); o.stop(AC.currentTime + 1.5);
    playNoise({ dur: 1.3, freq: 2400, q: 0.7, gain: 0.1, glideTo: 300 });
}

function playStaticBurst(dur) {
    playNoise({ dur, freq: 3000, q: 0.3, gain: 0.4, type: 'highpass', attack: 0.01 });
}

// ---------------------------------------------------------------- entity

function buildEntity() {
    const g = new THREE.Group();
    const mat = new THREE.MeshLambertMaterial({ color: 0x0a0805 });
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.24, 1.9, 8), mat);
    body.position.y = 1.35;
    body.scale.z = 0.6;
    g.add(body);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.16, 10, 8), mat);
    head.position.y = 2.45;
    head.scale.set(0.85, 1.5, 0.85);
    g.add(head);
    for (const side of [-1, 1]) {
        const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.03, 1.5, 6), mat);
        arm.position.set(side * 0.26, 1.5, 0);
        arm.rotation.z = side * 0.10;
        g.add(arm);
        const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.05, 1.0, 6), mat);
        leg.position.set(side * 0.11, 0.5, 0);
        g.add(leg);
    }
    g.visible = false;
    scene.add(g);
    return g;
}

const entity = buildEntity();

// ---------------------------------------------------------------- player + input

const player = {
    x: CELL / 2, z: CELL / 2, yaw: 0, pitch: 0,
    vx: 0, vz: 0, bob: 0, stepAcc: 0,
};

// make sure spawn cell has no pillar
if (pillarAt(0, 0)) player.x += CELL;

const keys = {};
let locked = false;
let started = false;

addEventListener('keydown', e => {
    keys[e.code] = true;
    if (e.code === 'KeyF' && started) {
        camLight.visible = !camLight.visible;
        camLight.intensity = camLight.visible ? 26 : 0;
        playNoise({ dur: 0.05, freq: 2000, q: 4, gain: 0.06 });
    }
});
addEventListener('keyup', e => keys[e.code] = false);
addEventListener('blur', () => { for (const k in keys) keys[k] = false; });

addEventListener('mousemove', e => {
    if (!locked) return;
    player.yaw -= e.movementX * 0.0023;
    player.pitch -= e.movementY * 0.0023;
    player.pitch = Math.max(-1.45, Math.min(1.45, player.pitch));
});

const startScreen = document.getElementById('start-screen');
const pausedHint = document.getElementById('paused-hint');

function requestLock() {
    try {
        const p = document.body.requestPointerLock();
        if (p && p.catch) p.catch(() => {});
    } catch (e) { /* retry on next click */ }
}

document.getElementById('enter-btn').addEventListener('click', e => {
    e.stopPropagation();
    initAudio();
    if (AC.state === 'suspended') AC.resume();
    startScreen.style.display = 'none';
    started = true;
    requestLock();
});

// any click re-acquires the lock if it was lost or the first request failed
document.getElementById('container').addEventListener('click', () => {
    if (started && !locked) requestLock();
});

document.addEventListener('pointerlockchange', () => {
    locked = document.pointerLockElement === document.body;
    if (started) pausedHint.classList.toggle('active', !locked);
});

// ---------------------------------------------------------------- events / scenes

const captionEl = document.getElementById('caption');
let captionTimer = null;

function caption(text, dur = 4000) {
    captionEl.textContent = text;
    captionEl.classList.add('show');
    clearTimeout(captionTimer);
    captionTimer = setTimeout(() => captionEl.classList.remove('show'), dur);
}

const state = {
    t: 0,
    lightFactor: 1,        // global light multiplier (blackouts)
    flicker: 0,            // >0 while flicker event active
    blackoutUntil: -1,
    humOffUntil: -1,
    nextEvent: 14,
    sightings: 0,
    entityMode: 'none',    // none | stare | chase
    entityTimer: 0,
    staticUntil: -1,
};

function forwardDir() {
    return [-Math.sin(player.yaw), -Math.cos(player.yaw)];
}

// find a spawn point roughly ahead of the player with line of sight
function findEntitySpawn(minD, maxD) {
    const [fx, fz] = forwardDir();
    let best = null, bestScore = -1;
    for (let attempt = 0; attempt < 80; attempt++) {
        // prefer ahead early, widen to full circle later
        const spread = attempt < 40 ? 1.4 : Math.PI * 2;
        const ang = (Math.random() - 0.5) * spread;
        const d = minD + Math.random() * (maxD - minD);
        const dx = fx * Math.cos(ang) - fz * Math.sin(ang);
        const dz = fx * Math.sin(ang) + fz * Math.cos(ang);
        const x = player.x + dx * d, z = player.z + dz * d;
        const i = Math.floor(x / CELL), j = Math.floor(z / CELL);
        const cx = i * CELL + CELL / 2, cz = j * CELL + CELL / 2;
        if (pillarAt(i, j)) continue;
        const dist = Math.hypot(cx - player.x, cz - player.z);
        if (dist < minD * 0.6) continue;
        if (lineOfSight(player.x, player.z, cx, cz)) {
            // furthest visible spot ahead wins
            const score = dist + (attempt < 40 ? 30 : 0);
            if (score > bestScore) { bestScore = score; best = [cx, cz]; }
        }
    }
    if (best) return best;
    // fallback: nearest open cell ~2 cells ahead — heard, then seen around a corner
    const i = Math.floor((player.x + fx * 9) / CELL), j = Math.floor((player.z + fz * 9) / CELL);
    if (!pillarAt(i, j)) return [i * CELL + CELL / 2, j * CELL + CELL / 2];
    return null;
}

function triggerEvent() {
    const elapsed = state.t;
    const roll = Math.random();
    const canSee = elapsed > 50;
    const canChase = state.sightings >= 2 && elapsed > 150;

    if (canChase && roll < 0.30) return startChase();
    if (canSee && roll < 0.55) return startSighting();
    if (roll < 0.70) {
        state.flicker = 1.6;
        caption('the lights…');
    } else if (roll < 0.85) {
        state.blackoutUntil = state.t + 2.2 + Math.random() * 1.6;
        playRumble();
        caption('');
    } else {
        playKnocks();
        caption('did you hear that');
    }
}

function startSighting() {
    const p = findEntitySpawn(14, 24);
    if (!p) { state.flicker = 1.2; return; }
    entity.position.set(p[0], 0, p[1]);
    entity.lookAt(player.x, 0, player.z);
    entity.visible = true;
    state.entityMode = 'stare';
    state.entityTimer = 5 + Math.random() * 4;
    state.sightings++;
    playRumble();
    caption('…something is standing there');
}

function startChase() {
    const p = findEntitySpawn(16, 22);
    if (!p) return startSighting();
    entity.position.set(p[0], 0, p[1]);
    entity.visible = true;
    state.entityMode = 'chase';
    state.entityTimer = 12;
    playScreech();
    caption('RUN', 2500);
}

const staticCut = document.getElementById('static-cut');
const staticCanvas = document.getElementById('static-canvas');
staticCanvas.width = 320; staticCanvas.height = 180;
const staticCtx = staticCanvas.getContext('2d');

function tapeDamage() {
    state.staticUntil = state.t + 2.2;
    staticCut.classList.add('active');
    playStaticBurst(2.2);
    entity.visible = false;
    state.entityMode = 'none';
    // noclip: relocate far away
    const ang = Math.random() * Math.PI * 2;
    const d = 240 + Math.random() * 240;
    let ni = Math.floor((player.x + Math.cos(ang) * d) / CELL);
    let nj = Math.floor((player.z + Math.sin(ang) * d) / CELL);
    if (pillarAt(ni, nj)) ni++;
    player.x = ni * CELL + CELL / 2;
    player.z = nj * CELL + CELL / 2;
    setTimeout(() => {
        staticCut.classList.remove('active');
        caption('…where am I now', 5000);
    }, 2200);
}

function updateEntity(dt) {
    if (state.entityMode === 'none') return;
    state.entityTimer -= dt;
    const dx = player.x - entity.position.x;
    const dz = player.z - entity.position.z;
    const dist = Math.hypot(dx, dz);

    if (state.entityMode === 'stare') {
        entity.lookAt(player.x, 0, player.z);
        // vanish if approached, timed out, or during a blackout
        if (dist < 7 || state.entityTimer <= 0) {
            entity.visible = false;
            state.entityMode = 'none';
            state.flicker = 1.4;
            playKnocks();
        }
    } else if (state.entityMode === 'chase') {
        const sp = 5.6;
        const ux = dx / (dist || 1), uz = dz / (dist || 1);
        let ex = entity.position.x + ux * sp * dt;
        let ez = entity.position.z + uz * sp * dt;
        [ex, ez] = collide(ex, ez);
        entity.position.set(ex, 0, ez);
        entity.lookAt(player.x, 0, player.z);
        entity.position.y = Math.abs(Math.sin(state.t * 14)) * 0.12;
        if (dist < 1.3) return tapeDamage();
        if (state.entityTimer <= 0) {
            entity.visible = false;
            state.entityMode = 'none';
            caption('…it stopped', 4000);
        }
    }
}

// ---------------------------------------------------------------- VHS overlay

const noiseCanvas = document.getElementById('noise');
noiseCanvas.width = 160; noiseCanvas.height = 90;
const noiseCtx = noiseCanvas.getContext('2d');
const timestampEl = document.getElementById('timestamp');
const trackingBar = document.getElementById('tracking-bar');

let tapeStart = null;

function updateVHS(t, frame) {
    if (frame % 2 === 0) {
        const img = noiseCtx.createImageData(160, 90);
        const d = img.data;
        for (let k = 0; k < d.length; k += 4) {
            const v = Math.random() * 255;
            d[k] = d[k + 1] = d[k + 2] = v;
            d[k + 3] = 255;
        }
        noiseCtx.putImageData(img, 0, 0);
    }
    // timestamp: tape starts SEP.23 1996 PM 1:57
    if (tapeStart === null) tapeStart = t;
    const secs = Math.floor(t - tapeStart) + 57 * 60 + 13 * 3600;
    const hh = Math.floor(secs / 3600) % 24, mm = Math.floor(secs / 60) % 60, ss = secs % 60;
    const ampm = hh >= 12 ? 'PM' : 'AM';
    const h12 = hh % 12 || 12;
    timestampEl.textContent =
        `SEP.23 1996  ${ampm} ${h12}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
    // occasional tracking bar sweep
    if (Math.random() < 0.0012) {
        trackingBar.style.transition = 'none';
        trackingBar.style.top = '108%';
        trackingBar.style.height = 14 + Math.random() * 30 + 'px';
        requestAnimationFrame(() => {
            trackingBar.style.transition = 'top 1.3s linear';
            trackingBar.style.top = '-10%';
        });
    }
    if (state.staticUntil > state.t) {
        const img = staticCtx.createImageData(320, 180);
        const d = img.data;
        for (let k = 0; k < d.length; k += 4) {
            const v = Math.random() * 255;
            d[k] = d[k + 1] = d[k + 2] = v;
            d[k + 3] = 255;
        }
        staticCtx.putImageData(img, 0, 0);
    }
}

// ---------------------------------------------------------------- lighting update

let lightRepickAcc = 1;

function updateLights(dt) {
    // global factor: blackout beats flicker
    let f = 1;
    if (state.t < state.blackoutUntil) {
        f = 0.04;
    } else if (state.flicker > 0) {
        state.flicker -= dt;
        f = Math.random() < 0.4 ? 0.15 + Math.random() * 0.3 : 1;
    }
    // subtle constant fluorescent shimmer
    f *= 0.965 + 0.035 * Math.sin(state.t * 47) * Math.sin(state.t * 13.7);
    state.lightFactor = f;

    ambient.intensity = 0.52 * f;
    hemi.intensity = 0.35 * f;
    panelMat.emissiveIntensity = Math.max(0.02, f);
    scene.fog.color.setHex(f < 0.3 ? 0x030302 : 0x0d0b03);

    // hum follows the lights
    if (AC && humGain) {
        const humOn = f > 0.3 && state.t > state.humOffUntil ? 1 : 0.05;
        humGain.gain.setTargetAtTime(humOn, AC.currentTime, 0.08);
    }

    // repick nearest panels for the point-light pool
    lightRepickAcc += dt;
    if (lightRepickAcc > 0.5) {
        lightRepickAcc = 0;
        const all = [];
        for (const ch of chunks.values())
            for (const p of ch.panels) {
                const d = (p.x - player.x) ** 2 + (p.z - player.z) ** 2;
                if (d < 500) all.push([d, p]);
            }
        all.sort((a, b) => a[0] - b[0]);
        for (let k = 0; k < LIGHT_POOL; k++) {
            if (k < all.length) pointLights[k].position.copy(all[k][1]);
            else pointLights[k].position.set(0, -100, 0);
        }
    }
    for (const l of pointLights)
        l.intensity = 5.5 * f * (0.9 + Math.random() * 0.1);
}

// ---------------------------------------------------------------- main loop

const clock = new THREE.Clock();
let frame = 0;

function tick() {
    requestAnimationFrame(tick);
    const dt = Math.min(clock.getDelta(), 0.05);
    frame++;

    if (started && locked) {
        state.t += dt;

        // movement
        const run = keys['ShiftLeft'] || keys['ShiftRight'];
        const speed = run ? 5.4 : 2.7;
        let mx = 0, mz = 0;
        if (keys['KeyW'] || keys['ArrowUp']) mz -= 1;
        if (keys['KeyS'] || keys['ArrowDown']) mz += 1;
        if (keys['KeyA'] || keys['ArrowLeft']) mx -= 1;
        if (keys['KeyD'] || keys['ArrowRight']) mx += 1;
        const ml = Math.hypot(mx, mz);
        if (ml > 0) { mx /= ml; mz /= ml; }
        const sin = Math.sin(player.yaw), cos = Math.cos(player.yaw);
        const wx = (mx * cos - mz * sin) * speed;
        const wz = (-mx * sin - mz * cos) * speed;
        let nx = player.x + wx * dt;
        let nz = player.z + wz * dt;
        [nx, nz] = collide(nx, nz);
        const moved = Math.hypot(nx - player.x, nz - player.z);
        player.x = nx; player.z = nz;

        // head bob + footsteps
        if (moved > 0.0005) {
            player.bob += moved * (run ? 1.9 : 1.6);
            player.stepAcc += moved;
            const stride = run ? 2.3 : 1.9;
            if (player.stepAcc > stride) {
                player.stepAcc = 0;
                playFootstep(run);
            }
        }

        // scheduled events
        if (state.t > state.nextEvent && state.entityMode === 'none' && state.staticUntil < state.t) {
            state.nextEvent = state.t + 20 + Math.random() * 22;
            triggerEvent();
        }

        updateEntity(dt);
        updateChunks(player.x, player.z);
        updateLights(dt);

        // camera: position + handheld sway
        const bobY = Math.sin(player.bob * 2) * 0.035 * (moved > 0.0005 ? 1 : 0);
        const swayY = Math.sin(state.t * 0.9) * 0.008 + Math.sin(state.t * 2.3) * 0.004;
        const swayYaw = Math.sin(state.t * 0.6) * 0.006;
        const swayPitch = Math.cos(state.t * 0.8) * 0.005;
        camera.position.set(player.x, EYE + bobY + swayY, player.z);
        camera.rotation.order = 'YXZ';
        camera.rotation.y = player.yaw + swayYaw;
        camera.rotation.x = player.pitch + swayPitch;
        camera.rotation.z = Math.sin(player.bob) * 0.006;

        // camcorder light follows view
        if (camLight.visible) {
            camLight.position.copy(camera.position);
            const dir = new THREE.Vector3(0, 0, -1).applyEuler(camera.rotation);
            camLight.target.position.copy(camera.position).add(dir.multiplyScalar(10));
        }

        updateVHS(state.t, frame);
    }

    renderer.render(scene, camera);
}

// debug/test hooks
window.BACKROOMS = {
    player, state, entity, startSighting, startChase, triggerEvent, tapeDamage,
    forceLock() { locked = true; started = true; startScreen.style.display = 'none'; },
};

// prebuild spawn chunks so first frame isn't empty
updateChunks(player.x, player.z);
camera.position.set(player.x, EYE, player.z);
tick();
