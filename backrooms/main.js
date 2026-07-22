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

// checkered vinyl for the kitchen
const checkerTex = makeCanvas(256, 256, (ctx, w, h) => {
    for (let x = 0; x < w; x += 32)
        for (let y = 0; y < h; y += 32) {
            ctx.fillStyle = ((x + y) / 32) % 2 ? '#c9bfa0' : '#4a4434';
            ctx.fillRect(x, y, 32, 32);
        }
    grain(ctx, w, h, 1400, 0.08, true);
});

// ceramic tile for the poolrooms
const tileTex = makeCanvas(256, 256, (ctx, w, h) => {
    ctx.fillStyle = '#7fa89c';
    ctx.fillRect(0, 0, w, h);
    const T = 32;
    for (let x = 0; x < w; x += T)
        for (let y = 0; y < h; y += T) {
            const v = 225 + Math.floor(Math.random() * 22);
            ctx.fillStyle = `rgb(${v - 8},${v},${v - 4})`;
            ctx.fillRect(x + 1, y + 1, T - 2, T - 2);
        }
    grain(ctx, w, h, 500, 0.05, true);
    // stains
    for (let k = 0; k < 4; k++) {
        const x = Math.random() * w, y = Math.random() * h, r = 14 + Math.random() * 30;
        const g = ctx.createRadialGradient(x, y, 2, x, y, r);
        g.addColorStop(0, 'rgba(70,110,95,0.16)');
        g.addColorStop(1, 'rgba(70,110,95,0)');
        ctx.fillStyle = g;
        ctx.fillRect(x - r, y - r, r * 2, r * 2);
    }
});

// water surface with caustic streaks
const waterTex = makeCanvas(256, 256, (ctx, w, h) => {
    ctx.fillStyle = '#1e6b58';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(190,255,230,0.35)';
    ctx.lineWidth = 2;
    for (let k = 0; k < 60; k++) {
        ctx.beginPath();
        let x = Math.random() * w, y = Math.random() * h;
        ctx.moveTo(x, y);
        for (let s = 0; s < 4; s++) {
            x += (Math.random() - 0.5) * 44;
            y += (Math.random() - 0.5) * 44;
            ctx.lineTo(x, y);
        }
        ctx.stroke();
    }
});

// ---------------------------------------------------------------- materials

const wallMat = new THREE.MeshLambertMaterial({ map: wallTex });
const carpetMat = new THREE.MeshLambertMaterial({ map: carpetTex });
const ceilMat = new THREE.MeshLambertMaterial({ map: ceilTex });
const panelMat = new THREE.MeshLambertMaterial({
    color: 0xfff6cf, emissive: 0xfff2b8, emissiveIntensity: 1.0,
});
const checkerMat = new THREE.MeshLambertMaterial({ map: checkerTex });
const tileMat = new THREE.MeshLambertMaterial({ map: tileTex });
const waterMat = new THREE.MeshLambertMaterial({
    map: waterTex, transparent: true, opacity: 0.62,
    emissive: 0x1d5c4a, emissiveIntensity: 0.5, side: THREE.DoubleSide,
});
const poolPanelMat = new THREE.MeshLambertMaterial({
    color: 0xeafff2, emissive: 0xcdf5e2, emissiveIntensity: 1.0,
});
const counterMat = new THREE.MeshLambertMaterial({ color: 0x8a7f5c });
const applianceMat = new THREE.MeshLambertMaterial({ color: 0xd0ccc0 });
const woodMat = new THREE.MeshLambertMaterial({ color: 0x5c3f28 });
const voidMat = new THREE.MeshLambertMaterial({ color: 0x030303 });

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
    // downward-facing rectangle (light panels, ceilings); ts = world meters per texture tile
    quadDown(x0, z0, x1, z1, y, ts) {
        const uw = ts ? (x1 - x0) / ts : 1, vh = ts ? (z1 - z0) / ts : 1;
        this.quad([x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1], [0, -1, 0], uw, vh);
    }
    // upward-facing rectangle (floors)
    quadUp(x0, z0, x1, z1, y, ts) {
        const uw = ts ? (x1 - x0) / ts : 1, vh = ts ? (z1 - z0) / ts : 1;
        this.quad([x0, y, z1], [x1, y, z1], [x1, y, z0], [x0, y, z0], [0, 1, 0], uw, vh);
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

// collect wall AABBs near a point: [x0, z0, x1, z1, y0, y1]
function nearbyBoxes(px, pz) {
    const boxes = [];
    const i0 = Math.floor(px / CELL), j0 = Math.floor(pz / CELL);
    const T = WALL_T / 2 + 0.02;
    for (let i = i0 - 1; i <= i0 + 1; i++)
        for (let j = j0 - 1; j <= j0 + 1; j++) {
            const cx = i * CELL, cz = j * CELL;
            if (wallEast(i, j)) boxes.push([cx + CELL - T, cz - T, cx + CELL + T, cz + CELL + T, 0, WALL_H]);
            if (wallSouth(i, j)) boxes.push([cx - T, cz + CELL - T, cx + CELL + T, cz + CELL + T, 0, WALL_H]);
            if (pillarAt(i, j)) boxes.push([cx + CELL / 2 - 0.37, cz + CELL / 2 - 0.37,
                                            cx + CELL / 2 + 0.37, cz + CELL / 2 + 0.37, 0, WALL_H]);
        }
    return boxes;
}

// boxes for the active world (level0 is procedural, specials are static lists)
function worldBoxes(px, pz) {
    if (world === 'level0') return nearbyBoxes(px, pz);
    const out = [];
    for (const b of currentLevel().colliders)
        if (px > b[0] - 1 && px < b[2] + 1 && pz > b[1] - 1 && pz < b[3] + 1) out.push(b);
    return out;
}

function collide(px, pz, feet = player.y, head = player.y + player.eye + 0.15) {
    for (let pass = 0; pass < 2; pass++) {
        for (const [x0, z0, x1, z1, y0, y1] of worldBoxes(px, pz)) {
            if (y1 !== undefined && (head < y0 || feet > y1)) continue;
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

function playDrip() {
    if (!AC) return;
    playNoise({ dur: 0.06, freq: 1900 + Math.random() * 900, q: 14, gain: 0.09 });
    setTimeout(() => playNoise({ dur: 0.25, freq: 900 + Math.random() * 300, q: 10, gain: 0.05 }), 70);
}

function playSplash(big) {
    playNoise({ dur: big ? 0.9 : 0.25, freq: 750, q: 0.8, gain: big ? 0.35 : 0.1, glideTo: 250, attack: 0.01 });
    if (big) playNoise({ dur: 0.6, freq: 120, q: 1, gain: 0.25, type: 'lowpass', attack: 0.01 });
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
    y: 0, vy: 0, eye: EYE,          // y = feet height; eye = current crouch-aware eye offset
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
    // one-time scripted destinations: the shrinking doors, then the kitchen/pools
    if (!state.doorsVisited && elapsed > 75) {
        state.doorsVisited = true;
        return noclipTo('doors');
    }
    if (!state.householdVisited && elapsed > 190) {
        state.householdVisited = true;
        return noclipTo('household');
    }
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

// ---------------------------------------------------------------- special levels
// Set-piece scenes from the movie: the shrinking doors, the household kitchen,
// and the Poolrooms below it (reached through the hole in the kitchen floor).

let world = 'level0';
const levels = {};          // name -> {group, colliders, lights, spawn, spawnYaw, caption}

const DX = 20000, DZ = 20000;   // doors corridor origin
const HX = 24000, HZ = 24000;   // household (kitchen + pools) origin

const DOOR_H = [3.0, 2.7, 2.45, 2.2, 1.95, 1.7, 1.5, 1.3, 1.1, 0.9];
const DOOR_W = [2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.05, 0.9, 0.75];

const DECK = -6.5, WATER = -6.9, POOL_BOT = -7.7, POOL_CEIL = -1.6;
const HOLE = [HX + 6, HZ + 3.2, HX + 7.6, HZ + 4.8];          // hole in kitchen floor
const POOL_A = [HX + 4, HZ + 1, HX + 9, HZ + 6];
const POOL_B = [HX - 11, HZ + 1, HX - 5, HZ + 6];

function currentLevel() { return levels[world]; }

function buildDoorsLevel() {
    const group = new THREE.Group();
    const colliders = [], lights = [];
    const walls = new MeshBuilder(), floors = new MeshBuilder(), ceils = new MeshBuilder();
    const frames = new MeshBuilder(), voids = new MeshBuilder(), panels = new MeshBuilder();

    const zl = DZ - 1.5, zr = DZ + 1.5;
    // side walls + entrance cap
    walls.boxSides(DX - 0.4, 0, zl - 0.3, DX + 40.2, 3.3, zl, 2.2, WALL_H);
    walls.boxSides(DX - 0.4, 0, zr, DX + 40.2, 3.3, zr + 0.3, 2.2, WALL_H);
    walls.boxSides(DX - 0.4, 0, zl, DX - 0.1, 3.3, zr, 2.2, WALL_H);
    colliders.push([DX - 0.5, zl - 0.4, DX + 40.3, zl, 0, 4],
                   [DX - 0.5, zr, DX + 40.3, zr + 0.4, 0, 4],
                   [DX - 0.5, zl, DX - 0.1, zr, 0, 4]);

    floors.quadUp(DX - 0.1, zl, DX + 40, zr, 0, 2.2);

    for (let k = 0; k < 10; k++) {
        const xk = DX + 4 * (k + 1), h = DOOR_H[k], w = DOOR_W[k];
        // partition: two side pieces + lintel above the opening
        walls.boxSides(xk - 0.13, 0, zl, xk + 0.13, 3.3, DZ - w / 2, 2.2, WALL_H);
        walls.boxSides(xk - 0.13, 0, DZ + w / 2, xk + 0.13, 3.3, zr, 2.2, WALL_H);
        walls.boxSides(xk - 0.13, h, DZ - w / 2, xk + 0.13, 3.3, DZ + w / 2, 2.2, WALL_H);
        colliders.push([xk - 0.14, zl, xk + 0.14, DZ - w / 2, 0, 4],
                       [xk - 0.14, DZ + w / 2, xk + 0.14, zr, 0, 4],
                       [xk - 0.14, DZ - w / 2, xk + 0.14, DZ + w / 2, h, 4]);
        // dark wooden door frame
        frames.boxSides(xk - 0.17, 0, DZ - w / 2 - 0.09, xk + 0.17, h + 0.07, DZ - w / 2 + 0.03, 1, 3);
        frames.boxSides(xk - 0.17, 0, DZ + w / 2 - 0.03, xk + 0.17, h + 0.07, DZ + w / 2 + 0.09, 1, 3);
        frames.boxSides(xk - 0.17, h, DZ - w / 2 - 0.09, xk + 0.17, h + 0.07, DZ + w / 2 + 0.09, 1, 3);
        // ceiling of the segment BEHIND this door steps down with it
        const ceilY = k === 0 ? 3.2 : Math.min(3.2, DOOR_H[k - 1] + 0.45);
        ceils.quadDown(xk - 4, zl, xk, zr, ceilY, 2.4);
        // sparse lights, thinning out toward the small end
        if (k % 3 === 0 && k < 7) {
            panels.quadDown(xk - 2.6, DZ - 0.4, xk - 1.4, DZ + 0.4, ceilY - 0.02);
            lights.push({ pos: new THREE.Vector3(xk - 2, ceilY - 0.2, DZ), color: 0xffeeaa });
        }
    }
    ceils.quadDown(DX + 40, zl, DX + 40.2, zr, DOOR_H[9] + 0.45, 2.4);

    // black void room past the last door
    voids.boxSides(DX + 40.2, 0, zl - 0.3, DX + 46.3, 3.3, zl, 4, 4);
    voids.boxSides(DX + 40.2, 0, zr, DX + 46.3, 3.3, zr + 0.3, 4, 4);
    voids.boxSides(DX + 46, 0, zl, DX + 46.3, 3.3, zr, 4, 4);
    voids.quadUp(DX + 40, zl, DX + 46, zr, 0, 4);
    voids.quadDown(DX + 40, zl, DX + 46, zr, 3.2, 4);
    colliders.push([DX + 40.2, zl - 0.4, DX + 46.4, zl, 0, 4],
                   [DX + 40.2, zr, DX + 46.4, zr + 0.4, 0, 4]);

    group.add(walls.build(wallMat), floors.build(carpetMat), ceils.build(ceilMat),
              frames.build(woodMat), voids.build(voidMat), panels.build(panelMat));
    scene.add(group);
    return {
        group, colliders, lights,
        spawn: [DX + 1.2, 0, DZ], spawnYaw: -Math.PI / 2,
        caption: 'the doors keep getting smaller',
    };
}

function buildHouseholdLevel() {
    const group = new THREE.Group();
    const colliders = [], lights = [];
    const walls = new MeshBuilder(), checker = new MeshBuilder(), ceil = new MeshBuilder();
    const tiles = new MeshBuilder(), fixtures = new MeshBuilder(), appliances = new MeshBuilder();
    const wood = new MeshBuilder(), voids = new MeshBuilder();
    const panels = new MeshBuilder(), poolPanels = new MeshBuilder(), water = new MeshBuilder();

    // ---- kitchen (y 0..2.7) ----
    const kx0 = HX, kz0 = HZ, kx1 = HX + 10, kz1 = HZ + 8;
    walls.boxSides(kx0 - 0.3, 0, kz0 - 0.3, kx1 + 0.3, 2.8, kz0, 2.2, WALL_H);
    walls.boxSides(kx0 - 0.3, 0, kz1, kx1 + 0.3, 2.8, kz1 + 0.3, 2.2, WALL_H);
    walls.boxSides(kx0 - 0.3, 0, kz0, kx0, 2.8, kz1, 2.2, WALL_H);
    walls.boxSides(kx1, 0, kz0, kx1 + 0.3, 2.8, kz1, 2.2, WALL_H);
    colliders.push([kx0 - 0.4, kz0 - 0.4, kx1 + 0.4, kz0, 0, 3],
                   [kx0 - 0.4, kz1, kx1 + 0.4, kz1 + 0.4, 0, 3],
                   [kx0 - 0.4, kz0, kx0, kz1, 0, 3],
                   [kx1, kz0, kx1 + 0.4, kz1, 0, 3]);
    // checkered floor with the hole cut out
    checker.quadUp(kx0, kz0, kx1, HOLE[1], 0, 1.3);
    checker.quadUp(kx0, HOLE[3], kx1, kz1, 0, 1.3);
    checker.quadUp(kx0, HOLE[1], HOLE[0], HOLE[3], 0, 1.3);
    checker.quadUp(HOLE[2], HOLE[1], kx1, HOLE[3], 0, 1.3);
    ceil.quadDown(kx0, kz0, kx1, kz1, 2.7, 2.4);
    panels.quadDown(HX + 4.2, HZ + 3.6, HX + 5.8, HZ + 4.4, 2.68);
    lights.push({ pos: new THREE.Vector3(HX + 5, 2.4, HZ + 4), color: 0xffe2a0 });

    // counters, stove, fridge, table
    appliances.boxSides(HX + 0.5, 0, HZ + 0.3, HX + 5.5, 0.95, HZ + 0.98, 2, 1);
    fixtures.quadUp(HX + 0.5, HZ + 0.3, HX + 5.5, HZ + 0.98, 0.95, 2);
    appliances.boxSides(HX + 6.2, 0, HZ + 0.3, HX + 7.0, 0.92, HZ + 0.98, 1, 1);
    appliances.boxSides(HX + 8.5, 0, HZ + 0.3, HX + 9.4, 1.85, HZ + 1.2, 1, 2);
    colliders.push([HX + 0.5, HZ + 0.3, HX + 5.5, HZ + 0.98, 0, 0.95],
                   [HX + 6.2, HZ + 0.3, HX + 7.0, HZ + 0.98, 0, 0.92],
                   [HX + 8.5, HZ + 0.3, HX + 9.4, HZ + 1.2, 0, 1.85]);
    wood.boxSides(HX + 2, 0.72, HZ + 4.6, HX + 3.4, 0.8, HZ + 5.8, 1.5, 0.1);
    wood.quadUp(HX + 2, HZ + 4.6, HX + 3.4, HZ + 5.8, 0.8, 1.5);
    wood.boxSides(HX + 2.6, 0, HZ + 5.1, HX + 2.8, 0.72, HZ + 5.3, 0.3, 1);
    colliders.push([HX + 2, HZ + 4.6, HX + 3.4, HZ + 5.8, 0, 0.8]);

    // jagged shaft down through the hole
    voids.boxSides(HOLE[0] - 0.1, POOL_CEIL, HOLE[1] - 0.1, HOLE[2] + 0.1, 0, HOLE[1], 1, 2);
    voids.boxSides(HOLE[0] - 0.1, POOL_CEIL, HOLE[3], HOLE[2] + 0.1, 0, HOLE[3] + 0.1, 1, 2);
    voids.boxSides(HOLE[0] - 0.1, POOL_CEIL, HOLE[1], HOLE[0], 0, HOLE[3], 1, 2);
    voids.boxSides(HOLE[2], POOL_CEIL, HOLE[1], HOLE[2] + 0.1, 0, HOLE[3], 1, 2);
    colliders.push([HOLE[0] - 0.15, HOLE[1] - 0.15, HOLE[2] + 0.15, HOLE[1], POOL_CEIL, 0],
                   [HOLE[0] - 0.15, HOLE[3], HOLE[2] + 0.15, HOLE[3] + 0.15, POOL_CEIL, 0],
                   [HOLE[0] - 0.15, HOLE[1], HOLE[0], HOLE[3], POOL_CEIL, 0],
                   [HOLE[2], HOLE[1], HOLE[2] + 0.15, HOLE[3], POOL_CEIL, 0]);

    // ---- poolrooms (deck at DECK, under the kitchen) ----
    const addChamber = (x0, z0, x1, z1, openings) => {
        // openings: {w:[z0,z1], e:[z0,z1]} gaps in west/east walls
        const seg = (bx0, bz0, bx1, bz1) => {
            tiles.boxSides(bx0, DECK, bz0, bx1, POOL_CEIL, bz1, 1.6, 2);
            colliders.push([bx0, bz0, bx1, bz1, DECK, POOL_CEIL]);
        };
        seg(x0 - 0.3, z0 - 0.3, x1 + 0.3, z0);          // north
        seg(x0 - 0.3, z1, x1 + 0.3, z1 + 0.3);          // south
        for (const [side, gap] of Object.entries(openings || {})) {
            const wx0 = side === 'w' ? x0 - 0.3 : x1, wx1 = side === 'w' ? x0 : x1 + 0.3;
            if (!gap) { seg(wx0, z0, wx1, z1); continue; }
            seg(wx0, z0, wx1, gap[0]);
            seg(wx0, gap[1], wx1, z1);
            // arch lintel above the opening
            tiles.boxSides(wx0, -3.2, gap[0], wx1, POOL_CEIL, gap[1], 1.6, 2);
            colliders.push([wx0, gap[0], wx1, gap[1], -3.2, POOL_CEIL]);
        }
    };
    const archA = [HZ + 3, HZ + 5.5], exitB = [HZ + 3.4, HZ + 4.6];
    addChamber(HX - 2, HZ - 2, HX + 12, HZ + 10, { w: archA, e: null });
    addChamber(HX - 14, HZ - 2, HX - 2, HZ + 10, { w: exitB, e: archA });

    // pool ceilings (chamber A's has the shaft hole cut out)
    tiles.quadDown(HX - 2, HZ - 2, HX + 12, HOLE[1], POOL_CEIL, 1.6);
    tiles.quadDown(HX - 2, HOLE[3], HX + 12, HZ + 10, POOL_CEIL, 1.6);
    tiles.quadDown(HX - 2, HOLE[1], HOLE[0], HOLE[3], POOL_CEIL, 1.6);
    tiles.quadDown(HOLE[2], HOLE[1], HX + 12, HOLE[3], POOL_CEIL, 1.6);
    tiles.quadDown(HX - 14, HZ - 2, HX - 2, HZ + 10, POOL_CEIL, 1.6);

    // decks with pools cut out, pool interiors, ramps
    const addPool = (P, rampWest) => {
        const [px0, pz0, px1, pz1] = P;
        // pool side walls (skip the ramp side)
        tiles.boxSides(px0, POOL_BOT, pz0 - 0.15, px1, DECK, pz0, 1.6, 1.5);
        tiles.boxSides(px0, POOL_BOT, pz1, px1, DECK, pz1 + 0.15, 1.6, 1.5);
        colliders.push([px0, pz0 - 0.15, px1, pz0, POOL_BOT, DECK - 0.1],
                       [px0, pz1, px1, pz1 + 0.15, POOL_BOT, DECK - 0.1]);
        const solidX = rampWest ? px1 : px0;
        tiles.boxSides(solidX - 0.15, POOL_BOT, pz0, solidX + 0.15, DECK, pz1, 1.6, 1.5);
        colliders.push([solidX - 0.15, pz0, solidX + 0.15, pz1, POOL_BOT, DECK - 0.1]);
        // bottom
        tiles.quadUp(px0, pz0, px1, pz1, POOL_BOT, 1.6);
        // ramp (1m wide strip sloping deck -> bottom)
        const rx0 = rampWest ? px0 : px1 - 1, rx1 = rampWest ? px0 + 1 : px1;
        const ya = rampWest ? DECK : POOL_BOT, yb = rampWest ? POOL_BOT : DECK;
        tiles.quad([rx0, ya, pz1], [rx1, yb, pz1], [rx1, yb, pz0], [rx0, ya, pz0],
                   [rampWest ? 0.66 : -0.66, 0.75, 0], (rx1 - rx0) / 1.6, (pz1 - pz0) / 1.6);
        // water
        water.quadUp(px0, pz0, px1, pz1, WATER, 3.2);
    };
    addPool(POOL_A, true);
    addPool(POOL_B, false);

    // deck floors around the pools
    const deckAround = (cx0, cz0, cx1, cz1, P) => {
        tiles.quadUp(cx0, cz0, cx1, P[1], DECK, 1.6);
        tiles.quadUp(cx0, P[3], cx1, cz1, DECK, 1.6);
        tiles.quadUp(cx0, P[1], P[0], P[3], DECK, 1.6);
        tiles.quadUp(P[2], P[1], cx1, P[3], DECK, 1.6);
    };
    deckAround(HX - 2, HZ - 2, HX + 12, HZ + 10, POOL_A);
    deckAround(HX - 14, HZ - 2, HX - 2, HZ + 10, POOL_B);

    // exit vestibule past chamber B — pure darkness
    voids.boxSides(HX - 16.3, DECK, HZ + 2.8, HX - 14.06, -4.3, HZ + 5.2, 4, 4);
    voids.quadUp(HX - 16.3, HZ + 2.8, HX - 14.06, HZ + 5.2, DECK + 0.001, 4);

    // pool lights: soft green-white panels
    for (const [lx, lz] of [[HX + 5, HZ + 4], [HX + 9.5, HZ + 7], [HX - 5, HZ + 4], [HX - 10, HZ + 7]]) {
        poolPanels.quadDown(lx - 0.9, lz - 0.45, lx + 0.9, lz + 0.45, POOL_CEIL - 0.02);
        lights.push({ pos: new THREE.Vector3(lx, POOL_CEIL - 0.4, lz), color: 0xbfffe0 });
    }

    const waterMesh = water.build(waterMat);
    group.add(walls.build(wallMat), checker.build(checkerMat), ceil.build(ceilMat),
              tiles.build(tileMat), fixtures.build(counterMat), appliances.build(applianceMat),
              wood.build(woodMat), voids.build(voidMat),
              panels.build(panelMat), poolPanels.build(poolPanelMat), waterMesh);
    scene.add(group);
    return {
        group, colliders, lights, waterMesh,
        spawn: [HX + 1.6, 0, HZ + 6.5], spawnYaw: Math.atan2(-(HOLE[0] - HX - 1.6), -(HOLE[1] - HZ - 6.5)),
        caption: '…someone’s kitchen?',
    };
}

function ensureLevel(name) {
    if (!levels[name])
        levels[name] = name === 'doors' ? buildDoorsLevel() : buildHouseholdLevel();
    return levels[name];
}

function inRect(x, z, r) { return x > r[0] && x < r[2] && z > r[1] && z < r[3]; }

function poolFloor(x, P, rampWest) {
    const rx0 = rampWest ? P[0] : P[2] - 1, rx1 = rampWest ? P[0] + 1 : P[2];
    if (x > rx0 && x < rx1) {
        const t = (x - rx0) / (rx1 - rx0);
        return rampWest ? DECK + (POOL_BOT - DECK) * t : POOL_BOT + (DECK - POOL_BOT) * t;
    }
    return POOL_BOT;
}

function floorAt(x, z) {
    if (world !== 'household') return 0;
    if (inRect(x, z, POOL_A)) return poolFloor(x, POOL_A, true);
    if (inRect(x, z, POOL_B)) return poolFloor(x, POOL_B, false);
    if (inRect(x, z, HOLE)) return POOL_BOT;   // shaft drops into pool A
    // stacked floors: kitchen above, pool deck below — disambiguate by player height
    if (player.y > -1.2 && inRect(x, z, [HX, HZ, HX + 10, HZ + 8])) return 0;
    return DECK;
}

function waterLevelAt(x, z) {
    if (world !== 'household') return null;
    if (inRect(x, z, POOL_A) || inRect(x, z, POOL_B) || inRect(x, z, HOLE)) return WATER;
    return null;
}

// how high the camera may sit (crouching through the shrinking doors)
function eyeCapAt(x) {
    if (world !== 'doors') return 99;
    let cap = 99;
    for (let k = 0; k < 10; k++) {
        const xk = DX + 4 * (k + 1);
        cap = Math.min(cap, DOOR_H[k] - 0.18 + Math.max(0, Math.abs(x - xk) - 0.35) * 1.6);
        // stay under the stepped-down ceiling of the segment behind each door
        if (x > xk && x < xk + (k === 9 ? 0.25 : 4.05)) cap = Math.min(cap, DOOR_H[k] + 0.30);
    }
    return cap;
}

const PROFILES = {
    level0:    { fog: [0x0d0b03, 7, 34], amb: 0.52, hemi: 0.35, hum: 1 },
    doors:     { fog: [0x0a0802, 5, 26], amb: 0.34, hemi: 0.20, hum: 0.55 },
    household: { fog: [0x07110d, 9, 46], amb: 0.55, hemi: 0.30, hum: 0.28 },
};

function applyProfile() {
    const p = PROFILES[world];
    scene.fog.color.setHex(p.fog[0]);
    scene.fog.near = p.fog[1];
    scene.fog.far = p.fog[2];
    scene.background.setHex(p.fog[0]);
}

function noclipTo(target) {
    playStaticBurst(1.8);
    state.staticUntil = state.t + 1.8;
    staticCut.classList.add('active');
    setTimeout(() => staticCut.classList.remove('active'), 1800);
    world = target;
    if (target === 'level0') {
        const ang = Math.random() * Math.PI * 2, d = 500 + Math.random() * 2000;
        let ni = Math.floor(Math.cos(ang) * d / CELL), nj = Math.floor(Math.sin(ang) * d / CELL);
        if (pillarAt(ni, nj)) ni++;
        player.x = ni * CELL + CELL / 2;
        player.z = nj * CELL + CELL / 2;
        player.y = 0;
        updateChunks(player.x, player.z);
        caption('back in the yellow rooms', 5000);
    } else {
        const lv = ensureLevel(target);
        [player.x, player.y, player.z] = lv.spawn;
        player.yaw = lv.spawnYaw;
        caption(lv.caption, 6000);
    }
    player.vy = 0;
    applyProfile();
    state.nextEvent = state.t + 18 + Math.random() * 15;
}

// per-frame logic while inside a special level
function updateSpecialWorld(dt) {
    if (world === 'doors') {
        if (player.x > DX + 44.5) noclipTo('level0');
    } else if (world === 'household') {
        if (player.x < HX - 14.6) noclipTo('level0');
        // animated water
        waterTex.offset.x = state.t * 0.014;
        waterTex.offset.y = Math.sin(state.t * 0.35) * 0.05;
        // cave drips down in the pools
        if (player.y < -3 && state.t > (state.nextDrip || 0)) {
            state.nextDrip = state.t + 1.2 + Math.random() * 4;
            playDrip();
        }
    }
}

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
    const prof = PROFILES[world];
    // global factor: blackout beats flicker (Level 0 events only)
    let f = 1;
    if (world === 'level0' && state.t < state.blackoutUntil) {
        f = 0.04;
    } else if (world === 'level0' && state.flicker > 0) {
        state.flicker -= dt;
        f = Math.random() < 0.4 ? 0.15 + Math.random() * 0.3 : 1;
    } else if (world === 'doors') {
        // the corridor's wiring is failing
        f = Math.random() < 0.05 ? 0.2 + Math.random() * 0.4 : 1;
    }
    // subtle constant fluorescent shimmer
    f *= 0.965 + 0.035 * Math.sin(state.t * 47) * Math.sin(state.t * 13.7);
    state.lightFactor = f;

    ambient.intensity = prof.amb * f;
    hemi.intensity = prof.hemi * f;
    panelMat.emissiveIntensity = Math.max(0.02, f);
    if (world === 'level0') scene.fog.color.setHex(f < 0.3 ? 0x030302 : 0x0d0b03);

    // hum follows the lights
    if (AC && humGain) {
        const humOn = (f > 0.3 && state.t > state.humOffUntil ? 1 : 0.05) * prof.hum;
        humGain.gain.setTargetAtTime(humOn, AC.currentTime, 0.08);
    }

    // repick nearest fixtures for the point-light pool
    lightRepickAcc += dt;
    if (lightRepickAcc > 0.5) {
        lightRepickAcc = 0;
        const all = [];
        if (world === 'level0') {
            for (const ch of chunks.values())
                for (const p of ch.panels) {
                    const d = (p.x - player.x) ** 2 + (p.z - player.z) ** 2;
                    if (d < 500) all.push([d, p, 0xffeeaa]);
                }
        } else {
            for (const l of currentLevel().lights) {
                const d = (l.pos.x - player.x) ** 2 + (l.pos.z - player.z) ** 2;
                all.push([d, l.pos, l.color]);
            }
        }
        all.sort((a, b) => a[0] - b[0]);
        for (let k = 0; k < LIGHT_POOL; k++) {
            if (k < all.length) {
                pointLights[k].position.copy(all[k][1]);
                pointLights[k].color.setHex(all[k][2]);
            } else pointLights[k].position.set(0, -100, 0);
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

        // crouch: eye follows the local ceiling cap (shrinking doors)
        const targetEye = Math.min(EYE, eyeCapAt(player.x));
        player.eye += (targetEye - player.eye) * Math.min(1, dt * 7);

        // movement
        const wl = waterLevelAt(player.x, player.z);
        const inWater = wl !== null && player.y < wl - 0.15;
        const run = keys['ShiftLeft'] || keys['ShiftRight'];
        let speed = (run ? 5.4 : 2.7) * (0.35 + 0.65 * (player.eye / EYE));
        if (inWater) speed *= 0.5;
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
        // steps taller than knee height act as walls (pool edges)
        if (floorAt(nx, player.z) > player.y + 0.5) nx = player.x;
        if (floorAt(nx, nz) > player.y + 0.5) nz = player.z;
        [nx, nz] = collide(nx, nz);
        const moved = Math.hypot(nx - player.x, nz - player.z);
        player.x = nx; player.z = nz;

        // gravity: fall through holes and off pool edges
        const fl = floorAt(player.x, player.z);
        if (player.y > fl + 0.002) player.vy -= 22 * dt;
        const wasAboveWater = wl !== null && player.y > wl;
        player.y += player.vy * dt;
        if (wl !== null && wasAboveWater && player.y <= wl && player.vy < -2) {
            playSplash(true);
            if (!state.pooled) { state.pooled = true; setTimeout(() => caption('…the pools', 6000), 1200); }
        }
        if (player.y <= fl) {
            if (player.vy < -9 && wl === null) playNoise({ dur: 0.18, freq: 120, q: 1, gain: 0.2 });
            player.y = fl;
            player.vy = 0;
        }

        // head bob + footsteps
        if (moved > 0.0005) {
            player.bob += moved * (run ? 1.9 : 1.6);
            player.stepAcc += moved;
            const stride = run ? 2.3 : 1.9;
            if (player.stepAcc > stride) {
                player.stepAcc = 0;
                if (inWater) playSplash(false);
                else playFootstep(run);
            }
        }

        // scheduled events (Level 0 only)
        if (world === 'level0' && state.t > state.nextEvent &&
            state.entityMode === 'none' && state.staticUntil < state.t) {
            state.nextEvent = state.t + 20 + Math.random() * 22;
            triggerEvent();
        }

        updateEntity(dt);
        if (world === 'level0') updateChunks(player.x, player.z);
        else updateSpecialWorld(dt);
        updateLights(dt);

        // camera: position + handheld sway
        const bobY = Math.sin(player.bob * 2) * 0.035 * (moved > 0.0005 ? 1 : 0);
        const swayY = Math.sin(state.t * 0.9) * 0.008 + Math.sin(state.t * 2.3) * 0.004;
        const swayYaw = Math.sin(state.t * 0.6) * 0.006;
        const swayPitch = Math.cos(state.t * 0.8) * 0.005;
        camera.position.set(player.x, player.y + player.eye + bobY + swayY, player.z);
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
    player, state, entity, startSighting, startChase, triggerEvent, tapeDamage, noclipTo,
    get world() { return world; },
    forceLock() { locked = true; started = true; startScreen.style.display = 'none'; },
};

// prebuild spawn chunks so first frame isn't empty
updateChunks(player.x, player.z);
camera.position.set(player.x, EYE, player.z);
tick();
