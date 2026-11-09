// The page: the kitchen table and the fly (fly-view.js, scene.js), its brain and nerve cord flashing beside it, and the visitor's sugar and swatter.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FlyView } from './fly-view.js';
import { buildKitchen, sugarDrop, swatter } from './scene.js';
import { isClear, TABLE } from './day.js';
import { WingSound } from './audio.js';

const css = name => getComputedStyle(document.body).getPropertyValue(name).trim();
if (new URLSearchParams(location.search).get('palette')) document.body.dataset.palette = new URLSearchParams(location.search).get('palette');
const P = { rest: css('--neuron'), restAlpha: 0.22, fire: css('--accent') };
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
const FLASH = 0.03;
const canvas = document.getElementById('view'), status = document.getElementById('status'), flyCanvas = document.getElementById('fly');
const worker = new Worker('worker.js', { type: 'module' });
const flyView = new FlyView(flyCanvas);
flyView.camera.far = 900; flyView.camera.fov = 32; flyView.camera.updateProjectionMatrix();
flyView.controls.minDistance = 0.8; flyView.controls.maxDistance = 130;
flyView.controls.enableZoom = false;   // the wheel walks through the page; zoom has its own buttons, ctrl + wheel and trackpad pinch
flyCanvas.style.touchAction = 'pan-y';
flyView.controls.target.set(-12, -16, 0.15);   // where the day starts, close enough to see a fly 0.25 cm long
flyView.camera.position.set(-14.2, -21.2, 2.2);
flyView.controls.update();
flyView.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
flyView.renderer.toneMapping = THREE.ACESFilmicToneMapping;
buildKitchen(flyView.scene);
const SWAT_R = 3, bat = swatter(SWAT_R);
flyView.scene.add(bat);
const drops = [];

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 1, 20000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enableZoom = false;
canvas.style.touchAction = 'pan-y';
let brainTurned = false;   // once the visitor has turned or zoomed the brain, a resize (on a phone, the address bar hiding) must not undo it
controls.addEventListener('start', () => { brainTurned = true; });
const material = new THREE.ShaderMaterial({
    uniforms: { uNow: { value: 0 }, uScale: { value: 1 }, uMin: { value: 1.5 }, uAlpha: { value: P.restAlpha }, uRest: { value: new THREE.Color(P.rest) }, uFire: { value: new THREE.Color(P.fire) } },
    vertexShader: /* glsl */`
        attribute float lastSpike;
        uniform float uNow, uScale, uMin, uAlpha; uniform vec3 uRest, uFire;
        varying vec4 vColor;
        void main() {
            float glow = exp(-max(uNow - lastSpike, 0.0) / ${FLASH.toFixed(3)});
            vColor = vec4(mix(uRest, uFire, glow), max(uAlpha, glow));
            vec4 mv = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = max((2.2 + 11.0 * glow) * uScale / -mv.z, uMin);
            gl_Position = projectionMatrix * mv;
        }`,
    fragmentShader: /* glsl */`
        varying vec4 vColor;
        void main() {
            float r = length(gl_PointCoord - 0.5);
            if (r > 0.5) discard;
            gl_FragColor = vec4(vColor.rgb, vColor.a * smoothstep(0.5, 0.2, r));
        }`,
    transparent: true, depthWrite: false,
});

const SPIKE_TEX = 512;   // holds every neuron's last spike time, so a whole neuron's branches can read it
let spikeTex = null, place = null, wires = null;
const wireMaterial = new THREE.ShaderMaterial({
    uniforms: { uSpikes: { value: null }, uNow: { value: 0 }, uRes: { value: new THREE.Vector2(1, 1) }, uWidth: { value: 1.2 }, uFireWidth: { value: 3 }, uAlpha: { value: 0.85 }, uSat: { value: 0.9 }, uLit: { value: 0.45 }, uBase: { value: new THREE.Color(P.rest) }, uTint: { value: 1.0 }, uShow: { value: 0.04 } },
    vertexShader: /* glsl */`
        attribute vec3 a, b; attribute float cell;
        uniform sampler2D uSpikes; uniform float uNow, uWidth, uFireWidth, uAlpha, uSat, uLit, uTint, uShow; uniform vec2 uRes; uniform vec3 uBase;
        varying vec4 vColor;
        vec3 hue(float h) { return clamp(abs(mod(h * 6.0 + vec3(0, 4, 2), 6.0) - 3.0) - 1.0, 0.0, 1.0); }
        void main() {
            float last = texelFetch(uSpikes, ivec2(int(mod(cell, ${SPIKE_TEX}.0)), int(cell) / ${SPIKE_TEX}), 0).r;
            float glow = exp(-max(uNow - last, 0.0) / ${(FLASH * 2).toFixed(3)});
            float h = fract(cell * 0.61803399);   // every neuron its own colour, as FlyWire shows them
            vec3 c = uLit + uSat * min(uLit, 1.0 - uLit) * (hue(h) * 2.0 - 1.0);
            float rest = fract(cell * 0.75487767) < uShow ? uAlpha : 0.0;   // a few neurons stay in view, so each can be followed; the others appear when they fire
            vColor = vec4(mix(mix(uBase, c, uTint), c, glow), mix(rest, 1.0, glow));
            if (vColor.a < 0.01) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
            vec4 pa = modelViewMatrix * vec4(a, 1.0), pb = modelViewMatrix * vec4(b, 1.0);
            vec4 ca = projectionMatrix * pa, cb = projectionMatrix * pb;
            vec2 d = (cb.xy / cb.w - ca.xy / ca.w) * uRes;
            d = length(d) > 1e-4 ? normalize(d) : vec2(1, 0);
            vec4 p = mix(ca, cb, position.x);
            p.xy += vec2(-d.y, d.x) * position.y * mix(uWidth, uFireWidth, glow) / uRes * p.w;
            gl_Position = p;
        }`,
    fragmentShader: /* glsl */`
        varying vec4 vColor;
        void main() { gl_FragColor = vColor; }`,
    transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide,
});

let lastSpike = null, points = null, t = 0, latest = null, spots = {};

function build(n) {
    const pos = n.pos, N = n.count;
    const axis = k => { const a = []; for (let i = k; i < 3 * N; i += 30) a.push(pos[i]); a.sort((x, y) => x - y); return [a[Math.floor(a.length * 0.001)], a[Math.floor(a.length * 0.999)]]; };
    const box = [axis(0), axis(1), axis(2)], v = new Float32Array(3 * N), side = n.side.names, si = n.side.index;
    let lx = 0, ln = 0, rx = 0, rn = 0;
    for (let i = 0; i < N; i++) { if (side[si[i]] === 'left') { lx += pos[3 * i]; ln++; } else if (side[si[i]] === 'right') { rx += pos[3 * i]; rn++; } }
    const flip = lx / ln > rx / rn ? 1 : -1, mid = box.map(b => (b[0] + b[1]) / 2);
    for (let i = 0; i < N; i++) {
        let inside = true;
        for (let k = 0; k < 3; k++) if (pos[3 * i + k] < box[k][0] || pos[3 * i + k] > box[k][1]) inside = false;
        if (!inside) { v[3 * i + 2] = -1e7; continue; }
        v[3 * i] = flip * (pos[3 * i] - mid[0]); v[3 * i + 1] = -(pos[3 * i + 1] - mid[1]); v[3 * i + 2] = -(pos[3 * i + 2] - mid[2]);
    }
    lastSpike = new Float32Array(SPIKE_TEX * SPIKE_TEX).fill(-1e9);
    spikeTex = new THREE.DataTexture(lastSpike, SPIKE_TEX, SPIKE_TEX, THREE.RedFormat, THREE.FloatType);
    place = (x, y, z) => [flip * (x - mid[0]), -(y - mid[1]), -(z - mid[2])];
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(v, 3));
    g.setAttribute('lastSpike', new THREE.BufferAttribute(lastSpike, 1).setUsage(THREE.DynamicDrawUsage));
    points = new THREE.Points(g, material); points.frustumCulled = false; scene.add(points);
    camera.userData.size = [box[0][1] - box[0][0], box[1][1] - box[1][0]];
    // where each named group sits in the view, for its label's line
    const where = cells => { const c = new THREE.Vector3(); for (const i of cells) c.add(new THREE.Vector3(v[3 * i], v[3 * i + 1], v[3 * i + 2])); return c.divideScalar(cells.length); };
    spots = { oDN1: where(n.life.oDN1), MN9: where(n.life.MN9), GF: where(n.life.GF), slow: where([...n.named.DNp04.left, ...n.named.DNp04.right, ...n.named.DNp11.left, ...n.named.DNp11.right]), dnL: where(n.named.DNp03.left), dnR: where(n.named.DNp03.right) };
    resize();
}

// Real neuron shapes (BANC's skeletons, a sample: tools/export_skeletons.py). A neuron without one still flashes as a dot.
async function buildWires(res) {
    const buf = await new Response(res.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    const [N, V] = new Uint32Array(buf, 0, 2), cells = new Uint32Array(buf, 8, N), counts = new Uint32Array(buf, 8 + 4 * N, N);
    const xyz = new Int16Array(buf, 8 + 8 * N, 3 * V), back = new Uint16Array(buf.slice(8 + 8 * N + 6 * V, 8 + 8 * N + 6 * V + 2 * V));
    const a = new Float32Array(3 * V), b = new Float32Array(3 * V), id = new Float32Array(V), v = points.geometry.attributes.position;
    let n = 0, at = 0;
    const out = [0, 0, 0];
    for (let c = 0; c < N; c++) {
        for (let i = 0; i < counts[c]; i++, at++) {
            if (!back[at]) continue;
            const q = at - back[at], A = place(xyz[3 * at] / 20, xyz[3 * at + 1] / 20, xyz[3 * at + 2] / 20), B = place(xyz[3 * q] / 20, xyz[3 * q + 1] / 20, xyz[3 * q + 2] / 20);
            a.set(A, 3 * n); b.set(B, 3 * n); id[n++] = cells[c];
        }
        v.setZ(cells[c], -1e7);   // drawn as a neuron now, not a dot
    }
    v.needsUpdate = true;
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, -1, 0, 1, -1, 0, 1, 1, 0, 0, -1, 0, 1, 1, 0, 0, 1, 0]), 3));
    g.setAttribute('a', new THREE.InstancedBufferAttribute(a.subarray(0, 3 * n), 3));
    g.setAttribute('b', new THREE.InstancedBufferAttribute(b.subarray(0, 3 * n), 3));
    g.setAttribute('cell', new THREE.InstancedBufferAttribute(id.subarray(0, n), 1));
    g.instanceCount = n;
    wires = new THREE.Mesh(g, wireMaterial); wires.frustumCulled = false; scene.add(wires);
    wireMaterial.uniforms.uSpikes.value = spikeTex;
    material.uniforms.uAlpha.value = 0;
}

let fitBrain = 1;
function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    if (camera.userData.size) {
        const [bw, bh] = camera.userData.size, tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
        const d = Math.max(bw / 0.82 / 2 / (tanV * camera.aspect), bh / 0.82 / 2 / tanV);
        if (!brainTurned) { camera.position.set(0, 0, d); controls.target.set(0, 0, 0); }
        camera.far = d * 5; controls.minDistance = d * 0.1; controls.maxDistance = d * 2.5;
        fitBrain = d;
    }
    const pr = renderer.getPixelRatio();
    material.uniforms.uScale.value = h * pr / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    material.uniforms.uMin.value = 1.2 * pr;
    wireMaterial.uniforms.uRes.value.set(w * pr, h * pr);
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);   // setSize clears the canvas: draw now, so a resize never shows it blank
}
new ResizeObserver(resize).observe(document.getElementById('brain-pane'));

let blurParts = [], blurWings = 0;
function buildBlur(m) {
    const byG = new Map(flyView.parts.map(q => [q.g, q]));
    for (const g of m.wings) {
        const q = byG.get(g);
        if (!q) continue;
        const copies = [];
        for (let k = 0; k < m.blurCopies; k++) {
            const mat = q.mesh.material.clone();
            Object.assign(mat, { transparent: true, opacity: Math.min(1, q.mesh.material.opacity) * 0.16, depthWrite: false });
            const c = new THREE.Mesh(q.mesh.geometry, mat);
            c.matrixAutoUpdate = false; c.frustumCulled = false; c.castShadow = false; c.visible = false;
            flyView.scene.add(c); copies.push(c);
        }
        blurParts.push({ i: m.wings.indexOf(g), mesh: q.mesh, copies });
    }
    blurWings = m.wings.length;
}
function poseBlur(m) {
    for (const { i, mesh, copies } of blurParts) {
        mesh.visible = !m.blur;
        copies.forEach((c, k) => {
            c.visible = !!m.blur;
            if (!m.blur) return;
            const a = (k * blurWings + i) * 12, r = m.blur, e = c.matrix.elements;
            e[0] = r[a + 3]; e[4] = r[a + 4]; e[8] = r[a + 5]; e[1] = r[a + 6]; e[5] = r[a + 7]; e[9] = r[a + 8]; e[2] = r[a + 9]; e[6] = r[a + 10]; e[10] = r[a + 11];
            e[12] = r[a]; e[13] = r[a + 1]; e[14] = r[a + 2]; e[3] = e[7] = e[11] = 0; e[15] = 1;
            c.matrixWorldNeedsUpdate = true;
        });
    }
}

const up = new THREE.Vector3(), look = new THREE.Vector3(), where = new THREE.Vector3();
let sound = null, soundOn = false, paused = false;
function poseWorld(m) {
    while (drops.length < m.foods.length) { const d = sugarDrop(); flyView.scene.add(d); drops.push(d); }
    drops.forEach((d, i) => {
        const f = m.foods[i]; d.visible = !!f;
        if (f) { d.position.set(f[0], f[1], 0); const s = f[2] * Math.sqrt(Math.max(0.1, f[3])); d.scale.set(s, s, s * 0.22); }
    });
    bat.visible = !!m.object;
    if (m.object) {
        // drawn only: the swing ends flat on the table (or on the fly), not through it. day.js's object keeps its straight path.
        const o = m.object, flat = Math.min(1, Math.max(0, o.p[2] / 6)), speed = Math.hypot(...o.v);
        bat.position.set(o.p[0], o.p[1], Math.max(o.p[2], o.hit ? 0.3 : 0.04));
        look.set(-o.v[0] * flat / speed, -o.v[1] * flat / speed, 1 - flat - o.v[2] * flat / speed).normalize();
        bat.quaternion.setFromUnitVectors(up.set(0, 0, 1), look);
    }
}

function onSpikes(m) {
    if (m.t < t) { lastSpike.fill(-1e9); points.geometry.attributes.lastSpike.needsUpdate = true; spikeTex.needsUpdate = true; }   // after Reset, or a neuron with a spike 'in the future' stays lit until the clock gets there
    t = m.t; latest = m;
    const s = m.spikes;
    for (let k = 0; k < s.length; k += 2) lastSpike[s[k + 1]] = s[k] * 1e-4;
    if (s.length) { points.geometry.attributes.lastSpike.needsUpdate = true; spikeTex.needsUpdate = true; }
    flyView.update(m);
    poseBlur(m);
    poseWorld(m);
    if (sound) {
        where.set(m.fly[0], m.fly[1], m.fly[2]);
        const dist = where.distanceTo(flyView.camera.position);
        sound.update(m.flapping && !m.slow && !paused, m.wingHz, dist, where.project(flyView.camera).x);
    }
}

const SAY = {
    walk: m => m.perched ? 'The fly stands on something. It will leave by air.' : m.eating ? 'The fly eats. Sugar decreases the firing of oDN1, and the mouth reaches down.' : m.goal ? (m.goal.why === 'hunger' ? 'The fly is hungry. It walks to the sugar.' : m.goal.why === 'escape' ? 'The fly moves away.' : 'The fly is bored. It walks to a new place.') : 'The fly stands.',
    rear: () => 'The fly rears up to jump.', launch: () => 'The fly takes off.', touch: () => 'The fly touches the table. The wings shut.',
    fly: m => m.goal?.why === 'hunger' ? 'The fly is hungry. It flies to the sugar.' : m.goal?.why === 'escape' ? 'The fly moves away.' : 'The fly is bored. It flies to a new place.', descend: () => 'The fly lands.',
};
function panel() {
    const m = latest;
    if (!m) return;
    const bar = (id, v) => { const b = document.getElementById(id), pct = (100 * Math.min(1, v)).toFixed(0); b.style.width = `${pct}%`; b.parentElement.setAttribute('aria-valuenow', pct); };
    bar('hunger', m.hunger); bar('bored', m.bored);
    const caption = document.getElementById('touch');
    const line = performance.now() < noticeUntil ? notice : m.object ? (m.object.hit ? 'Swatted.' : 'A swatter is coming!') : SAY[m.mode](m);
    if (caption.textContent !== line) caption.textContent = line;   // it is a live region: same text, no new announcement
    document.getElementById('why').href = '#' + (m.object ? 'swat' : m.eating ? 'oneb' : m.mode === 'fly' ? 'dodge' : m.mode === 'walk' ? 'what' : 'takeoff');
    for (const b of swats) b.disabled = !!m.object;         // one swatter at a time
    for (const [k, v] of Object.entries(m.rates)) { const el = document.querySelector(`[data-rate="${k}"]`); if (el) el.textContent = `${v.toFixed(0)} Hz`; }
    document.getElementById('counts').textContent = `Jumps from the swatter: ${m.counts.jumps} · Swats landed: ${m.counts.hits} · Put back on its feet: ${m.counts.putBacks}`;
    document.getElementById('spike-rate').textContent = `${(Math.round(m.spikeRate / 100) * 100).toLocaleString('en-US')} spikes/s`;
    document.getElementById('clock').textContent = `${t.toFixed(slow ? 3 : 1)} s · ${paused ? 'paused' : slow ? '1/50 speed' : m.speed < 0.95 ? `${m.speed.toFixed(2)}× real time` : 'real time'}`;
}

worker.onmessage = e => {
    const m = e.data;
    if (m.type === 'progress') status.textContent = `Loading the connectome… ${(m.loaded / 1e6).toFixed(0)} of ${(m.total / 1e6).toFixed(0)} MB`;
    else if (m.type === 'building') status.textContent = 'Building the fly…';
    else if (m.type === 'error') status.textContent = `Could not start the model: ${m.message}`;
    else if (m.type === 'body') { flyView.build(m); buildBlur(m); sound = new WingSound(m.pattern); }
    else if (m.type === 'ready') { status.textContent = ''; document.body.classList.add('ready'); arrive(); for (const b of document.querySelectorAll('.console button')) b.disabled = false; }
    else if (m.type === 'spikes' && points) onSpikes(m);
};

worker.onerror = e => { status.textContent = `The model stopped: ${e.message || 'a script did not load'}.`; };

// a click on the table (not a drag) drops sugar there
const ray = new THREE.Raycaster(), table = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), hit = new THREE.Vector3();
let down = null, noticeUntil = 0, notice = '';
const say = (text, ms) => { notice = text; noticeUntil = performance.now() + ms; panel(); };
const pressed = new Set();   // a second finger makes it a pinch, never a drop
flyCanvas.addEventListener('pointerdown', e => { pressed.add(e.pointerId); down = pressed.size === 1 && e.button === 0 ? [e.clientX, e.clientY] : null; });
flyCanvas.addEventListener('pointercancel', e => { pressed.delete(e.pointerId); down = null; });
flyCanvas.addEventListener('pointerup', e => {
    pressed.delete(e.pointerId);
    const from = down;
    down = null;
    if (!from) return;
    if (Math.hypot(e.clientX - from[0], e.clientY - from[1]) > 5) return;
    const r = flyCanvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), flyView.camera);
    if (!ray.ray.intersectPlane(table, hit)) return;
    if (Math.abs(hit.x) < TABLE.x && Math.abs(hit.y) < TABLE.y && isClear(hit.x, hit.y)) { worker.postMessage({ type: 'food', x: hit.x, y: hit.y }); document.getElementById('hint').classList.add('done'); say('You dropped sugar. When the fly is hungry, it goes to the nearest sugar.', 3000); }
    else say('Something stands there, or the point is too near the edge. Try an open area of the table.', 2500);
});
const swats = document.querySelectorAll('[data-swat]');
for (const b of swats) b.onclick = () => worker.postMessage({ type: 'swat', side: +b.dataset.swat });
let slow = false;
document.getElementById('slow').onclick = e => { slow = !slow; e.target.setAttribute('aria-pressed', slow); worker.postMessage({ type: 'slow', on: slow }); };
document.getElementById('sound').onclick = e => { soundOn = !soundOn; e.target.setAttribute('aria-pressed', soundOn); sound?.set(soundOn); };
document.getElementById('pause').onclick = e => { paused = !paused; e.target.textContent = paused ? 'Run' : 'Pause'; worker.postMessage({ type: 'pause', on: paused }); if (paused) sound?.update(false, 218, 1, 0); };
document.getElementById('reset').onclick = () => { noticeUntil = 0; worker.postMessage({ type: 'reset' }); };

// Arriving: the view comes down from the whole table to the fly, once. Any touch of the view ends it.
let intro = null;
const offset = new THREE.Vector3();
function arrive() {
    if (calm) return;
    intro = { start: performance.now(), base: flyView.camera.position.clone().sub(flyView.controls.target) };
}
flyView.controls.addEventListener('start', () => { intro = null; });
function arriving() {
    if (!intro) return;
    const u = Math.min(1, (performance.now() - intro.start) / 4500), e = 1 - (1 - u) ** 3, s = 4.5 + (1 - 4.5) * e;
    offset.copy(intro.base).multiplyScalar(s);
    flyView.camera.position.copy(flyView.controls.target).add(offset);
    if (u === 1) intro = null;
}

// Walking through: the room at the middle of the window sets what the world does. Reading, the fly moves over to make room for the text; in a room about the brain, the brain steps forward.
const rooms = [...document.querySelectorAll('.room')];
function walk() {
    const mid = innerHeight / 2, here = rooms.find(r => { const b = r.getBoundingClientRect(); return b.top < mid && b.bottom > mid; });
    document.body.dataset.room = here?.dataset.room || (here ? 'text' : '');
    document.body.classList.toggle('scrolled', scrollY > innerHeight * 0.3);
    flyView.shift = innerWidth < 760 ? innerWidth * 0.16 : here && innerWidth >= 1000 ? -Math.min(innerWidth * 0.2, 330) : 0;   // phone: the fly stands left of the brain
}
addEventListener('scroll', walk, { passive: true }); addEventListener('resize', walk);
walk();

// zoom: buttons, and ctrl + wheel (which is also what a trackpad pinch sends)
function zoom(k) {
    const c = flyView.controls, o = offset.copy(flyView.camera.position).sub(c.target);
    intro = null;
    o.setLength(Math.max(c.minDistance, Math.min(c.maxDistance, o.length() * k)));
    flyView.camera.position.copy(c.target).add(o);
}
document.getElementById('zoom-in').onclick = () => zoom(0.75);
document.getElementById('zoom-out').onclick = () => zoom(1 / 0.75);
flyCanvas.addEventListener('wheel', e => { if (e.ctrlKey) { e.preventDefault(); zoom(Math.exp(e.deltaY * 0.01)); } }, { passive: false });

// the lines from each label to its neurons
const leaders = document.getElementById('leaders'), labels = [...document.querySelectorAll('#neurons li')], lines = labels.map(() => { const l = document.createElementNS('http://www.w3.org/2000/svg', 'line'); leaders.appendChild(l); return l; });
const spot = new THREE.Vector3();
function drawLeaders() {
    const pane = document.getElementById('brain-pane').getBoundingClientRect(), view = canvas.getBoundingClientRect(), show = pane.width > 260;
    leaders.style.display = show ? '' : 'none';
    if (!show || !spots.GF) return;
    labels.forEach((li, i) => {
        const b = li.getBoundingClientRect(), edge = li.parentElement.getBoundingClientRect().right;   // from the plate's edge, so a line never crosses another label
        spot.copy(spots[li.dataset.group]).project(camera);
        const l = lines[i];
        l.setAttribute('x1', edge - pane.left); l.setAttribute('y1', b.top + b.height / 2 - pane.top);
        l.setAttribute('x2', view.left - pane.left + (spot.x + 1) / 2 * view.width); l.setAttribute('y2', view.top - pane.top + (1 - spot.y) / 2 * view.height);
    });
}

function frame() {
    requestAnimationFrame(frame);
    material.uniforms.uNow.value = t;
    wireMaterial.uniforms.uNow.value = t;
    controls.update();
    renderer.render(scene, camera);
    drawLeaders();
    arriving();
    flyView.render();
}
if (matchMedia('(pointer: coarse)').matches) document.getElementById('hint').textContent = 'Tap the table to drop sugar. Drag sideways to turn the view.';
build(await fetch('model/neurons.json').then(r => r.json()));
fetch('model/skeletons.bin').then(buildWires).catch(e => console.error(e));
requestAnimationFrame(frame);
setInterval(panel, 200);
window.flyWorld = { wire: wireMaterial.uniforms, colors(rest, fire, alpha) { material.uniforms.uRest.value.set(rest); material.uniforms.uFire.value.set(fire); material.uniforms.uAlpha.value = alpha; }, worker, flyView, get latest() { return latest; }, get sound() { return sound; } };
