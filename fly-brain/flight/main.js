// The flight page: draws every neuron of the brain and nerve cord, flashes it when it spikes, draws the flying fly
// (../fly-view.js), and sends the visitor's looming objects to worker.js.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FlyView } from '../fly-view.js';

// palette A of the brain page, its rest colour brighter: here neurons are smaller than a pixel
const P = { bg: '#030101', rest: '#c02438', restAlpha: 0.8, fire: '#ffffff' };
const FLASH = 0.03;
const WINDOW = 0.25;     // s of spikes behind each rate readout
const canvas = document.getElementById('view');
const status = document.getElementById('status');
const worker = new Worker('worker.js', { type: 'module' });
const flyView = new FlyView(document.getElementById('fly'), () => {});
flyView.follow = 0.5;     // a little behind the fly, so its turns show
flyView.setBackground('#20242b');
flyView.controls.target.set(0, 0, 1);
flyView.camera.position.set(-2.5, -0.5, 1.8);   // behind the fly, which starts toward +x, a little to its right
flyView.controls.minDistance = 0.8;
flyView.controls.maxDistance = 25;
flyView.camera.fov = 30;
flyView.camera.updateProjectionMatrix();
flyView.controls.update();

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(P.bg);
const camera = new THREE.PerspectiveCamera(32, 1, 1, 20000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
const material = new THREE.ShaderMaterial({
    uniforms: { uNow: { value: 0 }, uScale: { value: 1 }, uMin: { value: 1.5 }, uRest: { value: new THREE.Color(P.rest) }, uFire: { value: new THREE.Color(P.fire) } },
    vertexShader: /* glsl */`
        attribute float lastSpike;
        uniform float uNow, uScale, uMin; uniform vec3 uRest, uFire;
        varying vec4 vColor;
        void main() {
            float glow = exp(-max(uNow - lastSpike, 0.0) / ${FLASH.toFixed(3)});
            vColor = vec4(mix(uRest, uFire, glow), max(${P.restAlpha.toFixed(2)}, glow));
            vec4 mv = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = max((3.0 + 9.0 * glow) * uScale / -mv.z, uMin);   // um on screen
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

let neurons = null, lastSpike = null, points = null, t = 0;
const spikes = new Map();       // neuron -> spike times, for the neurons we read out
const state = { steer: true, turn: 0, hit: false, flying: false, clearance: null, dn: [0, 0] };

function build(n) {
    const pos = n.pos, N = n.count;
    // a few points lie far outside the fly: leave them out
    const axis = k => { const a = []; for (let i = k; i < 3 * N; i += 30) a.push(pos[i]); a.sort((x, y) => x - y); return [a[Math.floor(a.length * 0.001)], a[Math.floor(a.length * 0.999)]]; };
    const box = [axis(0), axis(1), axis(2)];
    const v = new Float32Array(3 * N);
    const side = n.side.names, si = n.side.index;
    let lx = 0, ln = 0, rx = 0, rn = 0;
    for (let i = 0; i < N; i++) {
        if (side[si[i]] === 'left') { lx += pos[3 * i]; ln++; } else if (side[si[i]] === 'right') { rx += pos[3 * i]; rn++; }
    }
    // from the fly's front, the fly's left is on the viewer's right
    const flip = lx / ln > rx / rn ? 1 : -1;
    const mid = [(box[0][0] + box[0][1]) / 2, (box[1][0] + box[1][1]) / 2, (box[2][0] + box[2][1]) / 2];
    for (let i = 0; i < N; i++) {
        let inside = true;
        for (let k = 0; k < 3; k++) if (pos[3 * i + k] < box[k][0] || pos[3 * i + k] > box[k][1]) inside = false;
        if (!inside) { v[3 * i + 2] = -1e7; continue; }
        v[3 * i] = flip * (pos[3 * i] - mid[0]);
        v[3 * i + 1] = -(pos[3 * i + 1] - mid[1]);
        v[3 * i + 2] = -(pos[3 * i + 2] - mid[2]);
    }
    lastSpike = new Float32Array(N).fill(-1e9);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(v, 3));
    g.setAttribute('lastSpike', new THREE.BufferAttribute(lastSpike, 1).setUsage(THREE.DynamicDrawUsage));
    points = new THREE.Points(g, material);
    points.frustumCulled = false;
    scene.add(points);
    const h = box[1][1] - box[1][0], w = box[0][1] - box[0][0];
    camera.userData.size = [w, h];
    for (const dd of document.querySelectorAll('[data-cells]')) for (const i of cellsOf(dd.dataset.cells)) spikes.set(i, []);
    resize();
}

// "b3 left", "DNp03 right"
function cellsOf(spec) {
    const [type, s] = spec.split(' ');
    return (neurons.wing[type] || neurons.named[type])[s];
}

function resize() {
    const pane = document.getElementById('brain-pane'), w = pane.clientWidth, h = pane.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    if (camera.userData.size) {
        const [bw, bh] = camera.userData.size, tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
        const d = Math.max(bw / 0.9 / 2 / (tanV * camera.aspect), bh / 0.9 / 2 / tanV);
        camera.position.set(0, 0, d);
        controls.target.set(0, 0, 0);
        camera.far = d * 5;
    }
    // pixels per um at distance 1, in the canvas's real pixels, so a Retina screen draws the same picture
    const pr = renderer.getPixelRatio();
    material.uniforms.uScale.value = h * pr / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    material.uniforms.uMin.value = 2 * pr;
    camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(document.getElementById('brain-pane'));

// the wings as a slow shutter sees them: the real poses of the last wingbeat, each faint
let blurParts = [], blurCount = 0;
function buildBlur(m) {
    blurCount = m.blurCopies;
    const byG = new Map(flyView.parts.map(q => [q.g, q]));
    for (const g of m.wings) {
        const q = byG.get(g);
        if (!q) continue;
        const copies = [];
        for (let k = 0; k < blurCount; k++) {
            const mat = q.mesh.material.clone();
            Object.assign(mat, { transparent: true, opacity: Math.min(1, q.mesh.material.opacity) * 0.16, depthWrite: false });
            const c = new THREE.Mesh(q.mesh.geometry, mat);
            c.matrixAutoUpdate = false; c.frustumCulled = false; c.castShadow = false; c.visible = false;
            flyView.scene.add(c);
            copies.push(c);
        }
        blurParts.push({ i: m.wings.indexOf(g), mesh: q.mesh, copies });
    }
    blurWings = m.wings.length;
}
let blurWings = 0;
function poseBlur(m) {
    for (const { i, mesh, copies } of blurParts) {
        mesh.visible = !m.blur;
        copies.forEach((c, k) => {
            c.visible = !!m.blur;
            if (!m.blur) return;
            const a = (k * blurWings + i) * 12, r = m.blur, e = c.matrix.elements;
            e[0] = r[a + 3]; e[4] = r[a + 4]; e[8] = r[a + 5];
            e[1] = r[a + 6]; e[5] = r[a + 7]; e[9] = r[a + 8];
            e[2] = r[a + 9]; e[6] = r[a + 10]; e[10] = r[a + 11];
            e[12] = r[a]; e[13] = r[a + 1]; e[14] = r[a + 2];
            e[3] = e[7] = e[11] = 0; e[15] = 1;
            c.matrixWorldNeedsUpdate = true;
        });
    }
}

let viewHeading = 0;
function onSpikes(m) {
    t = m.t;
    const s = m.spikes;
    for (let k = 0; k < s.length; k += 2) {
        const time = s[k] * 1e-4, i = s[k + 1];
        lastSpike[i] = time;
        const list = spikes.get(i);
        if (list) list.push(time);
    }
    if (s.length) points.geometry.attributes.lastSpike.needsUpdate = true;
    flyView.update(m);
    poseBlur(m);
    // the camera swings round behind the fly, more slowly than it turns, so its turns show; dragging still works
    const dh = Math.atan2(Math.sin(m.heading - viewHeading), Math.cos(m.heading - viewHeading)) * 0.06;
    viewHeading += dh;
    const c = flyView.camera.position, tg = flyView.controls.target;
    const x = c.x - tg.x, y = c.y - tg.y, cs = Math.cos(dh), sn = Math.sin(dh);
    c.x = tg.x + x * cs - y * sn;
    c.y = tg.y + x * sn + y * cs;
    Object.assign(state, { turn: m.turn, hit: m.hit, flying: m.flying, clearance: m.clearance, dn: m.dn, speed: m.speed, active: m.active });
}

function panel() {
    for (const dd of document.querySelectorAll('[data-cells]')) {
        const cells = cellsOf(dd.dataset.cells);
        let n = 0;
        for (const i of cells) {
            const l = spikes.get(i);
            while (l.length && l[0] < t - WINDOW) l.shift();
            n += l.length;
        }
        dd.textContent = `${(n / cells.length / WINDOW).toFixed(0)} Hz`;
    }
    document.getElementById('turn').textContent = `${state.turn.toFixed(1)} rad/s`;
    document.getElementById('n-active').textContent = state.active ? `${state.active.toLocaleString()} active` : '';
    document.getElementById('clock').textContent = `${t.toFixed(slow ? 3 : 1)} s · ${slow ? '1/50 speed' : state.speed < 0.95 ? `${(state.speed ?? 1).toFixed(2)}× real time` : 'real time'}`;
    document.getElementById('touch').textContent = state.hit ? `The object hit the fly.${state.steer ? '' : ' The brain does not steer.'}`
        : state.flying ? 'An object is coming.' : state.clearance !== null ? `It missed by ${state.clearance.toFixed(1)} cm.` : 'Flying. Send an object.';
    document.getElementById('steer').setAttribute('aria-pressed', state.steer);
}

worker.onmessage = e => {
    const m = e.data;
    if (m.type === 'progress') status.textContent = `Loading the connectome… ${(m.loaded / 1e6).toFixed(0)} of ${(m.total / 1e6).toFixed(0)} MB`;
    else if (m.type === 'error') status.textContent = `Could not start the model: ${m.message}`;
    else if (m.type === 'body') { flyView.build(m); buildBlur(m); }
    else if (m.type === 'ready') { status.textContent = ''; document.body.classList.add('ready'); setTimeout(() => worker.postMessage({ type: 'loom', side: 'left' }), 1500); }
    else if (m.type === 'spikes' && points) onSpikes(m);
};

for (const b of document.querySelectorAll('[data-loom]')) b.onclick = () => worker.postMessage({ type: 'loom', side: b.dataset.loom });
document.getElementById('steer').onclick = () => { state.steer = !state.steer; worker.postMessage({ type: 'steer', on: state.steer }); panel(); };
let slow = false;
document.getElementById('slow').onclick = e => { slow = !slow; e.target.setAttribute('aria-pressed', slow); worker.postMessage({ type: 'slow', on: slow }); };
let paused = false;
document.getElementById('pause').onclick = e => { paused = !paused; e.target.textContent = paused ? 'Run' : 'Pause'; worker.postMessage({ type: 'pause', on: paused }); };
document.getElementById('reset').onclick = () => { worker.postMessage({ type: 'reset' }); lastSpike.fill(-1e9); t = 0; for (const l of spikes.values()) l.length = 0; };

function frame() {
    material.uniforms.uNow.value = t;
    controls.update();
    renderer.render(scene, camera);
    flyView.render();
    requestAnimationFrame(frame);
}

neurons = await fetch('model/neurons.json').then(r => r.json());
build(neurons);
requestAnimationFrame(frame);
setInterval(panel, 200);
window.flyFlight = { state, worker, flyView };
