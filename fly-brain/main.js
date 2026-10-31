// The page: the kitchen table and the fly (fly-view.js, scene.js), its brain and nerve cord flashing beside it, and the visitor's sugar and swatter.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FlyView } from './fly-view.js';
import { buildKitchen, sugarDrop, swatter } from './scene.js';
import { isClear, TABLE } from './day.js';

const P = { bg: '#030101', rest: '#c02438', restAlpha: 0.8, fire: '#ffffff' };
const FLASH = 0.03;
const canvas = document.getElementById('view'), status = document.getElementById('status'), flyCanvas = document.getElementById('fly');
const worker = new Worker('worker.js', { type: 'module' });
const flyView = new FlyView(flyCanvas, () => {});
flyView.floor.visible = flyView.grid.visible = false;
flyView.follow = 0.12;
flyView.camera.far = 900; flyView.camera.fov = 32; flyView.camera.updateProjectionMatrix();
flyView.controls.minDistance = 0.8; flyView.controls.maxDistance = 130;
flyView.controls.target.set(-12, -16, 0.5);
flyView.camera.position.set(-20, -30, 9);
flyView.controls.update();
flyView.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
flyView.renderer.toneMapping = THREE.ACESFilmicToneMapping;
buildKitchen(flyView.scene);
const SWAT_R = 3, bat = swatter(SWAT_R);
flyView.scene.add(bat);
const drops = [];

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
            gl_PointSize = max((3.0 + 9.0 * glow) * uScale / -mv.z, uMin);
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

let lastSpike = null, points = null, t = 0, latest = null;

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
    lastSpike = new Float32Array(N).fill(-1e9);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(v, 3));
    g.setAttribute('lastSpike', new THREE.BufferAttribute(lastSpike, 1).setUsage(THREE.DynamicDrawUsage));
    points = new THREE.Points(g, material); points.frustumCulled = false; scene.add(points);
    camera.userData.size = [box[0][1] - box[0][0], box[1][1] - box[1][0]];
    resize();
}

function resize() {
    const pane = document.getElementById('brain-pane'), w = pane.clientWidth, h = pane.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    if (camera.userData.size) {
        const [bw, bh] = camera.userData.size, tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
        const d = Math.max(bw / 0.9 / 2 / (tanV * camera.aspect), bh / 0.9 / 2 / tanV);
        camera.position.set(0, 0, d); controls.target.set(0, 0, 0); camera.far = d * 5;
    }
    const pr = renderer.getPixelRatio();
    material.uniforms.uScale.value = h * pr / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    material.uniforms.uMin.value = 2 * pr;
    camera.updateProjectionMatrix();
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

const up = new THREE.Vector3(), look = new THREE.Vector3();
function poseWorld(m) {
    while (drops.length < m.foods.length) { const d = sugarDrop(); flyView.scene.add(d); drops.push(d); }
    drops.forEach((d, i) => {
        const f = m.foods[i]; d.visible = !!f;
        if (f) { d.position.set(f[0], f[1], 0); const s = f[2] * Math.sqrt(Math.max(0.1, f[3])); d.scale.set(s, s, s * 0.22); }
    });
    bat.visible = !!m.object;
    if (m.object) {
        const o = m.object; bat.position.set(o.p[0], o.p[1], o.p[2]);
        look.set(o.v[0], o.v[1], o.v[2]).normalize();
        bat.quaternion.setFromUnitVectors(up.set(0, 0, 1), look.negate());
    }
}

function onSpikes(m) {
    t = m.t; latest = m;
    const s = m.spikes;
    for (let k = 0; k < s.length; k += 2) lastSpike[s[k + 1]] = s[k] * 1e-4;
    if (s.length) points.geometry.attributes.lastSpike.needsUpdate = true;
    flyView.update(m);
    poseBlur(m);
    poseWorld(m);
}

const SAY = {
    walk: m => m.eating ? 'Eating. Sugar has cut its walking neuron, and its mouth is reaching down.' : m.goal ? (m.goal.why === 'hunger' ? 'Hungry. Walking to the sugar.' : m.goal.why === 'escape' ? 'Getting away.' : 'Bored. Walking somewhere new.') : 'Standing.',
    rear: () => 'Rearing up to jump.', launch: () => 'Taking off.', touch: () => 'Touchdown. Wings shut.',
    fly: m => m.goal?.why === 'hunger' ? 'Hungry. Flying to the sugar.' : m.goal?.why === 'escape' ? 'Getting away.' : 'Bored. Flying somewhere new.', descend: () => 'Landing.',
};
function panel() {
    const m = latest;
    if (!m) return;
    const bar = (id, v) => { document.getElementById(id).style.width = `${(100 * Math.min(1, v)).toFixed(0)}%`; };
    bar('hunger', m.hunger); bar('bored', m.bored);
    document.getElementById('touch').textContent = m.object ? (m.object.hit ? 'Swatted.' : 'A swatter is coming!') : SAY[m.mode](m);
    for (const [k, v] of Object.entries(m.rates)) { const el = document.querySelector(`[data-rate="${k}"]`); if (el) el.textContent = `${v.toFixed(0)} Hz`; }
    document.getElementById('counts').textContent = `Jumps from the swatter: ${m.counts.jumps} · Swats landed: ${m.counts.hits} · Put back on its feet: ${m.counts.putBacks} · Brain cleared: ${m.counts.calms}`;
    document.getElementById('n-active').textContent = `${m.active.toLocaleString()} active`;
    document.getElementById('clock').textContent = `${t.toFixed(slow ? 3 : 1)} s · ${slow ? '1/50 speed' : m.speed < 0.95 ? `${m.speed.toFixed(2)}× real time` : 'real time'}`;
}

worker.onmessage = e => {
    const m = e.data;
    if (m.type === 'progress') status.textContent = `Loading the connectome… ${(m.loaded / 1e6).toFixed(0)} of ${(m.total / 1e6).toFixed(0)} MB`;
    else if (m.type === 'error') status.textContent = `Could not start the model: ${m.message}`;
    else if (m.type === 'body') { flyView.build(m); buildBlur(m); }
    else if (m.type === 'ready') { status.textContent = ''; document.body.classList.add('ready'); }
    else if (m.type === 'spikes' && points) onSpikes(m);
};

// a click on the table (not a drag) drops sugar there
const ray = new THREE.Raycaster(), table = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), hit = new THREE.Vector3();
let down = null;
flyCanvas.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
flyCanvas.addEventListener('pointerup', e => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    const r = flyCanvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), flyView.camera);
    if (ray.ray.intersectPlane(table, hit) && Math.abs(hit.x) < TABLE.x && Math.abs(hit.y) < TABLE.y && isClear(hit.x, hit.y)) worker.postMessage({ type: 'food', x: hit.x, y: hit.y });
});
for (const b of document.querySelectorAll('[data-swat]')) b.onclick = () => worker.postMessage({ type: 'swat', side: +b.dataset.swat });
let slow = false;
document.getElementById('slow').onclick = e => { slow = !slow; e.target.setAttribute('aria-pressed', slow); worker.postMessage({ type: 'slow', on: slow }); };
let paused = false;
document.getElementById('pause').onclick = e => { paused = !paused; e.target.textContent = paused ? 'Run' : 'Pause'; worker.postMessage({ type: 'pause', on: paused }); };
document.getElementById('reset').onclick = () => { worker.postMessage({ type: 'reset' }); lastSpike.fill(-1e9); t = 0; };

function frame() {
    material.uniforms.uNow.value = t;
    controls.update();
    renderer.render(scene, camera);
    flyView.render();
    requestAnimationFrame(frame);
}
build(await fetch('model/neurons.json').then(r => r.json()));
requestAnimationFrame(frame);
setInterval(panel, 200);
window.flyWorld = { worker, flyView, get latest() { return latest; } };
