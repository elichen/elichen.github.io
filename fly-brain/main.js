// The page: draws every neuron at its cell body, flashes it when it spikes, draws the fly's body
// (fly-view.js), and sends the visitor's drops, activations and silencings to worker.js.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FlyView } from './fly-view.js';

// Palettes for the 3D view and the page accent, each sampled from one image (see styles.css).
const PALETTES = {
    // A · Confocal: Jenett, Schindelin & Heisenberg, "Drosophila melanogaster brain expression patterns" (CC BY 2.0)
    A: { bg: '#030101', rest: '#68101c', restAlpha: 0.55, fire: '#c8bfca', stim: '#d76cd5', silenced: '#393b3b', select: '#a85e80', shell: '#a52c1c', shellAlpha: 0.12, additive: true, accent: '#a52c1c' },
    // B · Fly on slate: André Karwath, "Drosophila melanogaster - side" (CC BY-SA 2.5)
    B: { bg: '#a0a1a5', rest: '#7f716f', restAlpha: 0.18, fire: '#da3809', stim: '#b56c37', silenced: '#341e19', select: '#341e19', shell: '#604f4c', shellAlpha: 0.08, additive: false, accent: '#cd260e' },
    // C · Scanning electron micrograph: Segura & Vieto, "Detalle a ojo compuesto de Drosophila melanogaster" (CC BY-SA 4.0)
    C: { bg: '#010000', rest: '#67412a', restAlpha: 0.55, fire: '#fcf3e5', stim: '#d66428', silenced: '#240301', select: '#d8c0ad', shell: '#8c6a54', shellAlpha: 0.12, additive: true, accent: '#830d05' },
    // D · Fire lookup table: Gerit Linneweber, "Visualization of the Drosophila blood brain barrier" (CC BY 2.0)
    D: { bg: '#f1f0fc', rest: '#908cf9', restAlpha: 0.16, fire: '#d4212a', stim: '#fcd635', silenced: '#4d37e6', select: '#8716a5', shell: '#6a62f5', shellAlpha: 0.10, additive: false, accent: '#6826cb' },
};

const BIN = 0.1;            // s per rate bin
const BINS = 10;            // 1 s of rate history
const FLASH = 0.03;         // s for a spike's flash to fade to 1/e
const viewport = document.getElementById('viewport');
const canvas = document.getElementById('view');
const brainPane = document.getElementById('brain-pane');
const status = document.getElementById('status');
const params = new URLSearchParams(location.search);

const state = {
    n: 0,
    neurons: null,
    t: 0,
    speed: 1,
    active: 0,
    paused: false,
    rate: 150,
    drop: 'sugar',          // taste of the drop on the floor, or null
    near: false,            // the next drop lands at the mouth, not ahead of the fly
    touch: false,           // the drop touches the labellum, so its taste neurons get input
    walk: true,             // P9 gets input
    move: { walk: 0, turn: 0 },
    sets: new Set(),        // names of activated sensory sets
    single: new Set(),      // neurons activated one by one
    silenced: new Set(),
    selected: -1,
    palette: params.get('palette') || 'A',
};

const worker = new Worker('worker.js', { type: 'module' });
const flyView = new FlyView(document.getElementById('fly'), pos => worker.postMessage({ type: 'wick', pos }));
// the neurons a drop of each taste drives (Shiu et al.'s sets)
const tasteSets = name => name.split('+').flatMap(t => state.neurons.sets[t].index);
let counts = null;          // spikes per neuron per bin, BINS bins
let binOf = -1;
let lastSpike = null;
let positions = null;

// ---------------------------------------------------------------- scene

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 1, 10000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.rotateSpeed = 0.6;

const pointMaterial = new THREE.ShaderMaterial({
    uniforms: {
        uNow: { value: 0 },
        uFlash: { value: FLASH },
        uScale: { value: 1 },
        uRest: { value: new THREE.Color() },
        uRestAlpha: { value: 0.35 },
        uFire: { value: new THREE.Color() },
        uStim: { value: new THREE.Color() },
        uSilenced: { value: new THREE.Color() },
        uSelect: { value: new THREE.Color() },
    },
    vertexShader: /* glsl */`
        attribute float lastSpike;
        attribute float flags;       // 1 activated, 2 silenced, 4 selected, 8 no position
        uniform float uNow, uFlash, uScale, uRestAlpha;
        uniform vec3 uRest, uFire, uStim, uSilenced, uSelect;
        varying vec4 vColor;
        void main() {
            float glow = exp(-max(uNow - lastSpike, 0.0) / uFlash);
            float f = flags;
            bool selected = mod(floor(f / 4.0), 2.0) > 0.5;
            bool silenced = mod(floor(f / 2.0), 2.0) > 0.5;
            bool stim = mod(f, 2.0) > 0.5;
            vec3 base = stim ? uStim : silenced ? uSilenced : uRest;
            float alpha = stim || silenced ? 0.9 : uRestAlpha;
            vColor = vec4(mix(base, uFire, glow), max(alpha, glow));
            float size = 2.2 + 7.0 * glow + (stim ? 2.0 : 0.0);   // µm
            if (selected) { vColor = vec4(uSelect, 1.0); size = 12.0; }
            vec4 mv = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = f >= 8.0 ? 0.0 : max(size * uScale / -mv.z, 1.5);
            gl_Position = projectionMatrix * mv;
        }`,
    fragmentShader: /* glsl */`
        varying vec4 vColor;
        void main() {
            vec2 c = gl_PointCoord - 0.5;
            float r = length(c);
            if (r > 0.5) discard;
            gl_FragColor = vec4(vColor.rgb, vColor.a * smoothstep(0.5, 0.2, r));
        }`,
    transparent: true,
    depthWrite: false,
});

const shellMaterial = new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color() }, uAlpha: { value: 0.1 } },
    vertexShader: /* glsl */`
        varying vec3 vN; varying vec3 vView;
        void main() {
            vec4 mv = modelViewMatrix * vec4(position, 1.0);
            vN = normalize(normalMatrix * normal);
            vView = normalize(-mv.xyz);
            gl_Position = projectionMatrix * mv;
        }`,
    fragmentShader: /* glsl */`
        uniform vec3 uColor; uniform float uAlpha;
        varying vec3 vN; varying vec3 vView;
        void main() {
            float rim = 1.0 - abs(dot(normalize(vN), normalize(vView)));
            gl_FragColor = vec4(uColor, uAlpha * (0.25 + 2.5 * pow(rim, 3.0)));
        }`,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
});

let points = null;
const center = new THREE.Vector3();

// FlyWire (µm): the fly's left is at low x, y grows ventral, z grows posterior. The view starts
// at the fly's face, dorsal up, so the fly's left is on the viewer's right.
function toWorld(x, y, z, out, k) {
    out[k] = -x; out[k + 1] = -y; out[k + 2] = -z;
}

function buildPoints(pos) {
    const n = pos.length;
    positions = new Float32Array(3 * n);
    const flags = new Float32Array(n);
    lastSpike = new Float32Array(n).fill(-1e9);
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (let i = 0; i < n; i++) {
        if (!pos[i]) { flags[i] = 8; continue; }
        toWorld(pos[i][0], pos[i][1], pos[i][2], positions, 3 * i);
        box.expandByPoint(v.fromArray(positions, 3 * i));
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setAttribute('lastSpike', new THREE.BufferAttribute(lastSpike, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('flags', new THREE.BufferAttribute(flags, 1).setUsage(THREE.DynamicDrawUsage));
    g.boundingBox = box;
    g.boundingSphere = box.getBoundingSphere(new THREE.Sphere());
    points = new THREE.Points(g, pointMaterial);
    points.frustumCulled = false;
    scene.add(points);
    box.getCenter(center);
    const size = box.getSize(new THREE.Vector3());
    brainWidth = size.x;
    camera.near = size.x * 0.05;
    camera.far = size.x * 10;
}

let brainWidth = 0;
let moved = false;          // the visitor turned or zoomed the view
controls.addEventListener('start', () => { moved = true; });
// front view with the brain's width filling ~90% of the viewport (its height is ~55% of its width)
function fit() {
    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const d = Math.max(brainWidth / 0.9 / 2 / (tanV * camera.aspect), brainWidth * 0.55 / 0.9 / 2 / tanV);
    controls.target.copy(center);
    camera.position.set(center.x, center.y, center.z + d);
    camera.updateProjectionMatrix();
}

function buildShell(np) {
    const v = new Float32Array(np.vertices.length);
    for (let k = 0; k < v.length; k += 3) toWorld(np.vertices[k], np.vertices[k + 1], np.vertices[k + 2], v, k);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(v, 3));
    g.setIndex(np.faces);
    g.computeVertexNormals();
    const mesh = new THREE.Mesh(g, shellMaterial);
    mesh.renderOrder = -1;
    scene.add(mesh);
}

function colors(name) {
    const p = PALETTES[name] || PALETTES.A;
    state.palette = name;
    scene.background = new THREE.Color(p.bg);
    flyView.setBackground(p.bg);
    viewport.style.background = p.bg;
    const u = pointMaterial.uniforms;
    u.uRest.value.set(p.rest);
    u.uRestAlpha.value = p.restAlpha;
    u.uFire.value.set(p.fire);
    u.uStim.value.set(p.stim);
    u.uSilenced.value.set(p.silenced);
    u.uSelect.value.set(p.select);
    pointMaterial.blending = p.additive ? THREE.AdditiveBlending : THREE.NormalBlending;
    pointMaterial.needsUpdate = true;
    shellMaterial.uniforms.uColor.value.set(p.shell);
    shellMaterial.uniforms.uAlpha.value = p.shellAlpha;
    viewport.classList.toggle('light', !p.additive);
    document.documentElement.style.setProperty('--accent', p.accent);
}

function resize() {
    const w = brainPane.clientWidth, h = brainPane.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    // pixels per µm at unit distance
    pointMaterial.uniforms.uScale.value = h * renderer.getPixelRatio() / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    if (brainWidth && !moved) fit();
}
new ResizeObserver(resize).observe(brainPane);

// ---------------------------------------------------------------- names

// names the paper uses for neurons that FlyWire types differently
const PAPER_NAMES = { MN9: 'MN9', aBN1: 'aBN1', aDN1: 'aDN1', aDN2: 'aDN2' };
const neuronName = i => {
    const nr = state.neurons;
    for (const k in PAPER_NAMES) if (nr.named[k] === i) return `${PAPER_NAMES[k]} (${nr.cell_type.names[nr.cell_type.index[i]]})`;
    const type = nr.cell_type.names[nr.cell_type.index[i]];
    return type || 'unnamed ' + (nr.cell_class.names[nr.cell_class.index[i]] || nr.super_class.names[nr.super_class.index[i]] || 'neuron');
};
const field = (name, i) => state.neurons[name].names[state.neurons[name].index[i]].replace(/_/g, ' ');

function describe(i) {
    const parts = [field('super_class', i), field('cell_class', i), field('cell_sub_class', i)].filter(Boolean);
    const unique = parts.filter((p, k) => parts.indexOf(p) === k);
    const side = field('side', i);
    const nt = field('nt', i);
    return `${unique.join(' · ')}${side ? ` · ${side} side` : ''}${nt ? ` · predicted ${nt}` : ''}`;
}

// ---------------------------------------------------------------- spikes and rates

function rateOf(i) {
    let s = 0;
    for (let b = 0; b < BINS; b++) s += counts[i * BINS + b];
    return s / (BINS * BIN);
}

function onSpikes(m) {
    const bin = Math.floor(m.t / BIN);
    if (bin !== binOf) {
        // clear the bins we move into
        for (let b = binOf + 1; b <= bin && b <= binOf + BINS; b++) {
            const k = ((b % BINS) + BINS) % BINS;
            for (let i = 0; i < state.n; i++) counts[i * BINS + k] = 0;
        }
        binOf = bin;
    }
    const s = m.spikes;
    for (let k = 0; k < s.length; k += 2) {
        const t = s[k] * 1e-4, i = s[k + 1];
        lastSpike[i] = t;
        const b = Math.floor(t / BIN) % BINS;
        if (counts[i * BINS + b] < 65535) counts[i * BINS + b]++;
    }
    if (s.length) points.geometry.attributes.lastSpike.needsUpdate = true;
    state.t = m.t;
    state.speed = m.speed;
    state.active = m.active;
    flyView.update(m);
    state.move = { walk: m.walk, turn: m.turn };
    if (m.touch !== state.touch) {
        state.touch = m.touch;
        refreshFlags();
    }
}

// ---------------------------------------------------------------- stimulation

function stimulated() {
    const on = new Set(state.single);
    for (const name of state.sets) for (const i of state.neurons.sets[name].index) on.add(i);
    return on;
}

// neurons getting input now: the activated ones, and the drop's taste neurons while it touches
function driven() {
    const on = new Set(applied);
    if (state.drop && state.touch) for (const i of tasteSets(state.drop)) on.add(i);
    if (state.walk) for (const i of cellsOf(WALK_INPUT)) on.add(i);
    return on;
}

const WALK_INPUT = 'DNp09', WALK_HZ = 100;      // P9, as in forage.js's tests
const cellsOf = type => { const k = state.neurons.cell_type.names.indexOf(type); return state.neurons.cell_type.index.flatMap((x, i) => x === k ? [i] : []); };

let applied = new Set();
function applyRates() {
    const on = stimulated();
    const off = [...applied].filter(i => !on.has(i));
    if (off.length) worker.postMessage({ type: 'rate', neurons: off, hz: 0 });
    if (on.size) worker.postMessage({ type: 'rate', neurons: [...on], hz: state.rate });
    applied = on;
    worker.postMessage({ type: 'walk', hz: state.walk ? WALK_HZ : 0 });
    worker.postMessage({ type: 'drop', taste: state.drop, hz: state.rate, near: state.near });
    flyView.setTaste(state.drop);
    refreshFlags();
}

function refreshFlags() {
    const f = points.geometry.attributes.flags;
    for (let i = 0; i < state.n; i++) f.array[i] = state.neurons.pos[i] ? 0 : 8;
    for (const i of driven()) f.array[i] += 1;
    for (const i of state.silenced) f.array[i] += 2;
    if (state.selected >= 0) f.array[state.selected] += 4;
    f.needsUpdate = true;
    for (const b of document.querySelectorAll('.chip[data-set]')) b.setAttribute('aria-pressed', state.sets.has(b.dataset.set));
    for (const b of document.querySelectorAll('.chip[data-drop]')) b.setAttribute('aria-pressed', (b.dataset.drop || null) === state.drop);
    document.getElementById('walk').setAttribute('aria-pressed', state.walk);
    document.getElementById('touch').textContent = !state.drop ? 'No drop. Choose one below.'
        : state.touch ? `Tasting the ${state.drop.replace('+', ' and ')} drop` : state.walk ? 'Walking. A rule, not the brain, turns it toward the smell.' : 'Standing';
}

function silence(i, on) {
    if (on) state.silenced.add(i); else state.silenced.delete(i);
    worker.postMessage({ type: 'silence', neurons: [i], on });
    refreshFlags();
}

// ---------------------------------------------------------------- panel

const topList = document.getElementById('top');
const card = document.getElementById('card');

function renderPanel() {
    if (!counts) return;
    // neurons that fire in response, not the activated ones
    const rated = [];
    let firing = 0;
    const input = driven();
    for (let i = 0; i < state.n; i++) {
        let s = 0;
        for (let b = 0; b < BINS; b++) s += counts[i * BINS + b];
        if (s) {
            firing++;
            if (!input.has(i)) rated.push([s / (BINS * BIN), i]);
        }
    }
    rated.sort((a, b) => b[0] - a[0]);
    document.getElementById('n-active').textContent = `${firing.toLocaleString()} firing`;
    const rows = rated.slice(0, 10).map(([r, i]) =>
        `<li data-i="${i}"><span class="bar" style="--w:${Math.min(1, r / 300)}"></span><span class="nm">${neuronName(i)}</span><span class="cls">${field('super_class', i)}</span><span class="hz">${r.toFixed(0)} Hz</span></li>`);
    topList.innerHTML = rows.join('') || `<li class="empty">${input.size ? 'No other neuron fires.' : 'Nothing fires. Touch a drop to the labellum, or activate a set of neurons above.'}</li>`;
    for (const dd of document.querySelectorAll('[data-named]')) dd.textContent = `${rateOf(state.neurons.named[dd.dataset.named]).toFixed(0)} Hz`;
    // a muscle's motor neurons, mean rate
    for (const dd of document.querySelectorAll('[data-motor]')) {
        const cells = dd.dataset.motor.split(',').flatMap(t => state.neurons.motor[t]);
        dd.textContent = `${(cells.reduce((s, i) => s + rateOf(i), 0) / cells.length).toFixed(0)} Hz`;
    }
    for (const dd of document.querySelectorAll('[data-type]')) {
        const cells = cellsOf(dd.dataset.type);
        dd.textContent = `${(cells.reduce((s, i) => s + rateOf(i), 0) / cells.length).toFixed(0)} Hz`;
    }
    document.getElementById('walk-speed').textContent = `${state.move.walk.toFixed(1)} cm/s`;
    document.getElementById('walk-turn').textContent = `${state.move.turn.toFixed(1)} rad/s`;
    if (state.selected >= 0) document.getElementById('card-rate').textContent = `${rateOf(state.selected).toFixed(0)} Hz over the last second`;
    const clock = document.getElementById('clock');
    clock.textContent = `${state.t.toFixed(1)} s` + (state.paused ? ' · paused' : state.speed < 0.95 ? ` · ${state.speed.toFixed(2)}× real time` : ' · real time');
}

function select(i) {
    state.selected = i;
    refreshFlags();
    if (i < 0) { card.hidden = true; return; }
    card.hidden = false;
    document.getElementById('card-name').textContent = neuronName(i);
    document.getElementById('card-what').textContent = describe(i);
    const id = state.neurons.ids[i];
    const codex = document.getElementById('card-codex');
    codex.href = `https://codex.flywire.ai/app/cell_details?root_id=${id}&data_version=783`;
    codex.title = `FlyWire ID ${id}`;
    document.getElementById('card-stim').setAttribute('aria-pressed', state.single.has(i));
    document.getElementById('card-silence').setAttribute('aria-pressed', state.silenced.has(i));
    renderPanel();
}

// the article's experiments; each starts from rest. near: the fly stands with the drop at its mouth.
const EXPERIMENTS = {
    forage: { drop: 'sugar', rate: 150, walk: true, show: 'MN9' },
    'forage-bitter': { drop: 'bitter', rate: 150, walk: true },
    sugar: { drop: 'sugar', near: true, rate: 150, show: 'MN9' },
    'sugar-bitter': { drop: 'sugar+bitter', near: true, rate: 150, show: 'MN9' },
    water: { drop: 'water', near: true, rate: 200, show: 'MN9' },
    'jo-ce': { sets: ['JO-CE'], rate: 150, show: 'aBN1' },
    'jo-f': { sets: ['JO-F'], rate: 150, show: 'aBN1' },
    silence: { drop: 'sugar', near: true, rate: 100, silence: ['CB0553'], show: 'CB0553' },
    runaway: { sets: ['LB3b-right'], rate: 150 },
};

function experiment(name) {
    const x = EXPERIMENTS[name];
    resetAll();
    for (const s of x.sets || []) state.sets.add(s);
    state.drop = x.drop || null;
    state.near = !!x.near;
    state.walk = !!x.walk;
    state.rate = x.rate;
    rateInput.value = x.rate;
    rateOut.textContent = `${x.rate} Hz`;
    for (const k of x.silence || []) silence(state.neurons.named[k], true);
    applyRates();
    select(x.show ? state.neurons.named[x.show] : -1);
    viewport.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

for (const b of document.querySelectorAll('[data-exp]')) b.onclick = () => experiment(b.dataset.exp);

topList.addEventListener('click', e => {
    const li = e.target.closest('li[data-i]');
    if (li) select(Number(li.dataset.i));
});
document.getElementById('card-close').onclick = () => select(-1);
document.getElementById('card-stim').onclick = () => {
    const i = state.selected;
    if (state.single.has(i)) state.single.delete(i); else state.single.add(i);
    applyRates();
    select(i);
};
document.getElementById('card-silence').onclick = () => {
    const i = state.selected;
    silence(i, !state.silenced.has(i));
    select(i);
};

for (const b of document.querySelectorAll('.chip[data-drop]')) {
    b.onclick = () => {
        state.drop = b.dataset.drop || null;
        state.near = false;
        applyRates();
    };
}

document.getElementById('walk').onclick = () => {
    state.walk = !state.walk;
    applyRates();
};

for (const b of document.querySelectorAll('.chip[data-set]')) {
    b.onclick = () => {
        const name = b.dataset.set;
        if (state.sets.has(name)) state.sets.delete(name); else state.sets.add(name);
        applyRates();
    };
}

const rateInput = document.getElementById('rate');
const rateOut = document.getElementById('rate-out');
rateInput.oninput = () => {
    state.rate = Number(rateInput.value);
    rateOut.textContent = `${state.rate} Hz`;
    applyRates();
};
rateOut.textContent = `${state.rate} Hz`;

document.getElementById('pause').onclick = e => {
    state.paused = !state.paused;
    e.target.textContent = state.paused ? 'Run' : 'Pause';
    worker.postMessage({ type: 'pause', on: state.paused });
};

document.getElementById('reset').onclick = () => resetAll();

function resetAll() {
    state.sets.clear();
    state.drop = null;
    state.near = false;
    state.walk = false;
    state.touch = false;
    state.single.clear();
    for (const i of state.silenced) worker.postMessage({ type: 'silence', neurons: [i], on: false });
    state.silenced.clear();
    applyRates();
    worker.postMessage({ type: 'reset' });
    counts.fill(0);
    lastSpike.fill(-1e9);
    points.geometry.attributes.lastSpike.needsUpdate = true;
    binOf = -1;
    select(-1);
}

// ---------------------------------------------------------------- picking

const raycaster = new THREE.Raycaster();
const tip = document.getElementById('tip');

function pick(e) {
    const r = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    raycaster.params.Points.threshold = camera.position.distanceTo(controls.target) * 0.006;
    const hits = raycaster.intersectObject(points);
    // nearest to the ray, preferring neurons that are firing
    let best = -1, bestScore = Infinity;
    for (const h of hits) {
        if (state.neurons.pos[h.index] == null) continue;
        const score = h.distanceToRay * (state.t - lastSpike[h.index] < 0.2 ? 0.3 : 1);
        if (score < bestScore) { bestScore = score; best = h.index; }
    }
    return best;
}

let down = null;
canvas.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', e => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    select(pick(e));
    document.body.classList.add('clicked');
});
let hoverAt = 0;
canvas.addEventListener('pointermove', e => {
    if (!points || e.buttons || performance.now() - hoverAt < 60) return;
    hoverAt = performance.now();
    const i = pick(e);
    if (i < 0) { tip.hidden = true; return; }
    const r = brainPane.getBoundingClientRect();
    tip.hidden = false;
    tip.textContent = neuronName(i);
    tip.style.left = `${e.clientX - r.left + 12}px`;
    tip.style.top = `${e.clientY - r.top + 12}px`;
});
canvas.addEventListener('pointerleave', () => { tip.hidden = true; });

// ---------------------------------------------------------------- start

function frame() {
    controls.update();
    pointMaterial.uniforms.uNow.value = state.t;
    renderer.render(scene, camera);
    flyView.render();
    requestAnimationFrame(frame);
}

worker.onmessage = e => {
    const m = e.data;
    if (m.type === 'progress') status.textContent = `Loading the connectome… ${(m.loaded / 1e6).toFixed(0)} of ${(m.total / 1e6).toFixed(0)} MB`;
    else if (m.type === 'error') status.textContent = `Could not start the model: ${m.message}`;
    else if (m.type === 'body') flyView.build(m);
    else if (m.type === 'ready') {
        status.textContent = '';
        document.body.classList.add('ready');
        workerReady = true;
        start();
    } else if (m.type === 'spikes' && counts) onSpikes(m);
};

let workerReady = false;
function start() {
    if (workerReady && state.neurons) applyRates();
}

const [neurons, neuropils] = await Promise.all([
    fetch('model/neurons.json').then(r => r.json()),
    fetch('model/neuropils.json').then(r => r.json()),
]);
state.neurons = neurons;
state.n = neurons.count;
counts = new Uint16Array(state.n * BINS);
buildPoints(neurons.pos);
buildShell(neuropils);
colors(state.palette);
resize();
refreshFlags();
requestAnimationFrame(frame);
setInterval(renderPanel, 250);
refreshFlags();
start();

window.flyBrain = { state, colors, select, experiment, worker, flyView, PALETTES };
document.getElementById('fly').addEventListener('pointerdown', () => document.body.classList.add('dragged'));
