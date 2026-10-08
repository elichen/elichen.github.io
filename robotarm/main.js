// The page: loads MuJoCo (WebAssembly), the Panda and the two trained policies, then
// runs physics and the policies on the main thread (a control step costs well under a
// millisecond) and draws with three.js.
//
// The pick policy lifts the box, the throw policy throws it along the arrow (a new
// random direction each time), and after it lands a new box drops. You can grab the
// box and drop it anywhere; the arm goes and gets it.
import * as THREE from 'three';
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.js';
import { PandaEnv, Throws, SUBSTEPS, CENTER, makeRandom } from './panda-env.js';
import { Policy } from './policy.js';
import { SceneRenderer } from './render.js';

const $ = id => document.getElementById(id);
const status = $('status');
const random = makeRandom();

async function loadFiles() {
    const { files: list } = await fetch('robot/manifest.json').then(r => r.json());
    const files = {};
    await Promise.all(list.map(async f => {
        const r = await fetch('robot/' + f);
        files[f] = f.endsWith('.xml') ? await r.text() : new Uint8Array(await r.arrayBuffer());
    }));
    return files;
}

async function loadPolicy(dir) {
    const [meta, bin] = await Promise.all([
        fetch(`${dir}/policy.json`).then(r => r.json()),
        fetch(`${dir}/policy.bin`).then(r => r.arrayBuffer()),
    ]);
    return new Policy(meta, new Float32Array(bin));
}

let env, pick, thrower, throws, view;
let substep = 0, simTime = 0, last = 0, speed = 1;
let pending = null;      // seconds until the next box, after a landing
let drag = null;         // the box, while you hold it
let awayTime = 0;        // seconds the box has been resting out of reach
const thrown = { count: 0, last: null, best: null };

async function init() {
    try {
        const [mujoco, files, p, t] = await Promise.all([loadMujoco(), loadFiles(), loadPolicy('model/pick'), loadPolicy('model/throw')]);
        env = new PandaEnv(mujoco, files, random);
        pick = p;
        thrower = t;
        throws = new Throws(env);
        env.reset();
        newBox();
        view = new SceneRenderer($('view'), env);
        setupPointer();
        status.hidden = true;
        window.demo = { env, throws, view, advance };   // for poking at it from the console
        requestAnimationFrame(frame);
    } catch (e) {
        status.textContent = `Couldn't start the simulation: ${e.message}`;
        throw e;
    }
}

function frame(now) {
    const dt = Math.min(0.05, (now - (last || now)) / 1000);
    last = now;
    advance(dt * speed);
    view.draw(dt * speed, throws.heading);
    requestAnimationFrame(frame);
}

// Run the simulation forward by `seconds`: physics in 5 ms steps, a policy every 4th (50 Hz)
function advance(seconds) {
    simTime += seconds;
    const sub = env.model.opt.timestep;
    while (simTime >= sub) {
        simTime -= sub;
        if (substep % SUBSTEPS === 0) env.act(throws.action(pick, thrower));
        env.substep();
        substep++;
        if (substep % SUBSTEPS === 0) controlTick();
    }
}

// Bookkeeping once per control step
function controlTick() {
    if (pending !== null) {
        pending -= env.dt;
        if (pending <= 0) { pending = null; view.boxLeaves(); newBox(); }
        return;
    }
    if (drag) return;   // the clock waits while you hold the box
    if (throws.phase === 'pick') {
        const b = env.boxPos();
        const reachable = Math.abs(b[0] - CENTER[0]) < 0.3 && Math.abs(b[1] - CENTER[1]) < 0.3 && b[2] > -0.05;
        awayTime = reachable ? 0 : awayTime + env.dt;
        if (awayTime > 1.5) { note('Out of reach, so here is a new box'); view.boxLeaves(); newBox(); return; }
    }
    const e = throws.update();
    if (!e || e.event === 'handover') return;
    if (e.event === 'landed') {
        thrown.count++;
        thrown.last = e.distance;
        const best = thrown.best === null || e.distance > thrown.best;
        if (best) thrown.best = e.distance;
        view.landed(e.point, e.distance, best);
        if (e.distance < 0.5) note('Dropped it');
        pending = 1.6;   // watch it settle
        showStats();
    } else {
        view.boxLeaves();
        newBox();
    }
}

function newBox() {
    throws.next();
    throws.heading = random.range(-Math.PI, Math.PI);
    awayTime = 0;
}

const fmt = d => d === null ? '–' : `${d.toFixed(2)} m`;
function showStats() {
    $('throws').textContent = thrown.count;
    $('last-throw').textContent = fmt(thrown.last);
    $('best-throw').textContent = fmt(thrown.best);
}

let noteTimer;
function note(text) {
    const el = $('note');
    el.textContent = text;
    el.hidden = false;
    clearTimeout(noteTimer);
    noteTimer = setTimeout(() => { el.hidden = true; }, 2500);
}

// --- Grabbing the box

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const LIFT = 0.12;   // a grabbed box hovers this high above the floor point under the pointer
const UP = new THREE.Vector3(0, 0, 1);

function rayAt(e) {
    const r = view.canvas.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(pointer, view.camera);
    return raycaster.ray;
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const onBox = () => raycaster.intersectObject(view.box, false).length > 0;

function setupPointer() {
    const canvas = view.canvas;
    // The wheel scrolls the page; pinching (ctrl + wheel on trackpads) zooms
    canvas.addEventListener('wheel', e => { if (!e.ctrlKey) e.stopImmediatePropagation(); }, { capture: true });
    // Registered before OrbitControls sees the event, so a grab doesn't also turn the view
    canvas.addEventListener('pointerdown', e => {
        rayAt(e);
        if (!onBox()) return;
        e.stopImmediatePropagation();
        try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic events in tests */ }
        const goal = new THREE.Vector3(...env.boxPos());
        drag = { goal, height: Math.max(goal.z, 0) + LIFT };
        env.pull = pullBox;
        pending = null;
        throws.reset();   // whatever it was doing, it starts over from picking up
        moveBox(raycaster.ray);
        canvas.classList.add('grabbing');
    }, { capture: true });

    canvas.addEventListener('pointermove', e => {
        const ray = rayAt(e);
        if (drag) moveBox(ray);
        else canvas.classList.toggle('can-grab', onBox());
    });

    const release = () => {
        if (!drag) return;
        env.pull = null;
        for (let i = 0; i < 6; i++) env.data.xfrc_applied[6 * env.boxBody + i] = 0;
        throws.reset();
        drag = null;
        canvas.classList.remove('grabbing');
    };
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', release);
}

// The box follows the floor point under the pointer, hovering at a fixed height
function moveBox(ray) {
    const p = new THREE.Vector3();
    if (!ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(UP, new THREE.Vector3(0, 0, drag.height)), p)) return;
    drag.goal.set(clamp(p.x, -0.2, 1.0), clamp(p.y, -0.6, 0.6), drag.height);
}

// A stiff, damped spring from the box's center to the goal, plus its weight.
// The free joint's qvel is linear velocity (world frame), then angular (box frame).
function pullBox() {
    const d = env.data, b = env.boxBody, mass = env.model.body_mass[b], v = env.boxDof;
    const k = 600, c = 2 * Math.sqrt(k), x = env.boxPos();
    for (let i = 0; i < 3; i++) {
        let f = mass * (k * (drag.goal.getComponent(i) - x[i]) - c * d.qvel[v + i]);
        if (i === 2) f += mass * 9.81;
        d.xfrc_applied[6 * b + i] = clamp(f, -25, 25);
    }
    // Calm the spin: a torque against the angular velocity, turned into the world frame
    const R = d.xmat;
    for (let i = 0; i < 3; i++) {
        let w = 0;
        for (let j = 0; j < 3; j++) w += R[9 * b + 3 * i + j] * d.qvel[v + 3 + j];
        d.xfrc_applied[6 * b + 3 + i] = -0.002 * w;
    }
}

// --- Controls

$('new-box').addEventListener('click', () => { if (env) { pending = null; view.boxLeaves(); newBox(); } });
$('slow').addEventListener('change', e => { speed = e.target.checked ? 0.25 : 1; });

init();
