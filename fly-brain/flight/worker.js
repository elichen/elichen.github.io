// Runs the brain + nerve cord model (engine.wasm on BANC) and the flying fly (MuJoCo, dodge.js) together, paced to real
// time, and posts every spike and the body's pose to the page.
// In: {type: 'loom', side} | {type: 'steer', on} | {type: 'reset'} | {type: 'pause', on} | {type: 'slow', on}
// Out: {type: 'progress', loaded, total} | {type: 'ready', n} | {type: 'body', geoms, meshes} | {type: 'spikes', ...}
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.js';
import { Dodger, BRAIN_STEPS } from './dodge.js';
import { WasmPolicy } from '../walker.js';

const BUDGET = 25;       // ms of work per slice, so messages get through
const STEP = 1e-4;       // s, the brain model's dt
const BEARING = 75 / 57.3;
const SLOW = 0.02;       // slow motion: 218 Hz wingbeats at about 4 per second
const BLUR = 12;         // wing poses kept for the blur, about one wingbeat
const BLUR_EVERY = 2;    // control steps (0.2 ms) between them
let timeScale = 1, wings = [], ring = null, ringAt = 0, ctrlCount = 0;
let w = null, d = null, paused = false, clockStart = 0, simStart = 0, samples = [], hit = false, untilGone = 0;

async function load() {
    const names = await fetch('body/manifest.json').then(r => r.json());
    const files = { 'fly.xml': await fetch('body/fly.xml').then(r => r.text()) };
    await Promise.all(names.map(async n => { files[n] = new Uint8Array(await fetch(`body/meshes/${n}`).then(r => r.arrayBuffer())); }));
    const [wasm, res, neurons, meta, weights, mujoco] = await Promise.all([
        fetch('../engine.wasm').then(r => r.arrayBuffer()).then(b => WebAssembly.instantiate(b)),
        fetch('model/banc.gz'),
        fetch('model/neurons.json').then(r => r.json()),
        fetch('body/meta.json').then(r => r.json()),
        fetch('body/policy.bin').then(r => r.arrayBuffer()),
        loadMujoco(),
    ]);
    const total = Number(res.headers.get('content-length')) || 28729403;
    let loaded = 0;
    const counted = res.body.pipeThrough(new TransformStream({
        transform(chunk, ctl) { loaded += chunk.length; postMessage({ type: 'progress', loaded, total }); ctl.enqueue(chunk); },
    }));
    const bytes = new Uint8Array(await new Response(counted.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    w = wasm.instance.exports;
    const p = w.alloc(bytes.length);
    new Uint8Array(w.memory.buffer, p, bytes.length).set(bytes);
    const n = w.init(p, bytes.length, (Math.random() * 2 ** 32) >>> 0);
    d = new Dodger(mujoco, files, meta, new WasmPolicy(w, new Float32Array(weights), [104, 256, 2, 12]), w, neurons);
    // every wing geom's pose, over the last wingbeat: the page draws them all, as a camera with a slow shutter would
    const m = d.model, body = n2 => d.mj.mj_name2id(m, d.mj.mjtObj.mjOBJ_BODY.value, n2);
    const wb = [body('walker/wing_left'), body('walker/wing_right')];
    for (let g = 0; g < m.ngeom; g++) if (m.geom_group[g] <= 2 && wb.includes(m.geom_bodyid[g])) wings.push(g);
    ring = new Float64Array(BLUR * wings.length * 12);
    d.onControl = () => {
        if (++ctrlCount % BLUR_EVERY) return;
        let o = ringAt * wings.length * 12;
        const px = d.data.geom_xpos, mx = d.data.geom_xmat;
        for (const g of wings) {
            for (let i = 0; i < 3; i++) ring[o++] = px[3 * g + i];
            for (let i = 0; i < 9; i++) ring[o++] = mx[9 * g + i];
        }
        ringAt = (ringAt + 1) % BLUR;
    };
    postMessage({ type: 'body', ...describe(), wings, blurCopies: BLUR });
    postMessage({ type: 'ready', n });
    restart();
    tick();
}

function describe() {
    const m = d.model, geoms = [], meshes = {};
    for (let g = 0; g < m.ngeom; g++) {
        if (m.geom_group[g] > 2) continue;
        const mat = m.geom_matid[g];
        const rgba = mat >= 0 ? Array.from(m.mat_rgba.slice(4 * mat, 4 * mat + 4)) : Array.from(m.geom_rgba.slice(4 * g, 4 * g + 4));
        const mesh = m.geom_type[g] === 7 ? m.geom_dataid[g] : -1;
        if (mesh >= 0 && !meshes[mesh]) {
            const v0 = m.mesh_vertadr[mesh], nv = m.mesh_vertnum[mesh], f0 = m.mesh_faceadr[mesh], nf = m.mesh_facenum[mesh];
            meshes[mesh] = { vert: Float32Array.from(m.mesh_vert.slice(3 * v0, 3 * (v0 + nv))), face: Uint32Array.from(m.mesh_face.slice(3 * f0, 3 * (f0 + nf))) };
        }
        geoms.push({ g, type: m.geom_type[g], size: Array.from(m.geom_size.slice(3 * g, 3 * g + 3)), mesh, rgba, name: d.mj.mj_id2name(m, d.mj.mjtObj.mjOBJ_GEOM.value, g) });
    }
    return { geoms, meshes };
}

const now = () => w.step_count() * STEP;
function restart() { clockStart = performance.now(); simStart = now(); }

function tick() {
    if (!paused) {
        const start = performance.now();
        const target = simStart + timeScale * (start - clockStart) / 1000;
        const out = [];
        while (now() + BRAIN_STEPS * STEP <= target + 1e-9 && performance.now() - start < BUDGET) {
            const s = d.step();
            if (s.length) out.push(s);
            if (d.object) {
                if (d.closest < 0) hit = true;
                if (d.t > untilGone) { d.object = null; d.placeObject(); }
            }
        }
        const lag = target - now();
        if (lag > 0.25 * timeScale) simStart -= lag;
        const real = performance.now();
        samples.push([real, now()]);
        while (samples.length > 2 && real - samples[0][0] > 1000) samples.shift();
        const [r0, s0] = samples[0];
        const speed = real > r0 ? (now() - s0) / ((real - r0) / 1000) / timeScale : 1;
        let len = 0;
        for (const p of out) len += p.length;
        const spikes = new Uint32Array(len);
        let o = 0;
        for (const p of out) { spikes.set(p, o); o += p.length; }
        const xpos = Float64Array.from(d.data.geom_xpos), xmat = Float64Array.from(d.data.geom_xmat), blur = ring.slice();
        postMessage({ type: 'spikes', spikes, t: now(), speed, active: w.active(), xpos, xmat, wick: null, fly: d.flyer.position(), heading: d.flyer.ref.heading, turn: d.turn, hit, flying: !!d.object,
            clearance: Number.isFinite(d.closest) ? d.closest : null, dn: [d.dn.left.rate, d.dn.right.rate], blur: timeScale === 1 ? blur : null, slow: timeScale < 1 }, [spikes.buffer, xpos.buffer, xmat.buffer, blur.buffer]);
    }
    setTimeout(tick, 8);
}

onmessage = e => {
    const m = e.data;
    if (!w) return;
    if (m.type === 'loom') {
        hit = false;
        d.launch((m.side === 'left' ? 1 : -1) * BEARING, 12, 30);
        untilGone = d.t + d.object.T + 0.5;
    }
    if (m.type === 'steer') d.steer = m.on;
    if (m.type === 'reset') { w.reset(); const s = d.steer; d.reset(); d.steer = s; hit = false; samples = []; restart(); }
    if (m.type === 'slow') { timeScale = m.on ? SLOW : 1; samples = []; restart(); }
    if (m.type === 'pause') { paused = m.on; samples = []; restart(); }
};

load().catch(err => postMessage({ type: 'error', message: String(err) }));
