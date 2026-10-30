// Runs the whole-brain model (engine.wasm) and the fly's body (MuJoCo, body.js) together, paced to
// real time, and posts every spike and the body's pose to the page.
// Messages in:  {type: 'rate', neurons, hz} | {type: 'silence', neurons, on} | {type: 'reset'} | {type: 'pause', on}
//               {type: 'drop', taste, hz} (taste null = no drop) | {type: 'wick', pos}
// Messages out: {type: 'progress', loaded, total} | {type: 'ready', n} | {type: 'body', geoms, meshes}
//               {type: 'spikes', spikes, t, speed, active, pose, touch, act, wick}
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.js';
import { FlyBody, Loop, STEP, CHUNK } from './body.js';

const BUDGET = 25;          // ms of work per slice, so messages get through
const DROP = [0.075, 0, -0.07];    // where a new drop is held: touching the labellum at rest and extended (cm)
const SETTLE = 2000;        // body steps from flybody's zero pose to the rest pose
let w = null, body = null, loop = null;
let paused = false;
let clockStart = 0;         // performance.now() when simulated time was simStart
let simStart = 0;
let samples = [];           // [real ms, simulated s] over the last second, for the speed readout
let wick = null;            // drop position, or null when no drop is held out

async function loadBody() {
    const names = await fetch('body/manifest.json').then(r => r.json());
    const files = { 'fly.xml': await fetch('body/fly.xml').then(r => r.text()) };
    await Promise.all(names.map(async n => { files[`meshes/${n}`] = new Uint8Array(await fetch(`body/meshes/${n}`).then(r => r.arrayBuffer())); }));
    return new FlyBody(await loadMujoco(), files);
}

async function load() {
    const [wasm, res, fly, neurons] = await Promise.all([
        fetch('engine.wasm').then(r => r.arrayBuffer()).then(b => WebAssembly.instantiate(b)),
        fetch('model/connectome.gz'),
        loadBody(),
        fetch('model/neurons.json').then(r => r.json()),
    ]);
    const total = Number(res.headers.get('content-length')) || 31341230;
    let loaded = 0;
    const counted = res.body.pipeThrough(new TransformStream({
        transform(chunk, ctl) {
            loaded += chunk.length;
            postMessage({ type: 'progress', loaded, total });
            ctl.enqueue(chunk);
        },
    }));
    const bytes = new Uint8Array(await new Response(counted.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    w = wasm.instance.exports;
    const p = w.alloc(bytes.length);
    new Uint8Array(w.memory.buffer, p, bytes.length).set(bytes);
    const n = w.init(p, bytes.length, (Math.random() * 2 ** 32) >>> 0);
    body = fly;
    const S = neurons.sets;
    loop = new Loop(w, body, neurons.motor, { sugar: S.sugar.index, bitter: S.bitter.index, water: S.water.index, 'sugar+bitter': [...S.sugar.index, ...S.bitter.index] });
    body.step(SETTLE);
    postMessage({ type: 'body', ...describeBody() });
    postMessage({ type: 'ready', n });
    restartClock();
    tick();
}

// the visible geoms with their meshes, for the page to draw
function describeBody() {
    const m = body.model;
    const geoms = [], meshes = {};
    for (let g = 0; g < m.ngeom; g++) {
        if (m.geom_group[g] > 2) continue;
        const mat = m.geom_matid[g];
        const rgba = mat >= 0 ? Array.from(m.mat_rgba.slice(4 * mat, 4 * mat + 4)) : Array.from(m.geom_rgba.slice(4 * g, 4 * g + 4));
        const mesh = m.geom_type[g] === 7 ? m.geom_dataid[g] : -1;
        if (mesh >= 0 && !meshes[mesh]) {
            const v0 = m.mesh_vertadr[mesh], nv = m.mesh_vertnum[mesh], f0 = m.mesh_faceadr[mesh], nf = m.mesh_facenum[mesh];
            meshes[mesh] = { vert: Float32Array.from(m.mesh_vert.slice(3 * v0, 3 * (v0 + nv))), face: Uint32Array.from(m.mesh_face.slice(3 * f0, 3 * (f0 + nf))) };
        }
        geoms.push({ g, type: m.geom_type[g], size: Array.from(m.geom_size.slice(3 * g, 3 * g + 3)), mesh, rgba, name: body.mj.mj_id2name(m, body.mj.mjtObj.mjOBJ_GEOM.value, g) });
    }
    return { geoms, meshes };
}

function pose() {
    const d = body.data;
    return { xpos: Float64Array.from(d.geom_xpos), xmat: Float64Array.from(d.geom_xmat) };
}

function now() { return w.step_count() * STEP; }

function restartClock() {
    clockStart = performance.now();
    simStart = now();
}

function tick() {
    if (!paused) {
        const start = performance.now();
        const target = simStart + (start - clockStart) / 1000;
        const out = [];
        while (now() + CHUNK * STEP <= target + 1e-9 && performance.now() - start < BUDGET) {
            const s = loop.chunk();
            if (s.length) out.push(s);
        }
        // more than 0.25 s behind: the CPU cannot keep up, so run slower than real time
        const lag = target - now();
        if (lag > 0.25) simStart -= lag;
        const real = performance.now();
        samples.push([real, now()]);
        while (samples.length > 2 && real - samples[0][0] > 1000) samples.shift();
        const [r0, s0] = samples[0];
        const speed = real > r0 ? (now() - s0) / ((real - r0) / 1000) : 1;
        const spikes = concat(out);
        const { xpos, xmat } = pose();
        postMessage({ type: 'spikes', spikes, t: now(), speed, active: w.active(), xpos, xmat, touch: loop.touch, act: loop.act, wick },
            [spikes.buffer, xpos.buffer, xmat.buffer]);
    }
    setTimeout(tick, 8);
}

function concat(parts) {
    let len = 0;
    for (const p of parts) len += p.length;
    const all = new Uint32Array(len);
    let o = 0;
    for (const p of parts) { all.set(p, o); o += p.length; }
    return all;
}

onmessage = e => {
    const m = e.data;
    if (!w) return;
    if (m.type === 'rate') for (const i of m.neurons) w.set_rate(i, m.hz);
    if (m.type === 'silence') for (const i of m.neurons) w.set_silenced(i, m.on);
    if (m.type === 'drop') {
        if (m.taste === loop.taste) {          // only the input rate changed
            loop.hz = m.hz;
            if (loop.touch) loop.input(true);
        } else {
            loop.setTaste(m.taste);
            loop.hz = m.hz;
            wick = m.taste ? [...DROP] : null;
            body.setWick(wick || body.away);
        }
    }
    if (m.type === 'wick' && wick) { wick = m.pos; body.setWick(wick); }
    if (m.type === 'reset') {
        w.reset();
        loop.reset();
        body.mj.mj_resetData(body.model, body.data);
        body.setWick(wick || body.away);
        body.step(SETTLE);
        samples = [];
        restartClock();
    }
    if (m.type === 'pause') { paused = m.on; samples = []; restartClock(); }
};

load().catch(err => postMessage({ type: 'error', message: String(err) }));
