// Runs the whole-brain model (engine.wasm) and the walking fly (MuJoCo, forage.js) together, paced to
// real time, and posts every spike and the body's pose to the page.
// Messages in:  {type: 'rate', neurons, hz} | {type: 'silence', neurons, on} | {type: 'reset'} | {type: 'pause', on}
//               {type: 'drop', taste, hz, near} (taste null = no drop; near = at the mouth) | {type: 'wick', pos} | {type: 'walk', hz}
// Messages out: {type: 'progress', loaded, total} | {type: 'ready', n} | {type: 'body', geoms, meshes}
//               {type: 'spikes', spikes, t, speed, active, xpos, xmat, touch, wick, fly, walk, turn}
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.js';
import { Forager, BRAIN_STEPS } from './forage.js';
import { WasmPolicy } from './walker.js';

const BUDGET = 25;          // ms of work per slice, so messages get through
const STEP = 1e-4;          // s, the brain model's dt
const AHEAD = 2.5;          // cm: where a new drop lands, ahead of the fly
let w = null, fly = null;
let paused = false;
let clockStart = 0;         // performance.now() when simulated time was simStart
let simStart = 0;
let samples = [];           // [real ms, simulated s] over the last second, for the speed readout

async function loadFiles() {
    const names = await fetch('walker/manifest.json').then(r => r.json());
    const files = { 'fly.xml': await fetch('walker/fly.xml').then(r => r.text()) };
    await Promise.all(names.map(async n => { files[n] = new Uint8Array(await fetch(`walker/meshes/${n}`).then(r => r.arrayBuffer())); }));
    return files;
}

async function load() {
    const [wasm, res, files, neurons, meta, weights, mujoco] = await Promise.all([
        fetch('engine.wasm').then(r => r.arrayBuffer()).then(b => WebAssembly.instantiate(b)),
        fetch('model/connectome.gz'),
        loadFiles(),
        fetch('model/neurons.json').then(r => r.json()),
        fetch('walker/meta.json').then(r => r.json()),
        fetch('walker/policy.bin').then(r => r.arrayBuffer()),
        loadMujoco(),
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
    const S = neurons.sets;
    const type = t => neurons.cell_type.index.flatMap((x, i) => x === neurons.cell_type.names.indexOf(t) ? [i] : []);
    fly = new Forager(mujoco, files, meta, new WasmPolicy(w, new Float32Array(weights)), w, neurons.motor,
        { sugar: S.sugar.index, bitter: S.bitter.index, water: S.water.index, 'sugar+bitter': [...S.sugar.index, ...S.bitter.index] },
        type('DNp09'), type('DNg97'));
    postMessage({ type: 'body', ...describeBody() });
    postMessage({ type: 'ready', n });
    restartClock();
    tick();
}

// the visible geoms with their meshes, for the page to draw
function describeBody() {
    const m = fly.model;
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
        geoms.push({ g, type: m.geom_type[g], size: Array.from(m.geom_size.slice(3 * g, 3 * g + 3)), mesh, rgba, name: fly.mj.mj_id2name(m, fly.mj.mjtObj.mjOBJ_GEOM.value, g) });
    }
    return { geoms, meshes };
}

function pose() {
    const d = fly.data;
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
        while (now() + BRAIN_STEPS * STEP <= target + 1e-9 && performance.now() - start < BUDGET) {
            const s = fly.step();
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
        const st = fly.state();
        postMessage({ type: 'spikes', spikes, t: now(), speed, active: w.active(), xpos, xmat, touch: st.touch, wick: fly.dropOn ? fly.dropXY : null,
            fly: [st.x, st.y, st.z], walk: st.speed, turn: st.yaw },
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
    if (m.type === 'walk') fly.setWalk(m.hz);
    if (m.type === 'drop') {
        fly.hz = m.hz;
        if (m.taste === fly.taste) {           // only the input rate changed
            if (fly.touch) fly.input(true);
        } else {
            fly.setTaste(m.taste);
            if (m.taste) fly.setDrop(m.near ? fly.mouth() : fly.ahead(AHEAD));
            else fly.setDrop(fly.dropXY, false);
        }
    }
    if (m.type === 'wick' && fly.dropOn) fly.setDrop(m.pos);
    if (m.type === 'reset') {
        w.reset();
        fly.setTaste(null);
        fly.reset();
        samples = [];
        restartClock();
    }
    if (m.type === 'pause') { paused = m.on; samples = []; restartClock(); }
};

load().catch(err => postMessage({ type: 'error', message: String(err) }));
