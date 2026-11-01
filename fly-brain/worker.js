// Runs the brain + nerve cord model and the fly's day (day.js) together, paced to real time, and posts spikes and the world to the page.
// In: {type: 'swat', side} | {type: 'food', x, y} | {type: 'reset'} | {type: 'pause', on} | {type: 'slow', on}
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.js';
import { Day } from './day.js';
import { Walker, WasmPolicy } from './walker.js';
import { Flyer } from './flyer.js';

const BUDGET = 25, STEP = 1e-4, SLOW = 0.02, BLUR = 12, BLUR_EVERY = 2;
let timeScale = 1, wings = [], ring = null, ringAt = 0, ctrlCount = 0;
let w = null, day = null, paused = false, dirty = false, clockStart = 0, simStart = 0, samples = [], spikeTotal = 0, spikeRate = 0;

const get = async url => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r; };

async function load() {
    const names = await get('body/manifest.json').then(r => r.json());
    const xml = await get('body/fly.xml').then(r => r.text());
    const files = {};
    await Promise.all(names.map(async n => { files[n] = new Uint8Array(await get(`body/meshes/${n}`).then(r => r.arrayBuffer())); }));
    const [bin, res, neurons, wm, fm, wp, fp, mujoco] = await Promise.all([
        get('engine.wasm').then(r => r.arrayBuffer()), get('model/banc.gz'), get('model/neurons.json').then(r => r.json()),
        get('policy/walk.json').then(r => r.json()), get('policy/fly.json').then(r => r.json()),
        get('policy/walk.bin').then(r => r.arrayBuffer()), get('policy/fly.bin').then(r => r.arrayBuffer()), loadMujoco(),
    ]);
    const total = Number(res.headers.get('content-length')) || 28729403;
    let loaded = 0;
    const counted = res.body.pipeThrough(new TransformStream({ transform(c, ctl) { loaded += c.length; postMessage({ type: 'progress', loaded, total }); ctl.enqueue(c); } }));
    const bytes = new Uint8Array(await new Response(counted.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    postMessage({ type: 'building' });
    const module = await WebAssembly.compile(bin);
    const fresh = () => new WebAssembly.Instance(module).exports;
    w = fresh();
    const p = w.alloc(bytes.length);
    new Uint8Array(w.memory.buffer, p, bytes.length).set(bytes);
    const n = w.init(p, bytes.length, (Math.random() * 2 ** 32) >>> 0);
    const vfs = new mujoco.MjVFS();
    for (const [name, b] of Object.entries(files)) vfs.addBuffer(name, b);
    const m = mujoco.MjModel.from_xml_string(xml, vfs), d = new mujoco.MjData(m);
    vfs.delete();
    const walker = new Walker(mujoco, m, d, wm, new WasmPolicy(fresh(), new Float32Array(wp)));
    const flyer = new Flyer(mujoco, m, d, fm, new WasmPolicy(fresh(), new Float32Array(fp), [104, 256, 2, 12]));
    day = new Day(mujoco, m, d, walker, flyer, w, neurons, (Math.random() * 2 ** 32) >>> 0);
    day.newDay();
    const body = n2 => mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY.value, n2), wb = [body('walker/wing_left'), body('walker/wing_right')];
    for (let g = 0; g < m.ngeom; g++) if (m.geom_group[g] <= 2 && wb.includes(m.geom_bodyid[g])) wings.push(g);
    ring = new Float64Array(BLUR * wings.length * 12);
    day.onControl = () => {
        if (++ctrlCount % BLUR_EVERY) return;
        let o = ringAt * wings.length * 12;
        const px = d.geom_xpos, mx = d.geom_xmat;
        for (const g of wings) {
            for (let i = 0; i < 3; i++) ring[o++] = px[3 * g + i];
            for (let i = 0; i < 9; i++) ring[o++] = mx[9 * g + i];
        }
        ringAt = (ringAt + 1) % BLUR;
    };
    postMessage({ type: 'body', ...describe(mujoco, m), wings, blurCopies: BLUR, pattern: fm.pattern });
    postMessage({ type: 'ready', n });
    restart();
    tick();
}

function describe(mujoco, m) {
    const geoms = [], meshes = {};
    for (let g = 0; g < m.ngeom; g++) {
        const name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM.value, g);
        if (m.geom_group[g] > 2 || m.geom_type[g] === 0 || name === 'looming') continue;   // the dodge test's sphere, parked under the table in the model; the page draws its own swatter
        const mat = m.geom_matid[g];
        const rgba = mat >= 0 ? Array.from(m.mat_rgba.slice(4 * mat, 4 * mat + 4)) : Array.from(m.geom_rgba.slice(4 * g, 4 * g + 4));
        const mesh = m.geom_type[g] === 7 ? m.geom_dataid[g] : -1;
        if (mesh >= 0 && !meshes[mesh]) {
            const v0 = m.mesh_vertadr[mesh], nv = m.mesh_vertnum[mesh], f0 = m.mesh_faceadr[mesh], nf = m.mesh_facenum[mesh];
            meshes[mesh] = { vert: Float32Array.from(m.mesh_vert.slice(3 * v0, 3 * (v0 + nv))), face: Uint32Array.from(m.mesh_face.slice(3 * f0, 3 * (f0 + nf))) };
        }
        geoms.push({ g, type: m.geom_type[g], size: Array.from(m.geom_size.slice(3 * g, 3 * g + 3)), mesh, rgba, name });
    }
    return { geoms, meshes };
}

const now = () => day.t;
function restart() { clockStart = performance.now(); simStart = now(); }

// behind real time, go on at once: a timer here would give away about a quarter of the CPU. A message, not a timer, so that clicks still get through.
const soon = new MessageChannel();
soon.port1.onmessage = () => tick();

function tick() {
    let behind = false;
    if (!paused || dirty) {   // a change made while paused still shows
        dirty = false;
        const start = performance.now();
        const target = simStart + timeScale * (start - clockStart) / 1000;
        const out = [];
        while (!paused && now() + 10 * STEP <= target + 1e-9 && performance.now() - start < BUDGET) {
            const s = day.step();
            if (s.length) out.push(s);
        }
        const lag = target - now();
        behind = !paused && lag > 10 * STEP;
        if (lag > 0.25 * timeScale) simStart -= lag;
        const real = performance.now();
        for (const p of out) spikeTotal += p.length / 2;
        samples.push([real, now(), spikeTotal]);
        while (samples.length > 2 && real - samples[0][0] > 1000) samples.shift();
        const [r0, s0, n0] = samples[0];
        if (now() > s0) spikeRate = (spikeTotal - n0) / (now() - s0);   // per simulated second; paused, the last value stays
        const speed = real > r0 ? (now() - s0) / ((real - r0) / 1000) / timeScale : 1;
        let len = 0;
        for (const p of out) len += p.length;
        const spikes = new Uint32Array(len);
        let o = 0;
        for (const p of out) { spikes.set(p, o); o += p.length; }
        const d = day.data, xpos = Float64Array.from(d.geom_xpos), xmat = Float64Array.from(d.geom_xmat), blur = ring.slice(), L = day.life;
        const flapping = L.mode !== 'walk' && L.mode !== 'rear' && L.mode !== 'touch';
        postMessage({ type: 'spikes', spikes, t: now(), speed, spikeRate, xpos, xmat, fly: day.F.position(),
            mode: L.mode, hunger: day.hunger, bored: day.bored, goal: day.goal && { x: day.goal.x, y: day.goal.y, why: day.goal.why }, eating: !!day.eating,
            foods: day.foods.map(f => [f.x, f.y, f.r, f.amount]), object: day.object && { p: day.object.p, v: day.object.v, radius: day.object.radius, hit: day.object.hit },
            rates: { oDN1: day.oDN1.rate, GF: day.GF.rate, MN9: day.mouth.MN9.rate, dnL: day.dnL.rate, dnR: day.dnR.rate },
            counts: { putBacks: day.putBacks || 0, jumps: day.jumps || 0, hits: day.hits || 0 }, blur: timeScale === 1 && flapping ? blur : null, slow: timeScale < 1, flapping, wingHz: day.F.freq }, [spikes.buffer, xpos.buffer, xmat.buffer, blur.buffer]);
    }
    if (behind) soon.port2.postMessage(0); else setTimeout(tick, 8);
}

onmessage = e => {
    const m = e.data;
    if (!w) return;
    dirty = true;
    if (m.type === 'swat' && !day.object) day.swat(m.side);   // one at a time: a second click must not teleport the swatter
    if (m.type === 'food') day.addFood(m.x, m.y);
    if (m.type === 'slow') { timeScale = m.on ? SLOW : 1; samples = []; restart(); }
    if (m.type === 'pause') { paused = m.on; samples = []; restart(); }
    if (m.type === 'reset') { day.newDay(); samples = []; restart(); }
};

load().catch(err => { console.error(err); postMessage({ type: 'error', message: err?.message || String(err) }); });
