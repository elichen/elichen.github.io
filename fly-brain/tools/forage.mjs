// The foraging loop in Node: the fly starts away from a drop, walks to it, and what happens there.
//   node tools/forage.mjs [seeds=5] [tastes=sugar,bitter,water] [distance cm=3] [P9 Hz=100] [taste Hz=150]
import fs from 'node:fs';
import zlib from 'node:zlib';
import loadMujoco from '@mujoco/mujoco';
import { Forager } from '../forage.js';
import { WasmPolicy, CONTROL_DT } from '../walker.js';

const [SEEDS = '5', which = 'sugar,bitter,water,sugar+bitter', DIST = '3', P9 = '100', HZ = '150'] = process.argv.slice(2);
const dir = new URL('..', import.meta.url).pathname;
const mujoco = await loadMujoco();
const files = { 'fly.xml': fs.readFileSync(`${dir}walker/fly.xml`, 'utf8') };
for (const f of fs.readdirSync(`${dir}walker/meshes`)) files[f] = new Uint8Array(fs.readFileSync(`${dir}walker/meshes/${f}`));
const meta = JSON.parse(fs.readFileSync(`${dir}walker/meta.json`));
const b = fs.readFileSync(`${dir}walker/policy.bin`);
const weights = new Float32Array(b.buffer, b.byteOffset, b.length / 4);
const n = JSON.parse(fs.readFileSync(`${dir}model/neurons.json`));
const cells = t => n.cell_type.index.flatMap((x, i) => x === n.cell_type.names.indexOf(t) ? [i] : []);
const S = n.sets;
const tastes = { sugar: S.sugar.index, bitter: S.bitter.index, water: S.water.index, 'sugar+bitter': [...S.sugar.index, ...S.bitter.index] };
const bytes = zlib.gunzipSync(fs.readFileSync(`${dir}model/connectome.gz`));
const wasm = await WebAssembly.compile(fs.readFileSync(`${dir}engine.wasm`));
const P9set = cells('DNp09'), oDN1 = cells('DNg97');

function trial(taste, seed, angle) {
    const e = new WebAssembly.Instance(wasm).exports;
    const p = e.alloc(bytes.length);
    new Uint8Array(e.memory.buffer, p, bytes.length).set(bytes);
    e.init(p, bytes.length, seed);
    const f = new Forager(mujoco, files, meta, new WasmPolicy(e, weights), e, n.motor, tastes, P9set, oDN1);
    f.hz = Number(HZ);
    f.setTaste(taste);
    const drop = [Number(DIST) * Math.cos(angle), Number(DIST) * Math.sin(angle)];
    f.setDrop(drop);
    f.setWalk(Number(P9));
    const r = { reach: null, stop: null, minRostrum: 9, near: 9, zMin: 9, fell: false, drive: 0, driveTouch: 0, nTouch: 0, mn9: 0 };
    const T = Number(process.env.T ?? 8), t0 = performance.now();
    for (let k = 1; k <= T / CONTROL_DT; k++) {
        f.step();
        const s = f.state(), t = k * CONTROL_DT;
        r.near = Math.min(r.near, Math.hypot(s.x - drop[0], s.y - drop[1]));
        r.zMin = Math.min(r.zMin, s.z);
        if (s.touch) { if (r.reach === null) r.reach = t; r.nTouch++; r.driveTouch += s.drive; r.mn9 += s.mn9; r.minRostrum = Math.min(r.minRostrum, s.rostrum); }
        else if (r.reach === null) r.drive = s.drive;
        if (r.reach !== null && r.stop === null && s.speed === 0) r.stop = t - r.reach;
        if (process.env.TRACE && k % Number(process.env.TRACE) === 0) console.log(t.toFixed(2), s.x.toFixed(2), s.y.toFixed(2), 'z', s.z.toFixed(3), 'speed', s.speed.toFixed(2), 'yaw', s.yaw.toFixed(2), 'oDN1', s.drive.toFixed(1), s.touch ? 'TOUCH' : '', 'rostrum', s.rostrum.toFixed(2), 'MN9', s.mn9.toFixed(0));
    }
    const s = f.state();
    r.end = Math.hypot(s.x - drop[0], s.y - drop[1]);
    r.speedEnd = s.speed;
    r.rate = T / ((performance.now() - t0) / 1000);
    return r;
}

for (const taste of which.split(',')) {
    const all = [];
    for (let s = 1; s <= Number(SEEDS); s++) {
        const angle = process.env.ANGLE ? Number(process.env.ANGLE) : s * 2.4 - 3;       // different bearings, some behind the fly
        const r = trial(taste, s, angle);
        all.push(r);
        console.log(`${taste} seed ${s} bearing ${(angle * 57.3).toFixed(0)}: reach ${r.reach === null ? 'never' : r.reach.toFixed(2) + ' s'}, closest ${r.near.toFixed(2)} cm; oDN1 on the way ${r.drive.toFixed(0)} Hz, touching ${r.nTouch ? (r.driveTouch / r.nTouch).toFixed(0) : '-'} Hz; stopped ${r.stop === null ? 'no' : r.stop.toFixed(2) + ' s after touch'}; rostrum ${r.nTouch ? r.minRostrum.toFixed(2) : '-'}, MN9 ${r.nTouch ? (r.mn9 / r.nTouch).toFixed(0) : '-'} Hz; end ${r.end.toFixed(2)} cm from drop, speed ${r.speedEnd.toFixed(1)}; touching ${(100 * r.nTouch * CONTROL_DT / Number(process.env.T ?? 8)).toFixed(0)}%; lowest thorax ${r.zMin.toFixed(3)}; ${r.rate.toFixed(2)}x real time`);
    }
    const mean = f => all.reduce((a, r) => a + f(r), 0) / all.length;
    const reached = all.filter(r => r.reach !== null);
    const T_ = Number(process.env.T ?? 8);
    console.log(`== ${taste}: reached ${reached.length}/${all.length} in ${mean(r => r.reach ?? 0).toFixed(1)} s; there at the end (<0.3 cm, speed 0) ${all.filter(r => r.end < 0.3 && r.speedEnd === 0).length}/${all.length}; oDN1 touching ${mean(r => r.nTouch ? r.driveTouch / r.nTouch : 0).toFixed(1)} Hz; MN9 touching ${mean(r => r.nTouch ? r.mn9 / r.nTouch : 0).toFixed(0)} Hz; deepest rostrum ${mean(r => r.nTouch ? r.minRostrum : 0).toFixed(2)}; share of time touching after reaching ${(100 * mean(r => r.reach ? r.nTouch * CONTROL_DT / (T_ - r.reach) : 0)).toFixed(0)}%; toppled (thorax < 0.09) ${all.filter(r => r.zMin < 0.09).length}`);
}
