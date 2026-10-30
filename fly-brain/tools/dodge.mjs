// Does the brain dodge? The object is aimed at where the fly will be if it flies straight, so a fly that doesn't turn gets hit.
//   node tools/dodge.mjs [seeds=10]
import fs from 'node:fs';
import zlib from 'node:zlib';
import loadMujoco from '@mujoco/mujoco';
import { Dodger } from '../flight/dodge.js';
import { WasmPolicy } from '../walker.js';

const SEEDS = Number(process.argv[2] ?? 10);
const dir = new URL('../flight/', import.meta.url).pathname;
const mujoco = await loadMujoco();
const files = { 'fly.xml': fs.readFileSync(`${dir}body/fly.xml`, 'utf8') };
for (const f of fs.readdirSync(`${dir}body/meshes`)) files[f] = new Uint8Array(fs.readFileSync(`${dir}body/meshes/${f}`));
const meta = JSON.parse(fs.readFileSync(`${dir}body/meta.json`));
const b = fs.readFileSync(`${dir}body/policy.bin`);
const weights = new Float32Array(b.buffer, b.byteOffset, b.length / 4);
const neurons = JSON.parse(fs.readFileSync(`${dir}model/neurons.json`));
const bytes = zlib.gunzipSync(fs.readFileSync(`${dir}model/banc.gz`));
const wasm = await WebAssembly.compile(fs.readFileSync(new URL('../engine.wasm', import.meta.url)));
const wingCells = {};
for (const t of ['b3', 'i1']) wingCells[t] = neurons.wing[t];

function trial(seed, bearing, mode, dist, speed) {
    const e = new WebAssembly.Instance(wasm).exports;
    const p = e.alloc(bytes.length);
    new Uint8Array(e.memory.buffer, p, bytes.length).set(bytes);
    e.init(p, bytes.length, seed);
    const d = new Dodger(mujoco, files, meta, new WasmPolicy(e, weights, [104, 256, 2, 12]), e, neurons);
    d.steer = mode !== 'none';
    d.turnSign = mode === 'toward' ? -1 : 1;
    d.launch(bearing, dist, speed);
    const T = d.object.T + 0.25;
    const startH = 0, t0 = performance.now();
    let zmin = 9, peak = [0, 0], maxYaw = 0, onset = null;
    const count = new Uint32Array(neurons.count);
    for (let k = 0; k < T / 0.001; k++) {
        const s = d.step();
        zmin = Math.min(zmin, d.flyer.position()[2]);
        peak = [Math.max(peak[0], d.dn.left.rate), Math.max(peak[1], d.dn.right.rate)];
        maxYaw = Math.max(maxYaw, Math.abs(d.turn));
        if (onset === null && d.saccades.length) onset = d.saccades[0].at;
        for (let i = 1; i < s.length; i += 2) count[s[i]]++;
    }
    const wing = {};
    for (const t in wingCells) wing[t] = wingCells[t].left.reduce((a, i) => a + count[i], 0) / T / wingCells[t].left.length + '/' + wingCells[t].right.reduce((a, i) => a + count[i], 0) / T / wingCells[t].right.length;
    const out = { n: d.saccades.length, closest: d.closest, heading: d.flyer.ref.heading * 57.3, zmin, peak, maxYaw, onset, intercept: d.object.T, rate: T / ((performance.now() - t0) / 1000), wing };
    d.data.delete(); d.model.delete();
    return out;
}

for (const [dist, speed] of [[12, 30], [12, 45]]) for (const deg of [30, 45, 75, 110]) {
    console.log(`-- object from ${deg} deg, ${dist} cm away, ${speed} cm/s`);
    for (const mode of ['brain', 'none', 'toward']) {
        const rs = [];
        for (let s = 1; s <= SEEDS; s++) {
            const left = s % 2 === 1;   // both sides
            const r = trial(s, (left ? 1 : -1) * deg / 57.3, mode, dist, speed);
            r.away = (left ? -1 : 1) * r.heading;
            rs.push(r);
        }
        const m = f => rs.reduce((a, r) => a + f(r), 0) / rs.length;
        console.log(`  ${mode.padEnd(7)} hit ${rs.filter(r => r.closest < 0).length}/${rs.length}; clearance ${m(r => r.closest).toFixed(2)} cm (lowest ${Math.min(...rs.map(r => r.closest)).toFixed(2)}); turned away ${m(r => r.away).toFixed(0)} deg; saccades ${m(r => r.n).toFixed(1)}; peak yaw ${m(r => r.maxYaw).toFixed(0)}; first turn at ${m(r => r.onset ?? 0).toFixed(2)} of ${rs[0].intercept.toFixed(2)} s; lowest z ${Math.min(...rs.map(r => r.zmin)).toFixed(2)}; ${m(r => r.rate).toFixed(2)}x`);
    }
}
