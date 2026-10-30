// The closed loop of body.js in Node: a drop is held at the labellum, then taken away, and the
// proboscis moves as the brain model drives it. The article's numbers.
//   node tools/feed.mjs [seeds=10] [tastes=sugar,bitter,water,sugar+bitter] [hz=150] [hold s=1] [after s=0.5]
//   TRACE=1 prints the joint angles every 50 ms; DROP=x,y,z moves the drop.
import fs from 'node:fs';
import zlib from 'node:zlib';
import path from 'node:path';
import loadMujoco from '@mujoco/mujoco';
import { FlyBody, Loop, STEP, CHUNK } from '../body.js';

const dir = path.dirname(new URL(import.meta.url).pathname) + '/..';
const [SEEDS = '10', which = 'sugar,bitter,water,sugar+bitter', HZ = '150', HOLD = '1', AFTER = '0.5'] = process.argv.slice(2);
// drop centre touching the labellum at rest and extended (cm, fly frame), as in worker.js
const DROP = process.env.DROP ? process.env.DROP.split(',').map(Number) : [0.075, 0, -0.07];
const EXTENDED = -0.3;      // rostrum angle (rad) counted as extended: ~45% of its reach under full drive

const mujoco = await loadMujoco();
const files = { 'fly.xml': fs.readFileSync(`${dir}/body/fly.xml`, 'utf8') };
for (const f of fs.readdirSync(`${dir}/body/meshes`)) files[`meshes/${f}`] = new Uint8Array(fs.readFileSync(`${dir}/body/meshes/${f}`));
const neurons = JSON.parse(fs.readFileSync(`${dir}/model/neurons.json`));
const S = neurons.sets;
const tastes = { sugar: S.sugar.index, bitter: S.bitter.index, water: S.water.index, 'sugar+bitter': [...S.sugar.index, ...S.bitter.index] };
const bytes = zlib.gunzipSync(fs.readFileSync(`${dir}/model/connectome.gz`));
const wasm = await WebAssembly.compile(fs.readFileSync(`${dir}/engine.wasm`));

function trial(taste, seed) {
    const w = new WebAssembly.Instance(wasm).exports;
    const p = w.alloc(bytes.length);
    new Uint8Array(w.memory.buffer, p, bytes.length).set(bytes);
    w.init(p, bytes.length, seed);
    const body = new FlyBody(mujoco, files);
    const loop = new Loop(w, body, neurons.motor, tastes);
    loop.hz = Number(HZ);
    body.step(2000);                 // settle into the rest pose
    body.setWick(DROP);
    loop.setTaste(taste);
    const dt = CHUNK * STEP, hold = Math.round(Number(HOLD) / dt), all = hold + Math.round(Number(AFTER) / dt);
    const r = { extendAt: null, retractAt: null, minRostrum: 0, minHaustellum: 0, maxLabellum: 0, touch: 0, mn9: 0 };
    for (let c = 1; c <= all; c++) {
        if (c === hold + 1) body.setWick(body.away);
        loop.chunk();
        const a = body.angles(), t = c * dt;
        if (c <= hold) {
            r.touch += loop.touch / hold;
            r.mn9 += loop.rates().MN9 / hold;
            r.minRostrum = Math.min(r.minRostrum, a.rostrum);
            r.minHaustellum = Math.min(r.minHaustellum, a.haustellum);
            r.maxLabellum = Math.max(r.maxLabellum, a.labrum_left);
            if (r.extendAt === null && a.rostrum < EXTENDED) r.extendAt = t;
        } else if (r.extendAt !== null && r.retractAt === null && a.rostrum > EXTENDED) r.retractAt = t - Number(HOLD);
        if (process.env.TRACE && c % 10 === 0) console.log(t.toFixed(2), loop.touch ? 'touch' : '     ', a.rostrum.toFixed(2), a.haustellum.toFixed(2), a.labrum_left.toFixed(2));
    }
    return r;
}

const ms = x => (1000 * x).toFixed(0);
const stat = xs => { const m = xs.reduce((s, x) => s + x, 0) / xs.length; return [m, Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, xs.length - 1))]; };
console.log(`drop held ${HOLD} s at ${HZ} Hz, then taken away for ${AFTER} s; ${SEEDS} seeds; extended = rostrum below ${EXTENDED} rad`);
for (const taste of which.split(',')) {
    const rs = [];
    const t0 = performance.now();
    for (let s = 1; s <= Number(SEEDS); s++) rs.push(trial(taste, s));
    const ext = rs.filter(r => r.extendAt !== null);
    const [lat, latSd] = stat(ext.map(r => r.extendAt));
    const ret = ext.filter(r => r.retractAt !== null).map(r => r.retractAt);
    const [rr] = stat(rs.map(r => r.minRostrum)), [hh] = stat(rs.map(r => r.minHaustellum)), [ll] = stat(rs.map(r => r.maxLabellum)), [mn9] = stat(rs.map(r => r.mn9)), [tc] = stat(rs.map(r => r.touch));
    console.log(`${taste.padEnd(13)} extended in ${ext.length}/${rs.length}` +
        (ext.length ? `, after ${ms(lat)} ± ${ms(latSd)} ms; back after removal in ${ret.length ? ms(stat(ret)[0]) + ' ms' : '—'}` : '') +
        `; deepest rostrum ${rr.toFixed(2)}, haustellum ${hh.toFixed(2)}, labellum open ${ll.toFixed(2)} rad; MN9 ${mn9.toFixed(0)} Hz; touching ${(100 * tc).toFixed(0)}%` +
        ` (${(rs.length * (Number(HOLD) + Number(AFTER)) / ((performance.now() - t0) / 1000)).toFixed(1)}x real time)`);
}
