// Short flights: a goal 7 to 14 cm away in any direction, also behind the fly. The turn right after takeoff is where flights fail.
//   node tools/hop.mjs <full body dir> [trials] [seed]
import fs from 'node:fs'; import zlib from 'node:zlib'; import loadMujoco from '@mujoco/mujoco';
import { Walker, WasmPolicy } from '../walker.js'; import { Flyer } from '../flyer.js'; import { Day } from '../day.js';
const dir = process.argv[2], N = +(process.argv[3] || 40), seed = +(process.argv[4] || 1);
const mujoco = await loadMujoco(), vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f, new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const m = mujoco.MjModel.from_xml_string(fs.readFileSync(`${dir}/fly.xml`, 'utf8'), vfs), d = new mujoco.MjData(m);   // an empty table
const wasm = await WebAssembly.compile(fs.readFileSync('engine.wasm')), bytes = zlib.gunzipSync(fs.readFileSync('model/banc.gz')), n = JSON.parse(fs.readFileSync('model/neurons.json'));
const load = (p, shape) => { const b = fs.readFileSync(p), e = new WebAssembly.Instance(wasm).exports; return new WasmPolicy(e, new Float32Array(b.buffer, b.byteOffset, b.length / 4), shape); };
const W = new Walker(mujoco, m, d, JSON.parse(fs.readFileSync('policy/walk.json')), load('policy/walk.bin')), F = new Flyer(mujoco, m, d, JSON.parse(fs.readFileSync('policy/fly.json')), load('policy/fly.bin', [104, 256, 2, 12]));
const b = new WebAssembly.Instance(wasm).exports, p = b.alloc(bytes.length); new Uint8Array(b.memory.buffer, p, bytes.length).set(bytes); b.init(p, bytes.length, seed);
const D = new Day(mujoco, m, d, W, F, b, n, seed * 7919);
const tally = { flew: 0, arrived: 0, putBacks: 0, flights: 0 }, bad = [];
for (let i = 0; i < N; i++) {
    const h = D.rnd() * 6.28, a = D.rnd() * 6.28, r = 7 + 7 * D.rnd();
    D.hunger = 0; D.bored = 0; D.putBacks = 0; D.reset(0, 0, h); D.calm();
    D.goal = { x: r * Math.cos(a), y: r * Math.sin(a), straight: true, why: 'test' };
    let t0 = D.t, flew = false, last = 'walk', flights = 0;
    while (D.t < t0 + 20 && D.goal && !D.putBacks) { D.hunger = 0; D.bored = 0; D.step(); if (D.life.mode === 'fly' && last !== 'fly') flights++; last = D.life.mode; if (last === 'fly') flew = true; }
    if (flew) tally.flew++; if (!D.goal && !D.putBacks) tally.arrived++; tally.putBacks += D.putBacks || 0; tally.flights += flights;
    if (D.putBacks) bad.push(`goal ${r.toFixed(0)} cm at ${(Math.atan2(Math.sin(a - h), Math.cos(a - h)) * 57.3).toFixed(0)} deg`);
}
console.log(JSON.stringify(tally), 'of', N, bad.length ? '| put back: ' + bad.join('; ') : '');
