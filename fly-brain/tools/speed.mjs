// How fast the day runs, by what the fly does: CPU seconds per simulated second, and the brain's spikes.
//   node tools/speed.mjs <full body dir>       NUMB=1: without the body's sensory neurons
import fs from 'node:fs'; import zlib from 'node:zlib'; import loadMujoco from '@mujoco/mujoco';
import { Walker, WasmPolicy } from '../walker.js'; import { Flyer } from '../flyer.js'; import { Day } from '../day.js'; import { addThings } from '../things.js';
const dir = process.argv[2];
const mujoco = await loadMujoco(), vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f, new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const m = mujoco.MjModel.from_xml_string(addThings(fs.readFileSync(`${dir}/fly.xml`, 'utf8')), vfs), d = new mujoco.MjData(m);
const wasm = await WebAssembly.compile(fs.readFileSync('engine.wasm')), bytes = zlib.gunzipSync(fs.readFileSync('model/banc.gz')), n = JSON.parse(fs.readFileSync('model/neurons.json'));
const load = (p, shape) => { const b = fs.readFileSync(p), e = new WebAssembly.Instance(wasm).exports; return new WasmPolicy(e, new Float32Array(b.buffer, b.byteOffset, b.length / 4), shape); };
const W = new Walker(mujoco, m, d, JSON.parse(fs.readFileSync('policy/walk.json')), load('policy/walk.bin')), F = new Flyer(mujoco, m, d, JSON.parse(fs.readFileSync('policy/fly.json')), load('policy/fly.bin', [104, 256, 2, 12]));
const b = new WebAssembly.Instance(wasm).exports, p = b.alloc(bytes.length); new Uint8Array(b.memory.buffer, p, bytes.length).set(bytes); b.init(p, bytes.length, 1);
const D = new Day(mujoco, m, d, W, F, b, n, 7); if (process.env.NUMB) D.numb = true;
D.newDay();
const cpu = {}, sim = {}, spk = {};
const run = (label, T, each) => { const t0 = D.t; while (D.t < t0 + T) { each?.(); const k = label ?? D.life.mode, t = performance.now(), s = D.step(); cpu[k] = (cpu[k] || 0) + performance.now() - t; sim[k] = (sim[k] || 0) + 1e-3; spk[k] = (spk[k] || 0) + s.length / 2; } };
run('stand', 5, () => { D.goal = null; D.bored = 0; D.hunger = 0; });
D.goal = { x: -12 + 4, y: -16 + 3, why: 'boredom' }; run('walk to a near goal', 3, () => { D.hunger = 0; });
D.goal = { x: 20, y: 10, why: 'boredom' }; run(null, 12, () => { D.hunger = 0; D.bored = 0; });
for (const k in cpu) console.log(k.padEnd(20), (cpu[k] / 1000 / sim[k]).toFixed(2), 's CPU per s;', Math.round(spk[k] / sim[k] / 1000) + 'k spikes/s;', sim[k].toFixed(1), 's');
