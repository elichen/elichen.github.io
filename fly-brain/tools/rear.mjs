// The rear-up as the day uses it: walk and turn, slow to a stand, wait, take off. Counts what each takeoff does.
//   node tools/rear.mjs <full body dir> [trials] [seed]      ESCAPE=1 for the giant fiber's short rear-up; TO='{"angle":30}' and PRE=0.05 try other settings
import fs from 'node:fs'; import loadMujoco from '@mujoco/mujoco';
import { Walker, WasmPolicy } from '../walker.js';
import { Flyer } from '../flyer.js';
import { Life, TAKEOFF, ESCAPE } from '../life.js';
const dir = process.argv[2], N = +(process.argv[3] || 40); let seed = +(process.argv[4] || 1); const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32; };
const mujoco = await loadMujoco(), vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f, new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const m = mujoco.MjModel.from_xml_string(fs.readFileSync(`${dir}/fly.xml`, 'utf8'), vfs), d = new mujoco.MjData(m);
const wasm = await WebAssembly.compile(fs.readFileSync('engine.wasm'));
const load = (p, shape) => { const b = fs.readFileSync(p), e = new WebAssembly.Instance(wasm).exports; return new WasmPolicy(e, new Float32Array(b.buffer, b.byteOffset, b.length / 4), shape); };
const wm = JSON.parse(fs.readFileSync('policy/walk.json')), fm = JSON.parse(fs.readFileSync('policy/fly.json'));
const W = new Walker(mujoco, m, d, wm, load('policy/walk.bin')), F = new Flyer(mujoco, m, d, fm, load('policy/fly.bin', [104, 256, 2, 12]));
const L = new Life(mujoco, m, d, W, F, { ...TAKEOFF, ...JSON.parse(process.env.TO || '{}') }); L.rnd = rnd;
const f = F.freeq, how = { ...(process.env.ESCAPE ? ESCAPE : L.TO) }; if (process.env.PRE) how.pre = +process.env.PRE; if (process.env.TO) Object.assign(how, JSON.parse(process.env.TO));
const reset = h => { mujoco.mj_resetData(m, d); m.opt.timestep = 2e-4; W.setState(wm.qpos0); for (const q of F.wq) d.qpos[q] = m.qpos_spring[q]; d.qpos[f + 3] = Math.cos(h / 2); d.qpos[f + 4] = d.qpos[f + 5] = 0; d.qpos[f + 6] = Math.sin(h / 2); mujoco.mj_forward(m, d); L.startWalking(); };
const wait = T => { let t = 0; while (t < T) t += L.step(); };
const tally = { flew: 0, calledOff: 0, tipped: 0, fellInFlight: 0 }, notes = [];
for (let i = 0; i < N; i++) {
    reset((rnd() - 0.5) * 6.2);
    // as day.js walks: ramp to 1.5-2 cm/s, turn, ramp down at 4 cm/s per s
    let v = 0; const top = 1.5 + 0.5 * rnd(), yaw = (rnd() - 0.5) * 6, T = 0.6 + rnd() * 1.5, stand = rnd() < 0.5 ? 0 : rnd() * 3;
    for (let t = 0; t < T; t += 0.002) { v = Math.min(top, v + 2 * 0.002); L.speed = v; L.yaw = v >= 0.8 ? Math.max(-2 * v, Math.min(2 * v, yaw)) : 0; L.step(); }
    while (v > 0.05) { v -= 4 * 0.002; L.speed = Math.max(0, v); L.yaw = 0; L.step(); }
    L.speed = 0; wait(stand);
    if (L.up() < 0.95) { i--; continue; }
    const a0 = L.aborts; L.takeOff(how);
    let t = 0, minUp = 1; while (L.mode === 'rear' && t < 0.3) { t += L.step(); minUp = Math.min(minUp, L.up()); }
    const rearUp = L.up(), rearPitch = -L.pitch() * 57.3;
    if (L.aborts > a0) { wait(1); if (L.up() < 0.8) { tally.tipped++; notes.push(`${i}: called off, tipped (up ${rearUp.toFixed(2)} pitch ${rearPitch.toFixed(0)} stand ${stand.toFixed(1)})`); } else { tally.calledOff++; notes.push(`${i}: called off (up ${rearUp.toFixed(2)} pitch ${rearPitch.toFixed(0)} stand ${stand.toFixed(1)})`); } continue; }
    t = 0; while (L.mode !== 'fly' && t < 1) t += L.step();
    L.speed = 14; wait(0.6);
    if (d.qpos[f + 2] > 0.3 && L.up() > 0.2 && Number.isFinite(d.qpos[f])) { tally.flew++; notes.push(`${i}: flew (rear up ${rearUp.toFixed(2)} pitch ${rearPitch.toFixed(0)})`); } else { tally.fellInFlight++; notes.push(`${i}: fell after launch (rear up ${rearUp.toFixed(2)} pitch ${rearPitch.toFixed(0)} stand ${stand.toFixed(1)})`); }
}
console.log(JSON.stringify(tally), 'of', N); if (process.env.NOTES) console.log(notes.join('\n'));
