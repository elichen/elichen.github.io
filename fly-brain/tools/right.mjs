// Can the fly get back on its feet by itself? Random search for one open-loop move of the legs and wings that turns a fallen fly over,
// after which the walking policy stands it up. No new policy.
//   node tools/right.mjs <full body dir> search <samples> <seed>            prints the best moves found, as JSON lines
//   node tools/right.mjs <full body dir> refine <file of JSON lines> <samples> <seed>     searches around the best
//   node tools/right.mjs <full body dir> test <file with one move> <trials> <seed>
import fs from 'node:fs'; import loadMujoco from '@mujoco/mujoco';
import { Walker, WasmPolicy } from '../walker.js'; import { Flyer } from '../flyer.js'; import { Life } from '../life.js';
const [dir, what, a3, a4, a5] = process.argv.slice(2);
let seed = +(what === 'search' ? a4 : a5) || 1; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32; };
const mujoco = await loadMujoco(), vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f, new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const m = mujoco.MjModel.from_xml_string(fs.readFileSync(`${dir}/fly.xml`, 'utf8'), vfs), d = new mujoco.MjData(m);
const wasm = await WebAssembly.compile(fs.readFileSync('engine.wasm'));
const load = (p, shape) => { const b = fs.readFileSync(p), e = new WebAssembly.Instance(wasm).exports; return new WasmPolicy(e, new Float32Array(b.buffer, b.byteOffset, b.length / 4), shape); };
const wm = JSON.parse(fs.readFileSync('policy/walk.json')), fm = JSON.parse(fs.readFileSync('policy/fly.json'));
const W = new Walker(mujoco, m, d, wm, load('policy/walk.bin')), F = new Flyer(mujoco, m, d, fm, load('policy/fly.bin', [104, 256, 2, 12]));
const L = new Life(mujoco, m, d, W, F), f = F.freeq;
const { fallen, rightingMove, PHASES } = await import('../righting.js');
const move = () => ({ phases: Array.from({ length: PHASES }, () => ({ T: 0.02 + 0.13 * rnd(), near: Array.from({ length: 8 }, rnd), far: Array.from({ length: 8 }, rnd), wingNear: [0, 1, 2].map(() => 2 * rnd() - 1), wingFar: [0, 1, 2].map(() => 2 * rnd() - 1), stickNear: rnd() < 0.5, stickFar: rnd() < 0.5 })) });
const nudge = (M, s) => ({ phases: M.phases.map(p => ({ T: Math.max(0.01, p.T + s * 0.05 * (rnd() - 0.5)), near: p.near.map(x => Math.min(1, Math.max(0, x + s * (rnd() - 0.5)))), far: p.far.map(x => Math.min(1, Math.max(0, x + s * (rnd() - 0.5)))), wingNear: p.wingNear.map(x => Math.min(1, Math.max(-1, x + 2 * s * (rnd() - 0.5)))), wingFar: p.wingFar.map(x => Math.min(1, Math.max(-1, x + 2 * s * (rnd() - 0.5)))), stickNear: rnd() < 0.1 ? !p.stickNear : p.stickNear, stickFar: rnd() < 0.1 ? !p.stickFar : p.stickFar })) });
// one trial: a fallen fly, the move, then the walking policy. Returns how upright it ends (1 = on its feet).
function trial(M, s) {
    fallen(mujoco, m, d, W, F, L, s);
    const up0 = L.up();
    m.opt.timestep = 5e-5; for (const next = rightingMove(L, M); next();) for (let i = 0; i < 4; i++) mujoco.mj_step(m, d);
    if (!Number.isFinite(d.qpos[f])) return { up0, up: -1 };
    if (L.up() < 0.3 && !process.env.WALK_ANYWAY) return { up0, up: L.up() - 1 };          // not turned over: no need to try the walk
    L.startWalking(); L.speed = 0; L.yaw = 0; let t = 0; while (t < 0.6) t += L.step();
    return { up0, up: Number.isFinite(d.qpos[f]) && d.qpos[f + 2] < 0.3 ? L.up() : -1 };
}
const states = n => Array.from({ length: n }, () => ({ roll: (rnd() < 0.5 ? -1 : 1) * (1.2 + 1.94 * rnd()), pitch: 0.5 * (rnd() - 0.5), yaw: 6.28 * rnd() }));
const score = (M, S) => S.reduce((x, s) => x + trial(M, s).up, 0) / S.length;
if (what === 'search') { const S = states(3), best = []; for (let i = 0; i < +a3; i++) { const M = move(), v = score(M, S); if (v > 0.2) { console.log(JSON.stringify({ v, M })); } } }
if (what === 'refine') {
    let pool = fs.readFileSync(a3, 'utf8').trim().split('\n').map(l => JSON.parse(l)).sort((x, y) => y.v - x.v).slice(0, 8), S = states(8);
    pool = pool.map(p => ({ M: p.M, v: score(p.M, S) }));
    for (let i = 0; i < +a4; i++) { const p = pool[Math.floor(rnd() * pool.length)], M = nudge(p.M, 0.3 * (1 - i / +a4) + 0.05), v = score(M, S); pool.push({ M, v }); pool.sort((x, y) => y.v - x.v); pool.length = 8; if (i % 20 === 0) console.error(i, pool.map(p => p.v.toFixed(2)).join(' ')); }
    for (const p of pool) console.log(JSON.stringify(p));
}
if (what === 'test') {
    const M = JSON.parse(fs.readFileSync(a3, 'utf8').trim().split('\n')[0]).M, S = states(+a4); let ok = 0, back = 0, okBack = 0, side = 0, okSide = 0;
    for (const s of S) { const r = trial(M, s), good = r.up > 0.9; if (good) ok++; if (r.up0 < -0.5) { back++; if (good) okBack++; } else { side++; if (good) okSide++; } }
    console.log(`on its feet ${ok} of ${S.length}; from its back ${okBack} of ${back}; from its side ${okSide} of ${side}`);
}
