// The things on the table are solid. A: put the fly on a flat top. B: send it low and straight at a thing.
//   node tools/things.mjs <full body dir> [trials per thing] [seed]
import fs from 'node:fs'; import zlib from 'node:zlib'; import loadMujoco from '@mujoco/mujoco';
import { Walker, WasmPolicy } from '../walker.js';
import { Flyer } from '../flyer.js';
import { Day, isClear, TABLE } from '../day.js';
import { addThings, THINGS as T, BOUNDS } from '../things.js';
const dir = process.argv[2], N = +(process.argv[3] || 5), seed = +(process.argv[4] || 1);
const mujoco = await loadMujoco(), vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f, new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const m = mujoco.MjModel.from_xml_string(addThings(fs.readFileSync(`${dir}/fly.xml`, 'utf8')), vfs), d = new mujoco.MjData(m);
const wasm = await WebAssembly.compile(fs.readFileSync('engine.wasm')), bytes = zlib.gunzipSync(fs.readFileSync('model/banc.gz')), n = JSON.parse(fs.readFileSync('model/neurons.json'));
const load = (p, shape) => { const b = fs.readFileSync(p), e = new WebAssembly.Instance(wasm).exports; return new WasmPolicy(e, new Float32Array(b.buffer, b.byteOffset, b.length / 4), shape); };
const W = new Walker(mujoco, m, d, JSON.parse(fs.readFileSync('policy/walk.json')), load('policy/walk.bin')), F = new Flyer(mujoco, m, d, JSON.parse(fs.readFileSync('policy/fly.json')), load('policy/fly.bin', [104, 256, 2, 12]));
const b = new WebAssembly.Instance(wasm).exports, p = b.alloc(bytes.length); new Uint8Array(b.memory.buffer, p, bytes.length).set(bytes); b.init(p, bytes.length, seed);
const D = new Day(mujoco, m, d, W, F, b, n, seed * 7919);
const f = F.freeq, at = () => F.position();
// A: the fly stands on a flat top (put there). It must stand, leave by air, and come down on the table.
for (const [name, x, y, h] of [['jar', T.jar.x, T.jar.y, T.jar.h], ['mug', T.mug.x, T.mug.y, T.mug.h], ['plate', T.plate.x, T.plate.y, T.plate.wall.at(-1)[1]]]) {
    const tally = { stood: 0, leftByAir: 0, onTheTable: 0, putBacks: 0 };
    for (let i = 0; i < N; i++) {
        D.hunger = 0; D.bored = 0; D.putBacks = 0; D.reset(x + (D.rnd() - 0.5), y + (D.rnd() - 0.5), D.rnd() * 6.28); D.calm();
        d.qpos[f + 2] += h; mujoco.mj_forward(m, d); D.life.startWalking();
        let t0 = D.t, stood = 0, left = false, down = false;
        while (D.t < t0 + 12 && !down) {
            D.hunger = 0; D.step(); const L = D.life;
            if (L.mode === 'walk' && L.floor > 0 && L.up() > 0.9) stood += 1e-3;
            if (L.mode === 'fly') left = true;
            if (left && L.mode === 'walk' && L.floor === 0 && L.t > 0.5 && L.up() > 0.9) down = true;
        }
        if (stood > 0.9) tally.stood++; if (left) tally.leftByAir++; if (down) tally.onTheTable++; tally.putBacks += D.putBacks || 0;
    }
    console.log('on the', name.padEnd(6), JSON.stringify(tally), 'of', N);
}
// B: a goal behind a thing, the way round it turned off, so the fly flies low into it. It must not go through.
const solidAt = q => [[T.jar.x, T.jar.y, T.jar.r, T.jar.h], [T.mug.x, T.mug.y, T.mug.r, T.mug.h], ...T.apples.map(([x, y, z, r]) => [x, y, r * 0.8, z])].some(([a, c, r, h]) => Math.hypot(q[0] - a, q[1] - c) < r - 0.4 && q[2] < h - 0.4);
for (const [name, x, y] of [['jar', T.jar.x, T.jar.y], ['mug', T.mug.x, T.mug.y], ['bowl', T.bowl.x, T.bowl.y]]) {
    const tally = { flew: 0, wentThrough: 0, pastIt: 0, putBacks: 0 };
    for (let i = 0; i < N; i++) {
        let sx, sy, a; do { a = D.rnd() * 6.28; sx = x + 19 * Math.cos(a); sy = y + 19 * Math.sin(a); } while (!isClear(sx, sy) || Math.abs(sx) > TABLE.x + 8 || Math.abs(sy) > TABLE.y + 8);
        D.hunger = 0; D.bored = 0; D.putBacks = 0; D.reset(sx, sy, a + Math.PI); D.calm();
        D.goal = { x: x - 3 * Math.cos(a), y: y - 3 * Math.sin(a), straight: true, why: 'test' };
        let t0 = D.t, flew = false, through = 0, past = false;
        while (D.t < t0 + 8) {
            D.hunger = 0; D.bored = 0; D.step(); const q = at();
            if (D.life.mode === 'fly') flew = true;
            if (solidAt(q)) through += 1e-3;
            if (flew && (q[0] - x) * Math.cos(a) + (q[1] - y) * Math.sin(a) < -2) past = true;
            if (!D.goal || (flew && D.life.mode === 'walk' && D.life.t > 0.3)) break;
        }
        if (flew) tally.flew++; if (through > 0.02) tally.wentThrough++; if (past) tally.pastIt++; tally.putBacks += D.putBacks || 0;
    }
    console.log('into the', name.padEnd(5), JSON.stringify(tally), 'of', N);
}
