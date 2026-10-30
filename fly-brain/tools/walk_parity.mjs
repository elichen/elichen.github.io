// walker.js in MuJoCo 3.14 wasm vs flybody in Python (tools/walk_ref.py).
//   node tools/walk_parity.mjs <model dir with fly.xml> <ref dir>
// 1. Teacher forced: put Python's state in, compare the observation and the policy's action.
// 2. Closed loop: does the port walk, and follow the path?
import fs from 'node:fs';
import path from 'node:path';
import loadMujoco from '@mujoco/mujoco';
import { Walker, WalkPolicy, WasmPolicy, CONTROL_DT } from '../walker.js';

const [dir, ref] = process.argv.slice(2);
const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f, new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
for (const f of []) vfs.addBuffer(f, new Uint8Array(fs.readFileSync(path.join(dir, f))));
const model = mujoco.MjModel.from_xml_string(fs.readFileSync(`${dir}/fly.xml`, 'utf8'), vfs);
const data = new mujoco.MjData(model);
const meta = JSON.parse(fs.readFileSync(`${ref}/meta.json`));
const b = fs.readFileSync(`${ref}/policy.bin`);
const weights = new Float32Array(b.buffer, b.byteOffset, b.length / 4);
const engine = new WebAssembly.Instance(await WebAssembly.compile(fs.readFileSync(new URL('../engine.wasm', import.meta.url)))).exports;
const js = new WalkPolicy(weights), wasm = new WasmPolicy(engine, weights);
const w = new Walker(mujoco, model, data, meta, wasm);
for (const [name, p] of [['js', js], ['wasm', wasm]]) {
    const t = performance.now();
    for (let i = 0; i < 200; i++) p.run(new Float32Array(741).fill(0.1));
    console.log(name, 'policy', ((performance.now() - t) / 200).toFixed(3), 'ms per run (real time allows 2 ms for everything)');
}
const load = (tag, n, m) => { const x = fs.readFileSync(`${ref}/${tag}_${n}.bin`); return [new Float64Array(x.buffer, x.byteOffset, x.length / 8), m]; };
const NAMES = meta.keys; const SIZES = meta.sizes;
console.log('nq', model.nq, 'nv', model.nv, 'na', model.na, 'timestep', model.opt.timestep);

for (const [tag, speed, yaw] of [['straight', 2, 0], ['turn', 2, 2]]) {
    const [O] = load(tag, 'obs'), [A] = load(tag, 'action'), [Q] = load(tag, 'qpos'), [V] = load(tag, 'qvel'), [ACT] = load(tag, 'act');
    const pq = Q.length / 600, pv = V.length / 600, pa = ACT.length / 600;
    w.speed = speed; w.yaw = yaw;
    const worst = Object.fromEntries(NAMES.map(k => [k, 0])); let actErr = 0;
    for (const k of [0, 1, 50, 200, 400]) {
        w.setState(Q.subarray(k * pq, (k + 1) * pq), V.subarray(k * pv, (k + 1) * pv), ACT.subarray(k * pa, (k + 1) * pa));
        mujoco.mj_forward(model, data);
        for (const s in w.buf) w.buf[s].fill(0);
        w.nbuf = 0; w.readSensors();
        // the path at step k, exact
        let x = 0, y = 0, th = 0;
        for (let i = 0; i < k; i++) { x += speed * CONTROL_DT * Math.cos(th); y += speed * CONTROL_DT * Math.sin(th); th += yaw * CONTROL_DT; }
        w.setRef(x, y, th);
        const n = w.writeObs();
        if (n !== 741) throw new Error('obs size ' + n);
        let at = 0;
        const ours = w.obs, py = O.subarray(k * 741, (k + 1) * 741);
        NAMES.forEach((name, i) => { for (let j = 0; j < SIZES[i]; j++, at++) worst[name] = Math.max(worst[name], Math.abs(ours[at] - py[at])); });
        // policy on Python's own observation: must match Python's action
        const a = wasm.run(Float32Array.from(py));
        for (let j = 0; j < 59; j++) actErr = Math.max(actErr, Math.abs(a[j] - A[k * 59 + j]));
    }
    console.log(tag, 'max |ours - python| per observation:', Object.fromEntries(Object.entries(worst).map(([k, v]) => [k.replace('walker/', ''), +v.toPrecision(2)])));
    console.log(tag, 'policy action, same input: max error', actErr.toExponential(2));
    // closed loop from Python's first state
    mujoco.mj_resetData(model, data);
    w.setState(Q.subarray(0, pq), V.subarray(0, pv), ACT.subarray(0, pa));
    data.time = 0; mujoco.mj_forward(model, data);
    for (const s in w.buf) w.buf[s].fill(0);
    w.nbuf = 0; w.setRef(0, 0, 0);
    const t0 = performance.now(), STEPS = 1000; let len = 0, last = [data.qpos[w.freeq], data.qpos[w.freeq + 1]], err = 0, maxErr = 0, zMin = 9, zMax = 0;
    for (let k = 0; k < STEPS; k++) {
        const r = { ...w.ref };
        w.step();
        const px = data.qpos[w.freeq], py = data.qpos[w.freeq + 1], z = data.qpos[w.freeq + 2];
        len += Math.hypot(px - last[0], py - last[1]); last = [px, py];
        const e = Math.hypot(px - w.ref.x, py - w.ref.y); err += e / STEPS; maxErr = Math.max(maxErr, e);
        zMin = Math.min(zMin, z); zMax = Math.max(zMax, z);
    }
    const s = (performance.now() - t0) / 1000;
    console.log(`${tag}: closed loop 2 s of walking took ${s.toFixed(2)} s (${(2 / s).toFixed(2)}x real time); path ${len.toFixed(2)} cm, asked ${speed * 2}; tracking error mean ${err.toFixed(3)} max ${maxErr.toFixed(3)} cm; thorax z ${zMin.toFixed(3)}-${zMax.toFixed(3)}; end (${last[0].toFixed(2)}, ${last[1].toFixed(2)}) path end (${w.ref.x.toFixed(2)}, ${w.ref.y.toFixed(2)})`);
}
