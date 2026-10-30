// flight/flyer.js in MuJoCo 3.14 wasm vs flybody in Python (tools/pack_flyer.py).
//   node tools/flight_parity.mjs flight/body <ref dir>
import fs from 'node:fs';
import loadMujoco from '@mujoco/mujoco';
import { Flyer, CONTROL_DT } from '../flight/flyer.js';
import { WalkPolicy, WasmPolicy } from '../walker.js';

const [dir, ref] = process.argv.slice(2);
const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
for (const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f, new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const model = mujoco.MjModel.from_xml_string(fs.readFileSync(`${dir}/fly.xml`, 'utf8'), vfs);
const data = new mujoco.MjData(model);
const meta = JSON.parse(fs.readFileSync(`${dir}/meta.json`));
const b = fs.readFileSync(`${dir}/policy.bin`);
const weights = new Float32Array(b.buffer, b.byteOffset, b.length / 4);
const engine = new WebAssembly.Instance(await WebAssembly.compile(fs.readFileSync(new URL('../engine.wasm', import.meta.url)))).exports;
const policy = new WasmPolicy(engine, weights, [104, 256, 2, 12]);
const f = new Flyer(mujoco, model, data, meta, policy);
const load = (tag, n) => { const x = fs.readFileSync(`${ref}/${tag}_${n}.bin`); return new Float64Array(x.buffer, x.byteOffset, x.length / 8); };
console.log('nq', model.nq, 'nv', model.nv, 'nu', model.nu, 'dt', model.opt.timestep, 'contact off', !!(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT.value));
for (const [tag, speed] of [['straight', 30], ['hover', 0]]) {
    const O = load(tag, 'obs'), A = load(tag, 'action'), Q = load(tag, 'qpos'), V = load(tag, 'qvel');
    const pq = Q.length / 600, pv = V.length / 600;
    f.speed = speed; f.yaw = 0;
    const worst = Object.fromEntries(meta.keys.map(k => [k, 0])); let actErr = 0;
    for (const k of [0, 1, 50, 300]) {
        f.setState(Q.subarray(k * pq, (k + 1) * pq), V.subarray(k * pv, (k + 1) * pv));
        mujoco.mj_forward(model, data);
        f.buf.fill(0); f.nbuf = 0; f.readSensors();
        f.ref = { x: speed * CONTROL_DT * k, y: 0, heading: 0 };
        const n = f.writeObs();
        if (n !== 104) throw new Error('obs ' + n);
        let at = 0;
        const py = O.subarray(k * 104, (k + 1) * 104);
        meta.keys.forEach((name, i) => { for (let j = 0; j < meta.sizes[i]; j++, at++) worst[name] = Math.max(worst[name], Math.abs(f.obs[at] - py[at])); });
        const a = policy.run(Float32Array.from(py));
        for (let j = 0; j < 12; j++) actErr = Math.max(actErr, Math.abs(a[j] - A[k * 12 + j]));
    }
    console.log(tag, 'max |ours - python|:', Object.fromEntries(Object.entries(worst).map(([k, v]) => [k.replace('walker/', ''), +v.toPrecision(2)])), 'policy', actErr.toExponential(1));
}
for (const [speed, yaw, T] of [[30, 0, 2], [10, 0, 2], [0, 0, 2], [30, 5, 2], [30, 10, 1.5], [30, 15, 1]]) {
    f.speed = speed; f.yaw = yaw;
    f.start();
    const t0 = performance.now(); const N = Math.round(T / CONTROL_DT);
    let maxE = 0, sumE = 0, zmin = 9;
    for (let k = 0; k < N; k++) {
        f.step();
        const p = f.position(), { p: want } = f.pose(f.ref.x, f.ref.y, f.ref.heading, f.yaw);
        const e = Math.hypot(p[0] - want[0], p[1] - want[1], p[2] - want[2]);
        maxE = Math.max(maxE, e); sumE += e; zmin = Math.min(zmin, p[2]);
    }
    const s = (performance.now() - t0) / 1000;
    console.log(`speed ${speed} yaw ${yaw}: ${T} s in ${s.toFixed(1)} s (${(T / s).toFixed(2)}x); error mean ${(sumE / N).toFixed(3)} max ${maxE.toFixed(3)} cm; lowest ${zmin.toFixed(2)}`);
}
