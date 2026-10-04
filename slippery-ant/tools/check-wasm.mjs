// 1) The wasm learner must match stream-ac.js on the same transitions (f32 vs f64 rounding only).
// 2) Speed of physics + learning with each.
//   node tools/check-wasm.mjs
import loadMujoco from '@mujoco/mujoco';
import { readFileSync } from 'fs';
import { AntEnv } from '../ant-env.js';
import { StreamAC } from '../stream-ac.js';
import { WasmLearner } from '../wasm-learner.js';

const here = new URL('..', import.meta.url);
const mujoco = await loadMujoco();
const env = new AntEnv(mujoco, readFileSync(new URL('ant.xml', here), 'utf8'));
const wasmBytes = readFileSync(new URL('stream-ac.wasm', here));

// --- 1. Equivalence: feed both learners the same transitions, with the wasm learner's actions
const js = new StreamAC(env.obsDim, env.nu);
const wasm = await WasmLearner.create(wasmBytes, js, 7);
let raw = env.reset();
let s = js.normalize(raw);
wasm.resetObs(raw);
let worstDelta = 0, steps = 0;
for (; steps < 300; steps++) {
    const a = Float64Array.from(wasm.act());
    js.act(s);
    js.a.set(a);                      // same action, so the gradients are comparable
    const r = env.step(a);
    const s2 = js.normalize(r.obs);
    const done = r.terminated || r.truncated;
    const dJs = js.learn(s, js.scaleReward(r.reward, done), s2, r.terminated, r.truncated);
    const dW = wasm.learn(r.obs, r.reward, r.terminated, r.truncated);
    worstDelta = Math.max(worstDelta, Math.abs(dJs - dW) / Math.max(1e-3, Math.abs(dJs)));
    s = s2;
    if (done) { raw = env.reset(); s = js.normalize(raw); wasm.resetObs(raw); }
}
const back = wasm.copyOut(new StreamAC(env.obsDim, env.nu));
const relDiff = (x, y) => {
    let num = 0, den = 0;
    for (let i = 0; i < x.length; i++) { num += (x[i] - y[i]) ** 2; den += x[i] ** 2; }
    return Math.sqrt(num / den);
};
console.log(`after ${steps} steps: worst TD-error mismatch ${worstDelta.toExponential(1)}, ` +
    `actor weights ${relDiff(js.actor.w, back.actor.w).toExponential(1)}, critic ${relDiff(js.critic.w, back.critic.w).toExponential(1)}, ` +
    `obs mean ${relDiff(js.obsStats.mean, back.obsStats.mean).toExponential(1)}`);

// --- 2. Speed: physics + act + learn, each learner for 6 s
const bench = (name, stepFn, resetFn) => {
    let raw = env.reset(); resetFn(raw);
    for (let i = 0; i < 2000; i++) { const r = stepFn(); if (r.terminated || r.truncated) resetFn(env.reset()); }
    let n = 0; const t0 = performance.now();
    while (performance.now() - t0 < 6000) {
        const r = stepFn(); n++;
        if (r.terminated || r.truncated) resetFn(env.reset());
    }
    console.log(`${name}: ${Math.round(n / ((performance.now() - t0) / 1000))} steps/s (physics + act + learn)`);
};
{
    const agent = new StreamAC(env.obsDim, env.nu);
    let s;
    bench('JavaScript', () => {
        agent.act(s);
        const r = env.step(agent.a);
        const s2 = agent.normalize(r.obs), done = r.terminated || r.truncated;
        agent.learn(s, agent.scaleReward(r.reward, done), s2, r.terminated, r.truncated);
        s = s2;
        return r;
    }, raw => { s = agent.normalize(raw); });
}
{
    const w = await WasmLearner.create(wasmBytes, new StreamAC(env.obsDim, env.nu), 3);
    bench('WebAssembly', () => {
        const r = env.step(w.act());
        w.learn(r.obs, r.reward, r.terminated, r.truncated);
        return r;
    }, raw => w.resetObs(raw));
}
{
    let n = 0; env.reset(); const a = new Float64Array(env.nu); const t0 = performance.now();
    while (performance.now() - t0 < 3000) { for (let i = 0; i < env.nu; i++) a[i] = Math.random() * 2 - 1; const r = env.step(a); n++; if (r.terminated || r.truncated) env.reset(); }
    console.log(`physics only: ${Math.round(n / ((performance.now() - t0) / 1000))} steps/s`);
}
