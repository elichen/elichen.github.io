// Checks the JS port against laps recorded from the JAX environment
// (model/test.json, written by tools/export.py):
//   1. the observation and the policy's action at the fixed start,
//   2. teacher-forced ticks: from each recorded state, one JS tick with the
//      recorded action must land on the next recorded state; observations
//      every 50 ticks must match,
//   3. closed loop: the JS policy driving the JS plant for a whole lap; it must
//      finish within a few ticks of the JAX lap (float64 here, float32 there).
//
//   node tools/parity.mjs [--model model/] [--tracks Suzuka,Spa]
import { readFileSync } from 'node:fs';
import { Car, observe, integrate, loadAll } from '../sim.js';

const arg = (name, fallback) => {
    const i = process.argv.indexOf(name);
    return i > 0 ? process.argv[i + 1] : fallback;
};
const base = new URL('../' + arg('--model', 'model/'), import.meta.url);
const read = async (name, kind) => {
    const buf = readFileSync(new URL(name, base));
    return kind === 'text' ? buf.toString() : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
};
const { tracks, policy } = await loadAll('', read);
const test = JSON.parse(await read('test.json', 'text'));
const only = arg('--tracks', null)?.split(',');

const maxAbs = (a, b) => Math.max(...Array.from(b, (x, i) => Math.abs(a[i] - x)));
let ok = true;
const check = (name, err, tol) => {
    const pass = err <= tol;
    ok &&= pass;
    if (!pass || process.argv.includes('-v')) console.log(`  ${pass ? 'ok  ' : 'FAIL'} ${name.padEnd(30)} ${err.toExponential(2)} (tol ${tol})`);
};

for (const ep of test.episodes) {
    if (only && !only.includes(ep.track)) continue;
    const track = tracks[ep.track];
    const car = new Car(track);
    check('start state', maxAbs(car.state, ep.ticks[0].state), 1e-4);
    check('start observation', maxAbs(car.observe(), ep.obs0), 1e-4);
    check('action at start', maxAbs(policy.act(ep.obs0), ep.ticks[0].action), 1e-4);

    let stateErr = 0, obsErr = 0, actErr = 0;
    const prev = new Float64Array(2);
    for (let t = 0; t + 1 < ep.ticks.length; t++) {
        const tick = ep.ticks[t];
        if (tick.obs) {
            prev.set(t ? ep.ticks[t - 1].action : [0, 0]);
            const obs = observe(track, Float64Array.from(tick.state), tick.index, prev);
            obsErr = Math.max(obsErr, maxAbs(obs, tick.obs));
            actErr = Math.max(actErr, maxAbs(policy.act(tick.obs), tick.action));
        }
        const state = Float64Array.from(tick.state);
        integrate(state, tick.action[0], tick.action[1]);
        // positions are ~1 km from the origin in float32 there: compare relative
        const next = ep.ticks[t + 1].state;
        stateErr = Math.max(stateErr, ...Array.from(state, (x, i) => Math.abs(x - next[i]) / (1 + Math.abs(next[i]))));
    }
    check('teacher-forced state (rel)', stateErr, 1e-5);
    check('observations', obsErr, 1e-4);
    check('actions', actErr, 1e-4);

    let ticks = 0;
    while (ticks < 8000) {
        const a = policy.act(car.observe());
        car.step(a[0], a[1]);
        ticks++;
        if (car.failed || car.progress >= track.length) break;
    }
    const lap = car.failed ? null : ticks;
    // the same lap, or the same failure at the same tick
    const pass = ep.result === 'lap' ? lap !== null && Math.abs(lap - ep.steps) <= 25 : car.failed === ep.result && Math.abs(ticks - ep.steps) <= 25;
    ok &&= pass;
    console.log(`${pass ? 'ok  ' : 'FAIL'} ${ep.track.padEnd(14)} JAX ${ep.result === 'lap' ? (ep.steps * 0.02).toFixed(2) + ' s' : ep.result}` +
        `  JS ${lap ? (lap * 0.02).toFixed(2) + ' s' : car.failed + ' at ' + car.progress.toFixed(0) + ' m'}` +
        `  (state ${stateErr.toExponential(1)}, obs ${obsErr.toExponential(1)}, act ${actErr.toExponential(1)})`);
}
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
