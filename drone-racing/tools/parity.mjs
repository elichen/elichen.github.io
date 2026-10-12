// Checks the JS port against episodes recorded from the MJX environment
// (model/test.json, written by tools/export.py):
//   1. the policy's action on the reset observation,
//   2. teacher-forced ticks: from each recorded state, one JS tick with the recorded
//      action and detector draws must land on the next recorded state and frame,
//   3. closed loop: the JS policy flying the JS plant from the recorded reset.
//
//   node tools/parity.mjs
import { readFileSync } from 'node:fs';
import { Plant, Course, Race, SUCCESS, quatToMat } from '../sim.js';
import { Policy } from '../policy.js';

const dir = new URL('../model/', import.meta.url);
const track = JSON.parse(readFileSync(new URL('track.json', dir)));
const test = JSON.parse(readFileSync(new URL('test.json', dir)));
const meta = JSON.parse(readFileSync(new URL('policy.json', dir)));
const bin = readFileSync(new URL('policy.bin', dir));
const policy = new Policy(meta, new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / 4));
const courses = Object.fromEntries(track.courses.map(c => [c.id, c]));

const maxAbs = (a, b) => Math.max(...Array.from(a, (x, i) => Math.abs(x - b[i])));
// Brax returns the Gaussian's mean; the environment clips it to [-1, 1]
const clip = a => a.map(x => Math.min(1, Math.max(-1, x)));
let ok = true;
const check = (name, err, tol) => {
    const pass = err <= tol;
    ok &&= pass;
    console.log(`  ${pass ? 'ok  ' : 'FAIL'} ${name.padEnd(34)} ${err.toExponential(2)} (tol ${tol})`);
};

for (const ep of test.episodes) {
    console.log(ep.course);
    const plant = new Plant(track.plant);
    const course = new Course(courses[ep.course], track.plant);
    let draws = [];
    const race = new Race(track, course, plant, () => draws.shift());
    plant.reset(ep.qpos.slice(0, 3), ep.qpos.slice(3), ep.qvel.slice(0, 3), ep.qvel.slice(3));
    race.start(0);
    check('reset observation', maxAbs(race.observation, ep.obs0), 1e-5);
    check('action on reset observation', maxAbs(policy.action(ep.obs0), clip(ep.ticks[0].action)), 1e-4);

    // Teacher-forced ticks
    let pos = 0, vel = 0, motor = 0, frame = 0, act = 0, events = 0;
    for (let t = 0; t < ep.ticks.length; t++) {
        const ref = ep.ticks[t];
        if (t > 0) {
            const prev = ep.ticks[t - 1];
            plant.p.set(prev.qpos.slice(0, 3)); plant.q.set(prev.qpos.slice(3));
            plant.v.set(prev.qvel.slice(0, 3)); plant.w.set(prev.qvel.slice(3));
            plant.motor.set(prev.motor); plant.rateFilter.set(prev.rate_filter);
            quatToMat(plant.q, plant.R);
        }
        act = Math.max(act, maxAbs(policy.action(race.observation), clip(ref.action)));
        draws = ref.draws.slice();
        race.tick(ref.action);
        pos = Math.max(pos, maxAbs(plant.p, ref.qpos.slice(0, 3)) + maxAbs(plant.q, ref.qpos.slice(3)));
        vel = Math.max(vel, maxAbs(plant.v, ref.qvel.slice(0, 3)) + maxAbs(plant.w, ref.qvel.slice(3)));
        motor = Math.max(motor, maxAbs(plant.motor, ref.motor));
        frame = Math.max(frame, maxAbs(race.observer.frame, ref.frame));
        if (race.active !== ref.active || race.lap !== ref.lap || race.terminal !== ref.terminal) events++;
    }
    check('tick: position + quaternion', pos, 1e-5);
    check('tick: velocity + body rates', vel, 1e-4);
    check('tick: motor state', motor, 1e-4);
    check('tick: observation frame', frame, 1e-4);
    check('tick: policy action', act, 1e-3);
    check('tick: gate/lap/terminal mismatches', events, 0);

    // Closed loop from the recorded reset (detector draws from the recording while they last)
    plant.reset(ep.qpos.slice(0, 3), ep.qpos.slice(3), ep.qvel.slice(0, 3), ep.qvel.slice(3));
    const recorded = ep.ticks.flatMap(t => t.draws);
    let k = 0;
    race.random = () => (k < recorded.length ? recorded[k++] : Math.random());
    race.start(0);
    let drift = 0, ticks = 0;
    for (; ticks < 1800 && !race.terminal; ticks++) {
        race.tick(policy.action(race.observation));
        if (ticks < 100) drift = Math.max(drift, maxAbs(plant.p, ep.ticks[ticks].qpos.slice(0, 3)));
    }
    const ref = test.summary.find(s => s.course === ep.course && s.seed === ep.seed);
    console.log(`  closed loop: terminal ${race.terminal} after ${ticks} ticks, ${race.passes} passes ` +
        `(MJX: terminal ${ref.terminal} after ${ref.ticks}, ${ref.passes} passes); ` +
        `position drift over 1.6 s ${drift.toExponential(2)} m`);
    ok &&= race.terminal === SUCCESS;
}
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
