// Run the simulation without a browser and print the bridge as Reid et al. (2015)
// measured it: distance from the junction, width, ants in the bridge, trail saved.
//   node tools/run.mjs --angle 20 --traffic 150 --minutes 30 --seed 1 [--every 60] [--json out.json]
import { Sim } from '../sim.js';
import { writeFileSync } from 'node:fs';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]);
    return acc;
}, []));
const params = args.params ? JSON.parse(args.params) : {};
const sim = new Sim({ angle: +(args.angle ?? 20), traffic: +(args.traffic ?? 150), seed: +(args.seed ?? 1), params });
const minutes = +(args.minutes ?? 30), every = +(args.every ?? 60);
const rows = [];
const t0 = performance.now();
let next = 0, stopped = null, gone = null;
const stopAt = args.stop ? +args.stop * 60 : Infinity;
while (sim.t < minutes * 60) {
    sim.step();
    // optionally stop the traffic and time how long the bridge takes to come apart
    if (sim.t >= stopAt && stopped === null) { stopped = sim.t; sim.traffic = 0; console.log(`# traffic stopped with ${sim.bridge.length} ants in the bridge`); }
    if (stopped !== null && gone === null) {
        const left = sim.bridge.length;
        if (left === 0) { gone = sim.t - stopped; console.log(`# bridge gone ${gone.toFixed(1)} s after the traffic stopped`); }
    }
    if (sim.t >= next) {
        const m = sim.measure();
        rows.push(m);
        if (!args.quiet) {
            console.log(`${(m.t / 60).toFixed(1).padStart(5)} min  d ${m.d.toFixed(2).padStart(5)} cm  width ${m.width.toFixed(2).padStart(5)}  ants ${String(m.ants).padStart(3)}  walkers ${String(m.walkers).padStart(3)}  saved ${m.saved.toFixed(2).padStart(5)} cm  joins ${m.joins} leaves ${m.leaves} falls ${m.falls} arrived ${m.arrivals}`);
        }
        next += every;
    }
}
const secs = (performance.now() - t0) / 1000;
console.log(`# ${minutes} min simulated in ${secs.toFixed(1)} s (${(minutes * 60 / secs).toFixed(0)}x real time)`);
if (args.json) writeFileSync(args.json, JSON.stringify({ args, rows }));
