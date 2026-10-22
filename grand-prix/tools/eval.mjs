// Laps every circuit from the fixed start in the JS simulation: lap times,
// top speed, and how much of the safe-straight time is spent at full pedal.
//
//   node tools/eval.mjs [--model model/] [--laps 3]
import { readFileSync } from 'node:fs';
import { Car, loadAll } from '../sim.js';

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
const laps = Number(arg('--laps', 3));

let first = 0, done = 0, crashes = 0;
for (const track of Object.values(tracks)) {
    const car = new Car(track);
    const times = [];
    let top = 0, full = 0, ticks = 0, last = 0;
    while (times.length < laps && ticks < laps * 9000) {
        const a = policy.act(car.observe());
        car.step(a[0], a[1]);
        ticks++;
        top = Math.max(top, car.state[3]);
        if (a[1] > 0.95) full++;
        if (car.failed) break;
        if (car.progress >= (times.length + 1) * track.length) {
            times.push(car.time - last);
            last = car.time;
        }
    }
    if (times.length) { first += times[0]; done++; }
    if (car.failed) crashes++;
    console.log(`${track.name.padEnd(14)} ${times.map(t => t.toFixed(2)).join('  ').padEnd(8 * laps)}` +
        `${car.failed ? `${car.failed} in lap ${times.length + 1}` : ''}`.padEnd(16) +
        ` top ${(3.6 * top).toFixed(0)} km/h, full pedal ${(100 * full / ticks).toFixed(0)}%`);
}
console.log(`first laps: ${done}/22 in ${first.toFixed(2)} s; ${crashes} crashed within ${laps} laps`);
