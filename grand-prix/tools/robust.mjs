// Robustness check on starts the policy never trained from: per circuit, N
// seeded random starts up to 1 m off the line, 90-108% of the start speed and
// up to 30 m further back. Counts laps finished and lap time.
//
//   node tools/robust.mjs [--model model/] [--starts 10]
import { readFileSync } from 'node:fs';
import { Car, loadAll } from '../sim.js';

const arg = (name, fallback) => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : fallback; };
const base = new URL('../' + arg('--model', 'model/'), import.meta.url);
const read = async (name, kind) => {
    const buf = readFileSync(new URL(name, base));
    return kind === 'text' ? buf.toString() : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
};
const { tracks, policy } = await loadAll('', read);
const N = Number(arg('--starts', 10));
let seed = 12345;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);

let done = 0, total = 0, time = 0;
const fails = [];
for (const t of Object.values(tracks)) {
    for (let k = 0; k < N; k++) {
        const car = new Car(t, (rand() * 2 - 1), rand() * 30);
        car.state[3] *= 0.9 + 0.18 * rand();
        const goal = car.progress + t.length;
        let ticks = 0;
        while (ticks < 9000 && !car.failed && car.progress < goal) {
            const a = policy.act(car.observe());
            car.step(a[0], a[1]);
            ticks++;
        }
        total++;
        if (!car.failed && car.progress >= goal) { done++; time += ticks * 0.02; }
        else fails.push(`${t.name}#${k} ${car.failed} at ${(car.progress).toFixed(0)} m`);
    }
}
console.log(`${done}/${total} laps finished from random starts (mean lap ${(time / Math.max(done, 1)).toFixed(2)} s)`);
if (fails.length) console.log('  failed: ' + fails.join(', '));
