// How hard can you turn with a key, against how hard the AI turns?
// AI: the most lateral g it uses in each speed band over all 22 laps.
// Key: hold full left at a steady speed through the page's input path
// (input.js key steering, race.js aids) and measure the steady lateral g.
//
//   node tools/turning.mjs
import { readFileSync } from 'node:fs';
import { Car, loadAll, integrate, tractionLimit, counterSteer, cornering, DT } from '../sim.js';

const base = new URL('../model/', import.meta.url);
const read = async (name, kind) => {
    const buf = readFileSync(new URL(name, base));
    return kind === 'text' ? buf.toString() : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
};
const { tracks, policy } = await loadAll('', read);
const BANDS = [60, 90, 120, 150, 180, 210, 240, 270];
const band = kmh => BANDS.findLastIndex(b => kmh >= b);

// lateral acceleration = dv/dt + u r (body frame), 95th percentile per band
const ai = BANDS.map(() => ({ g: [], steer: [] }));
for (const t of Object.values(tracks)) {
    const car = new Car(t);
    let v = 0;
    for (let k = 0; k < 9000 && !car.failed && car.progress < t.length; k++) {
        const a = policy.act(car.observe());
        car.step(a[0], a[1]);
        const ay = (car.state[4] - v) / DT + car.state[3] * car.state[5];
        v = car.state[4];
        const i = band(car.state[3] * 3.6);
        if (i >= 0 && k > 0) { ai[i].g.push(Math.abs(ay) / 9.81); ai[i].steer.push(Math.abs(a[0])); }
    }
}
const p95 = xs => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(0.95 * (s.length - 1))] ?? 0; };

const lockAt = (u, gain) => Math.min(1, gain * cornering(u).steer);  // as input.js

function key(kmh, { gain = 0.95, counter = true } = {}) {
    const s = new Float64Array([0, 0, 0, kmh / 3.6, 0, 0, 0, 0, 0]);
    let steer = 0, g = 0;
    for (let k = 0; k < 200; k++) {
        const lock = lockAt(s[3], gain), rate = 3.5 * lock;
        steer += Math.max(-rate * DT, Math.min(rate * DT, lock - steer));
        const cmd = counter ? counterSteer(s, steer) : steer;
        let pedal = Math.max(-1, Math.min(1, (kmh / 3.6 - s[3]) * 0.5));   // hold the speed
        if (pedal > 0) pedal = Math.min(pedal, tractionLimit(s));
        integrate(s, cmd, pedal);
        if (k > 150) g = Math.abs(s[3] * s[5]) / 9.81;
    }
    return { g, lock: lockAt(kmh / 3.6, gain) };
}

console.log('km/h   AI 95th-pct g (steer)   key g (lock)   key g without counter-steer');
BANDS.forEach((b, i) => {
    const k = key(b + 15), raw = key(b + 15, { counter: false });  // raw: no counter-steer
    console.log(`${String(b).padStart(3)}+   ${p95(ai[i].g).toFixed(2)} (${p95(ai[i].steer).toFixed(2)})        ${k.g.toFixed(2)} (${k.lock.toFixed(2)})    ${raw.g.toFixed(2)}`);
});
