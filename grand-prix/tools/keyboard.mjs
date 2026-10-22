// A scripted keyboard driver: it presses left/right when the network would
// steer more than a little, and gas/brake when its pedal is past ±0.3, through
// the same key handling as the page. Counts spins and lap times on every
// circuit, with and without the input assists (traction control, counter-steer).
//
//   node tools/keyboard.mjs [--gain 0.8]
import { readFileSync } from 'node:fs';
import { loadAll, cornering, DT } from '../sim.js';
import { Race } from '../race.js';

const arg = (name, fallback) => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : fallback; };
const base = new URL('../model/', import.meta.url);
const read = async (name, kind) => {
    const buf = readFileSync(new URL(name, base));
    return kind === 'text' ? buf.toString() : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
};
const { tracks, policy } = await loadAll('', read);

const lockAt = (u, gain) => Math.min(1, gain * cornering(u).steer);  // as input.js

function drive(track, gain, assists) {
    const race = new Race(track, policy, 1, { assists });
    race.start();
    let steer = 0, spins = 0;
    for (let k = 0; k < 9000 && !race.you.finish; k++) {
        const you = race.you;
        const a = you.out ? [0, 0] : policy.act(you.observe());
        const dir = a[0] > 0.02 ? 1 : a[0] < -0.02 ? -1 : 0;
        const lock = lockAt(you.state[3], gain), rate = (dir ? 3.5 : 6) * lock;
        steer += Math.max(-rate * DT, Math.min(rate * DT, dir * lock - steer));
        const pedal = a[1] > 0.3 ? 1 : a[1] < -0.3 ? -1 : 0;
        const wasOut = !!you.out;
        race.step({ steer, pedal });
        if (!wasOut && you.out && you.out.reason === 'spun') spins++;
    }
    return { spins, time: race.you.finish };
}

const configs = [
    ['no aids', 0.95, false],
    ['traction control', 0.95, { traction: true }],
    ['counter-steer', 0.95, { counter: true }],
    ['both (the page)', 0.95, true],
];
for (const [label, gain, assists] of configs) {
    let spins = 0, done = 0, total = 0;
    const rows = [];
    for (const t of Object.values(tracks)) {
        const r = drive(t, gain, assists);
        spins += r.spins;
        if (r.time) { done++; total += r.time; }
        rows.push(`${t.name.slice(0, 6)}=${r.spins}`);
    }
    console.log(`${label}: ${spins} spins on 22 tracks, ${done} finished\n   ${rows.join(' ')}`);
}
