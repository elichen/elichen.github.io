// Flies whole sessions the way the page does (pad, ferry, handoff, then laps until a
// crash or --laps) and reports handoffs, lap times and crashes per course.
//
//   node tools/eval.mjs [--flights 20] [--laps 10] [--course uzh_kidney_v1]
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { Policy } from '../policy.js';
import { Flight } from '../flight.js';

const { values: args } = parseArgs({ options: {
    flights: { type: 'string', default: '20' },
    laps: { type: 'string', default: '10' },
    course: { type: 'string' },
} });
const dir = new URL('../model/', import.meta.url);
const track = JSON.parse(readFileSync(new URL('track.json', dir)));
const meta = JSON.parse(readFileSync(new URL('policy.json', dir)));
const bin = readFileSync(new URL('policy.bin', dir));
const policy = new Policy(meta, new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / 4));

function mulberry32(seed) {
    return () => {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
const REASONS = ['none', 'success', 'ring strike', 'missed ring', 'ground', 'bounds', 'inversion', 'nonfinite'];
const laps = Number(args.laps), flights = Number(args.flights);
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;

for (const data of track.courses.filter(c => !args.course || c.id === args.course)) {
    const lapTimes = [], crashes = {}, handoffs = [], ferryStrikes = [];
    let lapsFlown = 0;
    for (let f = 0; f < flights; f++) {
        const flight = new Flight(track, data, { policy, random: mulberry32(1 + f) });
        let done = false, lapsHere = 0;
        for (let t = 0; t < 60 * 62.5 * 6 && !done; t++) {
            const prev = flight.plant.p.slice();
            const phase = flight.tick();
            if (phase !== 'race' && phase !== 'crashed' && phase !== 'pad' && phase !== 'spool'
                && flight.course.strike(prev, flight.plant.p) >= 0) ferryStrikes.push(phase);
            for (const e of flight.events.splice(0)) {
                if (e.type === 'handoff') handoffs.push(e.time);
                if (e.type === 'lap') { lapTimes.push(e.lapTime); lapsHere++; }
                if (e.type === 'crash') {
                    const why = REASONS[e.reason] + (e.ring >= 0 ? ` ${e.ring}` : '');
                    crashes[why] = (crashes[why] || 0) + 1;
                    done = true;
                }
            }
            if (lapsHere >= laps) done = true;
        }
        lapsFlown += lapsHere;
    }
    const crashCount = Object.values(crashes).reduce((s, x) => s + x, 0);
    console.log(`${data.name.padEnd(10)} handoff ${mean(handoffs).toFixed(1)} s after start · ` +
        `${lapsFlown} laps · lap ${mean(lapTimes).toFixed(2)} s ` +
        `(best ${Math.min(...lapTimes).toFixed(2)}) · ${crashCount} crashes ` +
        `(${(crashCount / Math.max(lapsFlown, 1) * 100).toFixed(1)}% per lap) ${JSON.stringify(crashes)}` +
        (ferryStrikes.length ? ` · FERRY STRIKES ${ferryStrikes.length}` : ''));
}
