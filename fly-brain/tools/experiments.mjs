// The article's table: Shiu et al.'s experiments run with engine.wasm (the page's engine) in Node.
//   node tools/experiments.mjs [trials]
// Mean firing rate (Hz) of the readout neuron over `trials` runs of 1 s from rest.
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
const here = new URL('..', import.meta.url).pathname;
const trials = Number(process.argv[2] || 20);
const neurons = JSON.parse(readFileSync(here + 'flywire/neurons.json'));
const bytes = gunzipSync(readFileSync(here + 'flywire/connectome.gz'));
const { instance } = await WebAssembly.instantiate(readFileSync(here + 'engine.wasm'));
const w = instance.exports;
const p = w.alloc(bytes.length);
new Uint8Array(w.memory.buffer, p, bytes.length).set(bytes);
w.init(p, bytes.length, 12345);
const named = neurons.named;

function run(sets, hz, silenced = []) {
    const on = sets.flatMap(s => neurons.sets[s].index);
    for (const i of on) w.set_rate(i, hz);
    for (const i of silenced) w.set_silenced(i, true);
    const count = new Float64Array(neurons.count);
    let active = 0;
    for (let t = 0; t < trials; t++) {
        w.reset();
        const seen = new Set();
        for (let k = 0; k < 100; k++) {
            const n = w.run(100);
            const s = new Uint32Array(w.memory.buffer, w.spikes(), 2 * n);
            for (let j = 1; j < s.length; j += 2) { count[s[j]]++; seen.add(s[j]); }
        }
        active += seen.size / trials;
    }
    for (const i of on) w.set_rate(i, 0);
    for (const i of silenced) w.set_silenced(i, false);
    return { rate: i => count[i] / trials, active };
}

const rows = [
    ['Sugar', ['sugar'], 'MN9'],
    ['Sugar + bitter', ['sugar', 'bitter'], 'MN9'],
    ['Sugar + Ir94e', ['sugar', 'Ir94e'], 'MN9'],
    ['Water', ['water'], 'MN9'],
    ['JO-CE', ['JO-CE'], 'aBN1'],
    ['JO-F', ['JO-F'], 'aBN1'],
];
const rates = [50, 100, 150, 200];
console.log(`| Activated | Readout | ${rates.map(r => `${r} Hz`).join(' | ')} |`);
for (const [label, sets, readout] of rows) {
    const cells = rates.map(hz => run(sets, hz).rate(named[readout]).toFixed(0));
    console.log(`| ${label} | ${readout} | ${cells.join(' | ')} |`);
}
const base = run(['sugar'], 100), cut = run(['sugar'], 100, [named.CB0553]);
console.log(`Sugar 100 Hz: MN9 ${base.rate(named.MN9).toFixed(1)} Hz; with CB0553 (right) silenced ${cut.rate(named.MN9).toFixed(1)} Hz`);
for (const [label, sets] of [['Shiu sugar', ['sugar']], ['LB3b right', ['LB3b-right']]]) {
    const r = run(sets, 150);
    let total = 0;
    for (let i = 0; i < neurons.count; i++) total += r.rate(i);
    console.log(`${label} 150 Hz: ${r.active.toFixed(0)} neurons fire, ${total.toFixed(0)} spikes/s, MN9 ${r.rate(named.MN9).toFixed(1)} Hz`);
}
