// Speed of engine.wasm in Node (V8, as in Chrome): one second of each stimulus set.
//   node tools/bench.mjs [seconds]
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
const here = new URL('..', import.meta.url).pathname;
const seconds = Number(process.argv[2] || 1);
const { instance } = await WebAssembly.instantiate(readFileSync(here + 'engine.wasm'));
const w = instance.exports;
const bytes = gunzipSync(readFileSync(here + 'model/connectome.gz'));
const p = w.alloc(bytes.length);
new Uint8Array(w.memory.buffer, p, bytes.length).set(bytes);
let t = performance.now();
const n = w.init(p, bytes.length, 1);
console.log(`${n} neurons parsed in ${((performance.now() - t) / 1000).toFixed(2)} s, memory ${(w.memory.buffer.byteLength / 1e6).toFixed(0)} MB`);
const neurons = JSON.parse(readFileSync(here + 'model/neurons.json'));
const runs = [['sugar', 150], ['JO-CE', 200], ['bitter', 150]];
for (const [name, hz] of runs) {
    w.reset();
    for (const i of neurons.sets[name].index) w.set_rate(i, hz);
    let spikes = 0;
    t = performance.now();
    for (let k = 0; k < seconds * 100; k++) spikes += w.run(100);
    const dt = (performance.now() - t) / 1000;
    console.log(`${name} ${hz} Hz: ${(dt / seconds).toFixed(3)} s per simulated s, ${(spikes / seconds).toFixed(0)} spikes/s, MN9 ${neurons.named.MN9}`);
    for (const i of neurons.sets[name].index) w.set_rate(i, 0);
}
// the runaway: LB3b on the fly's right (annotation side), not one of the paper's sets
const lb3b = neurons.cell_type.index.map((k, i) => [neurons.cell_type.names[k], neurons.side.names[neurons.side.index[i]], i])
    .filter(([ty, side]) => ty === 'LB3b' && side === 'right').map(r => r[2]);
w.reset();
for (const i of lb3b) w.set_rate(i, 150);
let spikes = 0;
t = performance.now();
for (let k = 0; k < seconds * 100; k++) spikes += w.run(100);
console.log(`LB3b right (${lb3b.length}) 150 Hz: ${((performance.now() - t) / 1000 / seconds).toFixed(3)} s per simulated s, ${(spikes / seconds).toFixed(0)} spikes/s`);
