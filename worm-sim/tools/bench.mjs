// Runs the wasm engine in Node: speed of the brain and body halves of a control step.
// usage: node tools/bench.mjs [seconds of body time] [agar]  (agar: crawling and food seeking, 100 ms of body per step)
import { readFileSync } from 'node:fs';
const dir = new URL('..', import.meta.url).pathname;
const bytes = (f) => readFileSync(dir + f);
const { instance } = await WebAssembly.instantiate(bytes('engine.wasm'));
const w = instance.exports;
const put = (buf) => {
  const a = new Float64Array(buf.buffer, buf.byteOffset, buf.byteLength / 8);
  const p = w.alloc(a.length);
  new Float64Array(w.memory.buffer, p, a.length).set(a);
  return [p, a.length];
};
const meta = JSON.parse(bytes('model/brain.json'));
const [bp, bn] = put(bytes('model/brain.bin'));
const [yp, yn] = put(bytes('model/body.bin'));
const [wp] = put(bytes('model/wout_offline.bin'));
const [cp] = put(new Float64Array([...meta.input, ...meta.output]));
w.init(bp, bn, yp, yn, wp, cp);
let t = performance.now();
w.warm_up(20);
console.log('warm-up 2 s of brain:', ((performance.now() - t) / 1000).toFixed(2), 's');
const secs = Number(process.argv[2] || 5);
const agar = process.argv[3] === 'agar';
if (agar) w.agar_on(3, -4, 1);
const perSec = agar ? 10 : 30;
let tb = 0, tn = 0;
for (let k = 0; k < secs * perSec; k++) {
  let a = performance.now(); w.think(); tn += performance.now() - a;
  a = performance.now(); w.moving(); tb += performance.now() - a;
}
const x = new Float64Array(w.memory.buffer, w.positions(), 3);
console.log(`${secs} s body time: brain ${(tn / 1000).toFixed(2)} s, body ${(tb / 1000).toFixed(2)} s -> ${(secs / ((tn + tb) / 1000)).toFixed(2)}x real time serial; brain ${(secs / tn * 1000).toFixed(2)}x, body ${(secs / tb * 1000).toFixed(2)}x alone`);
console.log('vertex 0', [...x].map((v) => v.toFixed(4)).join(' '));
