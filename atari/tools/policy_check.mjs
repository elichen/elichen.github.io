// policy.js (tfjs CPU) vs export.py's reference Q values (JAX, same exported weights).
//   node policy_check.mjs <model dir>
import { readFileSync } from 'node:fs';
import * as tf from '@tensorflow/tfjs';
import { Policy } from '../policy.js';

const dir = process.argv[2];
await tf.setBackend('cpu');
const meta = JSON.parse(readFileSync(`${dir}/policy.json`)), bin = readFileSync(`${dir}/policy.bin`);
const policy = new Policy(tf, meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.length));
const test = JSON.parse(readFileSync(`${dir}/test.json`));
let maxd = 0, agree = 0;
for (let i = 0; i < test.states.length; i++) {
  const q = policy.q(new Uint8Array(test.states[i])), qa = q.dataSync(); q.dispose();
  const ref = test.q[i];
  for (let a = 0; a < ref.length; a++) maxd = Math.max(maxd, Math.abs(qa[a] - ref[a]));
  const am = v => v.indexOf(Math.max(...v));
  agree += am([...qa]) === am(ref);
}
const ok = maxd < 1e-3 && agree === test.states.length;
console.log(`${test.states.length} states: max |dQ| ${maxd.toExponential(2)}, argmax agree ${agree}/${test.states.length}`, ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
