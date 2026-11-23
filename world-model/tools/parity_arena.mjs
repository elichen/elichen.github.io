// Verify JS MiniLM + Single Orthogonal Rotation R + BehaviorPolicy (TFJS CPU)
// against exact dequantized PyTorch FP32 reference outputs.
//   node world-model/tools/parity_arena.mjs

import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as tf from '../../atari/tools/node_modules/@tensorflow/tfjs/dist/tf.node.js';
import { MiniLM } from '../minilm.js';
import { ArenaRotationPolicy } from '../behavior-policy.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
await tf.setBackend('cpu');

const lmMeta = JSON.parse(readFileSync(`${root}/model/minilm.json`, 'utf8'));
const lmBin = readFileSync(`${root}/model/minilm.bin`);
const lm = new MiniLM(tf, lmMeta, lmBin.buffer.slice(lmBin.byteOffset, lmBin.byteOffset + lmBin.length));

const polMeta = JSON.parse(readFileSync(`${root}/model/arena_policy.json`, 'utf8'));
const polBin = readFileSync(`${root}/model/arena_policy.bin`);
const pol = new ArenaRotationPolicy(tf, polMeta, polBin.buffer.slice(polBin.byteOffset, polBin.byteOffset + polBin.length));

const ref = JSON.parse(readFileSync(`${root}/parity/arena_ref.json`, 'utf8'));

// 1. Verify EVERY rotation matrix R is truly orthogonal: max |R R^T - I_kT| < 1e-5
let maxOrtho = 0;
for (const nKey of [...pol.nOptions, 'untrained']) {
  maxOrtho = Math.max(maxOrtho, pol.orthoError(nKey));
}

// 2. Verify MiniLM + R + 3-step BehaviorPolicy GRU parity
let maxEmb = 0, maxQ = 0, maxAct = 0;
for (const tc of ref.cases) {
  const { emb, pieces } = await lm.encode(tc.text);
  for (let i = 0; i < emb.length; i++) maxEmb = Math.max(maxEmb, Math.abs(emb[i] - tc.emb[i]));

  const rot = pol.rotateEmbedding(emb, tc.text, pieces, { nKey: tc.nKey });
  for (let i = 0; i < rot.qUnit.length; i++) maxQ = Math.max(maxQ, Math.abs(rot.qUnit[i] - tc.q_unit[i]));

  pol.setActiveLatent(rot.zBeh);
  for (let step = 0; step < tc.obs_seq.length; step++) {
    const act = await pol.step(Float32Array.from(tc.obs_seq[step]), step / 79.0);
    for (let j = 0; j < act.length; j++) {
      maxAct = Math.max(maxAct, Math.abs(act[j] - tc.act_seq[step][j]));
    }
  }
}

const ok = maxOrtho < 1e-5 && maxEmb < 1e-3 && maxQ < 3e-3 && maxAct < 3e-3;
console.log(`Orthogonality (all R, ${pol.kT}x${pol.kV}): max |R R^T - I| = ${maxOrtho.toExponential(2)}`);
console.log(`MiniLM-L6-v2 (${ref.cases.length} prompts):    max |demb|      = ${maxEmb.toExponential(2)}`);
console.log(`Rotated z_beh (${pol.kV}D unit):     max |dq_unit|   = ${maxQ.toExponential(2)}`);
console.log(`BehaviorPolicy (3-step GRU):  max |dact|      = ${maxAct.toExponential(2)}`);
console.log(ok ? 'PARITY_ARENA: PASS' : 'PARITY_ARENA: FAIL');
lm.dispose();
pol.dispose();
process.exit(ok ? 0 : 1);
