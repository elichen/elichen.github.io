// Verify JS WorldModel (TFJS CPU) against exact quantized PyTorch reference outputs.
//   node tools/parity_model.mjs
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { inflateSync } from 'node:zlib';
import * as tf from '../../atari/tools/node_modules/@tensorflow/tfjs/dist/tf.node.js';
import { WorldModel } from '../model.js';
import { ActionBank } from '../planner.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
await tf.setBackend('cpu');

const meta = JSON.parse(readFileSync(`${root}/model/weights.json`, 'utf8'));
const bin = readFileSync(`${root}/model/weights.bin`);
const wm = new WorldModel(tf, meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.length));

const abMeta = JSON.parse(readFileSync(`${root}/model/abank.json`, 'utf8'));
const abBin = readFileSync(`${root}/model/abank.bin`);
const bank = new ActionBank(abMeta, abBin.buffer.slice(abBin.byteOffset, abBin.byteOffset + abBin.length), wm.actMean, wm.actStd);

const ref = JSON.parse(readFileSync(`${root}/parity/ref.json`, 'utf8'));

// Pure-JS 8-bit RGB PNG decoder using built-in node:zlib (zero subprocesses)
function readPngRgb(path) {
  const buf = readFileSync(path);
  let off = 8, w = 0, h = 0;
  const idats = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') { w = data.readUInt32BE(0); h = data.readUInt32BE(4); }
    else if (type === 'IDAT') idats.push(data);
    else if (type === 'IEND') break;
    off += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idats));
  const stride = w * 3;
  const out = new Uint8Array(h * stride);
  const paeth = (a, b, c) => {
    const p = a + b - c;
    const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
    return (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
  };
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)];
    const rOff = y * (stride + 1) + 1;
    const oOff = y * stride;
    for (let x = 0; x < stride; x++) {
      const v = raw[rOff + x];
      const a = x >= 3 ? out[oOff + x - 3] : 0;
      const b = y > 0 ? out[oOff - stride + x] : 0;
      const c = (x >= 3 && y > 0) ? out[oOff - stride + x - 3] : 0;
      let val = v;
      if (f === 1) val = (v + a) & 255;
      else if (f === 2) val = (v + b) & 255;
      else if (f === 3) val = (v + ((a + b) >> 1)) & 255;
      else if (f === 4) val = (v + paeth(a, b, c)) & 255;
      out[oOff + x] = val;
    }
  }
  return out;
}

let maxZ0 = 0, maxRo = 0, maxWs0 = 0, maxWs5 = 0, maxRollS = 0, maxRollLs = 0, maxRollIh = 0;

for (const tc of ref.neural_cases) {
  const rgb = readPngRgb(`${root}/parity/${tc.png}`);
  const { z0, s0, rawReadout } = await wm.encodeFrame(rgb, 3);
  for (let i = 0; i < 192; i++) maxZ0 = Math.max(maxZ0, Math.abs(z0[i] - tc.z0[i]));
  for (let i = 0; i < 10; i++) maxRo = Math.max(maxRo, Math.abs(rawReadout[i] - tc.raw[i]));

  const ws0T = wm.warmstartZTensor(s0);
  const ws0 = await ws0T.data();
  ws0T.dispose();
  for (let i = 0; i < 128; i++) maxWs0 = Math.max(maxWs0, Math.abs(ws0[i] - tc.z_ws0[i]));

  const aPrevFlat = new Float32Array(tc.a_prev.flat());
  const ws5T = wm.warmstartZTensor(s0, Float32Array.from(tc.s_prev), aPrevFlat);
  const ws5 = await ws5T.data();
  for (let i = 0; i < 128; i++) maxWs5 = Math.max(maxWs5, Math.abs(ws5[i] - tc.z_ws5[i]));

  const candFlat = new Float32Array(tc.cand_acts.flat(2));
  const { s, ls, ih } = wm.rolloutTensors(s0, ws5T, candFlat, 4, 10);
  const [sA, lsA, ihA] = await Promise.all([s.data(), ls.data(), ih.data()]);
  s.dispose(); ls.dispose(); ih.dispose(); ws5T.dispose();

  const refS = tc.roll_s.flat(2), refLs = tc.roll_ls.flat(2), refIh = tc.roll_ih.flat();
  for (let i = 0; i < refS.length; i++) maxRollS = Math.max(maxRollS, Math.abs(sA[i] - refS[i]));
  for (let i = 0; i < refLs.length; i++) maxRollLs = Math.max(maxRollLs, Math.abs(lsA[i] - refLs[i]));
  for (let i = 0; i < refIh.length; i++) maxRollIh = Math.max(maxRollIh, Math.abs(ihA[i] - refIh[i]));
}

// Sanity check action bank retrieval
const sampleCand = bank.retrieve(Float32Array.from(ref.neural_cases[0].z0), Float32Array.from(ref.neural_cases[0].s0), [0.45, 0, 0.02], { RS: 128, T: 10 });
const finiteBank = sampleCand.every(Number.isFinite) && sampleCand.length === 128 * 10 * 5;

// Note: z0 has magnitude ~10-20 after 2048-dim BatchNorm (1e-2 abs -> ~5e-4 relative);
// state/readout positions are in meters (< 5e-4 = 0.05 cm!).
const ok = maxZ0 < 1e-2 && maxRo < 5e-4 && maxWs0 < 5e-4 && maxWs5 < 5e-4 && maxRollS < 5e-4 && maxRollLs < 2e-3 && maxRollIh < 2e-3 && finiteBank;
console.log(`ViT+Proj z0 (5 frames):  max |dz0|   = ${maxZ0.toExponential(2)}`);
console.log(`Readout2 (5 frames):     max |dro|   = ${maxRo.toExponential(2)}`);
console.log(`Warmstart step-0:        max |dws0|  = ${maxWs0.toExponential(2)}`);
console.log(`Warmstart 5-step Kalman: max |dws5|  = ${maxWs5.toExponential(2)}`);
console.log(`Rollout 4x10 state:      max |ds|    = ${maxRollS.toExponential(2)}`);
console.log(`Rollout 4x10 log_sig:    max |dls|   = ${maxRollLs.toExponential(2)}`);
console.log(`Rollout 4x10 ih_logit:   max |dih|   = ${maxRollIh.toExponential(2)}`);
console.log(`ActionBank N=${bank.N}:       finite=${finiteBank}`);
console.log(ok ? 'PARITY_MODEL: PASS' : 'PARITY_MODEL: FAIL');
wm.dispose();
process.exit(ok ? 0 : 1);
