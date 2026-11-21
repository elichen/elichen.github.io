#!/usr/bin/env node
// Record controlled grokking experiments using grok.wasm for the article's static figures.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const wasmPath = path.join(__dirname, '..', 'grok.wasm');
const outPath = path.join(__dirname, '..', 'data', 'experiments.json');

const wasmBytes = fs.readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(wasmBytes);
const e = instance.exports;
const mem = () => new Float32Array(e.memory.buffer);
const evalLen = e.get_eval_len();

function runExperiment({ name, op = 0, frac = 0.35, lr = 3e-3, wd = 2.2, seed = 1, maxEpoch = 3000, stepEvery = 50, switchWdAt = null, switchWdTo = null }) {
  e.init_model(op, frac, lr, wd, seed);
  const series = [];
  const sample = (ep) => {
    const ptr = e.evaluate() >> 2;
    const ev = Array.from(mem().subarray(ptr, ptr + evalLen));
    series.push({
      ep,
      trLoss: +ev[0].toFixed(4),
      trAcc: +ev[1].toFixed(4),
      teLoss: +ev[2].toFixed(4),
      teAcc: +ev[3].toFixed(4),
      wnorm: +ev[4].toFixed(2),
      resLoss: +ev[5].toFixed(4),
      excLoss: +ev[6].toFixed(4),
      top5: +ev[7].toFixed(4),
      gini: +ev[8].toFixed(4),
    });
  };
  sample(0);
  for (let ep = stepEvery; ep <= maxEpoch; ep += stepEvery) {
    if (switchWdAt !== null && ep - stepEvery === switchWdAt) {
      e.set_hparams(lr, switchWdTo);
    }
    e.step_n(stepEvery);
    sample(ep);
  }
  const ptr = e.evaluate() >> 2;
  const ev = Array.from(mem().subarray(ptr, ptr + evalLen));
  const top5K = ev.slice(9, 14).map((x) => Math.round(x));
  let top5Mask = 0;
  for (const k of top5K) top5Mask |= (1 << (k - 1));
  const fullMask = (1 << 29) - 1;
  const soloPtr = e.eval_ablated(top5Mask) >> 2;
  const soloTeAcc = +mem()[soloPtr + 1].toFixed(4);
  const mutePtr = e.eval_ablated(fullMask ^ top5Mask) >> 2;
  const muteTeAcc = +mem()[mutePtr + 1].toFixed(4);

  const last = series[series.length - 1];
  console.log(`${name.padEnd(14)} end teAcc=${last.teAcc} teLoss=${last.teLoss} res=${last.resLoss} wnorm=${last.wnorm} top5=${last.top5} k=${JSON.stringify(top5K)} solo=${soloTeAcc} mute=${muteTeAcc}`);
  return { name, op, frac, lr, wd, seed, maxEpoch, top5K, soloTeAcc, muteTeAcc, series };
}

const results = {
  prime: 59,
  dModel: 64,
  nHeads: 4,
  dMlp: 192,
  nParams: e.get_n_params(),
  runs: {
    default: runExperiment({ name: 'default', op: 0, frac: 0.35, wd: 2.2, seed: 1, maxEpoch: 3000 }),
    noDecay: runExperiment({ name: 'noDecay', op: 0, frac: 0.35, wd: 0.0, seed: 1, maxEpoch: 3000 }),
    lowData: runExperiment({ name: 'lowData', op: 0, frac: 0.25, wd: 2.2, seed: 1, maxEpoch: 3000 }),
    delayedDecay: runExperiment({ name: 'delayedDecay', op: 0, frac: 0.35, wd: 0.0, seed: 1, maxEpoch: 3000, switchWdAt: 800, switchWdTo: 2.2 }),
  },
};

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(results));
console.log(`Wrote ${outPath} (${(fs.statSync(outPath).size / 1024).toFixed(1)} KB)`);
