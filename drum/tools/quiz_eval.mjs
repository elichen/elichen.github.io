// How often does the network win the page's quiz? Random rounds of four
// held-out drums from different categories, judged exactly as the page does.
// usage: node drum/tools/quiz_eval.mjs [rounds]
import { readFileSync } from 'node:fs';
import { drum } from '../fem.js';
import { Listener } from '../listener.js';
import { resample } from '../shapes.js';
import { toUnitArea, align, overlap } from '../compare.js';

const dir = new URL('../model/', import.meta.url);
const meta = JSON.parse(readFileSync(new URL('listener.json', dir), 'utf8'));
const bin = readFileSync(new URL(meta.file, dir));
const net = new Listener(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const shapes = JSON.parse(readFileSync(new URL('quiz.json', dir), 'utf8'));
const rounds = +(process.argv[2] || 200);

let seed = 7;
const rand = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);
let wins = 0;
for (let r = 0; r < rounds; r++) {
  const picks = [];
  while (picks.length < 4) {
    const s = shapes[Math.floor(rand() * shapes.length)];
    if (!picks.some((p) => p.cat === s.cat)) picks.push(s);
  }
  const answer = Math.floor(rand() * 4);
  const d = drum(Float64Array.from(picks[answer].outline));
  const h = net.hear(d.lam, 40);
  const top = h.confidence.indexOf(Math.max(...h.confidence));
  const scores = picks.map((p) => {
    const truth = toUnitArea(resample(Float64Array.from(p.outline), 64));
    return overlap(truth, toUnitArea(align(h.outlines[top], truth)));
  });
  if (scores.indexOf(Math.max(...scores)) === answer) wins++;
}
console.log(`network won ${wins} of ${rounds} rounds (${((100 * wins) / rounds).toFixed(1)}%), chance is 25%`);
