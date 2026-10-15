// Long runs that measure the beads' Brownian motion.
//   node tools/measure.mjs --L 640,400 --beads 5 --a 10 --time 2000 [--every 10] [--T 1] [--out name]
// Prints the beads' temperature (equipartition), mean squared displacement and
// velocity autocorrelation against lag; --out writes tools/runs/<name>.json.
import * as fs from 'node:fs';
import { getDevice } from './gpu.mjs';
import { Sim, initialState } from '../sim.js';

const arg = (name, def) => {
  const i = process.argv.indexOf(`--${name}`);
  return i < 0 ? def : process.argv[i + 1];
};
const L = arg('L', '640,400').split(',').map(Number);
const nb = +arg('beads', 5), a = +arg('a', 10), density = +arg('density', 0.7);
const T = +arg('T', 1), time = +arg('time', 2000), every = +arg('every', 10), dt = +arg('dt', 0.005);
const seed = +arg('seed', 1), tauT = +arg('tauT', 20), mass = +arg('mass', density * Math.PI * a * a);
const out = arg('out', null);

// beads on a grid, as far apart as the box allows
const cols = Math.ceil(Math.sqrt(nb * L[0] / L[1])), rows = Math.ceil(nb / cols);
const beads = Array.from({ length: nb }, (_, k) => ({
  x: ((k % cols) + 0.5) * L[0] / cols, y: (Math.floor(k / cols) + 0.5) * L[1] / rows, a, m: mass,
}));

const device = await getDevice();
const ringSize = 4096;
const sim = new Sim(device, { L, dt, T, tauT: 1, recordEvery: every, ringSize }, initialState({ L, density, beads, T, seed }));
sim.run(6000);
sim.tauT = tauT;
sim.writeParams();
const start = (await sim.readState()).beads[0].step / every;   // first record index after the melt

const records = Math.round(time / (dt * every));
const X = beads.map(() => new Float64Array(2 * records)), V = beads.map(() => new Float64Array(2 * records));
let got = 0, temps = 0, nt = 0;
const t0 = performance.now();
while (got < records) {
  const take = Math.min(ringSize - 8, records - got);
  sim.run(take * every);
  const recs = await sim.readRecords(start + 1 + got, start + 1 + got + take);
  recs.forEach((rec, r) => rec.forEach((b, k) => {
    X[k][2 * (got + r)] = b.x[0] + b.wrap[0] * L[0];
    X[k][2 * (got + r) + 1] = b.x[1] + b.wrap[1] * L[1];
    V[k][2 * (got + r)] = b.v[0];
    V[k][2 * (got + r) + 1] = b.v[1];
  }));
  got += take;
  const st = await sim.readState();
  temps += st.T; nt++;
  if (st.overflow) throw new Error('grid overflow');
  process.stderr.write(`\r${(100 * got / records).toFixed(0)}%  ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
process.stderr.write('\n');

const tau = dt * every;
const lags = [];
for (let l = 1; l < records / 4; l = Math.max(l + 1, Math.round(l * 1.25))) lags.push(l);

function msd(x, lag) {
  let s = 0;
  const n = x.length / 2 - lag;
  for (let i = 0; i < n; i++) s += (x[2 * (i + lag)] - x[2 * i]) ** 2 + (x[2 * (i + lag) + 1] - x[2 * i + 1]) ** 2;
  return s / n;
}
function vacf(v, lag) {
  let s = 0;
  const n = v.length / 2 - lag;
  for (let i = 0; i < n; i++) s += v[2 * (i + lag)] * v[2 * i] + v[2 * (i + lag) + 1] * v[2 * i + 1];
  return s / n;
}
const mean = xs => xs.reduce((p, q) => p + q, 0) / xs.length;
const sem = xs => { const m = mean(xs); return Math.sqrt(xs.reduce((p, q) => p + (q - m) ** 2, 0) / (xs.length * (xs.length - 1) || 1)); };

const beadT = beads.map((b, k) => mass * vacf(V[k], 0) / 2);
const fluidT = temps / nt;
console.log(`box ${L.join('×')}  molecules ${sim.n}  beads ${nb} × a=${a} M=${mass.toFixed(1)}  ${time} τ`);
console.log(`liquid T ${fluidT.toFixed(4)}   bead T (½M<v²>) ${mean(beadT).toFixed(4)} ± ${sem(beadT).toFixed(4)}`);
console.log(`  lag τ     MSD      ±      ballistic   MSD/4t    slope   VACF/VACF0`);
const rows_ = [];
let prev = null;
const c0 = mean(beads.map((b, k) => vacf(V[k], 0)));
for (const l of lags) {
  const per = beads.map((b, k) => msd(X[k], l));
  const m = mean(per), t = l * tau;
  const c = mean(beads.map((b, k) => vacf(V[k], l))) / c0;
  const slope = prev ? Math.log(m / prev.m) / Math.log(t / prev.t) : NaN;
  rows_.push({ t, msd: m, err: sem(per), vacf: c, slope });
  prev = { m, t };
}
for (const r of rows_.filter((r, i) => i % 3 === 0 || i === rows_.length - 1)) {
  console.log(`${r.t.toFixed(2).padStart(8)} ${r.msd.toExponential(3).padStart(10)} ${r.err.toExponential(1).padStart(8)} ` +
    `${(2 * T / mass * r.t * r.t).toExponential(2).padStart(10)} ${(r.msd / (4 * r.t)).toFixed(4).padStart(8)} ` +
    `${r.slope.toFixed(3).padStart(7)} ${r.vacf.toFixed(4).padStart(9)}`);
}
// how random a Perrin-style path sampled every Δ looks: correlation of successive steps
const msdAt = t => {
  const l = Math.round(t / tau);
  return mean(beads.map((b, k) => msd(X[k], l)));
};
const corr = [];
for (const D of [1, 3, 10, 30, 100, 300]) {
  if (2 * D / tau > records / 4) break;
  const m1 = msdAt(D), m2 = msdAt(2 * D);
  corr.push(`Δ=${D}τ: rms ${Math.sqrt(m1).toFixed(2)}σ, step corr ${((m2 - 2 * m1) / (2 * m1)).toFixed(2)}`);
}
console.log(corr.join('\n'));
if (out) {
  fs.mkdirSync(new URL('./runs/', import.meta.url), { recursive: true });
  fs.writeFileSync(new URL(`./runs/${out}.json`, import.meta.url), JSON.stringify({
    L, nb, a, mass, density, T, dt, every, time, n: sim.n, fluidT, beadT, rows: rows_,
  }));
}
process.exit(0);
