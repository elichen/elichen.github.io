// Tests Einstein's relation D·γ = kT in the simulation.
//   D: from the beads' mean squared displacement while they jiggle freely
//   γ: from the drift of one bead pulled by a constant force (γ = F / v)
// node tools/einstein.mjs [--L 220,90] [--free 50000] [--pulls 20] [--force 0.5] [--pulltime 4000] [--T 1] [--out name]
import * as fs from 'node:fs';
import { getDevice } from './gpu.mjs';
import { Sim, initialState, rng } from '../sim.js';

const arg = (name, def) => {
  const i = process.argv.indexOf(`--${name}`);
  return i < 0 ? def : process.argv[i + 1];
};
const L = arg('L', '220,90').split(',').map(Number);
const nb = +arg('beads', 5), a = +arg('a', 4), density = 0.7, dt = 0.005;
const T = +arg('T', 1), freeTime = +arg('free', 50000), pulls = +arg('pulls', 20);
const F = +arg('force', 0.5), pullTime = +arg('pulltime', 4000), settle = +arg('settle', 300);
const seed = +arg('seed', 1), out = arg('out', null);
const every = 50, ringSize = 4096, tau = every * dt;
const mass = density * Math.PI * a * a;

const cols = Math.ceil(Math.sqrt(nb * L[0] / L[1])), rows = Math.ceil(nb / cols);
const beads = Array.from({ length: nb }, (_, k) => ({
  x: ((k % cols) + 0.5) * L[0] / cols, y: (Math.floor(k / cols) + 0.5) * L[1] / rows, a, m: mass,
}));
const device = await getDevice();
const sim = new Sim(device, { L, dt, T, tauT: 1, recordEvery: every, ringSize }, initialState({ L, density, beads, T, seed }));
sim.run(6000);
sim.tauT = 20;
sim.writeParams();
sim.run(Math.round(500 / dt));

// Runs `time` τ and returns each bead's unwrapped positions and velocities, one row per record.
let temps = 0, nTemps = 0;
async function collect(time) {
  const first = sim.steps / every + 1, count = Math.round(time / tau);
  const X = Array.from({ length: nb }, () => new Float64Array(2 * count));
  const V = Array.from({ length: nb }, () => new Float64Array(2 * count));
  for (let got = 0; got < count;) {
    const take = Math.min(ringSize - 8, count - got);
    sim.run(take * every);
    const recs = await sim.readRecords(first + got, first + got + take);
    recs.forEach((rec, r) => rec.forEach((b, k) => {
      X[k][2 * (got + r)] = b.x[0] + b.wrap[0] * L[0];
      X[k][2 * (got + r) + 1] = b.x[1] + b.wrap[1] * L[1];
      V[k][2 * (got + r)] = b.v[0];
      V[k][2 * (got + r) + 1] = b.v[1];
    }));
    got += take;
    for (const x of X) {                 // a bead can't move several σ between records
      for (let i = 1; i < got; i++) {
        if (Math.hypot(x[2 * i] - x[2 * i - 2], x[2 * i + 1] - x[2 * i - 1]) > 3) throw new Error(`bead jumped at record ${i}`);
      }
    }
    const st = await sim.readState();
    if (st.overflow) throw new Error('grid overflow');
    temps += st.T; nTemps++;
  }
  return { X, V, count };
}

const mean = xs => xs.reduce((p, q) => p + q, 0) / xs.length;
const sem = xs => { const m = mean(xs); return Math.sqrt(xs.reduce((p, q) => p + (q - m) ** 2, 0) / (xs.length * (xs.length - 1))); };
function msd(x, lag) {
  let s = 0;
  const n = x.length / 2 - lag;
  for (let i = 0; i < n; i++) s += (x[2 * (i + lag)] - x[2 * i]) ** 2 + (x[2 * (i + lag) + 1] - x[2 * i + 1]) ** 2;
  return s / n;
}

// --- free jiggling
const t0 = performance.now();
const free = await collect(freeTime);
const beadT = free.V.map(v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return 0.5 * mass * s / (v.length / 2); });
const lagTimes = [1, 3, 10, 30, 100, 300, 1000, 2000, 3000];
const Dt = lagTimes.map(t => {
  const per = free.X.map(x => msd(x, Math.round(t / tau)) / (4 * t));
  return { t, D: mean(per), err: sem(per) };
});
// long-time D from the slope of MSD between 300 τ and 3000 τ, per bead
const slopeD = free.X.map(x => {
  const ts = [300, 600, 1000, 1500, 2000, 3000], ms = ts.map(t => msd(x, Math.round(t / tau)));
  const mt = mean(ts), mm = mean(ms);
  let num = 0, den = 0;
  ts.forEach((t, i) => { num += (t - mt) * (ms[i] - mm); den += (t - mt) ** 2; });
  return num / den / 4;
});
const D = mean(slopeD), Derr = sem(slopeD);
console.log(`box ${L.join('×')}  molecules ${sim.n}  beads ${nb} × a=${a} M=${mass.toFixed(1)}  free ${freeTime} τ  (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
console.log(`bead temperature ½M<v²> = ${mean(beadT).toFixed(4)} ± ${sem(beadT).toFixed(4)}   liquid ${(temps / nTemps).toFixed(4)}`);
for (const r of Dt) console.log(`  MSD/4t at ${String(r.t).padStart(5)} τ: ${r.D.toFixed(4)} ± ${r.err.toFixed(4)}`);
console.log(`D (slope of MSD, 300-3000 τ) = ${D.toFixed(4)} ± ${Derr.toFixed(4)}`);

// --- pulling one bead at a time
const rand = rng(seed + 7);
const speeds = [];
for (let r = 0; r < pulls; r++) {
  const b = r % nb, th = 2 * Math.PI * rand();
  const dir = [Math.cos(th), Math.sin(th)];
  sim.pullBead = b;
  sim.pull = [F * dir[0], F * dir[1]];
  sim.writeParams();
  sim.run(Math.round(settle / dt));
  const p = await collect(pullTime);
  const x = p.X[b];
  let st = 0, ss = 0, sts = 0, stt = 0;
  for (let i = 0; i < p.count; i++) {
    const t = i * tau, s = x[2 * i] * dir[0] + x[2 * i + 1] * dir[1];
    st += t; ss += s; sts += t * s; stt += t * t;
  }
  const v = (p.count * sts - st * ss) / (p.count * stt - st * st);
  speeds.push(v);
  sim.pullBead = -1;
  sim.pull = [0, 0];
  sim.writeParams();
  sim.run(Math.round(settle / dt));
  process.stderr.write(`\rpull ${r + 1}/${pulls}: v = ${v.toFixed(4)}   `);
}
process.stderr.write('\n');
const v = mean(speeds), verr = sem(speeds);
const gamma = F / v, gammaErr = gamma * verr / v;
const kT = D * gamma, kTerr = kT * Math.hypot(Derr / D, verr / v);
console.log(`pull F = ${F}: drift v = ${v.toFixed(4)} ± ${verr.toFixed(4)}  →  γ = F/v = ${gamma.toFixed(2)} ± ${gammaErr.toFixed(2)}`);
console.log(`Einstein: D·γ = ${kT.toFixed(3)} ± ${kTerr.toFixed(3)}   (temperature of the liquid: ${(temps / nTemps).toFixed(3)})`);
console.log(`total ${((performance.now() - t0) / 60000).toFixed(1)} min`);
if (out) {
  fs.mkdirSync(new URL('./runs/', import.meta.url), { recursive: true });
  fs.writeFileSync(new URL(`./runs/${out}.json`, import.meta.url), JSON.stringify({
    L, n: sim.n, nb, a, mass, T, F, freeTime, pullTime, beadT, Dt, slopeD, D, Derr, speeds, v, verr, gamma, gammaErr, kT, kTerr,
    liquidT: temps / nTemps,
  }, null, 1));
}
process.exit(0);
