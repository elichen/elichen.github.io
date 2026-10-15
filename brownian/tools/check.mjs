// Checks the WebGPU simulation against the double-precision reference and
// measures its speed.
//   node tools/check.mjs            correctness (small box) + speed
//   node tools/check.mjs --speed    speed only
import { getDevice } from './gpu.mjs';
import { Sim, initialState } from '../sim.js';
import * as ref from './reference.mjs';

const args = process.argv.slice(2);
const device = await getDevice();
let failed = false;
const check = (name, ok, detail) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}  ${detail}`);
  if (!ok) failed = true;
};

const beadMass = (a, density) => density * Math.PI * a * a;

function maxDiff(a, b, L) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    let d = a[i] - b[i];
    if (L) d -= L[i % 2] * Math.round(d / L[i % 2]);
    m = Math.max(m, Math.abs(d));
  }
  return m;
}

function asRef(st) {
  return {
    n: st.n, pos: Float64Array.from(st.pos), vel: Float64Array.from(st.vel), lambda: st.lambda, p: [...st.p],
    beads: st.beads.map(b => ({ x: [...b.x], v: [...b.v], wrap: [...b.wrap], a: b.a, m: b.m, ext: [0, 0] })),
  };
}

function momentum(st) {
  let px = 0, py = 0;
  for (let i = 0; i < st.n; i++) { px += st.vel[2 * i]; py += st.vel[2 * i + 1]; }
  for (const b of st.beads) { px += b.m * b.v[0]; py += b.m * b.v[1]; }
  return Math.hypot(px, py);
}

async function correctness() {
  const L = [60, 40], density = 0.7;
  const beads = [{ x: 15, y: 20, a: 5 }, { x: 40, y: 18, a: 5 }].map(b => ({ ...b, m: beadMass(b.a, density) }));
  const init = initialState({ L, density, beads, seed: 1 });
  const sim = new Sim(device, { L, dt: 0.005, T: 1, tauT: 0.5, recordEvery: 1, ringSize: 64 }, init);
  sim.run(4000);
  const melted = await sim.readAll();
  check('melt', Math.abs(melted.T - 1) < 0.05 && melted.overflow === 0, `n=${sim.n} T=${melted.T.toFixed(4)} overflow=${melted.overflow}`);
  check('thermostat keeps momentum', momentum(melted) < 5e-3, `|p| ${momentum(melted).toExponential(2)}`);

  // trajectories: GPU vs double precision from the same state, no thermostat
  sim.thermostat = false;
  sim.writeParams();
  const cfg = { L, dt: 0.005, thermostat: false };
  let cpu = asRef(melted);
  for (const [steps, tol] of [[1, 1e-4], [10, 1e-3], [100, 3e-2]]) {
    const before = sim.steps;
    while (sim.steps - before < steps) {
      sim.run(1);
      cpu = ref.step(cpu, cfg);
    }
    const gpu = await sim.readAll();
    const dx = maxDiff(gpu.pos, cpu.pos, L), dv = maxDiff(gpu.vel, cpu.vel);
    const bx = Math.max(...gpu.beads.map((b, k) => maxDiff(b.x, cpu.beads[k].x, L)));
    const bv = Math.max(...gpu.beads.map((b, k) => maxDiff(b.v, cpu.beads[k].v)));
    check(`match after ${sim.steps - before} more steps`, Math.max(dx, dv, bx, bv) < tol,
      `molecules |dx| ${dx.toExponential(2)} |dv| ${dv.toExponential(2)}  beads |dx| ${bx.toExponential(2)} |dv| ${bv.toExponential(2)}`);
    cpu = asRef(gpu);
  }

  // energy and momentum without a thermostat
  const energy = async () => {
    const a = await sim.readAll();
    sim.run(1);
    const b = await sim.readAll();
    let ke = 0;
    for (let i = 0; i < 2 * a.n; i++) ke += 0.25 * (a.vel[i] ** 2 + b.vel[i] ** 2);
    a.beads.forEach((B, k) => {
      for (const d of [0, 1]) ke += 0.25 * B.m * (B.v[d] ** 2 + b.beads[k].v[d] ** 2);
    });
    return { e: (ke + ref.potentialEnergy(a, L)) / a.n, p: momentum(b), T: b.T, overflow: b.overflow };
  };
  const e0 = await energy();
  let worst = 0, last;
  for (let k = 0; k < 10; k++) {
    sim.run(5000);
    last = await energy();
    worst = Math.max(worst, Math.abs(last.e - e0.e));
  }
  check('energy per molecule, 50k steps', worst < 2e-3, `E0 ${e0.e.toFixed(5)}  worst drift ${worst.toExponential(2)}  T ${last.T.toFixed(4)}`);
  check('momentum', last.p < 1e-2, `|p| ${e0.p.toExponential(2)} -> ${last.p.toExponential(2)}`);
  check('no grid overflow', last.overflow === 0, `${last.overflow}`);

  // the ring holds what the state says
  const st = await sim.readState();
  const k = st.beads[0].step;
  const [rec] = await sim.readRecords(k, k + 1);
  check('ring record', rec[0].x[0] === st.beads[0].x[0] && rec[1].v[1] === st.beads[1].v[1], `step ${k}`);
}

async function speed() {
  for (const [Lx, Ly] of (args.includes('--small') ? [[120, 75], [160, 100], [200, 125], [240, 150]] : [[200, 125], [400, 250], [640, 400], [960, 600]])) {
    const L = [Lx, Ly], density = 0.7;
    const beads = [0, 1, 2, 3, 4].map(k => ({ x: (k + 0.5) * Lx / 5, y: Ly / 2, a: 4 }))
      .map(b => ({ ...b, m: beadMass(b.a, density) }));
    const sim = new Sim(device, { L, dt: 0.005, T: 1, tauT: 1 }, initialState({ L, density, beads, seed: 2 }));
    sim.run(200);
    await device.queue.onSubmittedWorkDone();
    const chunk = 100;
    let steps = 0;
    const t0 = performance.now();
    while (performance.now() - t0 < 3000) {
      sim.run(chunk);
      await device.queue.onSubmittedWorkDone();
      steps += chunk;
    }
    const sec = (performance.now() - t0) / 1000;
    const st = await sim.readState();
    console.log(`n=${String(sim.n).padStart(7)}  ${Math.round(steps / sec).toString().padStart(6)} steps/s  ` +
      `${(steps * sim.n / sec / 1e6).toFixed(0)} M molecule-steps/s  T=${st.T.toFixed(3)} overflow=${st.overflow}`);
  }
}

if (!args.includes('--speed')) await correctness();
await speed();
process.exit(failed ? 1 : 0);
