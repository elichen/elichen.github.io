// Throw mode on MuJoCo's WebAssembly build, exactly as the page runs it.
//
//   node tools/eval.mjs [--pick model/pick] [--throw model/throw] [--throws 400] [--seeds 1,2,3,4]
//
// Each seed is one continuous run: a box drops, the pick policy lifts it, the throw
// policy throws it in a random direction, and after it lands the next box drops.
// Reports how often the handover and the landing happen and how far throws go
// (distance from the base along the aim direction, where the box first lands).
import loadMujoco from '@mujoco/mujoco';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import { PandaEnv, Throws, makeRandom } from '../panda-env.js';
import { Policy } from '../policy.js';

const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : dflt; };
const throwsWanted = +opt('throws', 400);
const seeds = opt('seeds', '1,2,3,4').split(',').map(Number);
const root = new URL('..', import.meta.url).pathname;

function loadPolicy(dir) {
    dir = resolve(root, dir);
    const meta = JSON.parse(readFileSync(`${dir}/policy.json`));
    const buf = readFileSync(`${dir}/policy.bin`);
    const p = new Policy(meta, new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4));
    try {
        const test = JSON.parse(readFileSync(`${dir}/test.json`));
        let worst = 0;
        test.obs.forEach((o, i) => p.action(o).forEach((a, j) => { worst = Math.max(worst, Math.abs(a - test.act[i][j])); }));
        console.log(`${dir}: policy.js vs Brax max |action difference| ${worst.toExponential(2)}`);
    } catch { /* no test vectors */ }
    return p;
}
const pick = loadPolicy(opt('pick', 'model/pick'));
const thrower = loadPolicy(opt('throw', 'model/throw'));

const manifest = JSON.parse(readFileSync(`${root}robot/manifest.json`));
const files = {};
for (const f of manifest.files) files[f] = f.endsWith('.xml') ? readFileSync(`${root}robot/${f}`, 'utf8') : readFileSync(`${root}robot/${f}`);
const mujoco = await loadMujoco();

const results = [];
for (const seed of seeds) {
    const random = makeRandom(seed);
    const env = new PandaEnv(mujoco, files, random);
    const loop = new Throws(env);
    env.reset();
    loop.next();
    loop.heading = random.range(-Math.PI, Math.PI);
    while (results.filter(r => r.seed === seed).length < throwsWanted / seeds.length) {
        env.step(loop.action(pick, thrower));
        const e = loop.update();
        if (!e || e.event === 'handover') continue;
        results.push({ seed, phase: loop.phase, heading: loop.heading, ...e });
        loop.next();
        loop.heading = random.range(-Math.PI, Math.PI);
    }
}

const n = results.length, landed = results.filter(r => r.event === 'landed');
const pickMiss = results.filter(r => r.event === 'missed' && r.phase === 'pick').length;
const throwMiss = results.filter(r => r.event === 'missed' && r.phase === 'throw').length;
const d = landed.map(r => r.distance).sort((a, b) => a - b);
const q = p => d.length ? d[Math.floor(p * (d.length - 1))].toFixed(2) : '-';
const mean = d.reduce((a, b) => a + b, 0) / d.length;
const sd = Math.sqrt(d.reduce((a, b) => a + (b - mean) ** 2, 0) / (d.length - 1));
console.log(`${n} rounds: ${landed.length} thrown, ${pickMiss} never handed over, ${throwMiss} handed over but never landed`);
// How far off the arrow's direction each landing was, in degrees
const off = landed.map(r => Math.abs(((Math.atan2(r.point[1], r.point[0]) - r.heading + 3 * Math.PI) % (2 * Math.PI)) - Math.PI) * 180 / Math.PI).sort((a, b) => a - b);
console.log(`off the arrow: median ${off[Math.floor((off.length - 1) / 2)].toFixed(0)}°, 90th pct ${off[Math.floor(0.9 * (off.length - 1))].toFixed(0)}°`);
console.log(`throw distance (m): mean ${mean.toFixed(2)} ± ${(1.96 * sd / Math.sqrt(d.length)).toFixed(2)} (95% CI), ` +
    `10th pct ${q(0.1)}, median ${q(0.5)}, 90th pct ${q(0.9)}, best ${q(1)}; under 0.5 m: ${d.filter(x => x < 0.5).length}`);
// By aim direction, in 8 sectors (0° = straight ahead of the robot, +x)
const sectors = Array.from({ length: 8 }, () => []);
for (const r of landed) sectors[Math.floor(((r.heading + Math.PI) / (2 * Math.PI)) * 8) % 8].push(r.distance);
console.log('median by aim: ' + sectors.map((s, i) => {
    const deg = Math.round(-180 + 22.5 + 45 * i);
    s.sort((a, b) => a - b);
    return `${deg}°: ${s.length ? s[Math.floor((s.length - 1) / 2)].toFixed(2) : '-'}`;
}).join(', '));
