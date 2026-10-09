// Run many simulations in parallel and summarise them like Reid et al. (2015), Fig. 2:
// distance the bridge moved over time, by angle and by traffic.
//   node tools/sweep.mjs [--angles 12,20,40,60] [--traffics 75,150,225] [--seeds 3] [--minutes 30]
//                        [--params '{"lockChance":0.5}'] [--out runs.json]
import { spawn } from 'node:child_process';
import { readFileSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir, cpus } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]);
    return acc;
}, []));
const angles = (args.angles ?? '12,20,40,60').split(',').map(Number);
const traffics = (args.traffics ?? '150').split(',').map(Number);
const seeds = +(args.seeds ?? 3), minutes = +(args.minutes ?? 30);
const here = dirname(fileURLToPath(import.meta.url));
const tmp = mkdtempSync(join(process.env.TMPDIR || tmpdir(), 'antsweep-'));

const jobs = [];
for (const angle of angles) for (const traffic of traffics) for (let seed = 1; seed <= seeds; seed++) jobs.push({ angle, traffic, seed });

const run = (job, k) => new Promise((resolve) => {
    const out = join(tmp, `${k}.json`);
    const argv = [join(here, 'run.mjs'), '--angle', job.angle, '--traffic', job.traffic, '--seed', job.seed,
        '--minutes', minutes, '--every', 30, '--quiet', '--json', out];
    if (args.params) argv.push('--params', args.params);
    const ch = spawn(process.execPath, argv.map(String), { stdio: ['ignore', 'ignore', 'inherit'] });
    ch.on('exit', () => resolve({ ...job, rows: JSON.parse(readFileSync(out)).rows }));
});

const results = [];
let next = 0;
const workers = Array.from({ length: Math.min(jobs.length, Math.max(1, cpus().length - 2)) }, async () => {
    while (next < jobs.length) { const k = next++; results.push(await run(jobs[k], k)); }
});
await Promise.all(workers);

const at = (rows, min) => rows.reduce((b, r) => Math.abs(r.t - min * 60) < Math.abs(b.t - min * 60) ? r : b);
const mean = (xs) => { xs = xs.filter(Number.isFinite); return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN; };
const marks = [2, 5, 10, 20, minutes].filter((m, i, a) => m <= minutes && a.indexOf(m) === i);
console.log(`angle traffic | d (cm) at ${marks.join(', ')} min | ants | width | saved | falls`);
for (const angle of angles) for (const traffic of traffics) {
    const rs = results.filter(r => r.angle === angle && r.traffic === traffic);
    const win = (rows, m) => mean(rows.filter(r => Math.abs(r.t - m * 60) <= 60).map(r => r.d));
    const ds = marks.map(m => mean(rs.map(r => win(r.rows, m))).toFixed(1).padStart(5)).join(' ');
    const last = rs.map(r => at(r.rows, minutes));
    console.log(`${String(angle).padStart(4)}° ${String(traffic).padStart(5)}  | ${ds} | ${mean(last.map(l => l.ants)).toFixed(0).padStart(4)} | ${mean(last.map(l => l.width)).toFixed(1).padStart(4)} | ${mean(last.map(l => l.saved)).toFixed(1).padStart(5)} | ${mean(last.map(l => l.falls)).toFixed(1)}`);
}
if (args.out) writeFileSync(args.out, JSON.stringify({ args, results }));
