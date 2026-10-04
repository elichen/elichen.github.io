// Turn a run's per-episode CSV into the compact JSON the article's charts load.
//   node tools/export-run.mjs <run> [<run> ...] [--as name] [--frozen]
// Concatenates runs end to end (steps offset), marks every friction change.
import { readFileSync, writeFileSync, mkdirSync } from 'fs';

const argv = process.argv.slice(2);
const as = argv.includes('--as') ? argv[argv.indexOf('--as') + 1] : null;
const frozen = argv.includes('--frozen');
const runs = argv.filter((a, i) => !a.startsWith('--') && argv[i - 1] !== '--as');
const here = new URL('..', import.meta.url);
const label = mu => mu >= 1.5 ? 'rubber' : mu <= 0.03 ? 'ice' : `μ ${mu}`;

const points = [], events = [];
let offset = 0, lastMu = null;
for (const run of runs) {
    const rows = readFileSync(new URL(`tools/runs/${run}.csv`, here), 'utf8').trim().split('\n').slice(1);
    let maxStep = 0;
    for (const row of rows) {
        const [step, ret, , , mu] = row.split(',').map(Number);
        // An episode belongs to the friction it ended on; mark where the friction first changed
        if (mu !== lastMu) {
            const prev = points.length ? points[points.length - 1][0] : 0;
            events.push([lastMu === null ? 0 : prev, label(mu)]);
            lastMu = mu;
        }
        points.push([offset + step, Math.round(ret), mu]);
        maxStep = step;
    }
    offset += maxStep;
}
mkdirSync(new URL('data/', here), { recursive: true });
const name = as || runs.join('+');
writeFileSync(new URL(`data/${name}.json`, here), JSON.stringify({ points, events, frozen }));
console.log(`data/${name}.json: ${points.length} episodes, ${events.length} changes, ${(offset / 1e6).toFixed(1)}M steps`);
