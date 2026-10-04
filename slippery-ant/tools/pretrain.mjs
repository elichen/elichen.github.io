// Pretrain (or keep training) the Ant with Stream-AC in Node, on the same MuJoCo
// build and the same code the page runs.
//
//   node tools/pretrain.mjs --steps 3000000 --seed 1 --friction 2@0 --out ant-mu2
//   node tools/pretrain.mjs --load ant-mu2 --steps 4000000 --friction 0.02@0,2@1000000 --out switch
//   add --wasm to run the per-step math in stream-ac.wasm (~1.6x faster, same results up to f32 rounding)
//   add --frozen to only act (no learning), e.g. to measure the pretrained policy on ice
//
// Writes tools/runs/<out>.csv (one row per episode) and tools/runs/<out>.{json,bin}
// (the agent). Copy a checkpoint into model/ to ship it.
import loadMujoco from '@mujoco/mujoco';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { AntEnv } from '../ant-env.js';
import { StreamAC } from '../stream-ac.js';
import { WasmLearner } from '../wasm-learner.js';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
    if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1]]);
    return acc;
}, []));
const steps = +(args.steps || 1e6);
const seed = +(args.seed || 1);
const out = args.out || 'run';
const schedule = (args.friction || '').split(',').filter(Boolean).map(x => x.split('@').map(Number));

// Seeded Math.random (mulberry32) so runs are reproducible
let rs = seed >>> 0;
Math.random = () => {
    rs = (rs + 0x6D2B79F5) >>> 0;
    let t = rs;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const here = new URL('..', import.meta.url);
const runs = new URL('tools/runs/', here);
mkdirSync(runs, { recursive: true });

const mujoco = await loadMujoco();
const env = new AntEnv(mujoco, readFileSync(new URL('ant.xml', here), 'utf8'));

let agent;
if (args.load) {
    const dir = args.load.includes('/') ? '' : 'tools/runs/';
    const meta = JSON.parse(readFileSync(new URL(`${dir}${args.load}.json`, here), 'utf8'));
    const buf = readFileSync(new URL(`${dir}${args.load}.bin`, here));
    agent = StreamAC.load(meta, new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4));
    console.log(`loaded ${args.load} (${meta.obsCount.toLocaleString()} steps of normalizer history)`);
} else {
    agent = new StreamAC(env.obsDim, env.nu);
}

// One interface over both learners: start(raw obs), act(), learn(raw next obs, reward, terminated, truncated)
const frozen = 'frozen' in args;
let learner;
if ('wasm' in args) {
    const w = await WasmLearner.create(readFileSync(new URL('stream-ac.wasm', here)), agent, seed);
    w.learning = !frozen;
    learner = { start: raw => w.resetObs(raw), act: () => w.act(), learn: (...x) => w.learn(...x), sync: () => w.copyOut(agent) };
} else {
    let s;
    learner = {
        start: raw => { s = agent.normalize(raw, !frozen); },
        act: () => agent.act(s),
        learn: (obs, reward, terminated, truncated) => {
            const s2 = agent.normalize(obs, !frozen);
            if (!frozen) agent.learn(s, agent.scaleReward(reward, terminated || truncated), s2, terminated, truncated);
            s = s2;
        },
        sync: () => {}
    };
}

const save = name => {
    if (frozen) return;
    learner.sync();
    const { meta, data } = agent.save();
    writeFileSync(new URL(`${name}.json`, runs), JSON.stringify(meta));
    writeFileSync(new URL(`${name}.bin`, runs), Buffer.from(data.buffer));
};

const rows = ['step,return,length,forward,friction'];
learner.start(env.reset());
let ret = 0, len = 0, fwd = 0;
let block = [], t0 = performance.now();
for (let step = 1; step <= steps; step++) {
    for (const [mu, at] of schedule) {
        if (step - 1 === at) {
            env.setFriction(mu);
            console.log(`${out} friction -> ${mu} at ${at}`);
        }
    }
    const r = env.step(learner.act());
    learner.learn(r.obs, r.reward, r.terminated, r.truncated);
    const done = r.terminated || r.truncated;
    ret += r.reward; fwd += r.forward; len++;
    if (done) {
        rows.push(`${step},${ret.toFixed(1)},${len},${(fwd / len).toFixed(3)},${env.friction}`);
        block.push(ret);
        ret = 0; len = 0; fwd = 0;
        learner.start(env.reset());
    }
    if (step % 100000 === 0) {
        const avg = block.reduce((x, y) => x + y, 0) / Math.max(block.length, 1);
        const rate = Math.round(100000 / ((performance.now() - t0) / 1000));
        console.log(`${out} ${(step / 1e6).toFixed(1)}M | avg return ${avg.toFixed(0)} over ${block.length} episodes | ${rate} steps/s`);
        block = [];
        t0 = performance.now();
    }
    if (step % 1000000 === 0) save(out);
}
save(out);
writeFileSync(new URL(`${out}.csv`, runs), rows.join('\n') + '\n');
console.log(`${out} saved`);
