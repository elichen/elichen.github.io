// Physics and learning run here, off the main thread, so the page stays smooth
// at full speed. The page sends commands; this posts poses (about 60 a second)
// and one message per finished episode.
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.js';
import { AntEnv } from './ant-env.js';
import { StreamAC } from './stream-ac.js';
import { WasmLearner } from './wasm-learner.js';

const SPEEDS = { 1: 1, 10: 10, max: Infinity };  // multiples of real time (20 steps/s)
let env, learner, pristine;
let speed = 1, running = false;
let step = 0, epReturn = 0, epLength = 0, epForward = 0;
let lastFrame = 0, budgetStart = 0, simTime = 0, wallStart = 0;
// Measured speed: steps and time spent stepping over the last second
let rateSteps = 0, rateStart = 0, busy = 0, rate = 0, busyFraction = 0;
const deltas = [];

async function init({ friction }) {
    const [mujoco, xml, meta, bin, wasm] = await Promise.all([
        loadMujoco(),
        fetch('ant.xml').then(r => r.text()),
        fetch('model/agent.json').then(r => r.json()),
        fetch('model/agent.bin').then(r => r.arrayBuffer()),
        fetch('stream-ac.wasm').then(r => r.arrayBuffer())
    ]);
    env = new AntEnv(mujoco, xml);
    env.setFriction(friction);
    pristine = StreamAC.load(meta, new Float32Array(bin));
    learner = await WasmLearner.create(wasm, pristine, (Math.random() * 2 ** 32) >>> 0);
    learner.resetObs(env.reset());

    const m = env.model;
    postMessage({
        type: 'ready',
        ngeom: m.ngeom,
        geomType: Array.from(m.geom_type),
        geomSize: Array.from(m.geom_size),
        dt: env.dt,
        pretrainedSteps: meta.obsCount
    });
    running = true;
    wallStart = performance.now();
    loop();
}

// One environment step plus one learning step
function tick() {
    const r = env.step(learner.act());
    const delta = learner.learn(r.obs, r.reward, r.terminated, r.truncated);
    step++;
    epReturn += r.reward;
    epForward += r.forward;
    epLength++;
    if (learner.learning) deltas.push(Math.abs(delta));
    if (r.terminated || r.truncated) {
        postMessage({
            type: 'episode', step, ret: epReturn, length: epLength, speed: epForward / epLength,
            friction: env.friction, learning: learner.learning, fell: r.terminated
        });
        epReturn = 0; epLength = 0; epForward = 0;
        learner.resetObs(env.reset());
    }
}

function postFrame() {
    const d = env.data;
    let meanDelta = 0;
    for (const x of deltas) meanDelta += x;
    meanDelta = deltas.length ? meanDelta / deltas.length : 0;
    deltas.length = 0;
    const xpos = Float32Array.from(d.geom_xpos), xmat = Float32Array.from(d.geom_xmat);
    postMessage({
        type: 'frame', xpos, xmat, step, epReturn, epLength,
        torso: [d.qpos[0], d.qpos[1], d.qpos[2]],
        value: learner.lastValue, meanDelta, push: learner.actorPush, criticPush: learner.criticPush,
        rate, busyFraction
    }, [xpos.buffer, xmat.buffer]);
}

// Real time and 10x pace themselves against the clock; Max steps flat out,
// yielding every ~12 ms so commands and frames still get through
function loop() {
    if (!running) return;
    budgetStart = performance.now();
    const before = step;
    const pace = SPEEDS[speed];
    if (pace === Infinity) {
        while (performance.now() - budgetStart < 12) tick();
    } else {
        const wantSteps = ((performance.now() - wallStart) / 1000) / env.dt * pace;
        let n = 0;
        while (simTime < wantSteps && n < 500) { tick(); simTime++; n++; }
        // Don't try to catch up after a stall (e.g. a background tab)
        if (simTime < wantSteps - 500) simTime = wantSteps;
    }
    const now = performance.now();
    busy += now - budgetStart;
    rateSteps += step - before;
    if (now - rateStart > 1000) {
        rate = rateSteps / ((now - rateStart) / 1000);
        busyFraction = busy / (now - rateStart);
        rateSteps = 0; busy = 0; rateStart = now;
    }
    if (now - lastFrame > 15) {
        postFrame();
        lastFrame = now;
    }
    schedule();
}

// setTimeout(0) is clamped to 4 ms after a few nestings; a MessageChannel isn't
const channel = new MessageChannel();
channel.port1.onmessage = () => (speed === 'max' ? loop() : setTimeout(loop, 4));
const schedule = () => channel.port2.postMessage(0);

function setSpeed(s) {
    speed = s;
    wallStart = performance.now();
    simTime = 0;
}

onmessage = async ({ data }) => {
    switch (data.type) {
        case 'init': await init(data); break;
        case 'friction': env.setFriction(data.value); break;
        case 'speed': setSpeed(data.value); break;
        case 'learning':
            learner.learning = data.value;
            // Traces from before a pause would credit the wrong steps
            learner.resetTraces();
            break;
        case 'reset-agent':
            learner.copyIn(pristine);
            learner.resetObs(env.reset());
            epReturn = 0; epLength = 0; epForward = 0;
            break;
        case 'pause': running = false; break;
        case 'resume': if (!running) { running = true; setSpeed(speed); loop(); } break;
    }
};
