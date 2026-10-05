// Physics and learning run here, off the main thread, so the page stays smooth
// at full speed. The page sends commands; this posts poses (about 60 a second)
// and one message per finished episode.
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.js';
import { G1Env } from './g1-env.js';
import { applyWorld } from './worlds.js';
import { StreamAC } from './stream-ac.js';
import { WasmLearner } from './wasm-learner.js';

const SPEEDS = { 0.05: 0.05, 1: 1, 10: 10, max: Infinity };  // multiples of real time; 0.05 is slow motion for checking gaits
let env, learner, pristine, torsoBody = 1;
let speed = 1, running = false;
let step = 0, epReturn = 0, epLength = 0, epForward = 0;
let lastFrame = 0, budgetStart = 0, simTime = 0, wallStart = 0;
// Measured speed: steps and time spent stepping over the last second
let rateSteps = 0, rateStart = 0, busy = 0, rate = 0, busyFraction = 0;
const deltas = [];

// The robot's files (robot/manifest.json), keyed by name for MuJoCo's virtual file system
async function loadRobot() {
    const { files: list } = await fetch('robot/manifest.json').then(r => r.json());
    const files = {};
    await Promise.all(list.map(async f => {
        const name = f.split('/').pop(), r = await fetch('robot/' + f);
        files[name] = f.endsWith('.xml') ? (await r.text()).replace(/meshdir="[^"]*"/, 'meshdir="."') : new Uint8Array(await r.arrayBuffer());
    }));
    return files;
}

async function init({ world }) {
    const [mujoco, files, meta, bin, wasm] = await Promise.all([
        loadMujoco(),
        loadRobot(),
        fetch('model/agent.json').then(r => r.json()),
        fetch('model/agent.bin').then(r => r.arrayBuffer()),
        fetch('stream-ac.wasm').then(r => r.arrayBuffer())
    ]);
    env = new G1Env(mujoco, files);
    applyWorld(env, world);
    pristine = StreamAC.load(meta, new Float32Array(bin));
    learner = await WasmLearner.create(wasm, pristine, (Math.random() * 2 ** 32) >>> 0);
    learner.resetObs(env.reset());

    // Visual geoms (group 2) and their meshes, sent once; the renderer poses them every frame
    const m = env.model, transfer = [], meshes = {};
    const bodyName = b => m.body(b).name;
    torsoBody = [...Array(m.nbody).keys()].find(b => bodyName(b) === 'torso_link');
    const visuals = [];
    for (let g = 0; g < m.ngeom; g++) {
        if (m.geom_group[g] !== 2 || m.geom_type[g] !== 7) continue;
        const mesh = m.geom_dataid[g], mat = m.geom_matid[g];
        if (!meshes[mesh]) {
            const v0 = m.mesh_vertadr[mesh], nv = m.mesh_vertnum[mesh], f0 = m.mesh_faceadr[mesh], nf = m.mesh_facenum[mesh];
            meshes[mesh] = { verts: Float32Array.from(m.mesh_vert.slice(3 * v0, 3 * (v0 + nv))), faces: Uint32Array.from(m.mesh_face.slice(3 * f0, 3 * (f0 + nf))) };
            transfer.push(meshes[mesh].verts.buffer, meshes[mesh].faces.buffer);
        }
        const rgba = mat >= 0 ? m.mat_rgba.slice(4 * mat, 4 * mat + 3) : m.geom_rgba.slice(4 * g, 4 * g + 3);
        visuals.push({ geom: g, mesh, body: bodyName(m.geom_bodyid[g]), color: Array.from(rgba) });
    }
    postMessage({ type: 'ready', ngeom: m.ngeom, visuals, meshes, dt: env.dt }, transfer);
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
            world: env.world, learning: learner.learning, fell: r.terminated
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
    // The torso body's own frame, for things attached to it (the backpack)
    const torsoPos = Array.from(d.xpos.slice(3 * torsoBody, 3 * torsoBody + 3)), torsoMat = Array.from(d.xmat.slice(9 * torsoBody, 9 * torsoBody + 9));
    postMessage({
        type: 'frame', xpos, xmat, step, epReturn, epLength,
        torso: [d.qpos[0], d.qpos[1], d.qpos[2]], torsoPos, torsoMat,
        // Pelvis heading (yaw) from its quaternion, so the camera can follow the way it faces
        yaw: Math.atan2(2 * (d.qpos[4] * d.qpos[5] + d.qpos[3] * d.qpos[6]), 1 - 2 * (d.qpos[5] * d.qpos[5] + d.qpos[6] * d.qpos[6])),
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
        case 'init':
            // Errors in async code don't reach the page's worker.onerror, so report them
            try { await init(data); } catch (e) { postMessage({ type: 'error', message: String(e?.message || e) }); }
            break;
        case 'world': applyWorld(env, data.value); break;
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
