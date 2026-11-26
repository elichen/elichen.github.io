// Node.js verification & benchmark suite for Shadow Hand Rubik's Cube policies on MuJoCo WASM.
// Usage:
//   node rubiks-hand/tools/eval.mjs --parity
//   node rubiks-hand/tools/eval.mjs --reorient 40
//   node rubiks-hand/tools/eval.mjs --twist 40
//   node rubiks-hand/tools/eval.mjs --solve 10 --scramble 5
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import loadMujoco from '../../robotarm/node_modules/@mujoco/mujoco/mujoco.js';
import { RubikHandEnv, CubeSolverController, makeRandom } from '../rubik-env.js';
import { FACE_INFO, QPOS_CENTERS, quatRotateVec, wrapPi, wrap90, round90 } from '../cube-state.js';
import { Policy } from '../policy.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function loadFiles() {
    const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'robot/manifest.json'), 'utf8'));
    const files = {};
    for (const f of manifest.files) {
        const p = path.join(ROOT, 'robot', f);
        files[f] = f.endsWith('.xml') ? fs.readFileSync(p, 'utf8') : new Uint8Array(fs.readFileSync(p));
    }
    return files;
}

function loadPolicy(name) {
    const dir = path.join(ROOT, 'model', name);
    const meta = JSON.parse(fs.readFileSync(path.join(dir, 'policy.json'), 'utf8'));
    const buf = fs.readFileSync(path.join(dir, 'policy.bin'));
    const weights = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    return { policy: new Policy(meta, weights), dir };
}

function checkParity(name) {
    const { policy, dir } = loadPolicy(name);
    const ref = JSON.parse(fs.readFileSync(path.join(dir, 'test.json'), 'utf8'));
    let maxErr = 0;
    for (let i = 0; i < ref.obs.length; i++) {
        const got = policy.action(Float64Array.from(ref.obs[i]));
        for (let j = 0; j < got.length; j++) {
            maxErr = Math.max(maxErr, Math.abs(got[j] - ref.act[i][j]));
        }
    }
    const ok = maxErr < 1e-5;
    console.log(`Parity [${name}]: maxErr=${maxErr.toExponential(2)} -> ${ok ? 'PASS' : 'FAIL'}`);
    return ok;
}

async function main() {
    const args = process.argv.slice(2);
    const getArg = (flag, def) => {
        const i = args.indexOf(flag);
        return i >= 0 && args[i + 1] ? Number(args[i + 1]) : def;
    };

    if (args.includes('--parity') || args.length === 0) {
        let allOk = true;
        for (const name of ['reorient', 'twist']) {
            if (fs.existsSync(path.join(ROOT, 'model', name, 'policy.json'))) {
                allOk = checkParity(name) && allOk;
            }
        }
        if (!allOk) process.exit(1);
    }

    const nReorient = getArg('--reorient', 0);
    const nTwist = getArg('--twist', 0);
    const nSolve = getArg('--solve', 0);
    const scrambleLen = getArg('--scramble', 5);
    const seed = getArg('--seed', 42);

    if (nReorient === 0 && nTwist === 0 && nSolve === 0) return;

    const mj = await loadMujoco();
    const files = loadFiles();
    const rng = makeRandom(seed);
    const env = new RubikHandEnv(mj, files, rng);

    if (nReorient > 0) {
        const { policy } = loadPolicy('reorient');
        let ok = 0, drops = 0, totalSteps = 0;
        for (let t = 0; t < nReorient; t++) {
            env.scramble(10);
            const target = rng.int(6);
            let done = false;
            for (let s = 0; s < 300; s++) {
                env.step(policy.action(env.reorientObs(target)));
                if (env.isDropped()) { drops++; done = true; break; }
                const nw = quatRotateVec(env.cubeQuat(), FACE_INFO[target].axis);
                if (nw[2] > 0.965 && env.maxFaceMisalignment() < 0.16) {
                    ok++; totalSteps += s + 1; done = true; break;
                }
            }
        }
        console.log(`Reorient (${nReorient} trials): ${ok}/${nReorient} (${(100*ok/nReorient).toFixed(1)}%), drops=${drops}, avgSteps=${ok ? (totalSteps/ok).toFixed(1) : '-'}`);
    }

    if (nTwist > 0) {
        const { policy: reorientPol } = loadPolicy('reorient');
        const { policy } = loadPolicy('twist');
        let ok = 0, drops = 0, totalSteps = 0;
        for (let t = 0; t < nTwist; t++) {
            env.scramble(10);
            const faceId = rng.int(6);
            // Bring random face to +Z via Reorient so Twist starts from a real handover grasp
            for (let s = 0; s < 200; s++) {
                env.step(reorientPol.action(env.reorientObs(faceId)));
                const nw = quatRotateVec(env.cubeQuat(), FACE_INFO[faceId].axis);
                const angv = Math.hypot(env.data.qvel[27], env.data.qvel[28], env.data.qvel[29]);
                if (nw[2] > 0.975 && env.maxFaceMisalignment() < 0.13 && angv < 2.2 && env.cubePosDist() < 0.045) break;
            }
            const dir = t % 2 === 0 ? -1 : 1;
            const goal = round90(env.data.qpos[QPOS_CENTERS + faceId]) + dir * Math.PI / 2;
            for (let s = 0; s < 250; s++) {
                env.step(policy.action(env.twistObs(faceId, goal, dir)));
                if (env.isDropped()) { drops++; break; }
                const rem = Math.abs(wrapPi(goal - env.data.qpos[QPOS_CENTERS + faceId]));
                const tv = Math.abs(env.data.qvel[30 + faceId]);
                let otherMax = 0;
                for (let i = 0; i < 6; i++) if (i !== faceId) otherMax = Math.max(otherMax, Math.abs(wrap90(env.data.qpos[QPOS_CENTERS + i])));
                const nw = quatRotateVec(env.cubeQuat(), FACE_INFO[faceId].axis);
                if (rem < 0.15 && tv < 2.2 && otherMax < 0.16 && nw[2] > 0.90) {
                    ok++; totalSteps += s + 1; break;
                }
            }
        }
        console.log(`Twist (${nTwist} trials): ${ok}/${nTwist} (${(100*ok/nTwist).toFixed(1)}%), drops=${drops}, avgSteps=${ok ? (totalSteps/ok).toFixed(1) : '-'}`);
    }

    if (nSolve > 0) {
        const { policy: reorient } = loadPolicy('reorient');
        const { policy: twist } = loadPolicy('twist');
        let solved = 0, drops = 0, totalMoves = 0, totalSimSec = 0;
        for (let t = 0; t < nSolve; t++) {
            const scr = env.scramble(scrambleLen);
            const ctrl = new CubeSolverController(env);
            const initPlan = ctrl.plan.length;
            let status = 'timeout';
            for (let s = 0; s < 12000; s++) {
                env.step(ctrl.action(reorient, twist));
                const ev = ctrl.update();
                if (ev?.event === 'solved') {
                    solved++;
                    totalMoves += ctrl.completedMoves.length;
                    totalSimSec += s * env.dt;
                    status = `SOLVED in ${ctrl.completedMoves.length} moves (${(s*env.dt).toFixed(1)}s sim, replans=${ctrl.replanCount})`;
                    break;
                }
                if (ev?.event === 'dropped') {
                    drops++;
                    status = `DROPPED after ${ctrl.completedMoves.length}/${initPlan} moves`;
                    break;
                }
            }
            console.log(`  Trial ${t + 1}/${nSolve} [${scr.join(' ')}]: ${status}`);
        }
        console.log(`End-to-end Solve (scramble=${scrambleLen}): ${solved}/${nSolve} (${(100*solved/nSolve).toFixed(1)}%), drops=${drops}, avgSimSec=${solved ? (totalSimSec/solved).toFixed(1) : '-'}`);
    }
}

main().catch(e => { console.error(e); process.exit(1); });
