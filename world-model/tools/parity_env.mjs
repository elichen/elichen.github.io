// Verify JS CubeEnv + DiffIKController on MuJoCo WASM against Python OGBench reference rollouts.
//   node tools/parity_env.mjs
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import loadMujoco from '../../robotarm/node_modules/@mujoco/mujoco/mujoco.js';
import { CubeEnv } from '../cube-env.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const manifest = JSON.parse(readFileSync(`${root}/robot/manifest.json`, 'utf8'));
const files = {};
for (const f of manifest.files) {
  files[f] = f.endsWith('.xml') ? readFileSync(`${root}/robot/${f}`, 'utf8') : readFileSync(`${root}/robot/${f}`);
}
const ref = JSON.parse(readFileSync(`${root}/parity/ref.json`, 'utf8'));
const mj = await loadMujoco();
const env = new CubeEnv(mj, files);

// 1. Standalone DiffIK parity
let maxIk = 0;
for (const tc of ref.ik_tests) {
  const got = env.ik.solve(tc.pos, tc.quat, tc.curr_qpos);
  for (let i = 0; i < 6; i++) maxIk = Math.max(maxIk, Math.abs(got[i] - tc.qpos_target[i]));
}

// 2. 50-step closed-loop physics + setControl parity across 2 episodes
let maxCtrl = 0, maxQpos = 0, maxEff = 0, maxCube = 0;
for (const ep of ref.rollout_refs) {
  env.setState(ep.qpos0, ep.qvel0);
  for (let k = 0; k < ep.actions.length; k++) {
    env.step(ep.actions[k]);
    const st = ep.steps[k];
    for (let i = 0; i < 7; i++) maxCtrl = Math.max(maxCtrl, Math.abs(env.data.ctrl[i] - st.ctrl[i]));
    for (let i = 0; i < env.model.nq; i++) maxQpos = Math.max(maxQpos, Math.abs(env.data.qpos[i] - st.qpos[i]));
    const eff = env.effectorPos(), cu = env.cubePos();
    maxEff = Math.max(maxEff, Math.hypot(eff[0] - st.effector_pos[0], eff[1] - st.effector_pos[1], eff[2] - st.effector_pos[2]));
    maxCube = Math.max(maxCube, Math.hypot(cu[0] - st.cube_pos[0], cu[1] - st.cube_pos[1], cu[2] - st.cube_pos[2]));
  }
}

const ok = maxIk < 1e-6 && maxEff < 5e-4 && maxCube < 5e-4;
console.log(`DiffIK (10 poses):   max |dq|     = ${maxIk.toExponential(2)}`);
console.log(`Physics (50 steps):  max |dctrl|  = ${maxCtrl.toExponential(2)}`);
console.log(`                     max |dqpos|  = ${maxQpos.toExponential(2)}`);
console.log(`                     max |deff|   = ${(maxEff * 1000).toFixed(4)} mm`);
console.log(`                     max |dcube|  = ${(maxCube * 1000).toFixed(4)} mm`);
console.log(ok ? 'PARITY_ENV: PASS' : 'PARITY_ENV: FAIL');
process.exit(ok ? 0 : 1);
