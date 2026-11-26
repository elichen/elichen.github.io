// Shadow Dexterous Hand + 26-cubelet 3x3x3 Rubik's Cube environment on MuJoCo WebAssembly.
// Shared by the web app and Node.js verification tools (parity.mjs, eval.mjs).
// Mirrors training/rubik_warp_env.py identically:
//   - 50 Hz control (5 physics substeps of 4 ms)
//   - EMA-filtered absolute joint targets (alpha = 0.4) across all 20 Shadow Hand actuators
//   - Full physical cube state observations (Reorient: 140 dims, Twist: 143 dims)

import {
    FACE_INFO, LETTER_TO_FACE, QPOS_CUBE_POS, QPOS_CUBE_QUAT, QPOS_CENTERS,
    applyMoveToQpos, readFaceletsFromQpos, isCubeSolvedQpos, topFaceFromCoreQuat,
    quatRotateVec, wrapPi, wrap90, round90,
} from './cube-state.js';
import { solveCube, expandQuarterTurns } from './kociemba.js';

export const SUBSTEPS = 5;
export const ACTION_EMA = 0.4;
export const OBS_REORIENT_DIM = 140;
export const OBS_TWIST_DIM = 143;

// Neutral palm-up cradle pose (20 actuator targets)
export const CRADLE_CTRL = Float64Array.from([
    0.0, 0.0,
    0.3, 0.9, 0.0, 0.2, 0.3,
    0.0, 0.5, 0.8,
    0.0, 0.5, 0.8,
    0.0, 0.5, 0.8,
    0.2, 0.0, 0.5, 0.8,
]);

const TIP_NAMES = ['tip_ff', 'tip_mf', 'tip_rf', 'tip_lf', 'tip_th'];

export function makeRandom(seed = (Math.random() * 2 ** 32) >>> 0) {
    let s = seed >>> 0;
    const uniform = () => {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return {
        uniform,
        range: (lo, hi) => lo + (hi - lo) * uniform(),
        int: n => Math.floor(uniform() * n),
    };
}

function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

// Smooth globally-analytic 90-degree-periodic yaw harmonic (sin(4*yaw), cos(4*yaw)) from 3x3 rotation matrix
export function yawHarmonic4(xmat, off) {
    let c4 = 0, s4 = 0;
    for (let j = 0; j < 3; j++) {
        const rx = xmat[off + j];      // R[0, j]
        const ry = xmat[off + 3 + j];  // R[1, j]
        const rx2 = rx * rx, ry2 = ry * ry;
        c4 += rx2 * rx2 + ry2 * ry2 - 6 * rx2 * ry2;
        s4 += 4 * (rx2 * rx * ry - rx * ry2 * ry);
    }
    return [0.5 * s4, 0.5 * c4];
}

export class RubikHandEnv {
    constructor(mujoco, files, random = makeRandom()) {
        this.mj = mujoco;
        this.random = random;
        this.vfs = new mujoco.MjVFS();
        const enc = new TextEncoder();
        for (const [name, data] of Object.entries(files)) {
            this.vfs.addBuffer(name, typeof data === 'string' ? enc.encode(data) : data);
        }
        const m = this.model = mujoco.MjModel.from_xml_string(files['shadow_rubik.xml'], this.vfs);
        this.data = new mujoco.MjData(m);

        const id = (type, name) => mujoco.mj_name2id(m, mujoco.mjtObj[type].value, name);
        this.cubeBody = id('mjOBJ_BODY', 'cube_core');
        this.graspSite = id('mjOBJ_SITE', 'grasp_site');
        this.tipSites = TIP_NAMES.map(n => id('mjOBJ_SITE', n));
        this.nu = m.nu;
        this.dt = m.opt.timestep * SUBSTEPS;

        this.lo = Float64Array.from({ length: m.nu }, (_, i) => m.actuator_ctrlrange[2 * i]);
        this.hi = Float64Array.from({ length: m.nu }, (_, i) => m.actuator_ctrlrange[2 * i + 1]);
        this.mid = Float64Array.from({ length: m.nu }, (_, i) => 0.5 * (this.lo[i] + this.hi[i]));
        this.half = Float64Array.from({ length: m.nu }, (_, i) => 0.5 * (this.hi[i] - this.lo[i]));

        this._obsReorient = new Float64Array(OBS_REORIENT_DIM);
        this._obsTwist = new Float64Array(OBS_TWIST_DIM);
        this.reset();
    }

    reset(scrambleMoves = []) {
        const { mj, model: m, data: d } = this;
        mj.mj_resetData(m, d);
        for (let i = 0; i < this.nu; i++) d.ctrl[i] = CRADLE_CTRL[i];
        // Initialize hand qpos close to cradle pose
        for (let i = 0; i < 7; i++) d.qpos[i] = CRADLE_CTRL[i];
        // Finger joints (each finger has J4, J3, J2, J1; actuator J0 drives J2+J1)
        const fMap = [[7, 7], [11, 10], [15, 13], [20, 17]];
        d.qpos[19] = CRADLE_CTRL[16]; // LFJ5
        for (const [qOff, aOff] of fMap) {
            d.qpos[qOff] = CRADLE_CTRL[aOff];
            d.qpos[qOff + 1] = CRADLE_CTRL[aOff + 1];
            d.qpos[qOff + 2] = 0.5 * CRADLE_CTRL[aOff + 2];
            d.qpos[qOff + 3] = 0.5 * CRADLE_CTRL[aOff + 2];
        }
        for (const mv of scrambleMoves) applyMoveToQpos(d.qpos, mv);
        mj.mj_forward(m, d);
    }

    scramble(nMoves = 20) {
        const faces = ['U', 'R', 'F', 'D', 'L', 'B'];
        const mods = ['', "'", '2'];
        const moves = [];
        let lastFace = -1;
        for (let i = 0; i < nMoves; i++) {
            let f = this.random.int(6);
            while (f === lastFace) f = this.random.int(6);
            lastFace = f;
            moves.push(faces[f] + mods[this.random.int(3)]);
        }
        this.reset(moves);
        return moves;
    }

    cubePos() {
        const p = this.data.qpos;
        return [p[QPOS_CUBE_POS], p[QPOS_CUBE_POS + 1], p[QPOS_CUBE_POS + 2]];
    }

    cubeQuat() {
        const p = this.data.qpos;
        let w = p[QPOS_CUBE_QUAT], x = p[QPOS_CUBE_QUAT + 1], y = p[QPOS_CUBE_QUAT + 2], z = p[QPOS_CUBE_QUAT + 3];
        if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
        return [w, x, y, z];
    }

    faceAngles() {
        const p = this.data.qpos;
        return [p[31], p[32], p[33], p[34], p[35], p[36]];
    }

    cubePosDist() {
        const p = this.cubePos();
        const g = this.data.site_xpos;
        const gi = 3 * this.graspSite;
        return Math.hypot(p[0] - g[gi], p[1] - g[gi + 1], p[2] - g[gi + 2]);
    }

    isDropped() {
        const p = this.cubePos();
        return this.cubePosDist() > 0.14 || p[2] < -0.05;
    }

    facelets() {
        return readFaceletsFromQpos(this.data.qpos);
    }

    isSolved() {
        return isCubeSolvedQpos(this.data.qpos);
    }

    maxFaceMisalignment() {
        let mx = 0;
        for (let i = 0; i < 6; i++) {
            const e = Math.abs(wrap90(this.data.qpos[QPOS_CENTERS + i]));
            if (e > mx) mx = e;
        }
        return mx;
    }

    _fillBaseObs(o) {
        const d = this.data;
        let k = 0;
        for (let i = 0; i < 24; i++) o[k++] = d.qpos[i];
        for (let i = 0; i < 24; i++) o[k++] = clamp(d.qvel[i] * 0.1, -5, 5);
        for (let i = 0; i < 20; i++) o[k++] = d.ctrl[i];

        const cx = d.qpos[QPOS_CUBE_POS], cy = d.qpos[QPOS_CUBE_POS + 1], cz = d.qpos[QPOS_CUBE_POS + 2];
        const sx = d.site_xpos;
        for (let t = 0; t < 5; t++) {
            const si = 3 * this.tipSites[t];
            o[k++] = (sx[si] - cx) * 10.0;
            o[k++] = (sx[si + 1] - cy) * 10.0;
            o[k++] = (sx[si + 2] - cz) * 10.0;
        }
        const gi = 3 * this.graspSite;
        o[k++] = (cx - sx[gi]) * 10.0;
        o[k++] = (cy - sx[gi + 1]) * 10.0;
        o[k++] = (cz - sx[gi + 2]) * 10.0;

        for (let i = 0; i < 3; i++) o[k++] = clamp(d.qvel[24 + i], -5, 5);
        const q = this.cubeQuat();
        o[k++] = q[0]; o[k++] = q[1]; o[k++] = q[2]; o[k++] = q[3];
        for (let i = 0; i < 3; i++) o[k++] = clamp(d.qvel[27 + i] * 0.2, -5, 5);

        const xmOff = 9 * this.cubeBody;
        for (let i = 0; i < 9; i++) o[k++] = d.xmat[xmOff + i];

        for (let i = 0; i < 6; i++) o[k++] = wrap90(d.qpos[QPOS_CENTERS + i]);
        for (let i = 0; i < 6; i++) o[k++] = Math.sin(4 * d.qpos[QPOS_CENTERS + i]);
        for (let i = 0; i < 6; i++) o[k++] = Math.cos(4 * d.qpos[QPOS_CENTERS + i]);
        for (let i = 0; i < 6; i++) o[k++] = clamp(d.qvel[30 + i] * 0.2, -5, 5);
        return k; // 129
    }

    reorientObs(targetFaceId) {
        const o = this._obsReorient;
        let k = this._fillBaseObs(o);
        const nWorld = quatRotateVec(this.cubeQuat(), FACE_INFO[targetFaceId].axis);
        o[k++] = nWorld[0]; o[k++] = nWorld[1]; o[k++] = nWorld[2];
        for (let i = 0; i < 6; i++) o[k++] = i === targetFaceId ? 1.0 : 0.0;
        const [s4, c4] = yawHarmonic4(this.data.xmat, 9 * this.cubeBody);
        o[k++] = s4; o[k++] = c4;
        return o;
    }

    twistObs(activeFaceId, goalAngle, twistDir) {
        const o = this._obsTwist;
        let k = this._fillBaseObs(o);
        const nWorld = quatRotateVec(this.cubeQuat(), FACE_INFO[activeFaceId].axis);
        o[k++] = nWorld[0]; o[k++] = nWorld[1]; o[k++] = nWorld[2];
        const rem = wrapPi(goalAngle - this.data.qpos[QPOS_CENTERS + activeFaceId]);
        o[k++] = rem;
        o[k++] = Math.sin(rem);
        o[k++] = Math.cos(rem);
        o[k++] = twistDir;
        o[k++] = clamp(this.data.qvel[30 + activeFaceId] * 0.2, -5, 5);
        for (let i = 0; i < 6; i++) {
            o[k++] = i === activeFaceId ? rem : -wrap90(this.data.qpos[QPOS_CENTERS + i]);
        }
        return o;
    }

    act(action) {
        const d = this.data;
        for (let i = 0; i < this.nu; i++) {
            const a = clamp(action[i], -1, 1);
            const target = this.mid[i] + this.half[i] * a;
            d.ctrl[i] = clamp((1 - ACTION_EMA) * d.ctrl[i] + ACTION_EMA * target, this.lo[i], this.hi[i]);
        }
    }

    substep() {
        this.pull?.();
        this.mj.mj_step(this.model, this.data);
    }

    step(action) {
        this.act(action);
        for (let k = 0; k < SUBSTEPS; k++) this.substep();
    }
}

// High-level Kociemba + Reorient + Twist closed-loop Rubik's Cube solver controller
export class CubeSolverController {
    constructor(env) {
        this.env = env;
        this.resetPlan();
    }

    resetPlan() {
        this.plan = [];          // remaining atomic quarter-turn tokens e.g. ["R", "U'", "F", "F"]
        this.completedMoves = [];
        this.phase = 'idle';     // 'idle' | 'reorient' | 'twist' | 'solved'
        this.targetFaceId = 4;
        this.goalAngle = 0;
        this.twistDir = -1;
        this.phaseSteps = 0;
        this.settleSteps = 0;
        this.unjamMode = false;
        this.straightenMode = false;
        this.totalSteps = 0;
        this.replanCount = 0;
        this.consecutiveBails = 0;
        this.consecutiveStraightens = 0;
        this.expectedCenters = null;
        this.replan();
    }

    _mostSkewedFace() {
        let faceId = 0, err = 0;
        for (let i = 0; i < 6; i++) {
            const e = Math.abs(wrap90(this.env.data.qpos[QPOS_CENTERS + i]));
            if (e > err) { err = e; faceId = i; }
        }
        return { faceId, err };
    }

    _snapCenters() {
        const c = new Int32Array(6);
        const h = Math.PI / 2;
        for (let i = 0; i < 6; i++) {
            c[i] = Math.round(this.env.data.qpos[QPOS_CENTERS + i] / h);
        }
        return c;
    }

    _centersDesynced() {
        if (!this.expectedCenters) return false;
        const cur = this._snapCenters();
        for (let i = 0; i < 6; i++) {
            if (((cur[i] - this.expectedCenters[i]) & 3) !== 0) return true;
        }
        return false;
    }

    replan() {
        this.phaseSteps = 0;
        this.settleSteps = 0;
        this.unjamMode = false;
        this.straightenMode = false;
        const { faceId: skewFace, err: skewErr } = this._mostSkewedFace();
        const fl = this.env.facelets();
        if (fl.includes('?') || skewErr > 0.22) {
            // Twist-to-Straighten: bring the jammed/skewed slice to +Z, then use Twist to snap it to round90!
            this.consecutiveStraightens++;
            this.expectedCenters = null;
            if (this.consecutiveStraightens >= 4) {
                this.consecutiveStraightens = 0;
                this.targetFaceId = (skewFace + 2) % 6;
                this.unjamMode = true;
                this.phase = 'reorient';
                return false;
            }
            this.straightenMode = true;
            this.targetFaceId = skewFace;
            const nw = quatRotateVec(this.env.cubeQuat(), FACE_INFO[skewFace].axis);
            const angv = Math.hypot(this.env.data.qvel[27], this.env.data.qvel[28], this.env.data.qvel[29]);
            if (nw[2] > 0.93 && this.env.cubePosDist() < 0.050 && angv < 3.0) {
                this._enterTwist(true);
            } else {
                this.phase = 'reorient';
            }
            return false;
        }
        this.consecutiveStraightens = 0;
        const { faceId: curTopId } = topFaceFromCoreQuat(this.env.cubeQuat());
        const sol = solveCube(fl, { maxTotalMoves: 22, timeoutMs: 180, startTopLetter: FACE_INFO[curTopId].letter });
        this.plan = expandQuarterTurns(sol);
        this.expectedCenters = this._snapCenters();
        if (this.plan.length === 0) {
            this.phase = 'solved';
            return true;
        }
        this._beginNextMove();
        return true;
    }

    _beginNextMove() {
        if (this.plan.length === 0) {
            this.phase = 'solved';
            return;
        }
        const mv = this.plan[0];
        this.targetFaceId = LETTER_TO_FACE[mv[0]];
        this.twistDir = mv.endsWith("'") ? 1.0 : -1.0;
        this.phaseSteps = 0;
        this.settleSteps = 0;
        this.straightenMode = false;
        const { faceId: topId, dotZ } = topFaceFromCoreQuat(this.env.cubeQuat());
        const aligned = this.env.maxFaceMisalignment() < 0.16;
        if (this.completedMoves.length > 0 && topId === this.targetFaceId && dotZ > 0.955 && aligned) {
            this._enterTwist(false);
        } else {
            this.phase = 'reorient';
        }
    }

    _enterTwist(forStraighten = false) {
        const cur = this.env.data.qpos[QPOS_CENTERS + this.targetFaceId];
        if (forStraighten) {
            this.goalAngle = round90(cur);
            this.twistDir = wrapPi(this.goalAngle - cur) >= 0 ? 1.0 : -1.0;
        } else {
            this.goalAngle = round90(cur) + this.twistDir * (Math.PI / 2);
        }
        this.phase = 'twist';
        this.phaseSteps = 0;
        this.settleSteps = 0;
    }

    action(reorientPolicy, twistPolicy) {
        if (this.phase === 'reorient') {
            return reorientPolicy.action(this.env.reorientObs(this.targetFaceId));
        }
        if (this.phase === 'twist') {
            return twistPolicy.action(this.env.twistObs(this.targetFaceId, this.goalAngle, this.twistDir));
        }
        // Hold cradle gently when solved
        const hold = new Float64Array(20);
        for (let i = 0; i < 20; i++) {
            hold[i] = (CRADLE_CTRL[i] - this.env.mid[i]) / this.env.half[i];
        }
        return hold;
    }

    update() {
        if (this.phase === 'solved' || this.phase === 'idle') return null;
        this.phaseSteps++;
        this.totalSteps++;

        if (this.env.isDropped()) {
            return { event: 'dropped' };
        }

        const q = this.env.cubeQuat();
        const nWorld = quatRotateVec(q, FACE_INFO[this.targetFaceId].axis);
        const dotZ = nWorld[2];
        const angv = Math.hypot(this.env.data.qvel[27], this.env.data.qvel[28], this.env.data.qvel[29]);
        const posDist = this.env.cubePosDist();

        if (this.phase === 'reorient') {
            const maxMis = this.env.maxFaceMisalignment();
            if (this.straightenMode) {
                if (maxMis < 0.14 && posDist < 0.055) {
                    this.replan();
                    return { event: 'replan' };
                }
                const dotReq = this.phaseSteps > 80 ? 0.88 : 0.94;
                const posReq = this.phaseSteps > 80 ? 0.060 : 0.048;
                if (dotZ > dotReq && posDist < posReq && angv < 3.0) {
                    this._enterTwist(true);
                    return { event: 'reoriented', face: FACE_INFO[this.targetFaceId].letter };
                }
                if (this.phaseSteps > 160) {
                    this.replan();
                    return { event: 'replan' };
                }
                return null;
            }
            if (this.unjamMode) {
                if (maxMis < 0.14 || this.phaseSteps > 60) {
                    this.replanCount++;
                    this.replan();
                    return { event: 'replan' };
                }
                return null;
            }
            // If a slice became heavily skewed mid-reorient (>0.28 rad), switch to Twist-to-Straighten
            if (maxMis > 0.28 && this.phaseSteps > 40) {
                this.replan();
                return { event: 'replan' };
            }
            // Canonical handover manifold: centered (<4.5 cm), upright (<12.8 deg), dead-still (<2.2 rad/s), faces locked (<7.5 deg)
            const dotReq = this.phaseSteps > 120 ? 0.94 : 0.975;
            const misReq = this.phaseSteps > 120 ? 0.18 : 0.13;
            const angReq = this.phaseSteps > 120 ? 4.0 : 2.2;
            const posReq = this.phaseSteps > 120 ? 0.065 : 0.045;
            if (dotZ > dotReq && maxMis < misReq && angv < angReq && posDist < posReq) {
                this.settleSteps = 0;
                const fl = this.env.facelets();
                if (!fl.includes('?') && fl === 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB') {
                    this.plan = [];
                    this.phase = 'solved';
                    return { event: 'solved' };
                }
                if (this._centersDesynced()) {
                    this.replanCount++;
                    this.replan();
                    return { event: 'replan' };
                }
                this._enterTwist(false);
                return { event: 'reoriented', face: FACE_INFO[this.targetFaceId].letter };
            }
            if (this.phaseSteps > 200) {
                if (dotZ > 0.88 && maxMis < 0.24) {
                    this._enterTwist(false);
                    return { event: 'reoriented', face: FACE_INFO[this.targetFaceId].letter };
                }
                if (maxMis > 0.22) {
                    this.replan();
                    return { event: 'replan' };
                }
                // Static grip equilibrium: momentarily tumble an orthogonal face up to re-grasp
                this.targetFaceId = (this.targetFaceId + 2) % 6;
                this.unjamMode = true;
                this.phaseSteps = 0;
                this.settleSteps = 0;
                return { event: 'replan' };
            }
            return null;
        }

        if (this.phase === 'twist') {
            const cur = this.env.data.qpos[QPOS_CENTERS + this.targetFaceId];
            const rem = Math.abs(wrapPi(this.goalAngle - cur));
            const topVel = Math.abs(this.env.data.qvel[30 + this.targetFaceId]);
            let otherMax = 0;
            for (let i = 0; i < 6; i++) {
                if (i === this.targetFaceId) continue;
                const e = Math.abs(wrap90(this.env.data.qpos[QPOS_CENTERS + i]));
                if (e > otherMax) otherMax = e;
            }
            if (this.straightenMode) {
                if ((rem < 0.14 && topVel < 2.5 && posDist < 0.055) || this.phaseSteps > 140) {
                    this.straightenMode = false;
                    this.consecutiveBails = 0;
                    const fl = this.env.facelets();
                    if (!fl.includes('?') && fl === 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB') {
                        this.plan = [];
                        this.phase = 'solved';
                        return { event: 'solved' };
                    }
                    this.replan();
                    return { event: 'replan' };
                }
                if (dotZ < 0.65 || posDist > 0.072) {
                    this.replan();
                    return { event: 'replan' };
                }
                return null;
            }
            // Canonical handover manifold exit: face settled (<7.5 deg, <1.8 rad/s), cube centered (<4.5 cm) & still (<2.5 rad/s)
            const remReq = this.phaseSteps > 140 ? 0.16 : 0.13;
            const tvReq = this.phaseSteps > 140 ? 2.8 : 1.8;
            const angReq = this.phaseSteps > 140 ? 4.0 : 2.5;
            const posReq = this.phaseSteps > 140 ? 0.065 : 0.045;
            const dotReq = this.phaseSteps > 140 ? 0.90 : 0.955;
            if (rem < remReq && topVel < tvReq && angv < angReq && posDist < posReq && otherMax < 0.15 && dotZ > dotReq) {
                this.settleSteps = 0;
                this.consecutiveBails = 0;
                const doneMove = this.plan.shift();
                this.completedMoves.push(doneMove);
                if (this.expectedCenters) {
                    this.expectedCenters[LETTER_TO_FACE[doneMove[0]]] += doneMove.endsWith("'") ? 1 : -1;
                }
                // Re-verify live physical cube state from MuJoCo!
                const fl = this.env.facelets();
                if (!fl.includes('?') && fl === 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB') {
                    this.plan = [];
                    this.phase = 'solved';
                    return { event: 'solved', move: doneMove };
                }
                if (this.plan.length === 0 || this._centersDesynced()) {
                    this.replanCount++;
                    this.replan();
                    return { event: 'replan', move: doneMove };
                }
                this._beginNextMove();
                return { event: 'twisted', move: doneMove };
            }
            // Early drift + tilt bailout: hand off to Reorient before cube leaves palm grasp!
            if (dotZ < 0.62 || posDist > 0.075 || this.phaseSteps > 220) {
                this.replanCount++;
                this.consecutiveBails++;
                if (this.consecutiveBails >= 6) {
                    // Break rare deterministic bailout loops with a brief orthogonal re-grasp tumble
                    this.consecutiveBails = 0;
                    this.targetFaceId = (this.targetFaceId + 2) % 6;
                    this.unjamMode = true;
                    this.phase = 'reorient';
                    this.phaseSteps = 0;
                    this.settleSteps = 0;
                    return { event: 'replan' };
                }
                this.replan();
                return { event: 'replan' };
            }
        }
        return null;
    }
}
