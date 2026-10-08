// Franka Panda picking up a box and throwing it, on MuJoCo's WebAssembly build.
// Shared by the page and tools/eval.mjs, so both run identical code.
//
// Picking mirrors MuJoCo Playground's PandaPickCube (pick.py) with the changes in
// training/panda_live.py: same model, 50 Hz control (4 physics steps of 5 ms), and
// each action nudges the 8 position targets (7 joints + gripper) by up to 0.04.
// Observation (66): qpos, qvel, gripper position and orientation, box orientation,
// box - gripper, target - box, target orientation - box orientation (the target is
// never turned), and position target - joint position.
const ACTION_SCALE = 0.04;
export const SUBSTEPS = 4;                             // physics steps per control step
export const CENTER = [0.5, 0, 0.03];                  // box start in the home keyframe
export const BOX_RANGE = 0.2;                          // box x, y within ±0.2 m of CENTER
export const TARGET_RANGE = 0.2, TARGET_Z = [0.2, 0.4]; // target: ±0.2 m, 0.2-0.4 m above CENTER
const S = Math.SQRT1_2;
const FACES = [[1, 0, 0, 0], [0, 1, 0, 0], [S, S, 0, 0], [S, -S, 0, 0], [S, 0, S, 0], [S, 0, -S, 0]];

function quatMul(a, b) {
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
}

// Small seeded generator, so evaluations repeat exactly
export function makeRandom(seed = (Math.random() * 2 ** 32) >>> 0) {
    let s = seed >>> 0;
    const uniform = () => {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
    return { uniform, normal, range: (lo, hi) => lo + (hi - lo) * uniform() };
}

export class PandaEnv {
    // files: { 'mjx_single_cube.xml': text, ..., 'meshes/link0_0.stl': Uint8Array }
    constructor(mujoco, files, random = makeRandom()) {
        this.mj = mujoco;
        this.random = random;
        this.vfs = new mujoco.MjVFS();
        const enc = new TextEncoder();
        for (const [name, data] of Object.entries(files)) {
            this.vfs.addBuffer(name, typeof data === 'string' ? enc.encode(data) : data);
        }
        const m = this.model = mujoco.MjModel.from_xml_string(files['mjx_single_cube.xml'], this.vfs);
        this.data = new mujoco.MjData(m);
        const id = (type, name) => mujoco.mj_name2id(m, mujoco.mjtObj[type].value, name);
        this.boxBody = id('mjOBJ_BODY', 'box');
        this.gripperSite = id('mjOBJ_SITE', 'gripper');
        this.mocap = m.body_mocapid[id('mjOBJ_BODY', 'mocap_target')];
        const boxJoint = m.body_jntadr[this.boxBody];
        this.boxQpos = m.jnt_qposadr[boxJoint];
        this.boxDof = m.jnt_dofadr[boxJoint];
        this.home = id('mjOBJ_KEY', 'home');
        this.nu = m.nu;
        this.lo = Float64Array.from({ length: m.nu }, (_, i) => m.actuator_ctrlrange[2 * i]);
        this.hi = Float64Array.from({ length: m.nu }, (_, i) => m.actuator_ctrlrange[2 * i + 1]);
        this.jointLo = Float64Array.from({ length: 7 }, (_, j) => m.jnt_range[2 * j]);
        this.jointHi = Float64Array.from({ length: 7 }, (_, j) => m.jnt_range[2 * j + 1]);
        this.dt = m.opt.timestep * SUBSTEPS;
        this.target = [...CENTER];
        this._obs = new Float64Array(66);
    }

    // Home pose with each arm joint moved by up to `jitter` radians, a new box and target
    reset(jitter = 0) {
        const { mj, model: m, data: d, random } = this;
        mj.mj_resetDataKeyframe(m, d, this.home);
        for (let j = 0; j < 7; j++) {
            const q = Math.min(this.jointHi[j], Math.max(this.jointLo[j], d.qpos[j] + jitter * random.range(-1, 1)));
            d.qpos[j] = q;
            d.ctrl[j] = q;
        }
        this.placeBox(...this.sampleBox());
        this.setTarget(this.sampleTarget());
        mj.mj_forward(m, d);
    }

    // Resting on a random face at a random yaw, or (half the time) dropped from 5-25 cm turned any way
    sampleBox() {
        const r = this.random;
        const x = CENTER[0] + r.range(-BOX_RANGE, BOX_RANGE), y = CENTER[1] + r.range(-BOX_RANGE, BOX_RANGE);
        if (r.uniform() < 0.5) {
            const q = [r.normal(), r.normal(), r.normal(), r.normal()], n = Math.hypot(...q);
            return [[x, y, 0.03 + r.range(0.05, 0.25)], q.map(v => v / n)];
        }
        const yaw = r.range(-Math.PI, Math.PI), face = Math.floor(r.uniform() * 6);
        return [[x, y, face < 2 ? 0.03 : 0.02], quatMul([Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], FACES[face])];
    }

    sampleTarget() {
        const r = this.random;
        return [CENTER[0] + r.range(-TARGET_RANGE, TARGET_RANGE), CENTER[1] + r.range(-TARGET_RANGE, TARGET_RANGE),
                CENTER[2] + r.range(TARGET_Z[0], TARGET_Z[1])];
    }

    placeBox(pos, quat) {
        const d = this.data, a = this.boxQpos;
        for (let i = 0; i < 3; i++) d.qpos[a + i] = pos[i];
        for (let i = 0; i < 4; i++) d.qpos[a + 3 + i] = quat[i];
        for (let i = 0; i < 6; i++) d.qvel[this.boxDof + i] = 0;
    }

    setTarget(pos) {
        this.target = [...pos];
        for (let i = 0; i < 3; i++) this.data.mocap_pos[3 * this.mocap + i] = pos[i];
    }

    boxPos() {
        const p = this.data.xpos, b = 3 * this.boxBody;
        return [p[b], p[b + 1], p[b + 2]];
    }

    gripperPos() {
        const p = this.data.site_xpos, s = 3 * this.gripperSite;
        return [p[s], p[s + 1], p[s + 2]];
    }

    // Distance from the box to the target, in meters
    error() {
        const b = this.boxPos();
        return Math.hypot(this.target[0] - b[0], this.target[1] - b[1], this.target[2] - b[2]);
    }

    obs() {
        const d = this.data, o = this._obs;
        let k = 0;
        for (let i = 0; i < this.model.nq; i++) o[k++] = d.qpos[i];
        for (let i = 0; i < this.model.nv; i++) o[k++] = d.qvel[i];
        const g = this.gripperPos(), b = this.boxPos();
        for (let i = 0; i < 3; i++) o[k++] = g[i];
        const sm = d.site_xmat, s = 9 * this.gripperSite;
        for (let i = 3; i < 9; i++) o[k++] = sm[s + i];
        const bm = d.xmat, m = 9 * this.boxBody;
        for (let i = 3; i < 9; i++) o[k++] = bm[m + i];
        for (let i = 0; i < 3; i++) o[k++] = b[i] - g[i];
        for (let i = 0; i < 3; i++) o[k++] = this.target[i] - b[i];
        const eye = [1, 0, 0, 0, 1, 0];
        for (let i = 0; i < 6; i++) o[k++] = eye[i] - bm[m + i];
        for (let i = 0; i < 8; i++) o[k++] = d.ctrl[i] - d.qpos[i];  // 7 arm joints, then finger_joint1
        return o;
    }

    // The throw policy's observation (59), as in training/panda_throw.py: no target,
    // but the direction to throw in, as an angle around the base
    throwObs(heading) {
        const d = this.data, o = this._throwObs ??= new Float64Array(59);
        let k = 0;
        for (let i = 0; i < this.model.nq; i++) o[k++] = d.qpos[i];
        for (let i = 0; i < this.model.nv; i++) o[k++] = d.qvel[i];
        const g = this.gripperPos(), b = this.boxPos();
        for (let i = 0; i < 3; i++) o[k++] = g[i];
        const sm = d.site_xmat, s = 9 * this.gripperSite;
        for (let i = 3; i < 9; i++) o[k++] = sm[s + i];
        const bm = d.xmat, m = 9 * this.boxBody;
        for (let i = 3; i < 9; i++) o[k++] = bm[m + i];
        for (let i = 0; i < 3; i++) o[k++] = b[i] - g[i];
        for (let i = 0; i < 8; i++) o[k++] = d.ctrl[i] - d.qpos[i];
        o[k++] = Math.cos(heading);
        o[k++] = Math.sin(heading);
        return o;
    }

    // One control step: nudge the position targets, then 4 physics steps
    step(action) {
        this.act(action);
        for (let k = 0; k < SUBSTEPS; k++) this.substep();
    }

    act(action) {
        const d = this.data;
        for (let i = 0; i < this.nu; i++) {
            d.ctrl[i] = Math.min(this.hi[i], Math.max(this.lo[i], d.ctrl[i] + ACTION_SCALE * action[i]));
        }
    }

    // One 5 ms physics step. The page sets `pull` to drag the box with the mouse.
    substep() {
        this.pull?.();
        this.mj.mj_step(this.model, this.data);
    }
}

// The page's loop. The pick policy lifts the box (carrying it toward a random target, as
// when the throw policy's start states were collected); once the box has stayed
// within 5 cm of the gripper and 10 cm up for 0.2 s, the throw policy takes over.
// A throw ends when the box first touches the floor after leaving the gripper, and
// its length is how far that point is from the robot's base along the aim direction.
export const HANDOVER = { held: 0.05, lifted: 0.10, steps: 10 };
export const RELEASED = 0.08, LANDED_Z = 0.045, PICK_TIME = 12, THROW_TIME = 3;
const STILL = new Float64Array(8);

export class Throws {
    constructor(env) {
        this.env = env;
        this.heading = 0;
        this.reset();
    }

    reset() {
        this.phase = 'pick';   // then 'throw', then 'landed' (the arm holds still)
        this.time = 0;
        this.heldSteps = 0;
        this.released = false;
    }

    // A new box and a new (hidden) target to carry it toward
    next() {
        const env = this.env;
        env.placeBox(...env.sampleBox());
        env.setTarget(env.sampleTarget());
        this.reset();
    }

    action(pick, thrower) {
        const env = this.env;
        if (this.phase === 'landed') return STILL;   // the throw policy never trained past the landing
        return this.phase === 'throw' ? thrower.action(env.throwObs(this.heading)) : pick.action(env.obs());
    }

    // After each control step. Returns null, { event: 'handover' },
    // { event: 'landed', distance, point } or { event: 'missed' }.
    update() {
        const env = this.env, b = env.boxPos(), g = env.gripperPos();
        if (this.phase === 'landed') return null;
        this.time += env.dt;
        const apart = Math.hypot(b[0] - g[0], b[1] - g[1], b[2] - g[2]);
        if (this.phase === 'pick') {
            this.heldSteps = apart < HANDOVER.held && b[2] > HANDOVER.lifted ? this.heldSteps + 1 : 0;
            if (this.heldSteps >= HANDOVER.steps) {
                this.phase = 'throw';
                this.time = 0;
                return { event: 'handover' };
            }
            return this.time >= PICK_TIME ? { event: 'missed' } : null;
        }
        this.released ||= apart > RELEASED;
        if (this.released && b[2] < LANDED_Z) {
            this.phase = 'landed';
            const distance = b[0] * Math.cos(this.heading) + b[1] * Math.sin(this.heading);
            return { event: 'landed', distance, point: [b[0], b[1]] };
        }
        return this.time >= THROW_TIME ? { event: 'missed' } : null;
    }
}
