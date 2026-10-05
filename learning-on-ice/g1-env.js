// Unitree G1 humanoid (MuJoCo Menagerie, BSD-3-Clause) on MuJoCo's WebAssembly
// build. Shared by the page (worker.js) and the Node tools, so both run
// identical physics.
//
// Control at 50 Hz (5 physics steps of 4 ms; Menagerie's default is 2 ms). Only the
// robot-floor contacts are simulated, not the robot's parts against each other.
// Both roughly triple the speed; walking gaits rarely need self-collisions.
//
// Set up like Unitree's own walking controller (unitree_rl_gym), so its policy can
// teach ours: the agent drives the 12 leg joints, whose motors use Unitree's PD
// gains, with targets = Unitree's default crouch + 0.25 · action (actions clipped
// to ±4). The waist and arms hold Menagerie's standing pose.
// Observation (36): pelvis height, gravity direction, linear and angular velocity
// (all in the pelvis frame, so they don't depend on which way it faces), 12 leg
// joint angles and velocities, and a gait clock (sin, cos of a 0.8 s cycle).
// Reward: 1 for staying up + 2 · exp(-(v - 0.8)² / 0.25) for walking forward near
// 0.8 m/s in its own frame - (sideways speed)² - 2 · (pelvis tilt)², where tilt is
// the horizontal part of the gravity direction (sin of the tilt angle).
// The episode ends if the pelvis drops below 0.5 m, rises above 1.0 m, or tilts
// more than 60°, or after 1000 steps (20 s).
const LEG_KP = [100, 100, 100, 150, 40, 40], LEG_KD = [2, 2, 2, 4, 2, 2];
const LEG_DEFAULT = [-0.1, 0, 0, 0.3, -0.2, 0];   // hip pitch, roll, yaw, knee, ankle pitch, roll
const ACTION_SCALE = 0.25, ACTION_LIMIT = 4, CLOCK_PERIOD = 0.8;

// Rotate world vector v into the frame of quaternion q = (w, x, y, z)
function toBody(q, v) {
    const [w, x, y, z] = q, [a, b, c] = [-x, -y, -z];
    const t = [2 * (b * v[2] - c * v[1]), 2 * (c * v[0] - a * v[2]), 2 * (a * v[1] - b * v[0])];
    return [v[0] + w * t[0] + (b * t[2] - c * t[1]),
            v[1] + w * t[1] + (c * t[0] - a * t[2]),
            v[2] + w * t[2] + (a * t[1] - b * t[0])];
}

export class G1Env {
    // files: { 'scene.xml': text, 'g1.xml': text, 'pelvis.STL': Uint8Array, ... }
    constructor(mujoco, files, { healthy = 1.0, forwardWeight = 2.0, targetSpeed = 0.8, lateralCost = 1.0, tiltCost = 2.0 } = {}) {
        this.mj = mujoco;
        this.vfs = new mujoco.MjVFS();
        const enc = new TextEncoder();
        for (const [name, data] of Object.entries(files)) {
            if (name !== 'scene.xml') this.vfs.addBuffer(name, typeof data === 'string' ? enc.encode(data) : data);
        }
        this.model = mujoco.MjModel.from_xml_string(files['scene.xml'], this.vfs);
        this.data = new mujoco.MjData(this.model);
        const m = this.model;
        m.opt.timestep = 0.004;
        for (let g = 0; g < m.ngeom; g++) {
            if (m.geom_bodyid[g] !== 0) { m.geom_contype[g] = 1; m.geom_conaffinity[g] = 0; }  // robot touches only the floor
        }
        // Actuators 0-11 are the legs (left then right), in Unitree's joint order
        for (let i = 0; i < 12; i++) {
            m.actuator_gainprm[10 * i] = LEG_KP[i % 6];
            m.actuator_biasprm[10 * i + 1] = -LEG_KP[i % 6];
            m.actuator_biasprm[10 * i + 2] = -LEG_KD[i % 6];
        }
        this.frameSkip = 5;
        this.dt = m.opt.timestep * this.frameSkip;
        this.nq = m.nq;
        this.nv = m.nv;
        this.nu = m.nu;
        this.actDim = 12;
        this.obsDim = 1 + 3 + 3 + 3 + 12 + 12 + 2;
        this.maxSteps = 1000;
        this.hp = { healthy, forwardWeight, targetSpeed, lateralCost, tiltCost };
        // Default pose per actuator: Unitree's crouch for the legs, the "stand" keyframe elsewhere
        this.center = Float64Array.from({ length: m.nu }, (_, i) =>
            i < 12 ? LEG_DEFAULT[i % 6] : m.key_qpos[m.jnt_qposadr[m.actuator_trnid[2 * i]]]);
        this.ctrlLo = Float64Array.from({ length: m.nu }, (_, i) => m.actuator_ctrlrange[2 * i]);
        this.ctrlHi = Float64Array.from({ length: m.nu }, (_, i) => m.actuator_ctrlrange[2 * i + 1]);
        this.friction = 1;
        this.t = 0;
    }

    // Sliding friction of every geom. MuJoCo uses the larger friction of two touching
    // geoms (or the higher-priority one: G1's feet have priority), so set them all.
    setFriction(mu) {
        const f = this.model.geom_friction;
        for (let i = 0; i < this.model.ngeom; i++) f[3 * i] = mu;
        this.friction = mu;
    }

    // Injured leg: one leg's hip, knee and ankle motors deliver at most `scale` of their torque
    setLegStrength(side, scale) {
        const m = this.model;
        if (!this.frc0) this.frc0 = Float64Array.from(m.jnt_actfrcrange);
        for (let j = 0; j < m.njnt; j++) {
            const name = m.jnt(j).name;
            const leg = ['hip', 'knee', 'ankle'].some(part => name.startsWith(`${side}_${part}`));
            m.jnt_actfrcrange[2 * j] = this.frc0[2 * j] * (leg ? scale : 1);
            m.jnt_actfrcrange[2 * j + 1] = this.frc0[2 * j + 1] * (leg ? scale : 1);
        }
        this.legStrength = { side, scale };
    }

    // Extra mass carried on the torso, in kg, with its inertia scaled to match
    setBackpack(kg) {
        const m = this.model;
        const torso = [...Array(m.nbody).keys()].find(b => m.body(b).name === 'torso_link');
        if (!this.torso0) this.torso0 = { mass: m.body_mass[torso], inertia: Array.from(m.body_inertia.slice(3 * torso, 3 * torso + 3)) };
        const mass = this.torso0.mass + kg, k = mass / this.torso0.mass;
        m.body_mass[torso] = mass;
        for (let j = 0; j < 3; j++) m.body_inertia[3 * torso + j] = this.torso0.inertia[j] * k;
        // Recompute derived constants on scratch data: mj_setConst overwrites the state it's given
        const scratch = new this.mj.MjData(m);
        this.mj.mj_setConst(m, scratch);
        scratch.delete();
        this.backpack = kg;
    }

    reset() {
        const { mj, model, data } = this;
        mj.mj_resetDataKeyframe(model, data, 0);
        const qpos = data.qpos, qvel = data.qvel;
        for (let i = 7; i < this.nq; i++) qpos[i] += Math.random() * 0.04 - 0.02;
        for (let i = 0; i < this.nv; i++) qvel[i] = Math.random() * 0.04 - 0.02;
        mj.mj_forward(model, data);
        this.t = 0;
        return this.obs();
    }

    // Gait clock: sin and cos of the phase in a 0.8 s cycle
    clock() {
        const p = 2 * Math.PI * ((this.t * this.dt) % CLOCK_PERIOD) / CLOCK_PERIOD;
        return [Math.sin(p), Math.cos(p)];
    }

    obs() {
        const d = this.data, q = d.qpos, v = d.qvel, o = new Float64Array(this.obsDim);
        const quat = [q[3], q[4], q[5], q[6]];
        const parts = [[q[2]], toBody(quat, [0, 0, -1]), toBody(quat, [v[0], v[1], v[2]]), [v[3], v[4], v[5]]];
        let k = 0;
        for (const p of parts) for (const x of p) o[k++] = x;
        for (let i = 7; i < 19; i++) o[k++] = q[i];
        for (let i = 6; i < 18; i++) o[k++] = v[i];
        for (const x of this.clock()) o[k++] = x;
        return o;
    }

    step(action) {
        const { mj, model, data, hp } = this;
        const ctrl = data.ctrl;
        for (let i = 0; i < this.nu; i++) {
            const a = i < 12 ? Math.max(-ACTION_LIMIT, Math.min(ACTION_LIMIT, action[i])) : 0;
            ctrl[i] = Math.max(this.ctrlLo[i], Math.min(this.ctrlHi[i], this.center[i] + ACTION_SCALE * a));
        }
        const x0 = data.qpos[0], y0 = data.qpos[1];
        for (let k = 0; k < this.frameSkip; k++) mj.mj_step(model, data);
        const q = data.qpos, z = q[2];
        // Pelvis facing (body x axis) in the ground plane, from the quaternion (w, x, y, z)
        const fx = 1 - 2 * (q[5] * q[5] + q[6] * q[6]), fy = 2 * (q[4] * q[5] + q[3] * q[6]);
        const norm = Math.hypot(fx, fy) || 1, hx = fx / norm, hy = fy / norm;
        const vx = (q[0] - x0) / this.dt, vy = (q[1] - y0) / this.dt;
        const forward = vx * hx + vy * hy, lateral = -vx * hy + vy * hx;
        // Pelvis up-vector z component from the quaternion (w, x, y, z): 1 - 2(x² + y²)
        const upZ = 1 - 2 * (q[4] * q[4] + q[5] * q[5]);
        const healthy = z >= 0.5 && z <= 1.0 && upZ >= 0.5;
        const g = toBody([q[3], q[4], q[5], q[6]], [0, 0, -1]), tilt2 = g[0] * g[0] + g[1] * g[1];
        this.t++;
        return {
            obs: this.obs(),
            reward: (healthy ? hp.healthy : 0) + hp.forwardWeight * Math.exp(-((forward - hp.targetSpeed) ** 2) / 0.25)
                - hp.lateralCost * lateral * lateral - hp.tiltCost * tilt2,
            forward,
            terminated: !healthy,
            truncated: healthy && this.t >= this.maxSteps
        };
    }
}
