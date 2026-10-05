// Gymnasium Humanoid-v5 on MuJoCo's WebAssembly build. Shared by the page
// (worker.js) and the Node tools, so both run identical physics. frame_skip 5 (dt 0.015 s); observation = qpos[2:], qvel, cinert[1:],
// cvel[1:], qfrc_actuator[6:], cfrc_ext[1:] (348 numbers); reward = 5 while
// upright + 1.25 · forward speed of the center of mass - 0.1·|a|² - contact cost;
// the episode ends when the torso height leaves [1.0, 2.0] or after 1000 steps.
export class HumanoidEnv {
    constructor(mujoco, xml) {
        this.mj = mujoco;
        this.model = mujoco.MjModel.from_xml_string(xml);
        this.data = new mujoco.MjData(this.model);
        this.frameSkip = 5;
        this.dt = this.model.opt.timestep * this.frameSkip;
        const m = this.model;
        this.nq = m.nq;
        this.nv = m.nv;
        this.nu = m.nu;
        this.nbody = m.nbody;
        this.obsDim = (m.nq - 2) + m.nv + (m.nbody - 1) * 10 + (m.nbody - 1) * 6 + (m.nv - 6) + (m.nbody - 1) * 6;
        this.initQpos = Float64Array.from(m.qpos0);
        this.ctrlLo = Float64Array.from({ length: m.nu }, (_, i) => m.actuator_ctrlrange[2 * i]);
        this.ctrlHi = Float64Array.from({ length: m.nu }, (_, i) => m.actuator_ctrlrange[2 * i + 1]);
        this.mass = Float64Array.from(m.body_mass);
        this.totalMass = this.mass.reduce((a, b) => a + b, 0);
        this.maxSteps = 1000;
        this.friction = m.geom_friction[0];
        this.t = 0;
    }

    // Sliding friction of every geom. MuJoCo uses the larger friction of the two
    // geoms in a contact, so lowering only the floor would change nothing.
    setFriction(mu) {
        const f = this.model.geom_friction;
        for (let i = 0; i < this.model.ngeom; i++) f[3 * i] = mu;
        this.friction = mu;
    }

    // Weaken one leg's motors (hip and knee), e.g. setLegStrength('right', 0.5); 1 restores
    setLegStrength(side, scale) {
        const m = this.model;
        if (!this.gear0) this.gear0 = Float64Array.from({ length: m.nu }, (_, i) => m.actuator_gear[6 * i]);
        for (let i = 0; i < m.nu; i++) {
            const name = m.actuator(i).name;
            const leg = name.startsWith(side + '_hip') || name === side + '_knee';
            m.actuator_gear[6 * i] = this.gear0[i] * (leg ? scale : 1);
        }
        this.legStrength = { side, scale };
    }

    // Extra mass carried on the torso, in kg, with its inertia scaled to match
    setBackpack(kg) {
        const m = this.model, torso = 1;
        if (!this.torso0) this.torso0 = { mass: m.body_mass[torso], inertia: Array.from(m.body_inertia.slice(3 * torso, 3 * torso + 3)) };
        const mass = this.torso0.mass + kg, k = mass / this.torso0.mass;
        m.body_mass[torso] = mass;
        for (let j = 0; j < 3; j++) m.body_inertia[3 * torso + j] = this.torso0.inertia[j] * k;
        // Recompute derived constants on scratch data: mj_setConst overwrites the state it's given
        const scratch = new this.mj.MjData(m);
        this.mj.mj_setConst(m, scratch);
        scratch.delete();
        this.mass = Float64Array.from(m.body_mass);
        this.totalMass = this.mass.reduce((a, b) => a + b, 0);
        this.backpack = kg;
    }

    massCenterX() {
        const xipos = this.data.xipos;
        let x = 0;
        for (let b = 0; b < this.nbody; b++) x += this.mass[b] * xipos[3 * b];
        return x / this.totalMass;
    }

    reset() {
        const { mj, model, data } = this;
        mj.mj_resetData(model, data);
        const qpos = data.qpos, qvel = data.qvel;
        for (let i = 0; i < this.nq; i++) qpos[i] = this.initQpos[i] + (Math.random() * 0.02 - 0.01);
        for (let i = 0; i < this.nv; i++) qvel[i] = Math.random() * 0.02 - 0.01;  // uniform, as in v5
        mj.mj_forward(model, data);
        this.t = 0;
        return this.obs();
    }

    obs() {
        const d = this.data, o = new Float64Array(this.obsDim);
        let k = 0;
        const put = (arr, from, to) => { for (let i = from; i < to; i++) o[k++] = arr[i]; };
        put(d.qpos, 2, this.nq);
        put(d.qvel, 0, this.nv);
        put(d.cinert, 10, 10 * this.nbody);
        put(d.cvel, 6, 6 * this.nbody);
        put(d.qfrc_actuator, 6, this.nv);
        put(d.cfrc_ext, 6, 6 * this.nbody);
        return o;
    }

    step(action) {
        const { mj, model, data } = this;
        const ctrl = data.ctrl;
        let ctrlCost = 0;
        for (let i = 0; i < this.nu; i++) {
            const a = Math.max(this.ctrlLo[i], Math.min(this.ctrlHi[i], action[i]));
            ctrl[i] = a;
            ctrlCost += a * a;
        }
        const x0 = this.massCenterX();
        for (let k = 0; k < this.frameSkip; k++) mj.mj_step(model, data);
        // Contact forces (cfrc_ext) aren't computed by mj_step; Gymnasium does this too
        mj.mj_rnePostConstraint(model, data);
        const forward = (this.massCenterX() - x0) / this.dt;
        const z = data.qpos[2];
        const healthy = z >= 1.0 && z <= 2.0;
        let contact = 0;
        const f = data.cfrc_ext;
        for (let i = 0; i < 6 * this.nbody; i++) contact += f[i] * f[i];
        contact = Math.min(5e-7 * contact, 10);
        this.t++;
        return {
            obs: this.obs(),
            reward: (healthy ? 5 : 0) + 1.25 * forward - 0.1 * ctrlCost - contact,
            forward,
            terminated: !healthy,
            truncated: healthy && this.t >= this.maxSteps
        };
    }
}

