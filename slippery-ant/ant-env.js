// Gymnasium Ant-v4 on MuJoCo's official WebAssembly build. Shared by the page
// (worker.js) and the Node pretraining tool, so both run identical physics.
//
// frame_skip 5 (dt 0.05 s), observation = qpos[2:] + qvel (27 numbers),
// reward = 1 (healthy) + forward x-velocity - 0.5·|a|², the episode ends when
// the torso height leaves [0.2, 1.0] (terminated) or after 1000 steps (truncated).
export class AntEnv {
    constructor(mujoco, xml) {
        this.mj = mujoco;
        this.model = mujoco.MjModel.from_xml_string(xml);
        this.data = new mujoco.MjData(this.model);
        this.frameSkip = 5;
        this.dt = this.model.opt.timestep * this.frameSkip;
        this.nq = this.model.nq;
        this.nv = this.model.nv;
        this.nu = this.model.nu;
        this.obsDim = this.nq - 2 + this.nv;
        this.initQpos = Float64Array.from(this.model.qpos0);
        this.maxSteps = 1000;
        this.friction = this.model.geom_friction[0];
        this.t = 0;
    }

    // Sliding friction of every geom. MuJoCo uses the larger friction of the two
    // geoms in a contact, so lowering only the floor would change nothing.
    setFriction(mu) {
        const f = this.model.geom_friction;
        for (let i = 0; i < this.model.ngeom; i++) f[3 * i] = mu;
        this.friction = mu;
    }

    reset() {
        const { mj, model, data } = this;
        mj.mj_resetData(model, data);
        const qpos = data.qpos, qvel = data.qvel;
        for (let i = 0; i < this.nq; i++) qpos[i] = this.initQpos[i] + (Math.random() * 0.2 - 0.1);
        for (let i = 0; i < this.nv; i++) qvel[i] = gauss() * 0.1;
        mj.mj_forward(model, data);
        this.t = 0;
        return this.obs();
    }

    obs() {
        const o = new Float64Array(this.obsDim), qpos = this.data.qpos, qvel = this.data.qvel;
        for (let i = 2; i < this.nq; i++) o[i - 2] = qpos[i];
        for (let i = 0; i < this.nv; i++) o[this.nq - 2 + i] = qvel[i];
        return o;
    }

    step(action) {
        const { mj, model, data } = this;
        const ctrl = data.ctrl;
        let ctrlCost = 0;
        for (let i = 0; i < this.nu; i++) {
            const a = Math.max(-1, Math.min(1, action[i]));
            ctrl[i] = a;
            ctrlCost += a * a;
        }
        const x0 = data.qpos[0];
        for (let k = 0; k < this.frameSkip; k++) mj.mj_step(model, data);
        const forward = (data.qpos[0] - x0) / this.dt;
        const z = data.qpos[2];
        const healthy = z >= 0.2 && z <= 1.0;
        this.t++;
        return {
            obs: this.obs(),
            reward: 1.0 + forward - 0.5 * ctrlCost,
            forward,
            terminated: !healthy,
            truncated: healthy && this.t >= this.maxSteps
        };
    }
}

export function gauss() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}
