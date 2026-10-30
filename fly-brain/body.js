// The whole-brain model in a tethered fly: flybody's proboscis (MuJoCo) driven by the model's
// proboscis motor neurons, and a drop whose contact with the labellum drives taste neurons.
// Shared by worker.js (page) and tools/feed.mjs (Node), so both run the same loop.
//
// Brain -> body. Each joint has the muscles McKellar et al. 2020 found move it, named by their
// motor neurons (BANC names them; tools/export_neurons.py). A muscle's activation is
// 1 - exp(-r / R_HALF) of its motor neurons' mean rate r, low-passed with TAU_MUSCLE; the joint's
// position actuator (flybody's, unchanged) aims at its range limit times the activation, extensor
// minus flexor. R_HALF and TAU_MUSCLE are our choices: no recordings give rate -> force for these
// muscles.
// Body -> brain. While the drop touches either labellar lobe, the drop's taste set (Shiu et al.'s
// neurons) gets Poisson input at the page's input rate; when it stops touching, the input stops.

export const STEP = 1e-4;          // s; the brain model's dt and flybody's timestep
export const CHUNK = 50;           // steps between brain <-> body exchanges (5 ms)
export const TAU_MUSCLE = 0.03;    // s
export const R_HALF = 40;          // Hz for activation 1 - 1/e

// joint: [extensor motor neurons, flexor motor neurons] (McKellar et al. 2020, eLife 9:e54978)
export const MUSCLES = {
    rostrum: [['MN9'], ['MN1', 'MN2Da', 'MN2Db', 'MN2V']],      // protractor m9; retractors m1, m2
    haustellum: [['MN4a', 'MN4b'], ['MN3L', 'MN3M']],           // extensor m4; flexor m3
    labrum_left: [['MN7'], []],                                  // m7 opens (abducts) the labellum
    labrum_right: [['MN7'], []],
};
// in flybody, the rostrum protracts and the haustellum extends toward negative angles
const EXTEND_SIGN = { rostrum: -1, haustellum: -1, labrum_left: 1, labrum_right: 1 };

export class FlyBody {
    // files: { 'fly.xml': text, 'meshes/x.stl': Uint8Array, ... }
    constructor(mujoco, files) {
        this.mj = mujoco;
        const vfs = new mujoco.MjVFS();
        const enc = new TextEncoder();
        for (const [name, data] of Object.entries(files)) vfs.addBuffer(name, typeof data === 'string' ? enc.encode(data) : data);
        const m = this.model = mujoco.MjModel.from_xml_string(files['fly.xml'], vfs);
        this.data = new mujoco.MjData(m);
        const id = (type, name) => mujoco.mj_name2id(m, mujoco.mjtObj[type].value, name);
        this.mocap = m.body_mocapid[id('mjOBJ_BODY', 'wick')];
        this.drop = id('mjOBJ_GEOM', 'drop');
        this.lobes = ['labrum_left_lower_collision', 'labrum_right_lower_collision'].map(n => id('mjOBJ_GEOM', n));
        this.joints = Object.keys(MUSCLES).map(name => {
            const a = id('mjOBJ_ACTUATOR', name);
            const lo = m.actuator_ctrlrange[2 * a], hi = m.actuator_ctrlrange[2 * a + 1];
            return { name, a, qpos: m.jnt_qposadr[id('mjOBJ_JOINT', name)], ext: EXTEND_SIGN[name] < 0 ? lo : hi, flex: EXTEND_SIGN[name] < 0 ? hi : lo };
        });
        this.away = [0.3, 0, -0.15];
        this.setWick(this.away);
    }

    setWick(p) {
        const mp = this.data.mocap_pos;
        for (let k = 0; k < 3; k++) mp[3 * this.mocap + k] = p[k];
    }

    // activation per joint: { rostrum: [extensor, flexor], ... }
    setMuscles(act) {
        const c = this.data.ctrl;
        for (const j of this.joints) {
            const [e, f] = act[j.name];
            c[j.a] = e * j.ext + f * j.flex;
        }
    }

    step(n) {
        for (let k = 0; k < n; k++) this.mj.mj_step(this.model, this.data);
    }

    // does the drop touch a labellar lobe?
    touching() {
        for (const g of this.lobes) {
            if (this.mj.mj_geomDistance(this.model, this.data, this.drop, g, 0.01, null) <= 0) return true;
        }
        return false;
    }

    angles() {
        return Object.fromEntries(this.joints.map(j => [j.name, this.data.qpos[j.qpos]]));
    }
}

// One brain + one body. brain: engine.wasm's exports after init(); motor: neurons.json 'motor';
// tastes: { sugar: [neuron indices], ... }
export class Loop {
    constructor(brain, body, motor, tastes) {
        this.brain = brain;
        this.body = body;
        this.tastes = tastes;
        this.taste = null;          // the drop's taste, or null for no drop
        this.hz = 150;
        this.touch = false;
        this.groups = {};           // motor neuron type -> { cells, rate }
        this.of = new Map();        // neuron -> group
        for (const [joint, [ext, flex]] of Object.entries(MUSCLES)) {
            for (const t of [...ext, ...flex]) {
                if (this.groups[t]) continue;
                const g = this.groups[t] = { cells: motor[t], rate: 0, count: 0 };
                for (const i of motor[t]) this.of.set(i, g);
            }
        }
        this.act = Object.fromEntries(Object.keys(MUSCLES).map(j => [j, [0, 0]]));
    }

    setTaste(taste) {
        this.input(false);
        this.taste = taste;
        this.touch = false;
    }

    // after the brain and body are reset: no input, muscles at rest
    reset() {
        this.input(false);
        this.touch = false;
        for (const g of Object.values(this.groups)) g.rate = 0;
    }

    input(on) {
        if (!this.taste) return;
        for (const i of this.tastes[this.taste]) this.brain.set_rate(i, on ? this.hz : 0);
    }

    // 5 ms of brain, then 5 ms of body; returns the brain's spikes as (step, neuron) pairs
    chunk() {
        const k = this.brain.run(CHUNK);
        const spikes = new Uint32Array(this.brain.memory.buffer, this.brain.spikes(), 2 * k).slice();
        for (const g of Object.values(this.groups)) g.count = 0;
        for (let s = 1; s < spikes.length; s += 2) {
            const g = this.of.get(spikes[s]);
            if (g) g.count++;
        }
        const dt = CHUNK * STEP, decay = Math.exp(-dt / TAU_MUSCLE);
        for (const g of Object.values(this.groups)) g.rate = g.rate * decay + (1 - decay) * g.count / g.cells.length / dt;
        const a = types => types.length ? 1 - Math.exp(-types.reduce((s, t) => s + this.groups[t].rate, 0) / types.length / R_HALF) : 0;
        for (const [joint, [ext, flex]] of Object.entries(MUSCLES)) this.act[joint] = [a(ext), a(flex)];
        this.body.setMuscles(this.act);
        this.body.step(CHUNK);
        const touch = this.taste !== null && this.body.touching();
        if (touch !== this.touch) {
            this.touch = touch;
            this.input(touch);
        }
        return spikes;
    }

    rates() {
        return Object.fromEntries(Object.entries(this.groups).map(([t, g]) => [t, g.rate]));
    }
}
