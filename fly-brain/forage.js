// The whole-brain model in a walking fly, on the way to a drop. Shared by worker.js and tools/forage.mjs.
//
// What comes from the connectome model:
//   walking drive  P9 (DNp09, a walking command neuron) gets Poisson input from the page; oDN1 (DNg97) follows.
//   stopping       sugar on the labellum cuts oDN1 (Sapkal et al. 2024), so the fly stops.
//   feeding        sugar -> MN9 and the other proboscis motor neurons -> flybody's mouth joints.
// What does not:
//   the legs       flybody's trained policy (walker.js) follows a speed and a turn rate.
//   oDN1 -> speed  our choice (SPEED_* below).
//   the turning    a rule, not the brain: turn toward the antenna that smells more (flies do this, Gaudry et
//                  al. 2013, but in this model odor input ignites the whole brain and cannot steer).
import { Walker, CONTROL_DT } from './walker.js';

// Brain -> mouth. Each joint has the muscles McKellar et al. 2020 (eLife 9:e54978) found move it, named by their
// motor neurons (BANC names them; tools/export_neurons.py). A muscle's activation is 1 - exp(-r / R_HALF) of its motor
// neurons' mean rate r, low-passed with TAU_MUSCLE; the joint's position actuator aims at its range limit times the
// activation, extensor minus flexor. Both constants are our choices: no recordings give rate -> force here.
export const TAU_MUSCLE = 0.03;     // s
export const R_HALF = 40;           // Hz for activation 1 - 1/e
export const MUSCLES = {            // joint: [extensor motor neurons, flexor motor neurons]
    rostrum: [['MN9'], ['MN1', 'MN2Da', 'MN2Db', 'MN2V']],      // protractor m9; retractors m1, m2
    haustellum: [['MN4a', 'MN4b'], ['MN3L', 'MN3M']],           // extensor m4; flexor m3
    labrum_left: [['MN7'], []],                                  // m7 opens (abducts) the labellum
    labrum_right: [['MN7'], []],
};

// in flybody, the rostrum protracts and the haustellum extends toward negative angles
const EXTEND_SIGN = { rostrum: -1, haustellum: -1, labrum_left: 1, labrum_right: 1 };
export const BRAIN_STEPS = 20;      // brain steps (0.1 ms) per control step
export const SPEED_MAX = 2;         // cm/s: the walking policy's range
export const SPEED_R0 = 6;          // Hz of oDN1: below this, stand
export const SPEED_R1 = 12;         // Hz of oDN1: at this and above, full speed
const TAU_DRIVE = 0.1;              // s, low-pass on oDN1
const ODOR_LENGTH = 1;              // cm: concentration ~ exp(-distance / ODOR_LENGTH)
const TURN_GAIN = 1500;             // rad/s per unit (L - R) / (L + R)
const TURN_MAX = 4;                 // rad/s: the policy tracks up to here
const SHARP_SPEED = 1;              // cm/s in a sharp turn
const TURN_MIN_SPEED = 0.8;         // cm/s: below this, no turning (it topples)
const TAU_TURN = 0.05;              // s
const RELEASE = 0.03;               // cm
export const DROP_Z = 0;            // a dome: half of it above the floor

export class Forager {
    // files: fly.xml + meshes by bare name; meta, policy: walker.js; brain: engine exports after init;
    // neurons: neurons.json; tastes: { sugar: [...], ... }; walkSet: P9, walkOut: oDN1
    constructor(mujoco, files, meta, policy, brain, motor, tastes, walkSet, walkOut) {
        Object.assign(this, { mj: mujoco, brain, tastes, walkSet, meta });
        const vfs = new mujoco.MjVFS();
        const enc = new TextEncoder();
        for (const [name, d] of Object.entries(files)) vfs.addBuffer(name, typeof d === 'string' ? enc.encode(d) : d);
        const m = this.model = mujoco.MjModel.from_xml_string(files['fly.xml'], vfs);
        const d = this.data = new mujoco.MjData(m);
        this.walker = new Walker(mujoco, m, d, meta, policy);
        const id = (t, n) => mujoco.mj_name2id(m, mujoco.mjtObj[t].value, n);
        this.mocap = m.body_mocapid[id('mjOBJ_BODY', 'drop')];
        this.dropGeom = id('mjOBJ_GEOM', 'drop');
        this.lobes = ['labrum_left_lower_collision', 'labrum_right_lower_collision'].map(n => id('mjOBJ_GEOM', 'walker/' + n));
        this.antennae = ['antenna_left', 'antenna_right'].map(n => id('mjOBJ_BODY', 'walker/' + n));
        this.joints = Object.keys(MUSCLES).map(name => {
            const a = id('mjOBJ_ACTUATOR', 'walker/' + name);
            const lo = m.actuator_ctrlrange[2 * a], hi = m.actuator_ctrlrange[2 * a + 1];
            return { name, a, qpos: m.jnt_qposadr[id('mjOBJ_JOINT', 'walker/' + name)], ext: EXTEND_SIGN[name] < 0 ? lo : hi, flex: EXTEND_SIGN[name] < 0 ? hi : lo };
        });
        this.groups = {};
        this.of = new Map();
        for (const [ext, flex] of Object.values(MUSCLES)) for (const t of [...ext, ...flex]) {
            if (this.groups[t]) continue;
            const g = this.groups[t] = { cells: motor[t], rate: 0, count: 0 };
            for (const i of motor[t]) this.of.set(i, g);
        }
        this.out = { cells: walkOut, rate: 0, count: 0 };
        for (const i of walkOut) this.of.set(i, this.out);
        this.hz = 150;       // taste input
        this.walkHz = 0;     // P9 input
        this.taste = null;
        this.reset();
    }

    reset() {
        const { mj, model, data, walker } = this;
        mj.mj_resetData(model, data);
        walker.setState(this.meta.qpos0);
        mj.mj_forward(model, data);
        for (const s in walker.buf) walker.buf[s].fill(0);
        walker.nbuf = 0;
        walker.setRef(0, 0, 0);
        walker.speed = 0; walker.yaw = 0;
        this.turn = 0;
        this.touch = false;
        this.dropOn = false;
        this.out.rate = 0;
        for (const g of Object.values(this.groups)) g.rate = 0;
        this.setDrop([4, 0], false);
    }

    // where the drop is, and whether there is one
    setDrop(xy, on = true) {
        this.dropOn = on;
        this.dropXY = xy;
        const p = this.data.mocap_pos, k = 3 * this.mocap;
        p[k] = xy[0]; p[k + 1] = xy[1]; p[k + 2] = on ? DROP_Z : -1;
    }

    // a spot under the labellum
    mouth() {
        const p = this.data.geom_xpos, g = this.lobes[0];
        return [p[3 * g], p[3 * g + 1]];
    }

    // a spot ahead of the fly, to one side or the other
    ahead(dist) {
        const q = this.data.qpos, f = this.walker.freeq;
        const th = 2 * Math.atan2(q[f + 6], q[f + 3]) + (Math.random() < 0.5 ? -1 : 1) * (0.3 + 0.5 * Math.random());
        return [q[f] + dist * Math.cos(th), q[f + 1] + dist * Math.sin(th)];
    }

    setTaste(taste) {
        this.input(false);
        this.taste = taste;
        this.touch = false;
    }

    setWalk(hz) {
        this.walkHz = hz;
        for (const i of this.walkSet) this.brain.set_rate(i, hz);
    }

    input(on) {
        if (this.taste) for (const i of this.tastes[this.taste]) this.brain.set_rate(i, on ? this.hz : 0);
    }

    // the rule: (L - R) / (L + R) of an odor falling off with distance from the drop
    smell() {
        if (!this.dropOn) return 0;
        const [x, y] = this.dropXY, p = this.data.xpos;
        const c = this.antennae.map(b => Math.exp(-Math.hypot(p[3 * b] - x, p[3 * b + 1] - y) / ODOR_LENGTH));
        return (c[0] - c[1]) / (c[0] + c[1]);
    }

    touching() {
        if (!this.taste || !this.dropOn) return false;
        // touch starts at contact and ends RELEASE clear of the drop, so a flicker doesn't end a meal
        const limit = this.touch ? RELEASE : 0;
        for (const g of this.lobes) if (this.mj.mj_geomDistance(this.model, this.data, this.dropGeom, g, 0.1, null) <= limit) return true;
        return false;
    }

    // one control step (2 ms) of brain and body; returns the brain's spikes as (step, neuron) pairs
    step() {
        const { brain, walker, data } = this, dt = CONTROL_DT;
        const k = brain.run(BRAIN_STEPS);
        const spikes = new Uint32Array(brain.memory.buffer, brain.spikes(), 2 * k).slice();
        const all = [...Object.values(this.groups), this.out];
        for (const g of all) g.count = 0;
        for (let s = 1; s < spikes.length; s += 2) {
            const g = this.of.get(spikes[s]);
            if (g) g.count++;
        }
        const decay = Math.exp(-dt / TAU_MUSCLE);
        for (const g of Object.values(this.groups)) g.rate = g.rate * decay + (1 - decay) * g.count / g.cells.length / dt;
        const dd = Math.exp(-dt / TAU_DRIVE);
        this.out.rate = this.out.rate * dd + (1 - dd) * this.out.count / this.out.cells.length / dt;
        // brain -> mouth
        const a = types => types.length ? 1 - Math.exp(-types.reduce((s, t) => s + this.groups[t].rate, 0) / types.length / R_HALF) : 0;
        for (const j of this.joints) {
            const [ext, flex] = MUSCLES[j.name];
            data.ctrl[j.a] = a(ext) * j.ext + a(flex) * j.flex;
        }
        // brain -> speed; the rule -> turn
        const speed = SPEED_MAX * Math.min(1, Math.max(0, (this.out.rate - SPEED_R0) / (SPEED_R1 - SPEED_R0)));
        const td = Math.exp(-dt / TAU_TURN);
        this.turn = this.turn * td + (1 - td) * TURN_GAIN * this.smell();
        // part of the rule: slow down for a sharp turn, or a drop inside the turning circle is never reached
        const sharp = Math.abs(this.turn) > TURN_MAX;
        const v = sharp ? Math.min(speed, SHARP_SPEED) : speed;
        const lim = v < TURN_MIN_SPEED ? 0 : TURN_MAX;
        walker.speed = v;
        walker.yaw = Math.max(-lim, Math.min(lim, this.turn));
        walker.step();
        const touch = this.touching();
        if (touch !== this.touch) { this.touch = touch; this.input(touch); }
        return spikes;
    }

    state() {
        const q = this.data.qpos, f = this.walker.freeq;
        return { x: q[f], y: q[f + 1], z: q[f + 2], speed: this.walker.speed, yaw: this.walker.yaw, drive: this.out.rate, touch: this.touch,
            rostrum: this.data.qpos[this.joints[0].qpos], mn9: this.groups.MN9.rate };
    }
}
