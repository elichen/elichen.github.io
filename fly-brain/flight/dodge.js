// The brain + nerve cord model (BANC) in a flying fly, dodging an object. Shared by the page and tools/dodge.mjs.
//
// From the connectome model: one-sided visual input -> DNp03 on that side (Croke et al. 2026), and the wing motor neurons
// that DNp03 reaches. What is not: the eye (below: no photoreceptors, a rule from the object's geometry to the input rate
// of the looming neurons), the saccade (below: the brain decides whether and which way, the shape is from recordings of
// real flies), and the wings (flybody's policy, flyer.js). The wing motor neurons fire but do not move these wings.
import { Flyer, CONTROL_DT } from './flyer.js';

export const BRAIN_STEPS = 10;      // brain steps (0.1 ms) per exchange, 1 ms
export const FLY_STEPS = 5;         // control steps (0.2 ms) per exchange
export const SPEED = 14;            // cm/s, the median of flybody's flight recordings; 30 pitches too far and tumbles
const LOOM_HZ = 100;                // full input rate of a looming neuron
const LOOM_FULL = 6;                // rad/s of angular expansion for the full rate
const FRONT = 0.1;                  // rad: within this of straight ahead, both eyes see it
export const TRIGGER = 60;          // Hz of DNp03 left - right that starts a saccade: several spikes (our choice)
export const SACCADE_PEAK = 43;     // rad/s, the median real saccade in flybody's recordings
const SACCADE_WIDTH = 0.04;         // s, full width at half maximum, same source
const REFRACTORY = 0.3;             // s between saccades (our choice)
const TAU_DRIVE = 0.05;             // s, low-pass on DNp03: one spike moves it 20 Hz

export class Dodger {
    // brain: engine exports after init; policy: WasmPolicy; files: fly.xml + meshes by bare name; neurons: neurons.json
    constructor(mujoco, files, meta, policy, brain, neurons) {
        Object.assign(this, { mj: mujoco, brain, neurons });
        const vfs = new mujoco.MjVFS();
        const enc = new TextEncoder();
        for (const [name, d] of Object.entries(files)) vfs.addBuffer(name, typeof d === 'string' ? enc.encode(d) : d);
        const m = this.model = mujoco.MjModel.from_xml_string(files['fly.xml'], vfs);
        const d = this.data = new mujoco.MjData(m);
        vfs.delete();
        this.flyer = new Flyer(mujoco, m, d, meta, policy);
        const id = (t, n) => mujoco.mj_name2id(m, mujoco.mjtObj[t].value, n);
        this.mocap = m.body_mocapid[id('mjOBJ_BODY', 'looming')];
        this.radius = m.geom_size[3 * id('mjOBJ_GEOM', 'looming')];
        this.head = id('mjOBJ_BODY', 'walker/head');
        this.eye = { left: [], right: [] };
        for (const s of ['left', 'right']) for (const k in neurons.looms) this.eye[s].push(...neurons.looms[k][s]);
        this.dn = { left: { cells: neurons.named.DNp03.left, rate: 0, count: 0 }, right: { cells: neurons.named.DNp03.right, rate: 0, count: 0 } };
        this.of = new Map();
        for (const g of Object.values(this.dn)) for (const i of g.cells) this.of.set(i, g);
        this.turnSign = 1;         // -1: the control that turns toward the object
        this.steer = true;         // false: the control that does not turn
        this.reset();
    }

    reset() {
        this.flyer.speed = SPEED;
        this.flyer.yaw = 0;
        this.flyer.start(0, 0, 0, Math.random());
        this.dn.left.rate = this.dn.right.rate = 0;
        this.turn = 0;
        this.saccades = [];        // { at, sign }, the latest last
        this.object = null;
        this.input = { left: 0, right: 0 };
        this.applyInput('left', 0); this.applyInput('right', 0);
        this.prevSize = null;
        this.closest = Infinity;
        this.t = 0;
    }

    // launch from a bearing (rad, left positive) relative to the fly, to meet the fly where it will be if it flies straight
    launch(bearing, dist = 12, speed = 30, radius = this.radius) {
        const [x, y, z] = this.flyer.position(), h = this.flyer.ref.heading;
        const v = this.flyer.speed, u = [Math.cos(h), Math.sin(h)];
        const a = h + bearing;
        const from = [x + dist * Math.cos(a), y + dist * Math.sin(a), z];
        // time to intercept: |from + speed t dir - (fly + v t u)| = 0
        const rx = from[0] - x, ry = from[1] - y;
        const A = speed * speed - v * v, B = 2 * v * (rx * u[0] + ry * u[1]), C = -(rx * rx + ry * ry);
        const T = (-B + Math.sqrt(B * B - 4 * A * C)) / (2 * A);
        const hit = [x + v * T * u[0], y + v * T * u[1]];
        const dx = hit[0] - from[0], dy = hit[1] - from[1], l = Math.hypot(dx, dy);
        this.object = { p: from, v: [speed * dx / l, speed * dy / l, 0], radius, T };
        this.prevSize = null;
        this.closest = Infinity;
        this.placeObject();
    }

    placeObject() {
        const p = this.data.mocap_pos, k = 3 * this.mocap, o = this.object;
        p[k] = o ? o.p[0] : 0; p[k + 1] = o ? o.p[1] : 0; p[k + 2] = o ? o.p[2] : -50;
    }

    applyInput(side, hz) {
        if (this.input[side] === hz) return;
        this.input[side] = hz;
        for (const i of this.eye[side]) this.brain.set_rate(i, hz);
    }

    // the rule that stands in for the eye: expansion rate of the object sets the input of the looming neurons on its side
    see(dt) {
        const o = this.object;
        if (!o) { this.applyInput('left', 0); this.applyInput('right', 0); return; }
        const [x, y] = this.flyer.position(), q = this.data.xpos, b = this.head;
        const hx = q[3 * b], hy = q[3 * b + 1];
        const dx = o.p[0] - hx, dy = o.p[1] - hy, d = Math.hypot(dx, dy, o.p[2] - q[3 * b + 2]);
        this.closest = Math.min(this.closest, d - o.radius);
        const size = 2 * Math.atan2(o.radius, Math.max(d, o.radius));
        const grow = this.prevSize === null ? 0 : (size - this.prevSize) / dt;
        this.prevSize = size;
        const h = this.flyer.ref.heading;
        let az = Math.atan2(dy, dx) - h;
        az = Math.atan2(Math.sin(az), Math.cos(az));
        const behind = Math.abs(az) > 2.5;          // past the fly: out of the eyes' field, roughly
        // in steps of 10 Hz: setting the rate of hundreds of neurons every millisecond is slow
        const hz = behind ? 0 : 10 * Math.round(LOOM_HZ / 10 * Math.min(1, Math.max(0, grow / LOOM_FULL)));
        this.applyInput('left', az > -FRONT ? hz : 0);
        this.applyInput('right', az < FRONT ? hz : 0);
    }

    // 1 ms of brain and body; returns the brain's spikes
    step() {
        const { brain, flyer } = this, dt = FLY_STEPS * CONTROL_DT;
        const k = brain.run(BRAIN_STEPS);
        const spikes = new Uint32Array(brain.memory.buffer, brain.spikes(), 2 * k).slice();
        this.dn.left.count = this.dn.right.count = 0;
        for (let s = 1; s < spikes.length; s += 2) {
            const g = this.of.get(spikes[s]);
            if (g) g.count++;
        }
        const dd = Math.exp(-dt / TAU_DRIVE);
        for (const g of Object.values(this.dn)) g.rate = g.rate * dd + (1 - dd) * g.count / g.cells.length / dt;
        // the brain's decision: DNp03 on the left turns the fly to its right (negative yaw), as Croke et al. saw
        const diff = this.dn.left.rate - this.dn.right.rate;
        const last = this.saccades[this.saccades.length - 1];
        if (this.steer && Math.abs(diff) > TRIGGER && (!last || this.t - last.at > REFRACTORY)) {
            this.saccades.push({ at: this.t, sign: -this.turnSign * Math.sign(diff) });
        }
        const sd = SACCADE_WIDTH / 2.355;
        let w = 0;
        for (const q of this.saccades) {
            const u = this.t - (q.at + 2 * sd);      // starts about now, peaks 2 widths later
            if (Math.abs(u) < 4 * sd) w += q.sign * SACCADE_PEAK * Math.exp(-u * u / (2 * sd * sd));
        }
        this.turn = w;
        flyer.yaw = this.turn;
        for (let i = 0; i < FLY_STEPS; i++) {
            flyer.step();
            this.onControl?.();
        }
        const o = this.object;
        if (o) {
            for (let i = 0; i < 3; i++) o.p[i] += o.v[i] * dt;
            this.placeObject();
        }
        this.see(dt);
        this.t += dt;
        return spikes;
    }
}
