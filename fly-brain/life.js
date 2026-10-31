// One fly's whole day on one body: walk (flybody's walking policy), take off, fly (flybody's flight policy), land, walk again.
// The two policies are unchanged and neither was trained for the handovers. The handovers are stated rules, here:
// takeoff = the legs rear the body up with the wings shut, then push while the flight policy runs the wings on a path that starts
// at the fly's own pose and blends into cruise. It cannot catch a level body (tools/catch.mjs: from 10 deg nose-up yes, from 0 no).
// A fly that is not nose-up after the rear-up does not launch. Landing = the path sinks to the floor, slows, legs reach out;
// on first touch the wings shut and the walking policy takes over.
import { CONTROL_DT as FLY_DT } from './flyer.js';
import { CONTROL_DT as WALK_DT } from './walker.js';

export const TAKEOFF = { pre: 0.0977, pamp: [0.8887, 0.1825, -1.2234], T: 0.0587, rise: 0.3675, t0: 0.0201, tp: 0.00535, amp: [0, 0.5462, 0.6799], minPitch: 12 };
// the path: fall from cruise at `rate` cm/s to `near` cm, still flying forward; brake to a hover over `slow` s, nose down to `pitch`; hold; sink to `floor` over `sink` s
export const LANDING = { rate: 8, near: 0.8, slow: 0.5, hold: 0.1, sink: 0.6, pitch: 35, floor: 0.2, legs: 0.2, still: 0.5, wait: 1 };
const CRUISE = -47.5 * Math.PI / 180, DEG = Math.PI / 180, WALK_H = 2e-4, FLY_H = 5e-5;
const smooth = x => { const u = Math.min(1, Math.max(0, x)); return u * u * (3 - 2 * u); };

export class Life {
    constructor(mujoco, model, data, walker, flyer, takeoff = TAKEOFF, landing = LANDING) {
        Object.assign(this, { mj: mujoco, model, data, W: walker, F: flyer, TO: takeoff, LA: landing });
        const m = model, nm = j => mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT.value, j);
        const act = n => mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR.value, n);
        this.legs = [...Array(m.njnt).keys()].filter(j => /coxa|femur|tibia|tarsus/.test(nm(j) || '')).map(j => {
            const n = nm(j), p = m.jnt_qposadr[j];
            return { n, q: p, a: act(n), pair: /T1/.test(n) ? 0 : /T2/.test(n) ? 1 : 2, coxa: /coxa/.test(n), tuck: m.qpos_spring[p] * 0.6, rest: m.qpos0[p] };
        }).filter(l => l.a >= 0);
        this.adhere = [...Array(m.nu).keys()].filter(a => /adhere_claw/.test(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR.value, a)));
        this.touch = ['T1', 'T2', 'T3'].flatMap(t => ['left', 'right'].map(s => m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR.value, `walker/touch_claw_${t}_${s}`)]));
        // +1 if a bigger angle moves the foot away from the thorax, from the rest pose
        const d = data, save = Float64Array.from(d.qpos);
        for (let i = 0; i < m.nq; i++) d.qpos[i] = m.qpos0[i];
        for (const l of this.legs) {
            const claw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY.value, `walker/claw_${['T1', 'T2', 'T3'][l.pair]}_${/left/.test(l.n) ? 'left' : 'right'}`);
            const reach = v => { const o = d.qpos[l.q]; d.qpos[l.q] = o + v; mujoco.mj_forward(m, d); const z = d.xpos[3 * flyer.root + 2] - d.xpos[3 * claw + 2]; d.qpos[l.q] = o; return z; };
            l.sign = reach(0.1) > reach(-0.1) ? 1 : -1;
        }
        d.qpos.set(save); mujoco.mj_forward(m, d);
        this.mode = 'walk'; this.t = 0; this.aborts = 0;
        this.noContact = mujoco.mjtDisableBit.mjDSBL_CONTACT.value; this.contactOff = false;
        this.speed = 0; this.yaw = 0;      // what walking or flying is asked to do now
        this.cruise = 1;                   // height flying aims at, cm above the table
    }

    pitch() { const f = this.F.freeq, q = this.data.qpos; return Math.asin(Math.max(-1, Math.min(1, 2 * (q[f + 3] * q[f + 5] - q[f + 6] * q[f + 4])))); }
    heading() { const f = this.F.freeq, q = this.data.qpos; return Math.atan2(2 * (q[f + 3] * q[f + 6] + q[f + 4] * q[f + 5]), 1 - 2 * (q[f + 5] ** 2 + q[f + 6] ** 2)); }
    up() { const f = this.F.freeq, q = this.data.qpos; return 1 - 2 * (q[f + 4] ** 2 + q[f + 5] ** 2); }

    // q and -q are the same pose, but the policies read the numbers: past 180 deg of heading the body's sign can differ from the path's,
    // and the policy throws the fly over. Each handover makes them agree.
    alignQuat(r) {
        const q = this.data.qpos, f = this.F.freeq;
        if (q[f + 3] * r[0] + q[f + 4] * r[1] + q[f + 5] * r[2] + q[f + 6] * r[3] < 0) for (let i = 3; i < 7; i++) q[f + i] = -q[f + i];
    }

    startWalking() {
        const { W, F, data: d } = this, f = F.freeq;
        this.alignQuat([Math.cos(this.heading() / 2), 0, 0, Math.sin(this.heading() / 2)]);
        this.model.opt.timestep = WALK_H;
        Object.values(W.buf).forEach(b => b.fill(0)); W.nbuf = 0;
        W.setRef(d.qpos[f], d.qpos[f + 1], this.heading());
        this.mode = 'walk'; this.t = 0;
    }

    takeOff() {
        if (this.mode !== 'walk') return;
        this.model.opt.timestep = FLY_H;
        for (const a of this.adhere) this.data.ctrl[a] = 0;
        this.from = this.legs.map(l => this.data.qpos[l.q]);
        this.mode = 'rear'; this.t = 0;
    }

    land() { if (this.mode === 'fly') { this.mode = 'descend'; this.t = 0; this.sinkAt = null; this.fromH = this.F.height; this.fromP = this.F.pitch; this.fromS = this.F.speed; this.from = this.legs.map(l => this.data.ctrl[l.a]); } }

    // how far the fly travels on its way down, if it starts now
    landingRun() { const A = Math.max(0, (this.F.height - this.LA.near) / this.LA.rate); return this.F.speed * (A + this.LA.slow / 2); }

    // runs the path from where the body is now
    pathFromHere(pitch, phase = this.F.phase) {
        const { F, data: d } = this, f = F.freeq, th = this.heading();
        F.freq = F.meta.base_freq; F.phase = phase; F.buf.fill(0); F.nbuf = 0; F.yaw = 0; F.speed = 0; F.pitch = pitch; F.height = 1;
        const { p, q } = F.pose(0, 0, th, 0);
        this.alignQuat(q);
        F.ref = { x: d.qpos[f] - p[0], y: d.qpos[f + 1] - p[1], heading: th };
        F.height = 1 - (p[2] - d.qpos[f + 2]);
    }

    touching() { const s = this.data.sensordata; return this.touch.some(a => s[a] > 0); }

    // one control step of whichever mode; returns the simulated seconds it took
    step() {
        const { mj, model: m, data: d, W, F, TO, LA } = this;
        // high in the air there is nothing to touch: leaving contacts out is what keeps flight near real time
        const f = F.freeq, off = this.mode === 'fly' && d.qpos[f + 2] > 1;
        if (off !== this.contactOff) { this.contactOff = off; if (off) m.opt.disableflags |= this.noContact; else m.opt.disableflags &= ~this.noContact; }
        if (this.mode === 'walk') { W.speed = this.speed; W.yaw = this.yaw; W.step(); this.t += WALK_DT; return WALK_DT; }
        if (this.mode === 'rear') {
            const u = smooth(this.t / TO.pre / 0.8);
            this.legs.forEach((l, i) => { d.ctrl[l.a] = this.from[i] * (1 - u) + (l.rest + (l.coxa ? 0 : l.sign * TO.pamp[l.pair])) * u; });
            for (let i = 0; i < 4; i++) mj.mj_step(m, d);
            this.t += FLY_DT;
            if (this.t >= TO.pre) {
                if (this.pitch() > -TO.minPitch * DEG) { this.aborts++; this.startWalking(); return FLY_DT; }
                this.set = this.legs.map(l => d.ctrl[l.a]);
                this.p0 = this.pitch(); this.pathFromHere(this.p0, Math.random()); this.h0 = F.height;
                this.mode = 'launch'; this.t = 0;
            }
            return FLY_DT;
        }
        if (this.mode === 'launch') {
            const t = this.t, s = smooth(t / TO.T);
            F.pitch = this.p0 + (CRUISE - this.p0) * s; F.height = this.h0 + TO.rise * s; F.speed = 14 * smooth((t - TO.T * 0.5) / TO.T);
            const u = smooth((t - TO.t0) / TO.tp), v = smooth((t - TO.t0 - TO.tp) / 0.06);
            this.legs.forEach((l, i) => { const c = this.set[i] + (l.coxa ? 0 : u * l.sign * TO.amp[l.pair]); d.ctrl[l.a] = c * (1 - v) + l.tuck * v; });
            F.step(); this.t += FLY_DT;
            if (this.t > TO.T + 0.15) { this.mode = 'fly'; this.t = 0; }
            return FLY_DT;
        }
        if (this.mode === 'fly') { F.yaw = this.yaw; F.height += Math.max(-LANDING.rate * FLY_DT, Math.min(LANDING.rate * FLY_DT, this.cruise - F.height)); F.step(); this.t += FLY_DT; return FLY_DT; }
        if (this.mode === 'descend') {
            const A = Math.max(0, (this.fromH - LA.near) / LA.rate), t = this.t, r = smooth((t - A) / LA.slow);
            // sink only from a still hover (first touch at 2.5 cm/s or more tips the fly over); a stated rule
            if (this.sinkAt === null && t > A + LA.slow + LA.hold && (Math.hypot(d.qvel[F.freev], d.qvel[F.freev + 1]) < LA.still || t > A + LA.slow + LA.hold + LA.wait)) this.sinkAt = t;
            const D = this.sinkAt ?? Infinity, l_ = smooth((t - A - LA.slow) / LA.legs);
            F.height = t < A ? this.fromH + (LA.near - this.fromH) * t / A : LA.near + (LA.floor - LA.near) * (t > D ? smooth((t - D) / LA.sink) : 0);
            F.speed = this.fromS * (1 - r); F.pitch = this.fromP + (-LA.pitch * DEG - this.fromP) * r; F.yaw = this.yaw * (1 - r);
            this.legs.forEach((l, i) => { d.ctrl[l.a] = this.from[i] * (1 - l_) + l.rest * l_; });
            F.step(); this.t += FLY_DT;
            if (this.touching() || t > A + LA.slow + LA.hold + LA.wait + LA.sink + 1) { this.mode = 'touch'; this.t = 0; this.touchAt = t; }
            return FLY_DT;
        }
        if (this.mode === 'touch') {     // wings shut, then the walking policy
            for (let i = 0; i < 6; i++) { const q = F.wq[i]; d.ctrl[F.ctrl[F.meta.wing_action[i]]] = Math.max(-3, Math.min(3, m.qpos_spring[q] - d.qpos[q])); }
            this.legs.forEach(l => { d.ctrl[l.a] = l.rest; });
            for (let i = 0; i < 4; i++) mj.mj_step(m, d);
            this.t += FLY_DT;
            if (this.t > 0.03) { for (let i = 0; i < 6; i++) d.ctrl[F.ctrl[F.meta.wing_action[i]]] = 0; this.startWalking(); }
            return FLY_DT;
        }
    }
}
