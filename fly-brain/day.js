// A fly's day on a kitchen table. Shared by worker.js and tools/day.mjs.
//
// From the connectome model (BANC, one brain and nerve cord for everything):
//   walking drive  P9 -> oDN1 sets walk or stand.        stopping   sugar on the mouth cuts oDN1 (Sapkal et al. 2024).
//   feeding        sugar -> MN9 and other mouth motor neurons -> mouth joints.
//   jump           an object growing in both eyes -> giant fiber (DNp01) -> takeoff.
//   dodge          an object growing in one eye -> DNp03 on that side -> saccade away.
// Rules, not the connectome:
//   the wish       hunger and boredom rise with time; they choose where to go and turn P9 on.
//   the way        walk or fly toward the goal: a compass, not the brain. Near: walk. Far: fly.
//   the eye        an object's expansion rate sets the looming neurons' input.
//   the movement   flybody's walking and flight policies, with the stated handovers in life.js.
import { Life } from './life.js';

// Brain -> mouth. Each joint has the muscles McKellar et al. 2020 (eLife 9:e54978) found move it, named by their
// motor neurons (BANC names them; tools/export_neurons.py). A muscle's activation is 1 - exp(-r / R_HALF) of its motor
// neurons' mean rate r, low-passed with TAU_MUSCLE; the joint's position actuator aims at its range limit times the
// activation, extensor minus flexor. Both constants are our choices: no recordings give rate -> force here.
const TAU_MUSCLE = 0.03;     // s
const R_HALF = 40;           // Hz for activation 1 - 1/e
const MUSCLES = {            // joint: [extensor motor neurons, flexor motor neurons]
    rostrum: [['MN9'], ['MN1', 'MN2Da', 'MN2Db', 'MN2V']],      // protractor m9; retractors m1, m2
    haustellum: [['MN4a', 'MN4b'], ['MN3L', 'MN3M']],           // extensor m4; flexor m3
    labrum_left: [['MN7'], []],                                  // m7 opens (abducts) the labellum
    labrum_right: [['MN7'], []],
};

export const TABLE = { x: 34, y: 22 };   // half sizes, cm: where the fly's goals may be
export const FOODS = [[-18, 6], [18, -15], [-27, -12], [8, 18]];
export const CRUISE = 12;                // cm above the table: clears what stands on it
// what stands on the table: x, y, radius, cm. Pictures only (scene.js); goals stay out, and a path across one flies high
export const KEEP_OUT = [[2, -3, 14.5], [-33, 16, 7.5], [28, 13, 12.5], [34, -7, 9], [36, -19, 5.5]];
export const isClear = (x, y) => KEEP_OUT.every(([a, b, r]) => Math.hypot(x - a, y - b) > r);
const crosses = (x0, y0, x1, y1) => KEEP_OUT.some(([a, b, r]) => { const dx = x1 - x0, dy = y1 - y0, u = Math.max(0, Math.min(1, ((a - x0) * dx + (b - y0) * dy) / (dx * dx + dy * dy || 1))); return Math.hypot(x0 + u * dx - a, y0 + u * dy - b) < r; });
const HUNGER_RATE = 1 / 45, BORED_RATE = 1 / 20, HUNGRY = 0.6, FULL = 0.05;   // per s; our choices
const EAT_RATE = 0.2;                    // hunger per s at full MN9 activation; our choice
const FLY_FROM = 6;                      // cm: farther than this, fly
const ARRIVED = 1.0;                     // cm: a wander goal is reached
const P9_HZ = 150, SUGAR_HZ = 150;       // tools/life_brain.mjs: oDN1 8.6 Hz, with sugar 3.8 Hz
const WALK_FROM = 7, STAND_BELOW = 5;    // Hz of oDN1, between those two
const TAU_ODN1 = 0.2, TAU_FAST = 0.05;   // s, rate filters
const GF_JUMP = 20;                      // Hz of giant fiber that starts a takeoff; both eyes 44, one eye 5
const DODGE = 60, REFRACTORY = 0.3, SACCADE_PEAK = 43, SD = 0.04 / 2.355;
const LOOM_HZ = 100, LOOM_FULL = 2, FRONT = 0.25;   // full rate at 2 rad/s: looming neurons respond well before contact (our choice)
const TURN_WALK = 4, TURN_FLY = 8, WALK_RAMP = 2;
const EXTEND = { rostrum: -1, haustellum: -1, labrum_left: 1, labrum_right: 1 };
const clamp = (x, a) => Math.max(-a, Math.min(a, x));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

export class Day {
    constructor(mujoco, model, data, walker, flyer, brain, neurons, seed = 1) {
        Object.assign(this, { mj: mujoco, model, data, brain, W: walker, F: flyer, n: neurons, rngState: seed >>> 0 || 1 });
        this.life = new Life(mujoco, model, data, walker, flyer);
        const L = neurons.life, m = model, id = (t, k) => mujoco.mj_name2id(m, mujoco.mjtObj[t].value, k);
        this.lobe = id('mjOBJ_GEOM', 'walker/labrum_left_lower_collision');
        this.head = id('mjOBJ_BODY', 'walker/head');
        this.joints = Object.keys(MUSCLES).map(name => {
            const a = id('mjOBJ_ACTUATOR', 'walker/' + name), lo = m.actuator_ctrlrange[2 * a], hi = m.actuator_ctrlrange[2 * a + 1];
            return { name, a, ext: EXTEND[name] < 0 ? lo : hi, flex: EXTEND[name] < 0 ? hi : lo };
        });
        this.of = new Map();
        const group = (cells, tau) => { const g = { cells, tau, rate: 0, count: 0 }; for (const i of cells) { if (!this.of.has(i)) this.of.set(i, []); this.of.get(i).push(g); } return g; };
        this.mouth = {}; for (const k in L.mouth) this.mouth[k] = group(L.mouth[k], TAU_MUSCLE);
        this.oDN1 = group(L.oDN1, TAU_ODN1); this.GF = group(L.GF, TAU_FAST);
        this.dnL = group(neurons.named.DNp03.left, TAU_FAST); this.dnR = group(neurons.named.DNp03.right, TAU_FAST);
        this.groups = [...Object.values(this.mouth), this.oDN1, this.GF, this.dnL, this.dnR];
        this.eye = { left: Object.values(neurons.looms).flatMap(v => v.left), right: Object.values(neurons.looms).flatMap(v => v.right) };
        this.foods = []; this.input = { left: 0, right: 0 }; this.brainBase = 0; this.calmAt = null;
        this.reset();
    }

    rnd() { let s = this.rngState; s ^= s << 13; s ^= s >>> 17; s ^= s << 5; this.rngState = s >>> 0; return this.rngState / 2 ** 32; }

    reset(x = -12, y = -16, heading = 0) {
        const { mj, model: m, data: d, W, F, life } = this, f = F.freeq;
        mj.mj_resetData(m, d); m.opt.timestep = 2e-4;
        W.setState(W.meta.qpos0);
        for (const q of F.wq) d.qpos[q] = m.qpos_spring[q];
        d.qpos[f] = x; d.qpos[f + 1] = y; d.qpos[f + 3] = Math.cos(heading / 2); d.qpos[f + 4] = d.qpos[f + 5] = 0; d.qpos[f + 6] = Math.sin(heading / 2);
        mj.mj_forward(m, d);
        life.startWalking(); life.cruise = CRUISE; life.speed = 0; life.yaw = 0;
        this.t = this.t ?? 0; this.bodyAhead = 0;
        this.hunger = this.hunger ?? 0.4; this.bored = this.bored ?? 0;
        this.goal = null; this.walking = false; this.eating = null; this.v = 0; this.setP9(0); this.setSugar(false);
        this.saccades = []; this.upsideDown = 0; this.retry = 0;
        this.object = null; this.prevSize = null; this.applyEye('left', 0); this.applyEye('right', 0);
    }

    // The model does not calm down by itself after a loom: at any input from 30 Hz up it stays at about 32,000 spikes/s for good (tools/life_after.mjs),
    // and in some runs the giant fiber keeps firing. So a second after an object has gone we clear the brain's state. A stated rule; real brains recover.
    calm() {
        this.brainBase += this.brain.step_count(); this.brain.reset(); this.calmAt = null; this.calms = (this.calms || 0) + 1;
        for (const g of this.groups) g.rate = 0;
        for (const i of this.n.life.P9) this.brain.set_rate(i, this.p9 || 0);
        for (const i of this.n.life.sugar) this.brain.set_rate(i, this.sugar ? SUGAR_HZ : 0);
        for (const k of ['left', 'right']) for (const i of this.eye[k]) this.brain.set_rate(i, this.input[k]);
    }

    setP9(hz) { if (this.p9 === hz) return; this.p9 = hz; for (const i of this.n.life.P9) this.brain.set_rate(i, hz); }
    setSugar(on) { if (this.sugar === on) return; this.sugar = on; for (const i of this.n.life.sugar) this.brain.set_rate(i, on ? SUGAR_HZ : 0); }
    applyEye(side, hz) { if (this.input[side] === hz) return; this.input[side] = hz; for (const i of this.eye[side]) this.brain.set_rate(i, hz); }

    addFood(x, y, r = 0.45, amount = 1) { this.foods.push({ x, y, r, amount }); }
    // an object coming down at the fly, from a side given by az (rad from straight above, in the table's frame)
    swat(fromSide = 0, dist = 30, speed = 30, radius = 3) {
        const f = this.F.freeq, q = this.data.qpos, h = this.life.heading(), a = h + fromSide;
        const tilt = Math.abs(fromSide) > 0 ? 0.8 : 0;
        const dir = [Math.sin(tilt) * Math.cos(a), Math.sin(tilt) * Math.sin(a), Math.cos(tilt)];
        const lead = this.life.mode === 'fly' ? this.F.speed * dist / speed : 0, aim = [q[f] + lead * Math.cos(h), q[f + 1] + lead * Math.sin(h), q[f + 2]];
        this.object = { p: aim.map((v, i) => v + dist * dir[i]), v: dir.map(v => -speed * v), radius, until: this.t + dist / speed + 0.25, hit: false, aim };
        this.prevSize = null;
    }

    see(dt) {
        const o = this.object;
        if (!o) { this.applyEye('left', 0); this.applyEye('right', 0); return; }
        const x = this.data.xpos, b = this.head, dx = o.p[0] - x[3 * b], dy = o.p[1] - x[3 * b + 1], dz = o.p[2] - x[3 * b + 2], dist = Math.hypot(dx, dy, dz);
        const size = 2 * Math.atan2(o.radius, Math.max(dist, o.radius)), grow = this.prevSize === null ? 0 : (size - this.prevSize) / dt;
        this.prevSize = size;
        const az = wrap(Math.atan2(dy, dx) - this.life.heading()), overhead = Math.hypot(dx, dy) < dz * 0.6;
        const hz = (!overhead && Math.abs(az) > 2.5) ? 0 : 10 * Math.round(LOOM_HZ / 10 * Math.min(1, Math.max(0, grow / LOOM_FULL)));
        this.applyEye('left', overhead || az > -FRONT ? hz : 0);
        this.applyEye('right', overhead || az < FRONT ? hz : 0);
    }

    somewhere(why) { for (;;) { const x = (2 * this.rnd() - 1) * TABLE.x, y = (2 * this.rnd() - 1) * TABLE.y; if (isClear(x, y)) return { x, y, why }; } }

    // the wish: which goal, by rule
    choose() {
        const { life } = this, p = life.F.position();
        if (this.eating) return;
        if (this.hunger > HUNGRY && this.foods.length) {
            const near = this.foods.reduce((a, b) => (Math.hypot(a.x - p[0], a.y - p[1]) < Math.hypot(b.x - p[0], b.y - p[1]) ? a : b));
            this.goal = { x: near.x, y: near.y, food: near, why: 'hunger' };
        } else if (!this.goal && this.bored > 1) {
            this.goal = this.somewhere('boredom');
            this.bored = 0;
        }
    }

    // 1 ms of brain; the body keeps up. Returns the spikes.
    step() {
        const { brain, life, data: d, F } = this, dt = 1e-3, f = F.freeq;
        const k = brain.run(10), spikes = new Uint32Array(brain.memory.buffer, brain.spikes(), 2 * k).slice();
        if (this.brainBase) for (let i = 0; i < spikes.length; i += 2) spikes[i] += this.brainBase;
        for (const g of this.groups) g.count = 0;
        for (let s = 1; s < spikes.length; s += 2) { const l = this.of.get(spikes[s]); if (l) for (const g of l) g.count++; }
        for (const g of this.groups) { const a = Math.exp(-dt / g.tau); g.rate = g.rate * a + (1 - a) * g.count / g.cells.length / dt; }
        this.t += dt;
        const air = life.mode !== 'walk' && life.mode !== 'rear';
        this.hunger = Math.min(1, this.hunger + HUNGER_RATE * dt);
        if (!this.goal) this.bored = Math.min(1.2, this.bored + BORED_RATE * dt);
        this.choose();
        const p = F.position();
        // mouth
        const act = ts => ts.length ? 1 - Math.exp(-ts.reduce((s, t) => s + (this.mouth[t]?.rate ?? 0), 0) / ts.length / R_HALF) : 0;
        for (const j of this.joints) { const [e, fl] = MUSCLES[j.name]; d.ctrl[j.a] = act(e) * j.ext + act(fl) * j.flex; }
        // eating: sugar under the mouth
        const lx = d.geom_xpos[3 * this.lobe], ly = d.geom_xpos[3 * this.lobe + 1];
        const under = !air && life.mode === 'walk' ? this.foods.find(o => Math.hypot(o.x - lx, o.y - ly) < o.r + (o === this.eating ? 0.3 : 0)) : null;
        if (under && this.hunger > FULL && (this.goal?.food === under || this.eating)) { this.eating = under; this.goal = null; }
        if (this.eating && (!under || under !== this.eating)) this.eating = null;
        this.setSugar(!!this.eating);
        if (this.eating) {
            const eat = EAT_RATE * act(['MN9']) * dt;
            this.hunger = Math.max(0, this.hunger - eat - HUNGER_RATE * dt);
            this.eating.amount -= eat * 0.5;
            if (this.hunger <= FULL || this.eating.amount <= 0) {
                if (this.eating.amount <= 0) this.foods.splice(this.foods.indexOf(this.eating), 1);
                this.eating = null; this.bored = 0;
            }
        }
        // the brain's walk or stand: P9 is on while there's a wish (rule); oDN1 carries it, and sugar cuts it
        this.setP9(this.goal || this.eating ? P9_HZ : 0);
        if (this.oDN1.rate >= WALK_FROM) this.walking = true; else if (this.oDN1.rate < STAND_BELOW) this.walking = false;
        // the way
        let dist = Infinity, bearing = 0;
        if (this.goal) { dist = Math.hypot(this.goal.x - p[0], this.goal.y - p[1]); bearing = wrap(Math.atan2(this.goal.y - p[1], this.goal.x - p[0]) - life.heading()); }
        if (this.goal && !this.goal.food && dist < ARRIVED + (air ? 4 : 0) && !air) this.goal = null;
        // the jump is the giant fiber's
        if (life.mode === 'walk' && this.GF.rate > GF_JUMP && !this.noJump) { this.goal = this.goal ?? this.somewhere('escape'); if (this.retry <= 0) { this.jumps = (this.jumps || 0) + 1; this.retry = 0.5; life.takeOff(); } }
        if (life.mode === 'walk') {
            this.retry -= dt;
            if (this.goal && dist > FLY_FROM && this.retry <= 0 && this.walking && life.t > 1 && life.up() > 0.95 && (this.v || 0) < 0.05) { life.takeOff(); this.retry = 0.5; }
            // the walking policy falls if turned below 0.8 cm/s or started at speed: speed ramps, and turning waits for it
            const go = this.goal && this.walking && !this.eating && dist <= FLY_FROM && life.t > 0.6, want = go ? (Math.abs(bearing) > 1 ? 1.5 : 2) : 0;
            this.v = (this.v || 0) + Math.max(-4 * dt, Math.min(WALK_RAMP * dt, want - (this.v || 0)));
            life.speed = this.v;
            life.yaw = go && this.v >= 0.8 ? clamp(4 * bearing, Math.min(TURN_WALK, 2 * this.v)) : 0;   // tools/walk_turns.mjs: 1 cm/s at 4 rad/s falls, 1.5 at 3 and 2 at 4 do not
        } else {
            life.yaw = this.goal ? clamp(3 * bearing, TURN_FLY) : 0;
            if (this.goal) life.cruise = dist < 25 && !crosses(p[0], p[1], this.goal.x, this.goal.y) ? Math.max(1.5, Math.min(CRUISE, dist * 0.25)) : CRUISE;
            // down when the goal is within the run, facing it or not: a goal inside the turning circle is never faced. Overshoot is walked or hopped back.
            if (life.mode === 'fly' && this.goal && dist < life.landingRun() + 1 && life.t > 0.5) life.land();
            if (life.mode === 'fly' && !this.goal) this.goal = this.somewhere('escape');
            const diff = this.dnL.rate - this.dnR.rate, last = this.saccades[this.saccades.length - 1];
            if (life.mode === 'fly' && Math.abs(diff) > DODGE && (!last || this.t - last.at > REFRACTORY)) this.saccades.push({ at: this.t, sign: -Math.sign(diff) });
            for (const q of this.saccades) { const u = this.t - (q.at + 2 * 0.04); if (Math.abs(u) < 4 * SD) life.yaw = q.sign * SACCADE_PEAK * Math.exp(-u * u / (2 * SD * SD)); }
        }
        // the body
        this.bodyAhead -= dt;
        while (this.bodyAhead < 0) { this.bodyAhead += life.step(); this.onControl?.(); }
        // the object
        const o = this.object;
        if (o) {
            for (let i = 0; i < 3; i++) o.p[i] += o.v[i] * dt;
            const b = F.position();
            const along = (b[0] - o.p[0]) * o.v[0] + (b[1] - o.p[1]) * o.v[1] + (b[2] - o.p[2]) * o.v[2], sp = Math.hypot(...o.v), off = Math.hypot(b[0] - o.p[0] - o.v[0] * along / sp / sp, b[1] - o.p[1] - o.v[1] * along / sp / sp, b[2] - o.p[2] - o.v[2] * along / sp / sp);
            if (Math.abs(along / sp) < 0.25 && off < o.radius) { o.hit = true; }
            if (o.hit && !o.counted) { o.counted = true; this.hits = (this.hits || 0) + 1; }
            if (this.t > o.until) { this.lastSwat = o.hit ? 'hit' : 'missed'; this.object = null; this.calmAt = this.t + 1; }
        }
        this.see(dt);
        if (this.calmAt !== null && this.t > this.calmAt && !this.object) this.calm();
        // put back: the policies can't right a fly on its back, or catch a tumble. Stated and counted on the page.
        const bad = !Number.isFinite(d.qpos[f]) || life.up() < 0.2 || p[2] < -0.2 || p[2] > 40 || (air && life.mode === 'fly' && life.t > 0.5 && p[2] < 0.25);
        this.upsideDown = Math.max(0, this.upsideDown + (bad ? dt : -dt / 3));      // a tumble flickers through upright: this adds it up
        if (this.upsideDown > 0.4 || !Number.isFinite(d.qpos[f])) {
            this.putBacks = (this.putBacks || 0) + 1;
            const ok = Number.isFinite(d.qpos[f]) && isClear(p[0], p[1]) && Math.abs(p[0]) < TABLE.x && Math.abs(p[1]) < TABLE.y, s = ok ? null : this.somewhere('');
            this.reset(ok ? p[0] : s.x, ok ? p[1] : s.y, this.rnd() * 6.28);
        }
        return spikes;
    }
}
