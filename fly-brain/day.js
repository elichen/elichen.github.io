// A fly's day on a kitchen table. Shared by worker.js and tools/day.mjs.
//
// From the connectome model (BANC, one brain and nerve cord for everything):
//   walking drive  P9 -> oDN1 sets walk or stand.        stopping   sugar on the mouth cuts oDN1 (Sapkal et al. 2024).
//   feeding        sugar -> MN9 and other mouth motor neurons -> mouth joints.
//   jump           an object growing in an eye -> giant fiber (DNp01) -> the fast takeoff; -> DNp04 and DNp11 -> the slow takeoff.
//   dodge          an object growing in one eye -> DNp03 on that side -> saccade away.
//   feeling        the body's own sensory neurons fire with the legs' joints and loads and with the wingbeat. Which neurons: BANC's annotation.
//                  How much they fire for a given angle or load: rules below. In the model they drive leg and wing motor neurons, which still move nothing.
// Rules, not the connectome:
//   the wish       hunger and boredom rise with time; they choose where to go and turn P9 on.
//   the way        walk or fly toward the goal: a compass, not the brain. Near: walk. Far: fly.
//   the eye        an object's expansion rate (LC4, LPLC1, LPLC4) and size (LPLC2) set the looming neurons' input.
//   the movement   flybody's walking and flight policies, with the stated handovers in life.js.
import { Life, ESCAPE } from './life.js';
import { nearThing, BOUNDS, SWATTER } from './things.js';

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
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
export const CRUISE = 12;                // cm above the table: clears what stands on it
// what stands on the table (things.js): x, y, radius, cm, with a margin. The fly collides with these things; goals stay out, and paths go round (way, below)
export const KEEP_OUT = BOUNDS.map(([x, y, r]) => [x, y, r + 1.5]);
export const isClear = (x, y) => KEEP_OUT.every(([a, b, r]) => Math.hypot(x - a, y - b) > r);
// The way to a goal, past the pictures: straight if nothing stands in the way, else along the tangent of the first thing in it, on the side nearer the goal.
// Returns the direction to go, and whether the way is clear. The descent takes 40 cm of flight, so flying over something at height would still bring the fly down onto it.
const ROUND = 2;   // cm of margin on the tangent
export function way(x0, y0, x1, y1) {
    const dx = x1 - x0, dy = y1 - y0, L2 = dx * dx + dy * dy || 1, goal = Math.atan2(dy, dx);
    let first = null, at = Infinity;
    for (const k of KEEP_OUT) {
        const [a, b, r] = k, u = Math.max(0, Math.min(1, ((a - x0) * dx + (b - y0) * dy) / L2));
        if (Math.hypot(x0 + u * dx - a, y0 + u * dy - b) < r && u < at) { first = k; at = u; }
    }
    if (!first) return { angle: goal, clear: true };
    const [a, b, r] = first, to = Math.atan2(b - y0, a - x0), d = Math.hypot(a - x0, b - y0), R = r + ROUND;
    const off = d > R ? Math.asin(R / d) : Math.PI / 2 + Math.min(0.8, R - d);   // inside the margin (an overshoot): round it and drift out
    return { angle: Math.abs(wrap(to + off - goal)) < Math.abs(wrap(to - off - goal)) ? to + off : to - off, clear: false };
}
const HUNGER_RATE = 1 / 45, BORED_RATE = 1 / 20, HUNGRY = 0.6, FULL = 0.05;   // per s; our choices
const EAT_RATE = 0.2;                    // hunger per s at full MN9 activation; our choice
const FLY_FROM = 6;                      // cm: farther than this, fly
const ARRIVED = 1.0;                     // cm: a wander goal is reached
const P9_HZ = 150, SUGAR_HZ = 150;
const WALK_FROM = 13, STAND_BELOW = 7;   // Hz of oDN1 through its 0.2 s filter: median about 15 under P9 (5th percentile about 8), about 5 with sugar
const TAU_ODN1 = 0.2, TAU_FAST = 0.05;   // s, rate filters
// The escape has two ways out, as in the fly: the giant fiber starts the fast takeoff (von Reyn et al. 2014), and DNp04 and DNp11 drive only the slow one
// (Dombrovski et al. 2023). In the fly one giant fiber spike is enough. The model's giant fiber also fires stray single spikes at weak input, so here it takes
// two spikes about 30 ms apart or less: 30 Hz through the 50 ms filter, where one spike gives 20. Both thresholds are our choices.
const FAST_JUMP = 30;                    // Hz, the giant fiber of one side
const SLOW_JUMP = 60;                    // Hz, mean of DNp04 and DNp11 on one side: they pass it at about 40% of full expansion input
const DODGE = 60, REFRACTORY = 0.3, SACCADE_PEAK = 43, SD = 0.04 / 2.355;
const LOOM_HZ = 100, LOOM_FULL = 2, FRONT = 0.25;   // full rate at 2 rad/s: looming neurons respond well before contact (our choice)
// Ache et al. 2019: the giant fiber adds a part that follows how fast the object grows (from LC4) and a part that is a Gaussian of its size (from LPLC2).
// The Gaussian's centre and width are our choices; one LPLC2 neuron sees at most 60 deg (Klapoetke et al. 2017).
const SIZE_PEAK = 60 * Math.PI / 180, SIZE_SD = 15 * Math.PI / 180, GROWS = 0.2;   // rad, rad, rad/s
const TURN_WALK = 4, TURN_FLY = 8, WALK_RAMP = 2;
const EXTEND = { rostrum: -1, haustellum: -1, labrum_left: 1, labrum_right: 1 };
const TILTED = 0.8;                      // up (cos of the body's tilt) below which a walking fly is not on its feet; walking stays above it
const clamp = (x, a) => Math.max(-a, Math.min(a, x));
// Body -> its sensory neurons, as firing rates up to FEEL_HZ. Each rule follows what the sensor is known to report; every size is our choice.
//   claw neurons of the femoral chordotonal organ: where the tibia is (Mamiya et al. 2018). Here: how far it is from the middle of its range.
//   hook neurons: that the tibia moves. Here: its angular speed.               hair plates: a joint near the end of its range. Here: the coxa.
//   campaniform sensilla: load on the leg. Here: the force on the foot.       touch neurons of the taste bristles: the foot is on the ground.
//   halteres and wing base: campaniform sensilla that fire with each wingbeat (Fayyazuddin & Dickinson 1996). Here: the wingbeat frequency, not locked to its phase.
// We do not know which claw and hook neurons prefer which direction, so all of a leg's fire alike.
const FEEL_HZ = 50, FEEL_EVERY = 10, CLAW_FULL = 0.8, HOOK_FULL = 40, PLATE_NEAR = 0.3, LOAD_FULL = 1, TOUCH_HZ = 30;   // Hz, ms, rad, rad/s, rad, dyn (the fly weighs 0.97), Hz
const LEGS = ['T1_left', 'T1_right', 'T2_left', 'T2_right', 'T3_left', 'T3_right'];   // the order of life.touch
// Each spike raises a neuron's threshold by 0.05 mV, decaying over 200 ms. Not in Shiu's model. Without it, looms left the model firing for ever in 5 of 6 runs; with it, in 0 of 192 (several inputs, 1 and 5 s). Our choice, by a sweep.
export const ADAPT = { step: 0.05, tau: 200 };

export class Day {
    constructor(mujoco, model, data, walker, flyer, brain, neurons, seed = 1) {
        Object.assign(this, { mj: mujoco, model, data, brain, W: walker, F: flyer, n: neurons, rngState: seed >>> 0 || 1 });
        this.life = new Life(mujoco, model, data, walker, flyer);
        brain.set_adaptation(ADAPT.step, ADAPT.tau);
        this.life.rnd = () => this.rnd();
        if (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, 'thing_bowl') >= 0) {     // the model has the things and the swatter (things.js)
            const o = () => this.object;
            this.life.near = (x, y, z) => nearThing(x, y, z) || (o() && Math.hypot(x - o().at[0], y - o().at[1], z - o().at[2]) < 1.5 * SWATTER.half + 1);
            this.swatter = { k: model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'swatter')], touch: model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR.value, 'swatter')] };
        }
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
        const N = neurons.named;
        this.slowL = group([...N.DNp04.left, ...N.DNp11.left], TAU_FAST); this.slowR = group([...N.DNp04.right, ...N.DNp11.right], TAU_FAST);
        this.gfL = group(L.GF.filter(i => N.DNp01.left.includes(i)), TAU_FAST); this.gfR = group(L.GF.filter(i => N.DNp01.right.includes(i)), TAU_FAST);
        this.groups = [...Object.values(this.mouth), this.oDN1, this.GF, this.gfL, this.gfR, this.dnL, this.dnR, this.slowL, this.slowR];
        // the eye's two kinds of input, per side: neurons that follow how fast an object grows, and LPLC2, which follows its size
        this.eye = { left: Object.values(neurons.looms).flatMap(v => v.left), right: Object.values(neurons.looms).flatMap(v => v.right), sizeleft: L.LPLC2.left, sizeright: L.LPLC2.right };
        this.foods = []; this.input = { left: 0, right: 0, sizeleft: 0, sizeright: 0 }; this.brainBase = 0;
        const joint = k => { const j = id('mjOBJ_JOINT', 'walker/' + k); return { q: m.jnt_qposadr[j], v: m.jnt_dofadr[j], lo: m.jnt_range[2 * j], hi: m.jnt_range[2 * j + 1] }; };
        const F_ = neurons.feel;
        this.feels = !!F_;
        if (F_) {
            this.legSense = LEGS.map((k, i) => ({ tibia: joint('tibia_' + k), coxa: joint('coxa_' + k), touch: this.life.touch[i], cells: F_.legs[k], hz: {}, speed: 0, load: 0 }));
            this.beatSense = { cells: [...F_.haltere.left, ...F_.haltere.right, ...F_.wing.left, ...F_.wing.right], hz: 0 };
        }
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
        this.saccade = null; this.upsideDown = 0; this.tilted = 0; this.retry = 0; this.tries = 0;
        this.object = null; this.prevSize = null; for (const k in this.eye) this.applyEye(k, 0);
    }

    // Reset: a new day from the start, as on loading the page.
    newDay() {
        this.brain.reset(); this.brainBase = 0; this.t = 0;
        this.foods = []; for (const [x, y] of FOODS) this.addFood(x, y);
        this.hunger = 0.4; this.bored = 0; this.putBacks = this.getUps = this.jumps = this.hits = 0;
        this.reset(); this.calm();
    }

    // Back to rest: for Reset, and for tools that want trials from rest. The day itself never needs it now that the model adapts (ADAPT).
    calm() {
        this.brainBase += this.brain.step_count(); this.brain.reset();
        for (const g of this.groups) g.rate = 0;
        for (const i of this.n.life.P9) this.brain.set_rate(i, this.p9 || 0);
        for (const i of this.n.life.sugar) this.brain.set_rate(i, this.sugar ? SUGAR_HZ : 0);
        for (const k in this.eye) for (const i of this.eye[k]) this.brain.set_rate(i, this.input[k]);
        if (this.feels) { for (const l of this.legSense) l.hz = {}; this.beatSense.hz = -1; this.feel(0); }   // the engine keeps its rates through a reset: nothing is assumed, feel() sets every one again
    }

    // the body's state -> its sensory neurons (the rules at the top). Speed and load are smoothed over about 5 ms; rates go in steps of 10 Hz.
    feel(ms) {
        const d = this.data, s = d.sensordata, brain = this.brain, mode = this.life.mode, on = this.numb ? 0 : 1;   // numb: a tool's control, with these neurons silent
        for (const l of this.legSense) { l.speed += 0.2 * (Math.abs(d.qvel[l.tibia.v]) - l.speed); l.load += 0.2 * (s[l.touch] - l.load); }
        if (ms % FEEL_EVERY) return;
        const steps = x => on * 10 * Math.round(FEEL_HZ / 10 * Math.min(1, Math.max(0, x)));
        for (const l of this.legSense) {
            const c = d.qpos[l.coxa.q];
            const want = { claw: steps(Math.abs(d.qpos[l.tibia.q] - (l.tibia.lo + l.tibia.hi) / 2) / CLAW_FULL), hook: steps(l.speed / HOOK_FULL), hairplate: steps(1 - Math.min(c - l.coxa.lo, l.coxa.hi - c) / PLATE_NEAR), campaniform: steps(l.load / LOAD_FULL), touch: l.load > 0.01 ? on * TOUCH_HZ : 0 };
            for (const k in want) if (want[k] !== l.hz[k]) { l.hz[k] = want[k]; for (const i of l.cells[k]) brain.set_rate(i, want[k]); }
        }
        const beat = mode === 'launch' || mode === 'fly' || mode === 'descend' ? on * 10 * Math.round(this.F.freq / 10) : 0, b = this.beatSense;
        if (beat !== b.hz) { b.hz = beat; for (const i of b.cells) brain.set_rate(i, beat); }
    }

    setP9(hz) { if (this.p9 === hz) return; this.p9 = hz; for (const i of this.n.life.P9) this.brain.set_rate(i, hz); }
    setSugar(on) { if (this.sugar === on) return; this.sugar = on; for (const i of this.n.life.sugar) this.brain.set_rate(i, on ? SUGAR_HZ : 0); }
    applyEye(side, hz) { if (this.input[side] === hz) return; this.input[side] = hz; for (const i of this.eye[side]) this.brain.set_rate(i, hz); }

    addFood(x, y, r = 0.45, amount = 1) { this.foods.push({ x, y, r, amount }); }
    // an object coming down at the fly: fromSide is the angle from the fly's heading, positive to its left, 0 = straight above
    swat(fromSide = 0, dist = 30, speed = 30, radius = SWATTER.half) {
        const f = this.F.freeq, q = this.data.qpos, h = this.life.heading(), a = h + fromSide;
        const tilt = Math.abs(fromSide) > 0 ? 0.8 : 0;
        const dir = [Math.sin(tilt) * Math.cos(a), Math.sin(tilt) * Math.sin(a), Math.cos(tilt)];
        const lead = this.life.mode === 'fly' ? this.F.speed * dist / speed : 0, aim = [q[f] + lead * Math.cos(h), q[f + 1] + lead * Math.sin(h), q[f + 2]];
        this.object = { p: aim.map((v, i) => v + dist * dir[i]), v: dir.map(v => -speed * v), radius, until: this.t + dist / speed + 0.25, hit: false, aim };
        this.prevSize = null; this.placeSwatter();
    }

    // Where the swatter is and how it is turned: its face follows its path, and comes flat as it nears the table. The page draws this, and the physics has it.
    placeSwatter() {
        const o = this.object, w = this.swatter, d = this.data;
        if (o) {
            const flat = Math.min(1, Math.max(0, o.p[2] / 6)), sp = Math.hypot(...o.v), n = [-o.v[0] * flat / sp, -o.v[1] * flat / sp, 1 - flat - o.v[2] * flat / sp], l = Math.hypot(...n);
            o.at = [o.p[0], o.p[1], Math.max(o.p[2], SWATTER.thick / 2)]; o.normal = n.map(v => v / l);
        }
        if (!w) return;
        const at = o ? o.at : [0, 0, -60], [x, y, z] = o ? o.normal : [0, 0, 1], c = Math.sqrt((1 + z) / 2) || 1;   // the turn from the z axis to the normal (the normal never points down)
        for (let i = 0; i < 3; i++) d.mocap_pos[3 * w.k + i] = at[i];
        d.mocap_quat[4 * w.k] = c; d.mocap_quat[4 * w.k + 1] = -y / (2 * c); d.mocap_quat[4 * w.k + 2] = x / (2 * c); d.mocap_quat[4 * w.k + 3] = 0;
    }

    see(dt) {
        const o = this.object;
        if (!o) { for (const k in this.eye) this.applyEye(k, 0); return; }
        const x = this.data.xpos, b = this.head, dx = o.p[0] - x[3 * b], dy = o.p[1] - x[3 * b + 1], dz = o.p[2] - x[3 * b + 2], dist = Math.hypot(dx, dy, dz);
        const size = 2 * Math.atan2(o.radius, Math.max(dist, o.radius)), grow = this.prevSize === null ? 0 : (size - this.prevSize) / dt;
        this.prevSize = size;
        const az = wrap(Math.atan2(dy, dx) - this.life.heading()), overhead = Math.hypot(dx, dy) < dz * 0.6;
        const seen = overhead || Math.abs(az) <= 2.5, steps = x => 10 * Math.round(LOOM_HZ / 10 * Math.min(1, Math.max(0, x)));   // steps of 10 Hz: setting hundreds of rates each ms is slow
        const hz = seen ? steps(grow / LOOM_FULL) : 0, big = seen && grow > GROWS ? steps(Math.exp(-((size - SIZE_PEAK) ** 2) / (2 * SIZE_SD ** 2))) : 0;
        const left = overhead || az > -FRONT, right = overhead || az < FRONT;
        this.applyEye('left', left ? hz : 0); this.applyEye('sizeleft', left ? big : 0);
        this.applyEye('right', right ? hz : 0); this.applyEye('sizeright', right ? big : 0);
    }

    somewhere(why, from = null, atLeast = 0) { for (;;) { const x = (2 * this.rnd() - 1) * TABLE.x, y = (2 * this.rnd() - 1) * TABLE.y; if (isClear(x, y) && (!from || Math.hypot(x - from[0], y - from[1]) >= atLeast)) return { x, y, why }; } }
    // where an escape goes: a place at least 12 cm away. A nearer one made the fly turn hard and come down as soon as it was in the air, and it fell.
    away() { return this.somewhere('escape', this.F.position(), 12); }

    // the wish: which goal, by rule
    choose() {
        const { life } = this, p = life.F.position();
        if (this.eating) return;
        if (this.goal?.why === 'escape' && life.mode !== 'walk') return;   // an escape first gets away: hunger must not call the fly back to the sugar it jumped from
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
        const under = life.mode === 'walk' ? this.foods.find(o => Math.hypot(o.x - lx, o.y - ly) < o.r + (o === this.eating ? 0.3 : 0)) : null;
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
        // the way (a goal marked straight is a tool's: tools/things.mjs sends the fly at a thing)
        let dist = Infinity, bearing = 0, clear = true;
        if (this.goal) { const w = this.goal.straight ? { angle: Math.atan2(this.goal.y - p[1], this.goal.x - p[0]), clear: true } : way(p[0], p[1], this.goal.x, this.goal.y); dist = Math.hypot(this.goal.x - p[0], this.goal.y - p[1]); bearing = wrap(w.angle - life.heading()); clear = w.clear; }
        if (this.goal && !this.goal.food && dist < ARRIVED && !air) this.goal = null;
        // the jump: the giant fiber starts the fast takeoff, also when it fires during a slow rear-up; DNp04 and DNp11 start the slow one
        if (!this.noJump && (life.mode === 'walk' || life.mode === 'rear')) {
            const fast = Math.max(this.gfL.rate, this.gfR.rate) > FAST_JUMP, slow = Math.max(this.slowL.rate, this.slowR.rate) > SLOW_JUMP;
            if (fast && life.hurry()) this.fastJumps = (this.fastJumps || 0) + 1;
            if (life.mode === 'walk' && (fast || slow)) {
                if (this.goal?.why !== 'escape') this.goal = this.away();
                if (this.retry <= 0) { this.jumps = (this.jumps || 0) + 1; if (fast) this.fastJumps = (this.fastJumps || 0) + 1; this.retry = 0.5; life.takeOff(fast ? ESCAPE : undefined); }
            }
        }
        if (life.mode === 'walk') {
            this.retry -= dt;
            // on top of something (an overshoot put it there): the walking policy can stand on a flat top but must not walk off it, so the fly leaves by air
            const perch = life.floor > 0;
            if (perch && !this.goal && !this.eating) this.goal = this.somewhere('boredom');
            if (this.goal && (dist > FLY_FROM || perch) && this.retry <= 0 && this.walking && life.t > 1 && life.up() > 0.95 && (this.v || 0) < 0.05) { life.takeOff(); this.retry = 0.5; }
            // the walking policy falls if turned below 0.8 cm/s or started at speed: speed ramps, and turning waits for it
            const go = this.goal && this.walking && !this.eating && !perch && dist <= FLY_FROM && life.t > 0.6, want = go ? (Math.abs(bearing) > 1 ? 1.5 : 2) : 0;
            this.v = (this.v || 0) + Math.max(-4 * dt, Math.min(WALK_RAMP * dt, want - (this.v || 0)));
            life.speed = this.v;
            life.yaw = go && this.v >= 0.8 ? clamp(4 * bearing, Math.min(TURN_WALK, 2 * this.v)) : 0;   // tools/walk_turns.mjs: 1 cm/s at 4 rad/s falls, 1.5 at 3 and 2 at 4 do not
        } else {
            this.v = life.speed = 0;   // on landing the walking speed ramps up from a stand again
            life.yaw = this.goal ? clamp(3 * bearing, TURN_FLY) : 0;
            // over or beside something that stands on the table (after a takeoff from its top, or an overshoot): stay high, and do not start down
            const over = life.near && !this.goal?.straight && nearThing(p[0], p[1], 0, 1);
            if (this.goal) life.cruise = dist < 25 && !over ? Math.max(1.5, Math.min(CRUISE, dist * 0.25)) : CRUISE;
            // down when the goal is in sight and within the run, facing it or not: a goal inside the turning circle is never faced. Overshoot is walked or hopped back.
            if (life.mode === 'fly' && this.goal && clear && !over && dist < life.landingRun() + 1 && life.t > 0.5) life.land();
            if (life.mode === 'fly' && !this.goal) this.goal = this.away();
            const diff = this.dnL.rate - this.dnR.rate, q = this.saccade;
            if (life.mode === 'fly' && Math.abs(diff) > DODGE && (!q || this.t - q.at > REFRACTORY)) this.saccade = { at: this.t, sign: -Math.sign(diff) };
            if (this.saccade) { const { at, sign } = this.saccade, u = this.t - (at + 2 * 0.04); if (Math.abs(u) < 4 * SD) life.yaw = sign * SACCADE_PEAK * Math.exp(-u * u / (2 * SD * SD)); }
        }
        // the body
        this.bodyAhead -= dt;
        while (this.bodyAhead < 0) { this.bodyAhead += life.step(); this.onControl?.(); }
        // the object
        const o = this.object;
        if (o) {
            if (o.p[2] > 0) for (let i = 0; i < 3; i++) o.p[i] += o.v[i] * dt;   // it stops on the table
            const b = F.position();
            const along = (b[0] - o.p[0]) * o.v[0] + (b[1] - o.p[1]) * o.v[1] + (b[2] - o.p[2]) * o.v[2], sp = Math.hypot(...o.v), off = Math.hypot(b[0] - o.p[0] - o.v[0] * along / sp / sp, b[1] - o.p[1] - o.v[1] * along / sp / sp, b[2] - o.p[2] - o.v[2] * along / sp / sp);
            // a hit: the swatter is solid, and its touch sensor reports a force. (A model with no swatter in it: the fly is inside the square as it passes.)
            if (this.swatter ? d.sensordata[this.swatter.touch] > 0 : Math.abs(along / sp) < 0.25 && off < o.radius) o.hit = true;
            if (o.hit && !o.counted) { o.counted = true; this.hits = (this.hits || 0) + 1; }
            if (this.t > o.until) { this.lastSwat = o.hit ? 'hit' : 'missed'; this.object = null; }
            this.placeSwatter();
        }
        this.see(dt);
        if (this.feels) this.feel(Math.round(this.t * 1000));
        // a fall: the policies can't catch a tumble. The fly gets up by itself; if it cannot, it is put back, stated and counted on the page.
        const bad = !Number.isFinite(d.qpos[f]) || life.up() < 0.2 || p[2] < -0.2 || p[2] > 40 || (life.mode === 'fly' && life.t > 0.5 && p[2] < 0.25);
        this.upsideDown = Math.max(0, this.upsideDown + (bad ? dt : -dt / 3));      // a tumble flickers through upright: this adds it up
        // a called-off rear-up can also leave it on its side or sitting back (up 0.3-0.7): the walking policy never stands up from there. Walking keeps up > 0.8.
        this.tilted = life.mode === 'walk' && life.up() < TILTED ? this.tilted + dt : 0;
        // down, and on the table: the fly gets up by itself (life.getUp). It has three tries; a try that ends upright for a second is counted.
        const lost = !Number.isFinite(d.qpos[f]) || p[2] < -0.2 || p[2] > 40;
        if (life.mode === 'getup') this.upsideDown = 0;
        if (this.tries && life.mode === 'walk' && life.t > 1 && life.up() > 0.9) { this.tries = 0; this.getUps = (this.getUps || 0) + 1; }
        if (!lost && (this.upsideDown > 0.4 || this.tilted > 1) && p[2] < 1 + life.floor && (this.tries || 0) < 3 && !this.helpless) {
            this.upsideDown = this.tilted = 0;
            if (life.up() <= 0.8) { this.tries = (this.tries || 0) + 1; life.getUp(); }
            else if (life.mode !== 'walk') { life.mode = 'touch'; life.t = 0; }      // it came down on its feet: the wings shut, and it walks
        } else if (lost || ((this.upsideDown > 0.4 || this.tilted > 1) && (p[2] < 1 + life.floor || life.mode === 'walk'))) {
            this.putBacks = (this.putBacks || 0) + 1; this.tries = 0;
            const ok = Number.isFinite(d.qpos[f]) && isClear(p[0], p[1]) && Math.abs(p[0]) < TABLE.x && Math.abs(p[1]) < TABLE.y, s = ok ? null : this.somewhere('');
            this.reset(ok ? p[0] : s.x, ok ? p[1] : s.y, this.rnd() * 6.28);
        }
        return spikes;
    }
}
