// The racing simulation, shared by the page and the Node tools: a JS port of
// racecar/env.py (the plant) and racecar/standalone/multitrack_env.py (track
// state and the policy's 73 inputs) from mjx-rl-experiments.
//
// The car is a transient bicycle model: longitudinal, lateral and yaw motion,
// front and rear tires on friction ellipses with load transfer, downforce and
// drag, power and brake limits, and rate-limited steering, throttle and brake.
// One control tick is 20 ms, integrated as two 10 ms midpoint steps. The tires
// are load-sensitive (training/plant.py; the repo's grew grip and stiffness in
// proportion to load, which made the car spin under braking).

export const DT = 0.02;
const STEER_MAX = 0.38, STEER_RATE_MAX = 1.8;
const CENTER_ALLOWANCE = 1.2, MAP_TOLERANCE = 0.15;
const SCALE = 0.76;  // the speed profile scale the policy was trained to read

// vehicle_params.py
const M = 746, IZZ = 850, LF = 2.0215318733873717, LR = 1.704468126612628;
const WB = LF + LR, H_COG = 0.3, RHO = 1.2, AREA = 1.5;
const CD = 0.995174643413709, CL = 2.9658746844902901, AERO_OFFSET = 0.09244576459714433;
const POWER = (623.7009898646145 + 120) * 1000, DRIVE_MAX = 18500;
const BRAKE_BIAS = 0.60, BRAKE_EFF = 0.98, MU_X = 1.86, MU_Y = 1.82;
const GRIP_FRONT = 0.96, GRIP_REAR = 1.05, ROLLING = 0.015, G = 9.81;

const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// ---- plant -----------------------------------------------------------------
// state: [x, y, yaw, u, v, yawRate, steer, throttle, brake]; u, v body-frame.
// grip scales tire capacity (1 on asphalt; the page lowers it on grass).

// training/plant.py: per-axle grip falls with load, cornering stiffness saturates
const FZ0 = 6000, LOAD_SENSITIVITY = 0.15, STIFFNESS_PEAK_LOAD = 1.6, SLIP_STIFFNESS = 12;
const STIFFNESS_REF = Math.sin(2 * Math.atan(1 / STIFFNESS_PEAK_LOAD));
const tireGrip = (mu0, fz) => mu0 * Math.max(1 - LOAD_SENSITIVITY * (fz / FZ0 - 1), 0.5);
function lateral(mu0, fz, alpha) {
    const peak = Math.max(tireGrip(mu0, fz) * fz, 1);
    const stiff = SLIP_STIFFNESS * mu0 * FZ0 * Math.sin(2 * Math.atan(fz / (STIFFNESS_PEAK_LOAD * FZ0))) / STIFFNESS_REF;
    return [peak * Math.tanh(stiff * alpha / peak), peak];
}

function ellipse(fx, fy, cx, cy) {
    const util = Math.hypot(fx / Math.max(cx, 1), fy / Math.max(cy, 1));
    const s = Math.min(1, 1 / Math.max(util, 1));
    return [fx * s, fy * s, util * s];
}

// Body forces from the tires and aero; also fills `out` with diagnostics.
export function forces(state, grip = 1, extraDrag = 0, out = null) {
    const [, , , u, v, r, steer, throttle, brake] = state;
    const safeU = Math.max(u, 4);
    const downforce = 0.5 * RHO * AREA * CL * (u * u + v * v);
    const drag = 0.5 * RHO * AREA * CD * u * Math.abs(u);
    const load = M * G + downforce;
    const rolling = (ROLLING + extraDrag) * load * Math.tanh(u / 2);
    const engine = throttle * Math.min(DRIVE_MAX, POWER / safeU);
    const alphaF = steer - Math.atan2(v + LF * r, safeU);
    const alphaR = -Math.atan2(v - LR * r, safeU);
    const fracStatic = LR / WB, fracAero = (LR - AERO_OFFSET) / WB;
    let ax = 0, fxTotal = 0, bodyY = 0, moment = 0, utilF = 0, utilR = 0, rear = null;
    for (let k = 0; k < 2; k++) {
        const transfer = M * H_COG * ax / WB;
        const fzF = clip(M * G * fracStatic + downforce * fracAero - transfer, 0.05 * load, 0.95 * load);
        const fzR = load - fzF;
        const cxF = tireGrip(MU_X * grip, fzF) * fzF, cxR = tireGrip(MU_X * grip, fzR) * fzR;
        const brakeForce = brake * BRAKE_EFF * (cxF + cxR);
        const [reqFyF, cyF] = lateral(GRIP_FRONT * MU_Y * grip, fzF, alphaF);
        const [reqFyR, cyR] = lateral(GRIP_REAR * MU_Y * grip, fzR, alphaR);
        const [fxF, fyF, uF] = ellipse(-brakeForce * BRAKE_BIAS, reqFyF, cxF, cyF);
        const [fxR, fyR, uR] = ellipse(engine - brakeForce * (1 - BRAKE_BIAS), reqFyR, cxR, cyR);
        const c = Math.cos(steer), s = Math.sin(steer);
        const frontX = fxF * c - fyF * s, frontY = fxF * s + fyF * c;
        fxTotal = fxR + frontX - drag - rolling;
        bodyY = fyR + frontY;
        moment = LF * frontY - LR * fyR;
        ax = fxTotal / M + r * v;
        utilF = uF; utilR = uR;
        rear = { cx: cxR, cy: cyR, fy: reqFyR };
    }
    if (out) Object.assign(out, { utilFront: utilF, utilRear: utilR, alphaFront: alphaF, alphaRear: alphaR, downforce, rear, engineMax: Math.min(DRIVE_MAX, POWER / safeU) });
    return [fxTotal, bodyY, moment];
}

function derivative(state, steerCmd, pedal, grip, extraDrag, d) {
    const [, , yaw, u, v, r, steer, throttle, brake] = state;
    const [fx, fy, mz] = forces(state, grip, extraDrag);
    d[0] = u * Math.cos(yaw) - v * Math.sin(yaw);
    d[1] = u * Math.sin(yaw) + v * Math.cos(yaw);
    d[2] = r;
    d[3] = fx / M + r * v;
    d[4] = fy / M - r * u;
    d[5] = mz / IZZ;
    d[6] = clip((steerCmd - steer) * 12, -STEER_RATE_MAX, STEER_RATE_MAX);
    d[7] = clip((Math.max(pedal, 0) - throttle) * 8, -5, 5);
    d[8] = clip((Math.max(-pedal, 0) - brake) * 12, -8, 8);
}

const k1 = new Float64Array(9), k2 = new Float64Array(9), mid = new Float64Array(9);

// One 20 ms control tick in place. steer and pedal are the policy's [-1, 1]
// actions (pedal > 0 throttle, < 0 brake).
export function integrate(state, steer, pedal, grip = 1, extraDrag = 0) {
    const steerCmd = STEER_MAX * clip(steer, -1, 1);
    pedal = clip(pedal, -1, 1);
    const h = DT / 2;
    for (let sub = 0; sub < 2; sub++) {
        derivative(state, steerCmd, pedal, grip, extraDrag, k1);
        for (let i = 0; i < 9; i++) mid[i] = state[i] + 0.5 * h * k1[i];
        derivative(mid, steerCmd, pedal, grip, extraDrag, k2);
        for (let i = 0; i < 9; i++) state[i] += h * k2[i];
        state[2] = wrap(state[2]);
        state[3] = clip(state[3], 2, 105);
        state[4] = clip(state[4], -30, 30);
        state[5] = clip(state[5], -3.5, 3.5);
        state[6] = clip(state[6], -STEER_MAX, STEER_MAX);
        state[7] = clip(state[7], 0, 1);
        state[8] = clip(state[8], 0, 1);
    }
    return state;
}

// ---- track -------------------------------------------------------------------
// Arrays from model/tracks/<name>.bin in the order of tracks.json "fields":
// racing line x, y; nearest centerline point and its tangent; asphalt widths
// right/left of the centerline; racing-line tangent, heading and curvature;
// two speed profiles; arclength s. The policy follows the racing line.

const POLICY_LOOKAHEAD = [15, 40, 80, 140, 220];
const MAP_LOOKAHEAD = [0, 50, 120, 220];
const EXT_POLICY_LOOKAHEAD = [300, 400, 520];
const EXT_MAP_LOOKAHEAD = [300, 400, 520];
const BRAKING_BINS = [[0, 30], [30, 70], [70, 140], [140, 240], [240, 350]];

export class Track {
    constructor(meta, fields, data) {
        Object.assign(this, meta);  // name, count, length, spacing
        const n = meta.count;
        fields.forEach((f, i) => { this[camel(f)] = data.subarray(i * n, (i + 1) * n); });
        // The environment computes index offsets in float32: int(L / spacing).
        const f = Math.fround, sp = f(meta.spacing);
        const offsets = list => list.map(L => Math.trunc(f(f(L) / sp)));
        this.policyOff = offsets(POLICY_LOOKAHEAD);
        this.mapOff = offsets(MAP_LOOKAHEAD);
        this.extPolicyOff = offsets(EXT_POLICY_LOOKAHEAD);
        this.extMapOff = offsets(EXT_MAP_LOOKAHEAD);
        this.bins = BRAKING_BINS.map(([a, b]) => {
            const d = [];
            for (let k = 0, m = Math.ceil((b + 0.1 - a) / 3); k < m; k++) d.push(a + 3 * k);
            return { d, off: offsets(d) };
        });
    }

    at(i) { return ((i % this.count) + this.count) % this.count; }

    nearest(x, y, prev) {
        let best = prev, bestD = Infinity;
        for (let o = -7; o < 10; o++) {
            const i = this.at(prev + o);
            const d = (this.x[i] - x) ** 2 + (this.y[i] - y) ** 2;
            if (d < bestD) { bestD = d; best = i; }
        }
        return best;
    }

    // s along the racing line, offset from the racing line, offset from the
    // centerline (left positive) and the distance left before the track limit.
    state(x, y, i) {
        const dx = x - this.x[i], dy = y - this.y[i];
        const tx = this.tangentX[i], ty = this.tangentY[i];
        const lateral = -ty * dx + tx * dy;
        const along = clip(tx * dx + ty * dy, -0.55 * this.spacing, 0.55 * this.spacing);
        const cdx = x - this.centerX[i], cdy = y - this.centerY[i];
        const center = -this.centerTangentY[i] * cdx + this.centerTangentX[i] * cdy;
        const available = (center >= 0 ? this.widthLeft[i] : this.widthRight[i]) + CENTER_ALLOWANCE;
        return { s: this.s[i] + along, lateral, center, margin: available - Math.abs(center) };
    }
}

const camel = s => s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());

// The policy's 73 inputs (MultiTrackRaceCar._observation with speed profile,
// extended preview, actuator history and braking preview).
export function observe(track, state, index, previousAction, out = new Float64Array(73)) {
    const [x, y, yaw, u, v, r, steer, throttle, brake] = state;
    const ts = track.state(x, y, index);
    const c = Math.cos(yaw), s = Math.sin(yaw);
    let k = 0;
    out[k++] = u / 100; out[k++] = v / 20; out[k++] = r / 3;
    out[k++] = Math.atan2(v, Math.max(u, 2)) / 0.35;
    out[k++] = ts.lateral / 6; out[k++] = ts.center / 7; out[k++] = ts.margin / 6;
    out[k++] = wrap(track.heading[index] - yaw) / 0.5;
    const path = (offsets, dists) => offsets.forEach((o, j) => {
        const i = track.at(index + o), dx = track.x[i] - x, dy = track.y[i] - y;
        out[k++] = (c * dx + s * dy) / dists[j];
        out[k++] = (-s * dx + c * dy) / dists[j];
    });
    const map = offsets => offsets.forEach(o => {
        const i = track.at(index + o);
        out[k++] = track.curvature[i] * 80; out[k++] = track.widthLeft[i] / 10; out[k++] = track.widthRight[i] / 10;
    });
    const speed = offsets => offsets.forEach(o => { out[k++] = SCALE * track.legacyTargetSpeed[track.at(index + o)] / 100; });
    path(track.policyOff, POLICY_LOOKAHEAD);
    map(track.mapOff);
    speed(track.mapOff);
    path(track.extPolicyOff, EXT_POLICY_LOOKAHEAD);
    map(track.extMapOff);
    speed(track.extMapOff);
    out[k++] = steer / STEER_MAX; out[k++] = throttle; out[k++] = brake;
    out[k++] = previousAction[0]; out[k++] = previousAction[1];
    out[k++] = clip(SCALE * SCALE * track.targetAccel[index] / 20, -1, 1);
    for (const { d, off } of track.bins) {
        let minSpeed = Infinity, minDist = 0;
        off.forEach((o, j) => {
            const v = SCALE * track.targetSpeed[track.at(index + o)];
            if (v < minSpeed) { minSpeed = v; minDist = d[j]; }
        });
        const need = Math.max((u * u - minSpeed * minSpeed) / (2 * Math.max(minDist, track.spacing)), 0);
        out[k++] = minSpeed / 100; out[k++] = minDist / 350; out[k++] = clip(need / 25, 0, 1);
    }
    return out;
}

// ---- policy ------------------------------------------------------------------
// Brax PPO actor: 73 -> 512 -> 512 -> 4 with swish; the deterministic action
// is tanh of the first two outputs (the last two are exploration noise).
export class Policy {
    constructor(meta, weights) {
        let k = 0;
        this.layers = meta.layers.map(({ in: n, out }) => {
            const w = weights.subarray(k, k += n * out), b = weights.subarray(k, k += out);
            return { n, out, w, b, y: new Float64Array(out) };
        });
        this.action = new Float64Array(2);
    }

    act(obs) {
        let x = obs;
        const last = this.layers.length - 1;
        this.layers.forEach(({ n, out, w, b, y }, l) => {
            for (let o = 0; o < out; o++) {
                let s = b[o];
                for (let i = 0, r = o * n; i < n; i++) s += w[r + i] * x[i];
                y[o] = l === last ? s : s / (1 + Math.exp(-s));
            }
            x = y;
        });
        this.action[0] = Math.tanh(x[0]);
        this.action[1] = Math.tanh(x[1]);
        return this.action;
    }
}

// ---- a car on a track -----------------------------------------------------------
// Progress and failure rules as MultiTrackRaceCar.step. The fixed start is the
// qualifier's: index 0 on the racing line at 0.985 x the 0.76-scaled profile
// speed, throttle 0.75. `lateral` moves the car sideways (left positive).
export class Car {
    constructor(track, lateral = 0, back = 0) {
        this.track = track;
        const i = track.at(-Math.round(back / track.spacing));
        const tx = track.tangentX[i], ty = track.tangentY[i];
        this.state = new Float64Array([
            track.x[i] - ty * lateral, track.y[i] + tx * lateral, track.heading[i],
            SCALE * track.targetSpeed[0] * 0.985, 0, 0, 0, 0.75, 0,
        ]);
        this.index = i;
        this.s = track.state(this.state[0], this.state[1], i).s;
        this.progress = i === 0 ? 0 : -(track.length - this.s);
        this.time = 0;
        this.previousAction = new Float64Array(2);
        this.obs = new Float64Array(73);
        this.margin = 0;
        this.failed = null;  // 'off' | 'spin' once the environment would end the episode
    }

    observe() { return observe(this.track, this.state, this.index, this.previousAction, this.obs); }

    step(steer, pedal, grip = 1, extraDrag = 0) {
        const st = this.state;
        integrate(st, steer, pedal, grip, extraDrag);
        this.previousAction[0] = clip(steer, -1, 1);
        this.previousAction[1] = clip(pedal, -1, 1);
        this.index = this.track.nearest(st[0], st[1], this.index);
        const ts = this.track.state(st[0], st[1], this.index);
        const L = this.track.length;
        let ds = ts.s - this.s;
        if (ds < -0.5 * L) ds += L;
        if (ds > 0.5 * L) ds -= L;
        this.progress += clip(ds, -2.5, 3);
        this.s = ts.s;
        this.margin = ts.margin;
        this.center = ts.center;
        this.time += DT;
        const beta = Math.abs(Math.atan2(st[4], Math.max(st[3], 2)));
        if (ts.margin < -MAP_TOLERANCE) this.failed = 'off';
        else if (beta > 0.5 || Math.abs(st[5]) > 3.25 || st[3] < 3) this.failed = 'spin';
        else this.failed = null;
        return ts;
    }
}

// The most throttle the rear tires can take on top of their cornering force
// (`limit` of the friction ellipse). Used only for your car: a key is all or
// nothing, a foot is not.
const tc = {};
export function tractionLimit(state, grip = 1, limit = 0.95) {
    forces(state, grip, 0, tc);
    const { cx, cy, fy } = tc.rear;
    const lateral = Math.min(Math.abs(fy) / Math.max(cy, 1), limit);
    return Math.min(1, cx * Math.sqrt(Math.max(limit * limit - lateral * lateral, 0)) / Math.max(tc.engineMax, 1));
}

// ---- your inputs --------------------------------------------------------------
// How hard this car can corner at each speed: the steering input that gives the
// most steady lateral g, with the speed held, before it slides.
// [km/h, steering input, lateral g]: tools/corner-table.mjs
export const CORNERING = [
    [20, 1.000, 0.33],
    [30, 1.000, 0.74],
    [40, 1.000, 1.26],
    [50, 1.000, 1.74],
    [60, 0.890, 1.88],
    [70, 0.745, 1.97],
    [80, 0.590, 2.13],
    [90, 0.470, 2.12],
    [100, 0.395, 2.19],
    [110, 0.345, 2.28],
    [120, 0.305, 2.36],
    [130, 0.275, 2.45],
    [140, 0.250, 2.55],
    [150, 0.230, 2.66],
    [160, 0.210, 2.75],
    [170, 0.195, 2.87],
    [180, 0.180, 2.98],
    [190, 0.165, 3.08],
    [200, 0.155, 3.21],
    [210, 0.140, 3.31],
    [220, 0.130, 3.44],
    [230, 0.120, 3.56],
    [240, 0.110, 3.71],
    [250, 0.100, 3.88],
    [260, 0.085, 3.91],
    [270, 0.075, 3.99],
    [280, 0.065, 4.08],
    [290, 0.055, 4.02],
    [300, 0.045, 3.94],
    [310, 0.035, 3.81],
    [320, 0.025, 3.64],
    [330, 0.015, 3.22],
    [340, 0.010, 2.78],
];

// {steer, g} at speed u (m/s), interpolated; above ~280 km/h the table can't
// hold speed in a corner, so the steering is kept from shrinking below 0.05.
export function cornering(u) {
    const kmh = u * 3.6, T = CORNERING;
    if (kmh <= T[0][0]) return { steer: T[0][1], g: T[0][2] };
    for (let i = 1; i < T.length; i++) {
        if (kmh <= T[i][0]) {
            const f = (kmh - T[i - 1][0]) / (T[i][0] - T[i - 1][0]);
            return { steer: Math.max(0.05, T[i - 1][1] + f * (T[i][1] - T[i - 1][1])), g: T[i - 1][2] + f * (T[i][2] - T[i - 1][2]) };
        }
    }
    const last = T[T.length - 1];
    return { steer: Math.max(0.05, last[1]), g: last[2] };
}

// Trail braking for your car: the share of full cornering your steering asks
// for (CORNERING) leaves sqrt(1 - share²) of the tires' grip for braking (the
// friction circle, with a 10% margin), so you can brake and turn in at once.
export function brakeLimit(state, steer) {
    const share = Math.min(1, Math.abs(steer) / cornering(Math.max(state[3], 4)).steer);
    return Math.sqrt(Math.max(0, 1 - (0.9 * share) ** 2));
}

// Counter-steer for your car: steers into a slide (toward where the car is
// going) and against rotation beyond what your input asks for. Your input asks
// for its share of the most the car can corner at this speed (CORNERING).
// deadband 0.1 rad: the slip of normal cornering at the limit (0.09-0.13), not a slide
export const COUNTER = { slide: 1.2, yaw: 0.3, deadband: 0.1 };
export function counterSteer(state, steer, { slide, yaw, deadband } = COUNTER) {
    const u = Math.max(state[3], 4);
    const limit = cornering(u);
    const wanted = Math.sign(steer) * Math.min(1, Math.abs(steer) / limit.steer) * limit.g * G / u;
    const r = state[5];
    const excess = r - clip(r, Math.min(0, wanted), Math.max(0, wanted));
    // cornering at the limit runs some slip; only correct beyond `deadband`
    const beta = Math.atan2(state[4], u);
    const slip = Math.sign(beta) * Math.max(0, Math.abs(beta) - deadband);
    return clip(steer + (slide * slip - yaw * excess) / STEER_MAX, -1, 1);
}

export async function loadAll(base, read) {
    const meta = JSON.parse(await read(base + 'tracks.json', 'text'));
    const tracks = {};
    for (const t of meta.tracks) {
        const bin = await read(base + `tracks/${t.name}.bin`, 'binary');
        tracks[t.name] = new Track(t, meta.fields, new Float32Array(bin));
    }
    const pmeta = JSON.parse(await read(base + 'policy.json', 'text'));
    const policy = new Policy(pmeta, new Float32Array(await read(base + 'policy.bin', 'binary')));
    return { meta, tracks, policy };
}
