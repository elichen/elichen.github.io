// The racing drone and the course, ported from drone.racing.env.RingRace in
// mjx-rl-experiments (qualification mode: nominal plant, clean sensing, no wind).
// Shared by the page and the Node tools, so both fly identical physics.
//
// World frame is z-up (east, north, up); the body frame is forward, left, up.
// The plant is one rigid body stepped like MuJoCo's Euler integrator at 4 ms:
// four rotors behind a rate loop (P gains on a filtered gyro), an attitude-
// priority mixer, first-order motor lag, and the RealFlight-fitted drag and
// rotor-inflow thrust terms. The policy acts every 16 ms (four substeps) with a
// collective-thrust and body-rate request that reaches the plant one tick late.

export const G = 9.81;
export const FRAME = 51;        // one actor observation frame
export const STACK = 4;         // frames stacked into the 204-value observation
export const TICK = 0.016;      // policy period, s

// Terminal reasons, as in drone.racing.contracts
export const NONE = 0, SUCCESS = 1, STRIKE = 2, MISS = 3, GROUND = 4, BOUNDS = 5,
    INVERSION = 6, NONFINITE = 7;

const SWEEP = 17;               // samples along a substep for the ring-strike test

// ---- small vector helpers ---------------------------------------------------------
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export function quatToMat(q, R = new Float64Array(9)) {
    const [w, x, y, z] = q;
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z);     R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y);     R[7] = 2 * (y * z + w * x);     R[8] = 1 - 2 * (x * x + y * y);
    return R;
}
// R^T v (world -> body)
function toBody(R, v, out) {
    out[0] = R[0] * v[0] + R[3] * v[1] + R[6] * v[2];
    out[1] = R[1] * v[0] + R[4] * v[1] + R[7] * v[2];
    out[2] = R[2] * v[0] + R[5] * v[1] + R[8] * v[2];
    return out;
}

// ---- the plant ----------------------------------------------------------------------
export class Plant {
    constructor(plant) {
        const c = plant.config;
        this.dt = plant.timestep;
        this.mass = plant.mass;                         // MJCF body mass
        this.inertia = plant.inertia;                   // MJCF diagonal inertia
        this.A = plant.allocation;                      // rotor thrusts -> [T, tx, ty, tz]
        this.Ainv = plant.allocation_inverse;
        this.max = plant.actuator_max;                  // 8.75 N per rotor
        this.hoverPerRotor = plant.hover_per_rotor;
        this.kp = c.nominal_rate_kp_nm_per_rad_s;
        this.rateAlpha = c.rate_sensor_filter_alpha;
        this.alphaUp = c.motor_alpha_up_nominal;
        this.alphaDown = c.motor_alpha_down_nominal;
        this.curve = 0.35;                              // nominal thrust-curve shape
        this.highCommandLoss = c.rotor_high_command_thrust_loss;
        this.axialLoss = c.rotor_axial_inflow_thrust_loss;
        this.edgewiseGain = c.rotor_edgewise_thrust_gain;
        this.airspeedLoss = c.airspeed_thrust_loss_at_20_m_s;
        this.dragQuadratic = c.drag_quadratic_nominal_n_per_mps2;
        this.dragLinear = c.drag_linear_xy_nominal_n_per_mps;
        // Action decoding (drone.racing.action): collective 0 is hover
        this.hoverThrust = c.nominal_mass_kg * G;
        this.maxThrust = 4 * c.published_max_thrust_per_rotor_n * c.thrust_efficiency;
        this.rateMax = [c.provisional_rate_limit_rp_rad_s, c.provisional_rate_limit_rp_rad_s,
            c.provisional_rate_limit_yaw_rad_s];

        this.p = new Float64Array(3);   // position, world
        this.q = new Float64Array(4);   // orientation (w, x, y, z), body -> world
        this.v = new Float64Array(3);   // velocity, world
        this.w = new Float64Array(3);   // angular velocity, body
        this.motor = new Float64Array(4);
        this.thrust = new Float64Array(4);
        this.rateFilter = new Float64Array(3);
        this.R = new Float64Array(9);
        this._air = new Float64Array(3);
        this._desired = new Float64Array(4);
        this._rp = new Float64Array(4);
        this.reset([0, 0, 1], [1, 0, 0, 0], [0, 0, 0]);
    }

    reset(p, q, v, w = [0, 0, 0]) {
        this.p.set(p); this.q.set(q); this.v.set(v); this.w.set(w);
        this.motor.fill(this.hoverPerRotor);
        this.thrust.fill(this.hoverPerRotor);
        this.rateFilter.fill(0);
        quatToMat(this.q, this.R);
    }

    // Normalized [collective, roll, pitch, yaw] request -> total thrust (N) and body rates
    decode(a, out = { thrust: 0, rates: new Float64Array(3) }) {
        const c = clamp(a[0], -1, 1);
        out.thrust = c >= 0 ? this.hoverThrust + c * Math.max(this.maxThrust - this.hoverThrust, 0)
            : this.hoverThrust * (1 + c);
        for (let i = 0; i < 3; i++) {
            const x = clamp(a[i + 1], -1, 1);
            out.rates[i] = this.rateMax[i] * (0.5 * x + 0.5 * x * x * x);
        }
        return out;
    }

    // The inverse of decode, for showing a thrust/rate command as stick positions
    encode(thrust, rates, out = new Float64Array(4)) {
        out[0] = thrust >= this.hoverThrust
            ? (thrust - this.hoverThrust) / (this.maxThrust - this.hoverThrust)
            : thrust / this.hoverThrust - 1;
        for (let i = 0; i < 3; i++) {
            const s = clamp(rates[i] / this.rateMax[i], -1, 1);
            let x = s;  // solve 0.5 x + 0.5 x^3 = s
            for (let k = 0; k < 6; k++) x -= (0.5 * x + 0.5 * x * x * x - s) / (0.5 + 1.5 * x * x);
            out[i + 1] = x;
        }
        for (let i = 0; i < 4; i++) out[i] = clamp(out[i], -1, 1);
        return out;
    }

    // Attitude-priority mixer (plant.desaturate_attitude_priority)
    mix(T, torque, out) {
        const A = this.Ainv, max = this.max, rp = this._rp;
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < 4; i++) {
            rp[i] = A[i][1] * torque[0] + A[i][2] * torque[1];
            lo = Math.min(lo, rp[i]); hi = Math.max(hi, rp[i]);
        }
        const shrink = Math.min(1, max / Math.max(hi - lo, 1e-9));
        lo *= shrink; hi *= shrink;
        for (let i = 0; i < 4; i++) {
            rp[i] *= shrink;
            out[i] = clamp(A[i][0] * T, -lo, max - hi) + rp[i];
        }
        // Yaw takes whatever room is left (plant.fit_actuator_delta)
        let fit = Infinity;
        for (let i = 0; i < 4; i++) {
            const d = A[i][3] * torque[2];
            if (d > 1e-9) fit = Math.min(fit, (max - out[i]) / (d + 1e-9));
            if (d < -1e-9) fit = Math.min(fit, out[i] / (-d + 1e-9));
        }
        fit = clamp(fit, 0, 1);
        for (let i = 0; i < 4; i++) {
            const v = out[i] + fit * A[i][3] * torque[2];
            out[i] = Number.isFinite(v) ? clamp(v, 0, max) : T / 4;
        }
        return out;
    }

    // One 4 ms physics step with a physical thrust (N) and body-rate (rad/s) request
    substep(T, rates) {
        const R = this.R, w = this.w, v = this.v, max = this.max, dt = this.dt;
        // Rate loop on a first-order filtered gyro
        const torque = [0, 0, 0];
        for (let i = 0; i < 3; i++) {
            this.rateFilter[i] += this.rateAlpha * (w[i] - this.rateFilter[i]);
            torque[i] = this.kp[i] * (rates[i] - this.rateFilter[i]);
        }
        const desired = this.mix(T, torque, this._desired);
        // Flight controller inverts the nominal quadratic thrust curve
        const c = this.curve;
        for (let i = 0; i < 4; i++) {
            const n = Math.max(desired[i], 0) / max;
            const root = (-(1 - c) + Math.sqrt((1 - c) * (1 - c) + 4 * c * n)) / (2 * c);
            desired[i] = max * clamp(root, 0, 1);
        }
        // Thrust lost to airspeed and to inflow through the rotor disks
        const air = toBody(R, v, this._air);
        const speed = Math.sqrt(dot(v, v));
        const f = clamp(speed / 20, 0, 1.5);
        const axial = clamp(air[2] / 20, -1, 1);
        const edgewise = clamp((air[0] * air[0] + air[1] * air[1]) / 400, 0, 2.25);
        const scale = clamp(1 - this.airspeedLoss * f * f, 0.65, 1.35)
            * clamp(1 - this.axialLoss * axial + this.edgewiseGain * edgewise, 0.5, 1.5);
        // Motors: first-order lag toward the command, then the thrust curve
        for (let i = 0; i < 4; i++) {
            const delta = desired[i] - this.motor[i];
            const alpha = delta > 0 ? this.alphaUp : this.alphaDown;
            this.motor[i] = clamp(this.motor[i] + alpha * delta, 0, max);
            const n = this.motor[i] / max;
            const nonlinear = max * (c * n * n + (1 - c) * n);
            this.thrust[i] = clamp(nonlinear * scale * (1 - this.highCommandLoss * n), 0, max);
        }
        const A = this.A, t = this.thrust;
        const total = A[0][0] * t[0] + A[0][1] * t[1] + A[0][2] * t[2] + A[0][3] * t[3];
        const tau = [1, 2, 3].map(r => A[r][0] * t[0] + A[r][1] * t[1] + A[r][2] * t[2] + A[r][3] * t[3]);
        // Body drag (quadratic, plus linear in the horizontal plane), world frame
        const s = Math.sqrt(dot(v, v) + 1e-9);
        const m = this.mass;
        const ax = (R[2] * total - this.dragQuadratic * v[0] * s - this.dragLinear * v[0]) / m;
        const ay = (R[5] * total - this.dragQuadratic * v[1] * s - this.dragLinear * v[1]) / m;
        const az = (R[8] * total - this.dragQuadratic * v[2] * s) / m - G;
        // Euler's equations in the body frame: I dw = tau - w x (I w)
        const I = this.inertia;
        const Iw = [I[0] * w[0], I[1] * w[1], I[2] * w[2]];
        const dw0 = (tau[0] - (w[1] * Iw[2] - w[2] * Iw[1])) / I[0];
        const dw1 = (tau[1] - (w[2] * Iw[0] - w[0] * Iw[2])) / I[1];
        const dw2 = (tau[2] - (w[0] * Iw[1] - w[1] * Iw[0])) / I[2];
        // MuJoCo's Euler integrator: velocities first, then positions with the new ones
        v[0] += dt * ax; v[1] += dt * ay; v[2] += dt * az;
        w[0] += dt * dw0; w[1] += dt * dw1; w[2] += dt * dw2;
        this.p[0] += dt * v[0]; this.p[1] += dt * v[1]; this.p[2] += dt * v[2];
        integrateQuat(this.q, w, dt);
        quatToMat(this.q, R);
    }
}

// q <- normalize(q * exp(w dt / 2)), angular velocity in the body frame
export function integrateQuat(q, w, dt) {
    const n = Math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    if (n < 1e-12) return;
    const h = 0.5 * n * dt, s = Math.sin(h) / n;
    const bw = Math.cos(h), bx = w[0] * s, by = w[1] * s, bz = w[2] * s;
    const [aw, ax, ay, az] = q;
    q[0] = aw * bw - ax * bx - ay * by - az * bz;
    q[1] = aw * bx + ax * bw + ay * bz - az * by;
    q[2] = aw * by - ax * bz + ay * bw + az * bx;
    q[3] = aw * bz + ax * by - ay * bx + az * bw;
    const l = Math.hypot(q[0], q[1], q[2], q[3]);
    for (let i = 0; i < 4; i++) q[i] /= l;
}

// ---- course geometry ----------------------------------------------------------------
export class Course {
    constructor(c, plant) {
        Object.assign(this, c);
        const cfg = plant.config;
        this.count = c.mask.filter(Boolean).length;
        this.sphere = cfg.airframe_sphere_radius_m;
        this.halfThickness = cfg.ring_half_thickness_m;
        this.floor = cfg.floor_margin_m;
        this.missDepth = cfg.active_ring_miss_depth_m;
        this.missMargin = cfg.active_ring_miss_radius_margin_m;
        this.inversionSteps = cfg.inversion_steps;
    }
    successor(i) { return this.open ? Math.min(i + 1, this.count - 1) : (i + 1) % this.count; }
    safeRadius(i) { return this.inner_radius[i] - this.sphere; }

    radial(p, i) {
        const c = this.centers[i], n = this.normals[i];
        const o = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        const a = dot(o, n);
        return Math.sqrt((o[0] - a * n[0]) ** 2 + (o[1] - a * n[1]) ** 2 + (o[2] - a * n[2]) ** 2 + 1e-12);
    }
    plane(p, i) {
        const c = this.centers[i];
        return dot([p[0] - c[0], p[1] - c[1], p[2] - c[2]], this.normals[i]);
    }

    // Did the segment cross ring i's plane forwards inside its safe radius?
    pass(a, b, i) {
        const d0 = this.plane(a, i), d1 = this.plane(b, i), den = d1 - d0;
        const f = clamp(-d0 / (den + 1e-7), 0, 1);
        const pt = [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
        const r = this.radial(pt, i);
        return { passed: d0 < 0 && d1 >= 0 && den > 1e-7 && r <= this.safeRadius(i), f, r, pt };
    }

    // Swept-sphere test against every ring's tube (contracts.swept_sphere_annulus_strike).
    // Returns the struck ring's index, or -1.
    strike(a, b) {
        const seg = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        const err = Math.sqrt(dot(seg, seg)) / (2 * (SWEEP - 1));
        for (let i = 0; i < this.mask.length; i++) {
            if (!this.mask[i]) continue;
            const c = this.centers[i], n = this.normals[i];
            const major = 0.5 * (this.inner_radius[i] + this.outer_radius[i]);
            const tube = Math.max(0.5 * (this.outer_radius[i] - this.inner_radius[i]), this.halfThickness);
            let best = Infinity;
            for (let k = 0; k < SWEEP; k++) {
                const t = k / (SWEEP - 1);
                const o = [a[0] + t * seg[0] - c[0], a[1] + t * seg[1] - c[1], a[2] + t * seg[2] - c[2]];
                const ax = dot(o, n);
                const rad = Math.sqrt((o[0] - ax * n[0]) ** 2 + (o[1] - ax * n[1]) ** 2 + (o[2] - ax * n[2]) ** 2 + 1e-12);
                best = Math.min(best, Math.sqrt(ax * ax + (rad - major) ** 2 + 1e-12));
            }
            if (best <= tube + this.sphere + err) return i;
        }
        return -1;
    }

    // Flew past the active ring's plane near the ring without passing it
    missed(p, i) {
        return this.plane(p, i) > this.missDepth && this.radial(p, i) <= this.outer_radius[i] + this.missMargin;
    }
}

// ---- the policy's view: mapped ring landmarks in the body frame ---------------------------
export class Observer {
    constructor(track) {
        const c = track.plant.config;
        this.fx = c.camera_fx_px; this.fy = c.camera_fy_px;
        this.width = c.camera_width_px; this.height = c.camera_height_px;
        this.cameraDecimation = c.camera_decimation;
        this.rateMax = [c.provisional_rate_limit_rp_rad_s, c.provisional_rate_limit_rp_rad_s,
            c.provisional_rate_limit_yaw_rad_s];
        this.stack = new Float32Array(FRAME * STACK);
        this.frame = new Float32Array(FRAME);
        this._lm = [new Float64Array(12), new Float64Array(12)];
    }

    // Four landmarks (top, right, bottom, left of the inner circle) of ring i, body frame
    landmarks(course, i, plant, out) {
        const c = course.centers[i], u = course.up[i], r = course.right[i], rad = course.inner_radius[i];
        const p = plant.p, R = plant.R, tmp = [0, 0, 0], b = [0, 0, 0];
        const signs = [[1, 0], [0, 1], [-1, 0], [0, -1]];
        for (let k = 0; k < 4; k++) {
            const [su, sr] = signs[k];
            for (let j = 0; j < 3; j++) tmp[j] = c[j] + rad * (su * u[j] + sr * r[j]) - p[j];
            toBody(R, tmp, b);
            out[3 * k] = b[0]; out[3 * k + 1] = b[1]; out[3 * k + 2] = b[2];
        }
        return out;
    }

    // observation.report_quality: how well the camera would see ring i
    quality(course, i, plant, lm) {
        let visible = 0, valid = 0;
        const center = [0, 0, 0];
        for (let k = 0; k < 4; k++) {
            const x = lm[3 * k], y = lm[3 * k + 1], z = lm[3 * k + 2];
            const d = Math.max(x, 1e-6);
            if (x > 0.15 && Math.abs(this.fx * y / d) <= 0.5 * this.width && Math.abs(this.fy * z / d) <= 0.5 * this.height) visible++;
            if (x > 0.15) valid++;
            center[0] += x / 4; center[1] += y / 4; center[2] += z / 4;
        }
        const dist = Math.sqrt(dot(center, center));
        const range = Math.exp(-Math.max(dist - 8, 0) / 12);
        const n = toBody(plant.R, course.normals[i], [0, 0, 0]);
        const obliquity = clamp(Math.abs(dot(center, n) / Math.max(dist, 1e-6)), 0, 1);
        const w = plant.w;
        const blur = Math.exp(-0.08 * Math.sqrt(dot(w, w)));
        return clamp((visible / 4) * (valid / 4) * range * obliquity * blur, 0, 1);
    }

    // One 51-value frame; meta = [detected, quality, age] for the active and next ring
    build(course, race, plant) {
        const f = this.frame, R = plant.R;
        for (let i = 0; i < 9; i++) f[i] = R[i];
        const vb = toBody(R, plant.v, [0, 0, 0]);
        for (let i = 0; i < 3; i++) f[9 + i] = clamp(vb[i] / 10, -2, 2);
        for (let i = 0; i < 3; i++) f[12 + i] = clamp(plant.w[i] / this.rateMax[i], -2, 2);
        for (let i = 0; i < 12; i++) f[15 + i] = clamp(race.lm[0][i] / 10, -2, 2);
        for (let i = 0; i < 12; i++) f[27 + i] = clamp(race.lm[1][i] / 10, -2, 2);
        const m = race.meta;
        f[39] = m.activeValid ? 1 : 0; f[40] = clamp(m.activeQuality, 0, 1); f[41] = clamp(m.activeAge / 0.5, 0, 1);
        f[42] = m.nextValid ? 1 : 0; f[43] = clamp(m.nextQuality, 0, 1); f[44] = clamp(m.nextAge / 0.5, 0, 1);
        f[45] = 0; f[46] = 0;
        for (let i = 0; i < 4; i++) f[47 + i] = clamp(race.lastAction[i], -1, 1);
        for (let i = 0; i < FRAME; i++) if (!Number.isFinite(f[i])) f[i] = 0;
        return f;
    }

    push(frame, initialize) {
        const s = this.stack;
        if (initialize) for (let k = 0; k < STACK; k++) s.set(frame, k * FRAME);
        else { s.copyWithin(0, FRAME); s.set(frame, (STACK - 1) * FRAME); }
        return s;
    }
}

// ---- one race: the policy flying the course (RingRace.reset / RingRace.step) ---------------
export class Race {
    // random: () => uniform [0, 1) for the detector model
    constructor(track, course, plant, random = Math.random) {
        this.course = course;
        this.plant = plant;
        this.observer = new Observer(track);
        this.random = random;
        this.lm = [new Float64Array(12), new Float64Array(12)];
        this.lastAction = new Float64Array(4);
        this.pending = new Float64Array(4);  // one-tick action delay
        this.cmd = { thrust: 0, rates: new Float64Array(3) };
        this.meta = {};
    }

    // Start racing from the plant's current state toward ring `active` (an episode reset)
    start(active = 0) {
        this.active = active;
        this.lap = 0;
        this.step = 0;
        this.passes = 0;
        this.inversion = 0;
        this.terminal = NONE;
        this.lastAction.fill(0);
        this.pending = new Float64Array(4);
        this.detectorBad = false;
        this.observe(true);
    }

    observe(initialize) {
        const { course, plant, observer, meta } = this;
        const next = course.successor(this.active);
        observer.landmarks(course, this.active, plant, this.lm[0]);
        observer.landmarks(course, next, plant, this.lm[1]);
        const qa = observer.quality(course, this.active, plant, this.lm[0]);
        const qn = observer.quality(course, next, plant, this.lm[1]);
        if (initialize) {
            Object.assign(meta, { activeValid: qa > 0, activeQuality: qa, activeAge: 0,
                nextValid: qn > 0, nextQuality: qn, nextAge: 0 });
            this.detectorBad = false;
        } else {
            // The ring detector reports every other tick and sometimes drops out for a while
            const due = this.step % observer.cameraDecimation === 0;
            const u0 = this.random(), u1 = this.random(), u2 = this.random();
            if (due) {
                const bad = this.detectorBad ? u0 < 0.78 : u0 < 0.008;
                this.detectorBad = bad;
                meta.activeValid = !bad && u1 < qa * 0.995;
                meta.nextValid = !bad && u2 < qn * 0.98;
                meta.activeQuality = qa;
                meta.nextQuality = qn;
            }
            meta.activeAge = due && meta.activeValid ? 0 : meta.activeAge + TICK;
            meta.nextAge = due && meta.nextValid ? 0 : meta.nextAge + TICK;
        }
        return observer.push(observer.build(course, this, plant), initialize);
    }

    get observation() { return this.observer.stack; }

    // Apply one policy action (4 values in [-1, 1]) for one 16 ms tick
    tick(action) {
        const { course, plant } = this;
        const applied = Float64Array.from(action, a => clamp(a, -1, 1));
        // The plant executes the previous tick's request
        const executed = this.pending;
        this.pending = applied;
        const cmd = plant.decode(executed, this.cmd);
        const active = this.active;
        let passed = false, struck = -1, crossing = null;
        const a = new Float64Array(3);
        for (let k = 0; k < 4; k++) {
            a.set(plant.p);
            plant.substep(cmd.thrust, cmd.rates);
            const result = course.pass(a, plant.p, active);
            if (result.passed) { passed = true; crossing = result; }
            const s = course.strike(a, plant.p);
            if (s >= 0 && struck < 0) struck = s;
        }
        const p = plant.p, R = plant.R;
        const ground = p[2] <= course.floor;
        const bounds = Math.hypot(p[0], p[1]) > course.flight_radius || p[2] > course.flight_ceiling;
        this.inversion = R[8] < 0 ? this.inversion + 1 : 0;
        const inversion = this.inversion >= course.inversionSteps;
        const finite = [...p, ...plant.q, ...plant.v, ...plant.w].every(Number.isFinite);
        const miss = !passed && course.missed(p, active);
        const failure = struck >= 0 || ground || bounds || inversion || !finite || miss;
        const accepted = passed && !failure;
        let success = false;
        if (accepted) {
            const wrapped = active === course.count - 1;
            this.active = (active + 1) % course.count;
            if (wrapped) this.lap++;
            success = this.lap >= course.laps;
            this.passes++;
            // The next ring's detection becomes the active one's
            const m = this.meta;
            Object.assign(m, { activeValid: m.nextValid, activeQuality: m.nextQuality,
                activeAge: m.nextAge, nextValid: false, nextQuality: 0, nextAge: 0.5 });
        }
        let t = NONE;
        if (success) t = SUCCESS;
        if (miss) t = MISS;
        if (inversion) t = INVERSION;
        if (bounds) t = BOUNDS;
        if (ground) t = GROUND;
        if (struck >= 0) t = STRIKE;
        if (!finite) t = NONFINITE;
        this.terminal = t;
        this.struck = struck;
        this.lastAction.set(applied);
        this.step++;
        this.observe(false);
        return { passed: accepted, gate: active, crossing, terminal: t, struck, executed };
    }
}
