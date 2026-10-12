// One flight from the pad: a scripted ferry (takeoff, staging, launch, as the
// RealFlight relay in mjx-rl-experiments does it) brings the drone to the trained
// handoff, 2.5 m before the first ring at 2.7 m/s, then the policy races lap after
// lap until it crashes. Shared by the page and tools.
import { G, TICK, NONE, STRIKE, Plant, Course, Race, integrateQuat, quatToMat } from './sim.js';

const HANDOFF_DISTANCE = 2.5;     // m before ring 0
const HANDOFF_SPEED = 2.7;        // m/s along its normal
const STAGING_DISTANCE = 6.0;     // m behind the handoff point
export const REST_HEIGHT = 0.032; // body origin above the ground when landed

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const norm = a => Math.sqrt(dot(a, a));

// Heading error (rad) from the body's forward axis to a desired horizontal direction
function headingError(R, desired) {
    const cx = R[0], cy = R[3];
    const cn = Math.hypot(cx, cy), dn = Math.hypot(desired[0], desired[1]);
    if (cn < 1e-6 || dn < 1e-6) return 0;
    const c = (cx * desired[0] + cy * desired[1]) / (cn * dn);
    const s = (cx * desired[1] - cy * desired[0]) / (cn * dn);
    return Math.atan2(s, clamp(c, -1, 1));
}

// Thrust along the current body z and body rates that tilt it toward `acc`
// (fly.py _geometric_command / _launch_command)
function attitudeCommand(plant, acc, forward, yawLimit, thrustOverride) {
    const R = plant.R, m = plant.mass;
    const n = Math.max(norm(acc), 1e-6);
    const up = [acc[0] / n, acc[1] / n, acc[2] / n];
    const cu = [R[2], R[5], R[8]];
    const e = [cu[1] * up[2] - cu[2] * up[1], cu[2] * up[0] - cu[0] * up[2], cu[0] * up[1] - cu[1] * up[0]];
    const rates = [
        clamp(2.7 * (R[0] * e[0] + R[3] * e[1] + R[6] * e[2]), -2.5, 2.5),
        clamp(2.7 * (R[1] * e[0] + R[4] * e[1] + R[7] * e[2]), -2.5, 2.5),
        clamp(2.0 * headingError(R, forward), -yawLimit, yawLimit),
    ];
    const thrust = clamp(thrustOverride ?? m * dot(acc, cu), 0.25 * m * G, Math.min(1.65 * m * G, 4 * plant.max));
    return { thrust, rates };
}

export class Flight {
    constructor(track, courseData, { policy, random = Math.random } = {}) {
        this.track = track;
        this.plant = new Plant(track.plant);
        // Every course is lapped until a crash (the policy sees no lap counter). The
        // open Multistory course is closed from its finish plane back to gate 0.
        const n0 = courseData.centers.length;
        const closed = courseData.open ? { ...courseData, open: false,
            previous_centers: [courseData.centers[n0 - 1], ...courseData.previous_centers.slice(1)] } : courseData;
        this.course = new Course(closed, track.plant);
        this.course.laps = Infinity;
        this.race = new Race(track, this.course, this.plant, random);
        this.policy = policy;
        const c = this.course, n = c.normals[0];
        this.handoff = sub(c.centers[0], n.map(x => HANDOFF_DISTANCE * x));
        this.staging = sub(this.handoff, n.map(x => STAGING_DISTANCE * x));
        this.pad = [this.staging[0], this.staging[1], 0];
        this.events = [];
        this.reset();
    }

    // Back on the pad, motors off, facing the first ring
    reset() {
        const n = this.course.normals[0];
        const yaw = Math.atan2(n[1], n[0]);
        this.plant.reset([this.pad[0], this.pad[1], REST_HEIGHT], [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], [0, 0, 0]);
        this.plant.motor.fill(0);
        this.plant.thrust.fill(0);
        this.grounded = true;
        this.phase = 'pad';
        this.phaseTime = 0;
        this.time = 0;
        this.settled = 0;
        this.stick = new Float64Array(4);
        this.stick[0] = -1;
        this.cmd = { thrust: 0, rates: [0, 0, 0] };
        this.lapStart = null;
        this.raceStart = null;
    }

    setPhase(phase) { this.phase = phase; this.phaseTime = 0; this.settled = 0; }
    emit(type, data = {}) { this.events.push({ type, time: this.time, ...data }); }

    // The ferry's command for this tick: fly to the phase's waypoint
    ferry() {
        const { plant, course } = this, m = plant.mass, n = course.normals[0];
        const p = plant.p, v = plant.v;
        let target = null;
        const limit = 1.8;
        switch (this.phase) {
            case 'climb': target = [this.pad[0], this.pad[1], this.staging[2]]; break;
            case 'stage': target = this.staging; break;
        }
        if (target) {
            const pe = sub(target, p);
            let hx = 0.8 * pe[0] - 1.8 * v[0], hy = 0.8 * pe[1] - 1.8 * v[1];
            const h = Math.hypot(hx, hy);
            if (h > limit) { hx *= limit / h; hy *= limit / h; }
            const vz = clamp(G + 3.2 * pe[2] - 2.4 * v[2], 0.55 * G, 1.45 * G);
            const cmd = attitudeCommand(plant, [hx, hy, vz], n, 0.6);
            if (this.phase === 'climb') cmd.thrust *= clamp(this.phaseTime / 1.0, 0.35, 1);
            return cmd;
        }
        // launch: build the handoff speed along ring 0's normal, centered on its axis
        const right = course.right[0], up = course.up[0];
        const o = sub(p, this.handoff);
        const behind = -dot(o, n), lateral = dot(o, right), vertical = dot(o, up);
        const fwd = dot(v, n), side = dot(v, right), rise = dot(v, up);
        const along = behind <= 1.0 ? 0 : clamp(1.5 * (HANDOFF_SPEED - fwd), -0.6, 1.5);
        const across = clamp(-1.0 * lateral - 1.8 * side, -1.5, 1.5);
        const lift = clamp(G - 3.2 * vertical - 2.4 * rise, 0.65 * G, 1.35 * G);
        const acc = [0, 1, 2].map(i => along * n[i] + across * right[i] + lift * up[i]);
        const cu = [plant.R[2], plant.R[5], plant.R[8]];
        return attitudeCommand(plant, acc, n, 0.6, m * dot(acc, up) / Math.max(0.72, dot(cu, up)));
    }

    // Is the drone inside the policy's trained handoff envelope (course_survey.handoff_envelope_errors)?
    inEnvelope() {
        const { plant, course } = this, n = course.normals[0];
        const o = sub(plant.p, course.centers[0]);
        const behind = -dot(o, n);
        const fwd = dot(plant.v, n);
        const cross = norm(sub(plant.v, n.map(x => fwd * x)));
        const level = Math.acos(clamp(plant.R[8], -1, 1));
        return behind >= 1.5 && behind <= 3.5 && Math.abs(dot(o, course.right[0])) <= 1.2 * behind / 3.5
            && Math.abs(dot(o, course.up[0])) <= 0.6 && fwd >= 2.2 && fwd <= 3.7 && cross <= 0.6
            && Math.abs(headingError(plant.R, n)) <= 0.18 && level <= 0.05;
    }

    // Advance 16 ms. Returns the phase.
    tick() {
        const { plant, course, race } = this;
        this.time += TICK;
        this.phaseTime += TICK;
        if (this.phase === 'pad') {
            if (this.phaseTime >= 0.6) this.setPhase('spool');
            return this.phase;
        }
        if (this.phase === 'crashed') return this.tumble();
        if (this.phase === 'race') return this.raceTick();

        let cmd;
        if (this.phase === 'spool') {
            // Props spin up on the ground before liftoff
            cmd = { thrust: 0.45 * plant.hoverThrust * clamp(this.phaseTime / 0.8, 0, 1), rates: [0, 0, 0] };
            if (this.phaseTime >= 1.2) this.setPhase('climb');
        } else {
            cmd = this.ferry();
        }
        this.cmd = cmd;
        plant.encode(cmd.thrust, cmd.rates, this.stick);
        for (let k = 0; k < 4; k++) this.physics(cmd.thrust, cmd.rates);

        const p = plant.p, speed = norm(plant.v);
        if (this.phase === 'climb' && p[2] > 0.7 * this.staging[2]) this.setPhase('stage');
        else if (this.phase === 'stage') {
            const good = norm(sub(this.staging, p)) < 0.4 && speed < 0.4
                && Math.abs(headingError(plant.R, course.normals[0])) < 0.15;
            this.settled = good ? this.settled + 1 : 0;
            if (this.settled >= 20) this.setPhase('launch');
        } else if (this.phase === 'launch') {
            const behind = -dot(sub(p, course.centers[0]), course.normals[0]);
            if (behind <= HANDOFF_DISTANCE && this.inEnvelope()) {
                race.start(0);
                this.raceStart = this.time;
                this.lapStart = null;
                this.setPhase('race');
                this.emit('handoff');
            } else if (behind < 1.6 || this.phaseTime > 12) {
                this.setPhase('stage');   // missed the envelope: back to staging, try again
            }
        }
        return this.phase;
    }

    raceTick() {
        const { race } = this;
        const action = this.policy.action(race.observation);
        this.stick.set(action);
        const r = race.tick(action);
        this.cmd = race.cmd;
        if (r.passed) {
            this.emit('gate', { gate: r.gate });
            if (r.gate === 0) {
                if (this.lapStart !== null) this.emit('lap', { lapTime: this.time - this.lapStart });
                this.lapStart = this.time;
            }
        }
        if (r.terminal !== NONE) {
            this.emit('crash', { reason: r.terminal, ring: r.struck });
            this.setPhase('crashed');
            // Power off. A ring knocks the drone back; either way it tumbles.
            const v = this.plant.v, w = this.plant.w, rnd = () => this.race.random() * 2 - 1;
            if (r.terminal === STRIKE) for (let i = 0; i < 3; i++) v[i] = -0.35 * v[i] + 0.8 * rnd();
            for (let i = 0; i < 3; i++) w[i] = 18 * rnd();
        }
        return this.phase;
    }

    // After a crash: unpowered fall, bounce and slide to rest, then back to the pad
    tumble() {
        const plant = this.plant, dt = plant.dt, p = plant.p, v = plant.v, w = plant.w;
        for (let k = 0; k < 4; k++) {
            v[2] -= G * dt;
            for (let i = 0; i < 3; i++) { v[i] *= 1 - 0.2 * dt; p[i] += v[i] * dt; w[i] *= 1 - 0.8 * dt; }
            integrateQuat(plant.q, w, dt);
            if (p[2] < 0.03) {
                p[2] = 0.03;
                if (v[2] < 0) v[2] *= -0.25;
                v[0] *= 0.8; v[1] *= 0.8;
                for (let i = 0; i < 3; i++) w[i] *= 0.7;
            }
        }
        quatToMat(plant.q, plant.R);
        for (let i = 0; i < 4; i++) { plant.thrust[i] *= 0.9; plant.motor[i] *= 0.9; }
        this.stick.fill(0); this.stick[0] = -1;
        this.cmd = { thrust: 0, rates: [0, 0, 0] };
        if (this.phaseTime > 3) { this.reset(); this.emit('reset'); }
        return this.phase;
    }

    // One physics substep for the ferry, with the drone resting on the ground until it lifts off
    physics(thrust, rates) {
        const plant = this.plant;
        if (this.grounded) {
            plant.substep(thrust, [0, 0, 0]);
            if (plant.p[2] > REST_HEIGHT) { this.grounded = false; return; }
            plant.p[2] = REST_HEIGHT;
            plant.v.fill(0); plant.w.fill(0); plant.rateFilter.fill(0);
            return;
        }
        plant.substep(thrust, rates);
    }
}
