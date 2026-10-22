// One race: the AI on pole on the racing line, you alongside one row back,
// both rolling at the qualifier's start speed. The cars can't touch: the AI
// has never seen another car, so they pass through each other.
import { Car, DT, tractionLimit, counterSteer, brakeLimit } from './sim.js';

const GRASS_GRIP = 0.6, GRASS_DRAG = 0.35;   // off the asphalt
const WALL = 13;                               // the barrier, m beyond the edge
const REJOIN_DELAY = 1.4;                      // s before a spun car is put back
const GAP_BUCKET = 5;                          // m of progress per gap sample

export class Race {
    // watch: the AI races alone (your car stays out of it)
    constructor(track, policy, laps, { assists = true, watch = false } = {}) {
        this.assists = assists;
        this.watch = watch;
        this.track = track;
        this.policy = policy;
        this.laps = laps;
        this.ai = new Car(track, 0, 0);
        this.ai.name = 'ai';
        // start on whichever side of the racing line has more asphalt
        const i = track.at(-Math.round(8 / track.spacing));
        const st = track.state(track.x[i], track.y[i], i);
        const room = [track.widthLeft[i] - st.center, track.widthRight[i] + st.center];
        const lateral = room[0] > room[1] ? Math.min(3.6, room[0] - 1.2) : -Math.min(3.6, room[1] - 1.2);
        this.you = new Car(track, lateral, 8);
        this.you.name = 'you';
        for (const car of [this.ai, this.you]) {
            car.finish = null;
            car.lapTimes = [];
            car.lapStart = 0;
            car.out = null;       // {reason, time} while waiting to rejoin
            car.onGrass = false;
            car.steerCmd = 0;
            car.pedalCmd = 0;
        }
        this.gapTimes = { ai: [0], you: [0] };  // time at each GAP_BUCKET of progress
        this.phase = 'grid';
        this.time = 0;
    }

    start() { this.phase = 'race'; }

    // input: { steer, pedal } in [-1, 1] for your car
    step(input) {
        if (this.phase === 'grid') return;
        this.time += DT;
        // the AI
        const ai = this.ai;
        if (!ai.finish || this.phase === 'race') {
            if (ai.out) this.waitOut(ai);
            else {
                const a = this.policy.act(ai.observe());
                ai.steerCmd = a[0]; ai.pedalCmd = a[1];
                ai.step(a[0], a[1]);
                if (ai.failed) this.goOut(ai, ai.failed === 'off' ? 'off the track' : 'spun');
            }
        }
        // you
        const you = this.you;
        if (this.watch) { /* no car of yours */ }
        else if (you.out) this.waitOut(you);
        else if (you.finish) {
            // after the flag the network drives your car on a slow cool-down lap
            const a = this.policy.act(you.observe());
            you.step(a[0], Math.min(a[1], you.state[3] > 30 ? -0.3 : 0.15));
            if (you.failed) this.goOut(you, 'spun');
        } else {
            const grip = you.onGrass ? GRASS_GRIP : 1, drag = you.onGrass ? GRASS_DRAG : 0;
            // Your inputs get help the AI drives without: traction control on the
            // throttle, a brake that leaves grip for your steering (trail braking),
            // and counter-steer (see sim.js). Your car's physics are the AI's.
            let pedal = input.pedal, steer = input.steer;
            const aid = this.assists === true ? { traction: true, counter: true } : this.assists || {};
            if (aid.traction && pedal > 0) pedal = Math.min(pedal, tractionLimit(you.state, grip));
            // trail braking: the brake leaves the tires the grip your steering asks for
            if (aid.counter && pedal < 0) pedal = Math.max(pedal, -brakeLimit(you.state, steer));
            if (aid.counter) steer = counterSteer(you.state, steer);
            you.steerCmd = steer; you.pedalCmd = pedal;
            you.step(steer, pedal, grip, drag);
            you.onGrass = you.margin < -0.15;
            const st = you.state;
            const beta = Math.abs(Math.atan2(st[4], Math.max(st[3], 2)));
            if (you.margin < -WALL) this.goOut(you, 'into the wall');
            else if (beta > 0.6 || Math.abs(st[5]) > 3.2) this.goOut(you, 'spun');
            else if (st[3] < 3.2 && this.time > 1) this.goOut(you, 'stopped');
        }
        for (const car of [ai, you]) this.laps_(car);
        if (ai.finish && (you.finish || this.watch) && this.phase !== 'finished') this.phase = 'finished';
    }

    laps_(car) {
        const L = this.track.length;
        const times = this.gapTimes[car.name];
        while ((times.length) * GAP_BUCKET <= car.progress) times.push(car.time);
        if (car.finish) return;
        const done = Math.floor(car.progress / L);
        while (car.lapTimes.length < done) {
            // interpolate the crossing inside the tick
            const over = (car.progress - (car.lapTimes.length + 1) * L) / Math.max(car.state[3] * DT, 1e-6);
            const t = car.time - Math.min(Math.max(over, 0), 1) * DT;
            car.lapTimes.push(t - car.lapStart);
            car.lapStart = t;
            if (car.lapTimes.length >= this.laps) {
                car.finish = t;
                if (!this.winner) this.winner = car.name;
                break;
            }
        }
    }

    goOut(car, reason) {
        car.out = { reason, time: car.time };
        car.state[3] = Math.max(car.state[3] * 0.3, 0);
    }

    // Wait, then put the car back on the racing line a little ahead, slow.
    waitOut(car) {
        car.time += DT;
        const st = car.state;
        st[3] = Math.max(st[3] - 15 * DT, 0);  // skid to a stop
        st[0] += Math.cos(st[2]) * st[3] * DT;
        st[1] += Math.sin(st[2]) * st[3] * DT;
        if (car.time - car.out.time < REJOIN_DELAY) return;
        const t = this.track, i = t.at(car.index + 3);
        st.set([t.x[i], t.y[i], t.heading[i], 18, 0, 0, 0, 0, 0]);
        const ts = t.state(t.x[i], t.y[i], i);
        let ds = ts.s - car.s;
        if (ds < -0.5 * t.length) ds += t.length;
        if (ds > 0.5 * t.length) ds -= t.length;
        car.progress += ds;
        car.s = ts.s;
        car.index = i;
        car.previousAction.fill(0);
        car.onGrass = false;
        car.out = null;
    }

    // Seconds `a` is behind `b` (negative: ahead), from when each passed the same point.
    gap(a = this.you, b = this.ai) {
        const ta = this.gapTimes[a.name], tb = this.gapTimes[b.name];
        if (a.progress <= b.progress) {
            const k = Math.floor(a.progress / GAP_BUCKET);
            return k < tb.length && k >= 0 ? a.time - tb[k] - (a.progress - k * GAP_BUCKET) / Math.max(a.state[3], 1) : 0;
        }
        const k = Math.floor(b.progress / GAP_BUCKET);
        return k < ta.length && k >= 0 ? -(b.time - ta[k] - (b.progress - k * GAP_BUCKET) / Math.max(b.state[3], 1)) : 0;
    }

    position() { return this.you.progress > this.ai.progress ? 1 : 2; }
}
