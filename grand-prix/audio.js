// Engine sound, synthesized. A 1.6 litre V6 fires three times per turn, so the
// note is rpm / 20 Hz; rpm follows road speed through eight gears. A sawtooth
// and a square an octave down, through a lowpass that opens with the throttle.
// The AI's engine is the same sound, quieter and duller with distance.
const GEAR_TOP = [0, 22, 32, 41, 50, 60, 70, 80, 200];
const RPM_MAX = 11800, RPM_MIN = 4200;

function rpmFor(u) {
    let g = 1;
    while (g < 8 && u >= GEAR_TOP[g]) g++;
    const lo = GEAR_TOP[g - 1] * 0.62, hi = Math.min(GEAR_TOP[g], 96);
    return RPM_MIN + (RPM_MAX - RPM_MIN) * Math.min(1, Math.max(0, (u - lo) / Math.max(hi - lo, 1)));
}

class Voice {
    constructor(ctx, out) {
        this.saw = ctx.createOscillator(); this.saw.type = 'sawtooth';
        this.sq = ctx.createOscillator(); this.sq.type = 'square';
        this.filter = ctx.createBiquadFilter(); this.filter.type = 'lowpass'; this.filter.Q.value = 3;
        this.gain = ctx.createGain(); this.gain.gain.value = 0;
        const sqGain = ctx.createGain(); sqGain.gain.value = 0.5;
        this.saw.connect(this.filter);
        this.sq.connect(sqGain).connect(this.filter);
        this.filter.connect(this.gain).connect(out);
        this.saw.start(); this.sq.start();
        this.ctx = ctx;
    }

    set(rpm, throttle, level, dull = 1) {
        const t = this.ctx.currentTime, f = rpm / 20;
        this.saw.frequency.setTargetAtTime(f, t, 0.03);
        this.sq.frequency.setTargetAtTime(f / 2, t, 0.03);
        this.filter.frequency.setTargetAtTime((500 + 2600 * throttle) * dull, t, 0.05);
        this.gain.gain.setTargetAtTime(level * (0.35 + 0.65 * throttle), t, 0.05);
    }

    stop() { this.saw.stop(); this.sq.stop(); }
}

export class Engine {
    constructor() { this.on = false; this.ctx = null; }

    toggle() {
        this.on = !this.on;
        if (this.on && !this.ctx) {
            this.ctx = new AudioContext();
            this.master = this.ctx.createGain();
            this.master.gain.value = 0.18;
            this.master.connect(this.ctx.destination);
        }
        if (this.ctx) this.on ? this.ctx.resume() : this.ctx.suspend();
        if (this.on && this.running) this.start();
        return this.on;
    }

    start() {
        this.running = true;
        if (!this.on || this.you) return;
        this.you = new Voice(this.ctx, this.master);
        this.ai = new Voice(this.ctx, this.master);
    }

    stop() {
        this.running = false;
        this.you?.stop(); this.ai?.stop();
        this.you = this.ai = null;
    }

    // watch: only the AI's engine, as the one you hear up close
    update(you, ai, camera, cars, watch = false) {
        if (!this.on || !this.you) return;
        if (watch) {
            this.you.set(rpmFor(ai.state[3]), ai.state[7], 0.9);
            this.ai.set(rpmFor(ai.state[3]), 0, 0);
            return;
        }
        this.you.set(rpmFor(you.state[3]), you.state[7], 0.9);
        const d = camera.position.distanceTo(cars.ai.position);
        const level = 0.9 / (1 + (d / 25) ** 2);
        this.ai.set(rpmFor(ai.state[3]), ai.state[7], level, 1 / (1 + d / 150));
    }
}
