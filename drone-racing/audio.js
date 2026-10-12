// Motor sound, synthesized from the simulated rotors. Each motor is one oscillator
// at its rotation frequency (rpm from thrust with RealFlight's fitted coefficient,
// T = 1.682e-8 N/rpm²) whose harmonics carry the tri-blade pass tone (3x), its
// overtones and the 14-pole motor's electrical whine (7x). Four motors at slightly
// different speeds beat against each other, as real quads do. Broadband prop noise
// rises with thrust. Heard from the pilot's spot the sound falls off with distance,
// loses its highs to the air and shifts with the drone's speed (Doppler).
const K_THRUST = 1.682e-8;      // N per rpm²
const SOUND_SPEED = 343;

// Harmonic amplitudes of one motor, by multiple of the rotation frequency. Onboard
// (FPV) the once-per-turn vibration dominates, as in onboard racing footage; from
// the ground the blade-pass tone and its overtones carry.
const HARMONICS = {
    far:     { 1: 0.3, 2: 0.22, 3: 1.0, 4: 0.16, 5: 0.1, 6: 0.38, 7: 0.1, 9: 0.12, 12: 0.05 },
    onboard: { 1: 1.0, 2: 0.55, 3: 0.5, 4: 0.15, 5: 0.07, 6: 0.13, 7: 0.05, 9: 0.04 },
};
function motorWave(ctx, onboard) {
    const n = 24, re = new Float32Array(n + 1), im = new Float32Array(n + 1);
    const table = HARMONICS[onboard ? 'onboard' : 'far'];
    for (let h = 1; h <= n; h++) im[h] = table[h] ?? 0.02 / h;
    return ctx.createPeriodicWave(re, im);
}

function noiseBuffer(ctx, seconds) {
    const b = ctx.createBuffer(1, ctx.sampleRate * seconds, ctx.sampleRate);
    const d = b.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    return b;
}

export class MotorSound {
    constructor() {
        this.ctx = null;
        this.volume = 0.8;
    }

    // Must be called from a user gesture. (An OfflineAudioContext can be passed for tests.)
    start(context) {
        if (this.ctx) { this.ctx.resume(); return; }
        const ctx = this.ctx = context ?? new AudioContext();
        this.master = ctx.createGain();
        this.master.gain.value = this.volume;
        const comp = ctx.createDynamicsCompressor();
        comp.threshold.value = -14; comp.ratio.value = 4; comp.attack.value = 0.005; comp.release.value = 0.2;
        this.master.connect(comp).connect(ctx.destination);

        // Drone voice: motors + prop noise -> air absorption -> pan -> distance
        this.distance = ctx.createGain();
        this.pan = ctx.createStereoPanner();
        this.air = ctx.createBiquadFilter();
        this.air.type = 'lowpass';
        this.air.Q.value = 0.5;
        this.air.connect(this.pan).connect(this.distance).connect(this.master);
        this.waves = { far: motorWave(ctx, false), onboard: motorWave(ctx, true) };
        this.motors = [0, 1, 2, 3].map(() => {
            const osc = ctx.createOscillator();
            osc.setPeriodicWave(this.waves.far);
            osc.frequency.value = 100;
            const gain = ctx.createGain();
            gain.gain.value = 0;
            osc.connect(gain).connect(this.air);
            osc.start();
            return { osc, gain };
        });
        const noise = ctx.createBufferSource();
        noise.buffer = noiseBuffer(ctx, 2);
        noise.loop = true;
        this.noiseBand = ctx.createBiquadFilter();
        this.noiseBand.type = 'bandpass';
        this.noiseBand.Q.value = 0.5;
        const noiseTop = ctx.createBiquadFilter();
        noiseTop.type = 'lowpass';
        noiseTop.frequency.value = 4000;
        this.noiseGain = ctx.createGain();
        this.noiseGain.gain.value = 0;
        noise.connect(this.noiseBand).connect(noiseTop).connect(this.noiseGain).connect(this.air);
        noise.start();

        // A faint outdoor bed, so silence between flights isn't dead
        const bed = ctx.createBufferSource();
        bed.buffer = noiseBuffer(ctx, 3);
        bed.loop = true;
        const bedFilter = ctx.createBiquadFilter();
        bedFilter.type = 'lowpass'; bedFilter.frequency.value = 420;
        const bedGain = ctx.createGain();
        bedGain.gain.value = 0.035;
        bed.connect(bedFilter).connect(bedGain).connect(this.master);
        bed.start();
        this.onboard = false;
    }

    stop() { if (this.ctx) this.ctx.suspend(); }
    get running() { return this.ctx?.state === 'running'; }

    setVolume(v) {
        this.volume = v;
        if (this.master) this.master.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05);
    }

    // thrust: four rotor thrusts (N); listener: { position, forward (unit, world), right };
    // source: { position, velocity }; onboard: the listener rides on the drone (FPV)
    update(thrust, source, listener, onboard) {
        if (!this.ctx) return;
        const ctx = this.ctx, t = ctx.currentTime, tc = 0.03;
        if (onboard !== this.onboard) {
            this.onboard = onboard;
            this.motors.forEach(m => m.osc.setPeriodicWave(onboard ? this.waves.onboard : this.waves.far));
        }
        let dist = 1.5, doppler = 1, pan = 0;
        if (!onboard) {
            const d = [0, 1, 2].map(i => source.position[i] - listener.position[i]);
            dist = Math.max(Math.hypot(...d), 0.5);
            const away = (d[0] * source.velocity[0] + d[1] * source.velocity[1] + d[2] * source.velocity[2]) / dist;
            doppler = SOUND_SPEED / (SOUND_SPEED + Math.max(-150, Math.min(150, away)));
            const side = (d[0] * listener.right[0] + d[1] * listener.right[1] + d[2] * listener.right[2]) / dist;
            pan = Math.max(-0.85, Math.min(0.85, side));
        }
        let total = 0;
        thrust.forEach((T, i) => {
            const rpm = Math.sqrt(Math.max(T, 0) / K_THRUST);
            const level = Math.pow(rpm / 11300, 2);   // louder with speed; 1 at hover
            const m = this.motors[i];
            m.osc.frequency.setTargetAtTime(Math.max(rpm / 60 * doppler, 1), t, tc);
            m.gain.gain.setTargetAtTime(0.05 * Math.min(level, 5) * (rpm > 200 ? 1 : 0), t, tc);
            total += T;
        });
        const mean = Math.sqrt(Math.max(total / 4, 0) / K_THRUST);
        this.noiseBand.frequency.setTargetAtTime(Math.max(mean / 60 * 2.5 * doppler, 80), t, tc);
        this.noiseGain.gain.setTargetAtTime(0.035 * Math.pow(mean / 11300, 3), t, tc);
        // Air absorbs the highs over distance; onboard, the camera's case muffles them
        this.air.frequency.setTargetAtTime(onboard ? 3200 : Math.max(1800, 9000 * Math.exp(-dist / 25)), t, tc);
        this.pan.pan.setTargetAtTime(pan, t, tc);
        this.distance.gain.setTargetAtTime(onboard ? 0.55 : Math.min(1.4, 4 / dist), t, tc);
    }

    // A crash: a short clatter of plastic and carbon
    crash(source, listener) {
        if (!this.running) return;
        const ctx = this.ctx, t = ctx.currentTime;
        const burst = ctx.createBufferSource();
        burst.buffer = noiseBuffer(ctx, 0.5);
        const band = ctx.createBiquadFilter();
        band.type = 'bandpass'; band.frequency.value = 1800; band.Q.value = 1.2;
        const g = ctx.createGain();
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(0.9, t + 0.005);
        g.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
        burst.connect(band).connect(g).connect(this.air);
        burst.start(t);
        burst.stop(t + 0.5);
    }
}
