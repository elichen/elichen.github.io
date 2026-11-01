// The sound of the wings, from the same wingbeat that moves them. The wave is one beat of the flight policy's wing pattern: the force on a wing
// grows with the square of its speed (lift, ω², twice per beat; drag, ω|ω|, once per beat; equal shares are our choice), and what the ear gets from a
// small flapping source is the rate of change of that force. The pitch is the policy's own wingbeat frequency. Volume falls with the camera's distance.
const HARMONICS = 24, LOUD = 0.35, NEAR = 6;   // cm: full volume inside this distance; volume and distance are our choices

// One beat of [yaw, roll, pitch] angles -> Fourier coefficients of the sound, from the stroke angle (yaw).
export function wingWave(pattern) {
    const N = pattern.length, at = i => pattern[(i + N) % N][0];
    const speed = Array.from({ length: N }, (_, i) => at(i + 1) - at(i - 1));
    const force = speed.map(w => w * w + w * Math.abs(w));
    const sound = force.map((_, i) => force[(i + 1) % N] - force[(i - 1 + N) % N]);
    const real = new Float32Array(HARMONICS + 1), imag = new Float32Array(HARMONICS + 1);
    for (let h = 1; h <= HARMONICS; h++) sound.forEach((s, i) => {
        real[h] += s * Math.cos(2 * Math.PI * h * i / N) / N;
        imag[h] += s * Math.sin(2 * Math.PI * h * i / N) / N;
    });
    return { real, imag };
}

export class WingSound {
    constructor(pattern) { this.wave = wingWave(pattern); this.ctx = null; }

    // browsers only start audio from a click, so this is called from the Sound button
    set(on) {
        if (on && !this.ctx) {
            const c = this.ctx = new AudioContext();
            this.osc = c.createOscillator();
            this.osc.setPeriodicWave(c.createPeriodicWave(this.wave.real, this.wave.imag));
            this.gain = c.createGain(); this.gain.gain.value = 0;
            this.pan = c.createStereoPanner();
            this.osc.connect(this.gain).connect(this.pan).connect(c.destination);
            this.osc.start();
        }
        if (this.ctx) on ? this.ctx.resume() : this.ctx.suspend();
    }

    // wings beating or not, Hz, cm from the camera, -1..1 left to right on the screen
    update(beating, hz, dist, pan) {
        if (!this.ctx) return;
        const at = this.ctx.currentTime, go = (p, v, tau) => { p.cancelScheduledValues(at); p.setTargetAtTime(v, at, tau); };
        go(this.gain.gain, beating ? LOUD * Math.min(1, NEAR / Math.max(dist, 1e-3)) : 0, 0.02);
        go(this.osc.frequency, hz, 0.02);
        go(this.pan.pan, Math.max(-1, Math.min(1, pan)), 0.05);
    }
}
