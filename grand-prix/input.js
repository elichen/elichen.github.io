// Your controls: keyboard, gamepad or touch, as steering and one pedal in [-1, 1]
// (the same two numbers the AI sets).
//
// Full steering lock at 300 km/h would spin any car, so a full key or stick asks
// for the steering at which this car corners hardest at its current speed
// (CORNERING in sim.js, measured from the simulation), a little less for a key.
// Only the size of the input is scaled; the car and its physics are the AI's.
import { cornering } from './sim.js';

const lockAt = (speed, gain) => Math.min(1, gain * cornering(speed).steer);

export class Controls {
    constructor(touchRoot) {
        this.keys = new Set();
        this.steer = 0;
        this.pedal = 0;
        this.used = 'keyboard';
        this.touch = { steer: null, gas: false, brake: false };
        const keyMap = {
            ArrowLeft: 'left', KeyA: 'left', ArrowRight: 'right', KeyD: 'right',
            ArrowUp: 'gas', KeyW: 'gas', ArrowDown: 'brake', KeyS: 'brake', Space: 'brake',
        };
        addEventListener('keydown', e => {
            const k = keyMap[e.code];
            if (!k || e.target.closest?.('input, select, textarea')) return;
            this.keys.add(k);
            this.used = 'keyboard';
            e.preventDefault();
        });
        addEventListener('keyup', e => { const k = keyMap[e.code]; if (k) this.keys.delete(k); });
        addEventListener('blur', () => this.keys.clear());
        if (touchRoot) this.bindTouch(touchRoot);
    }

    bindTouch(root) {
        const pad = root.querySelector('[data-touch=steer]');
        const knob = pad.querySelector('.knob');
        const set = e => {
            const r = pad.getBoundingClientRect();
            const x = Math.max(-1, Math.min(1, ((e.clientX - r.left) / r.width - 0.5) * 2));
            this.touch.steer = -x;
            knob.style.transform = `translateX(${x * r.width * 0.4}px)`;
            this.used = 'touch';
        };
        pad.addEventListener('pointerdown', e => { pad.setPointerCapture(e.pointerId); set(e); });
        pad.addEventListener('pointermove', e => { if (pad.hasPointerCapture(e.pointerId)) set(e); });
        const release = () => { this.touch.steer = 0; knob.style.transform = ''; };
        pad.addEventListener('pointerup', release);
        pad.addEventListener('pointercancel', release);
        for (const name of ['gas', 'brake']) {
            const b = root.querySelector(`[data-touch=${name}]`);
            b.addEventListener('pointerdown', e => { b.setPointerCapture(e.pointerId); this.touch[name] = true; b.classList.add('down'); this.used = 'touch'; });
            const up = () => { this.touch[name] = false; b.classList.remove('down'); };
            b.addEventListener('pointerup', up);
            b.addEventListener('pointercancel', up);
        }
    }

    gamepad() {
        for (const gp of navigator.getGamepads?.() ?? []) {
            if (!gp || !gp.connected) continue;
            const x = gp.axes[0] ?? 0;
            const rt = gp.buttons[7]?.value ?? 0, lt = gp.buttons[6]?.value ?? 0;
            const a = gp.buttons[0]?.pressed ? 1 : 0, b = gp.buttons[1]?.pressed || gp.buttons[2]?.pressed ? 1 : 0;
            const active = Math.abs(x) > 0.12 || rt > 0.05 || lt > 0.05 || a || b;
            if (active) this.used = 'gamepad';
            if (this.used !== 'gamepad') return null;
            const dz = Math.max(0, Math.abs(x) - 0.08) / 0.92;
            return { steer: -Math.sign(x) * dz ** 1.6, gas: Math.max(rt, a), brake: Math.max(lt, b) };
        }
        return null;
    }

    // Called every 20 ms tick with the car's speed (m/s); returns {steer, pedal}.
    read(speed, dt) {
        const gp = this.gamepad();
        if (gp) {
            this.steer = gp.steer * lockAt(speed, 1.0);
            this.pedal = gp.brake > 0.05 ? -gp.brake : gp.gas;
            return this;
        }
        if (this.used === 'touch' && this.touch.steer !== null) {
            const target = this.touch.steer * lockAt(speed, 0.95);
            this.steer += (target - this.steer) * Math.min(1, dt * 14);
            this.pedal = this.touch.brake ? -1 : this.touch.gas ? 1 : 0;
            return this;
        }
        // keyboard: the wheel turns toward the key's lock at ~4 locks/s and back
        const dir = (this.keys.has('left') ? 1 : 0) - (this.keys.has('right') ? 1 : 0);
        const lock = lockAt(speed, 0.95);
        const target = dir * lock;
        const rate = (dir === 0 ? 6 : 3.5) * lock;
        this.steer += Math.max(-rate * dt, Math.min(rate * dt, target - this.steer));
        this.pedal = this.keys.has('brake') ? -1 : this.keys.has('gas') ? 1 : 0;
        return this;
    }

    get any() { return this.keys.size > 0 || this.touch.gas || this.touch.brake; }
}
