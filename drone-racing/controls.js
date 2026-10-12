// Mouse, touch and keyboard input for the camera. Gestures are reported in pixels;
// the view decides what they do in each camera mode. `begin(kind)` is called once
// at the start of each gesture ('orbit', 'pan', 'zoom' or 'fly').
const KEYS = {
    KeyW: [1, 1], ArrowUp: [1, 1], KeyS: [1, -1], ArrowDown: [1, -1],
    KeyD: [0, 1], ArrowRight: [0, 1], KeyA: [0, -1], ArrowLeft: [0, -1],
    KeyE: [2, 1], KeyQ: [2, -1],
};
const DRAG_START = 3;   // px of movement before a press counts as a drag

export class CameraInput {
    constructor(canvas, { begin, orbit, pan, zoom }) {
        this.pointers = new Map();
        this.keys = new Set();
        this.move = [0, 0, 0];   // right, forward, up, each -1..1, from held keys
        let gesture = null, travel = 0, last = null;
        const start = kind => { if (gesture !== kind) { gesture = kind; begin(kind); } };
        const spread = () => {
            const [a, b] = [...this.pointers.values()];
            return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, d: Math.hypot(a.x - b.x, a.y - b.y) };
        };

        canvas.addEventListener('pointerdown', e => {
            canvas.setPointerCapture(e.pointerId);
            this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY,
                pan: e.button === 2 || e.button === 1 || e.shiftKey });
            gesture = null; travel = 0;
            last = this.pointers.size === 2 ? spread() : null;
        });
        canvas.addEventListener('pointermove', e => {
            const p = this.pointers.get(e.pointerId);
            if (!p) return;
            const dx = e.clientX - p.x, dy = e.clientY - p.y;
            p.x = e.clientX; p.y = e.clientY;
            if (this.pointers.size === 1) {
                travel += Math.abs(dx) + Math.abs(dy);
                if (travel < DRAG_START) return;
                const kind = p.pan ? 'pan' : 'orbit';
                start(kind);
                (kind === 'pan' ? pan : orbit)(dx, dy);
            } else if (this.pointers.size === 2) {
                // Two fingers: the midpoint drags the field, the spread zooms
                const now = spread();
                if (last) {
                    start('pan');
                    pan(now.x - last.x, now.y - last.y);
                    if (now.d > 0 && last.d > 0) zoom(last.d / now.d);
                }
                last = now;
            }
        });
        const up = e => {
            this.pointers.delete(e.pointerId);
            gesture = null;
            last = this.pointers.size === 2 ? spread() : null;
        };
        canvas.addEventListener('pointerup', up);
        canvas.addEventListener('pointercancel', up);
        canvas.addEventListener('contextmenu', e => e.preventDefault());
        canvas.addEventListener('wheel', e => {
            e.preventDefault();
            const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
            start('zoom');
            zoom(Math.exp(delta * (e.ctrlKey ? 0.01 : 0.0015)));   // ctrl: trackpad pinch
            gesture = null;
        }, { passive: false });

        addEventListener('keydown', e => {
            if (!(e.code in KEYS) || e.metaKey || e.ctrlKey || e.altKey) return;
            if (e.target.closest?.('input, textarea, select, .about')) return;
            e.preventDefault();
            if (!this.keys.has(e.code)) {
                this.keys.add(e.code);
                begin('fly');
                this.update();
            }
        });
        addEventListener('keyup', e => { if (this.keys.delete(e.code)) this.update(); });
        addEventListener('blur', () => { this.keys.clear(); this.update(); });
    }

    update() {
        this.move.fill(0);
        for (const code of this.keys) this.move[KEYS[code][0]] += KEYS[code][1];
        for (let i = 0; i < 3; i++) this.move[i] = Math.max(-1, Math.min(1, this.move[i]));
    }
}
