// The on-screen radio and telemetry. The radio is a Mode 2 transmitter: left stick
// throttle (up/down) and yaw, right stick pitch and roll. It shows what the pilot
// (the policy, or the scripted ferry before handoff) is commanding each tick.
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
};

const GIMBAL = 30;      // gimbal well radius in SVG units
const THROW = 21;       // how far a stick end travels from center

export class Transmitter {
    constructor(container) {
        const svg = el('svg', { viewBox: '0 0 220 150', class: 'radio', 'aria-hidden': 'true' }, container);
        el('rect', { x: 141, y: 0, width: 5, height: 30, rx: 2.5, class: 'antenna' }, svg);
        el('path', { class: 'body', d: 'M30 26 Q30 16 42 16 L178 16 Q190 16 190 26 L196 112 Q198 140 172 142 L150 142 Q138 142 132 132 L124 120 L96 120 L88 132 Q82 142 70 142 L48 142 Q22 140 24 112 Z' }, svg);
        el('path', { class: 'bevel', d: 'M38 28 Q38 22 46 22 L174 22 Q182 22 182 28' }, svg);
        // toggle switches on the shoulders
        for (const x of [44, 58, 162, 176]) {
            el('circle', { cx: x, cy: 30, r: 3.4, class: 'switch-base' }, svg);
            el('rect', { x: x - 1.2, y: 21, width: 2.4, height: 9, rx: 1.2, class: 'switch' }, svg);
        }
        // screen
        el('rect', { x: 84, y: 29, width: 52, height: 24, rx: 2, class: 'lcd' }, svg);
        this.mode = el('text', { x: 110, y: 39.5, class: 'lcd-text', 'text-anchor': 'middle' }, svg);
        this.timer = el('text', { x: 110, y: 48.5, class: 'lcd-text small', 'text-anchor': 'middle' }, svg);
        this.sticks = [[62, 78], [158, 78]].map(([cx, cy]) => {
            el('circle', { cx, cy, r: GIMBAL + 4, class: 'bezel' }, svg);
            el('circle', { cx, cy, r: GIMBAL, class: 'well' }, svg);
            for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]])
                el('line', { x1: cx + dx * (GIMBAL - 7), y1: cy + dy * (GIMBAL - 7), x2: cx + dx * (GIMBAL - 2), y2: cy + dy * (GIMBAL - 2), class: 'tick' }, svg);
            el('circle', { cx, cy, r: 4.5, class: 'pivot' }, svg);
            const shaft = el('line', { x1: cx, y1: cy, x2: cx, y2: cy, class: 'shaft' }, svg);
            const tip = el('circle', { cx, cy, r: 6.5, class: 'tip' }, svg);
            el('circle', { cx: 0, cy: 0, r: 2.4, class: 'tip-dot' }, svg);
            const dot = svg.lastChild;
            return { cx, cy, shaft, tip, dot };
        });
        el('text', { x: 62, y: 128, class: 'label', 'text-anchor': 'middle' }, svg).textContent = 'THR · YAW';
        el('text', { x: 158, y: 128, class: 'label', 'text-anchor': 'middle' }, svg).textContent = 'PIT · ROLL';
    }

    // throttle in [0, 1]; yaw, pitch, roll in [-1, 1] (stick right = yaw right / roll right,
    // stick up = pitch forward)
    set(throttle, yaw, pitch, roll) {
        const pos = [[yaw, 2 * throttle - 1], [roll, pitch]];
        this.sticks.forEach((s, i) => {
            const x = s.cx + THROW * Math.max(-1, Math.min(1, pos[i][0]));
            const y = s.cy - THROW * Math.max(-1, Math.min(1, pos[i][1]));
            s.shaft.setAttribute('x2', x); s.shaft.setAttribute('y2', y);
            s.tip.setAttribute('cx', x); s.tip.setAttribute('cy', y);
            s.dot.setAttribute('cx', x - 1.6); s.dot.setAttribute('cy', y - 1.6);
        });
    }

    screen(mode, timer) {
        if (this._mode !== mode) this.mode.textContent = this._mode = mode;
        if (this._timer !== timer) this.timer.textContent = this._timer = timer;
    }
}

export class Telemetry {
    constructor(container, rows) {
        const box = document.createElement('div');
        box.className = 'telemetry glass';
        box.innerHTML = '<div class="telemetry-title">Telemetry</div>';
        const dl = document.createElement('dl');
        this.values = {};
        for (const [key, label] of rows) {
            const dt = document.createElement('dt'); dt.textContent = label;
            const dd = document.createElement('dd'); dd.textContent = '–';
            dl.append(dt, dd);
            this.values[key] = dd;
        }
        box.appendChild(dl);
        container.appendChild(box);
        this.last = {};
    }
    set(key, text) {
        if (this.last[key] !== text) this.values[key].textContent = this.last[key] = text;
    }
}
