// Episode return against environment steps: each episode as a faint dot, a
// 20-episode average as the line, and a labeled hairline at every change.
// The same class draws the live chart and the article's recorded runs.
const css = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

function niceStep(span, target) {
    const raw = span / target, p = 10 ** Math.floor(Math.log10(raw));
    return [1, 2, 2.5, 5, 10].map(m => m * p).find(s => span / s <= target) || 10 * p;
}

export const fmtSteps = s => s >= 1e6 ? `${+(s / 1e6).toFixed(s % 1e6 ? 1 : 0)}M` : s >= 1e3 ? `${Math.round(s / 1e3)}k` : `${s}`;

export class ReturnChart {
    // opts.window: show only the last N steps (live); omit to show everything (recorded runs)
    constructor(canvas, tooltip, opts = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.tooltip = tooltip;
        this.window = opts.window || null;
        this.avgN = opts.avgN || 20;
        this.yMin = opts.yMin ?? -500;
        this.yMax = opts.yMax ?? 8000;
        this.describe = opts.describe || (p => `${fmtSteps(p.x)} steps`);
        this.empty = opts.empty || '';
        this.points = [];   // { x, y, avg, ... }
        this.events = [];   // { x, label }
        this.hover = null;
        this.margin = { left: 44, right: 40, top: 20, bottom: 22 };
        if (tooltip) {
            canvas.addEventListener('pointermove', e => this.onPointer(e));
            canvas.addEventListener('pointerleave', () => { this.hover = null; tooltip.hidden = true; this.draw(); });
        }
    }

    readColors() {
        this.c = {
            series: css('--series', '#2a78d6'),
            dot: css('--series-dot', 'rgba(42,120,214,0.3)'),
            grid: css('--grid', '#e1e0d9'),
            event: css('--event', '#9a9893'),
            text: css('--text-2', '#52514e'),
            muted: css('--text-3', '#898781'),
            surface: css('--surface', '#ffffff')
        };
        // Canvas fonts can't use CSS variables; resolve the family once
        this.font = `11px ${css('--mono', 'ui-monospace, monospace')}`;
    }

    add(point) {
        const recent = this.points.slice(-(this.avgN - 1));
        const avg = (recent.reduce((s, p) => s + p.y, 0) + point.y) / (recent.length + 1);
        this.points.push({ ...point, avg });
        if (this.window) {
            const cut = point.x - this.window * 1.2;
            while (this.points.length && this.points[0].x < cut) this.points.shift();
            while (this.events.length && this.events[0].x < cut) this.events.shift();
        }
    }

    // Recorded runs: points are already in order
    setData(points, events) {
        this.points = [];
        this.events = events;
        for (const p of points) this.add(p);
        this.draw();
    }

    mark(x, label) {
        this.events.push({ x, label });
    }

    range() {
        const last = this.points.length ? this.points[this.points.length - 1].x : 0;
        if (this.window) {
            const hi = Math.max(this.window, last + this.window * 0.04);
            return { lo: hi - this.window, hi };
        }
        return { lo: this.points.length ? Math.min(0, this.points[0].x) : 0, hi: Math.max(1, last) };
    }

    layout() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
        if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
            this.canvas.width = Math.round(w * dpr);
            this.canvas.height = Math.round(h * dpr);
        }
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const m = this.margin, { lo, hi } = this.range();
        return {
            w, h, lo, hi,
            x: s => m.left + (s - lo) / (hi - lo) * (w - m.left - m.right),
            y: r => m.top + (1 - (Math.max(this.yMin, Math.min(this.yMax, r)) - this.yMin) / (this.yMax - this.yMin)) * (h - m.top - m.bottom)
        };
    }

    draw() {
        if (!this.c) this.readColors();
        const ctx = this.ctx, { w, h, lo, hi, x, y } = this.layout(), m = this.margin, c = this.c;
        if (!w) return;
        ctx.clearRect(0, 0, w, h);
        ctx.font = this.font;
        ctx.lineWidth = 1;

        // Gridlines and y ticks
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        const ys = niceStep(this.yMax - this.yMin, 4);
        for (let r = Math.ceil(this.yMin / ys) * ys; r <= this.yMax; r += ys) {
            const yy = Math.round(y(r)) + 0.5;
            ctx.strokeStyle = c.grid;
            ctx.beginPath(); ctx.moveTo(m.left, yy); ctx.lineTo(w - m.right, yy); ctx.stroke();
            ctx.fillStyle = c.muted;
            ctx.fillText((r + 0).toLocaleString(), m.left - 6, yy);  // + 0 turns -0 into 0
        }
        // X ticks
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const xs = niceStep(hi - lo, Math.max(3, Math.floor((w - m.left - m.right) / 90)));
        for (let s = Math.ceil(lo / xs) * xs; s <= hi; s += xs) {
            if (s < 0) continue;
            ctx.fillText(fmtSteps(s), x(s), h - m.bottom + 6);
        }

        // Event hairlines, newest label first; labels that would overlap are skipped
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        let limit = w;
        const visible = this.events.filter(e => e.x >= lo && e.x <= hi);
        for (let i = visible.length - 1; i >= 0; i--) {
            const ex = Math.round(x(visible[i].x)) + 0.5;
            ctx.strokeStyle = c.event;
            ctx.beginPath(); ctx.moveTo(ex, m.top - 6); ctx.lineTo(ex, h - m.bottom); ctx.stroke();
            const tw = ctx.measureText(visible[i].label).width;
            const lx = Math.min(ex + 3, w - tw);
            if (lx + tw + 8 <= limit) {
                ctx.fillStyle = c.text;
                ctx.fillText(visible[i].label, lx, m.top - 8);
                limit = lx;
            }
        }

        const pts = this.points.filter(p => p.x >= lo);
        if (!pts.length) {
            if (this.empty) {
                ctx.fillStyle = c.muted;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(this.empty, (m.left + w - m.right) / 2, (m.top + h - m.bottom) / 2);
            }
            return;
        }
        ctx.fillStyle = c.dot;
        for (const p of pts) {
            ctx.beginPath(); ctx.arc(x(p.x), y(p.y), 2, 0, 2 * Math.PI); ctx.fill();
        }
        ctx.strokeStyle = c.series;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        pts.forEach((p, i) => i ? ctx.lineTo(x(p.x), y(p.avg)) : ctx.moveTo(x(p.x), y(p.avg)));
        ctx.stroke();

        // End dot with a surface ring, and the latest average as a direct label
        const last = pts[pts.length - 1];
        ctx.beginPath(); ctx.arc(x(last.x), y(last.avg), 4, 0, 2 * Math.PI);
        ctx.fillStyle = c.series; ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = c.surface; ctx.stroke();
        ctx.fillStyle = c.text;
        ctx.textBaseline = 'middle';
        ctx.fillText(Math.round(last.avg).toLocaleString(), Math.min(x(last.x) + 7, w - 34), y(last.avg));

        if (this.hover) {
            const hx = Math.round(x(this.hover.x)) + 0.5;
            ctx.strokeStyle = c.text; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(hx, m.top); ctx.lineTo(hx, h - m.bottom); ctx.stroke();
            ctx.beginPath(); ctx.arc(hx, y(this.hover.avg), 4, 0, 2 * Math.PI);
            ctx.fillStyle = c.series; ctx.fill();
            ctx.strokeStyle = c.surface; ctx.lineWidth = 2; ctx.stroke();
        }
    }

    onPointer(e) {
        if (!this.points.length) return;
        const { lo, hi, x } = this.layout();
        const rect = this.canvas.getBoundingClientRect(), px = e.clientX - rect.left;
        let best = null;
        for (const p of this.points) {
            if (p.x < lo || p.x > hi) continue;
            if (!best || Math.abs(x(p.x) - px) < Math.abs(x(best.x) - px)) best = p;
        }
        if (!best) return;
        this.hover = best;
        this.draw();
        const t = this.tooltip;
        t.replaceChildren();
        const v = document.createElement('strong');
        v.textContent = Math.round(best.y).toLocaleString();
        const r1 = document.createElement('div');
        r1.append(v, document.createTextNode(' return'));
        const r2 = document.createElement('div');
        r2.textContent = `${this.avgN}-episode average ${Math.round(best.avg).toLocaleString()}`;
        const r3 = document.createElement('div');
        r3.className = 'muted';
        r3.textContent = this.describe(best);
        t.append(r1, r2, r3);
        t.hidden = false;
        const tx = x(best.x);
        t.style.left = `${tx + 12 + t.offsetWidth > rect.width ? tx - 12 - t.offsetWidth : tx + 12}px`;
        t.style.top = `${this.margin.top}px`;
    }
}
