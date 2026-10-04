// Episode returns over time: each episode as a faint dot, the 10-episode average
// as a line, and a hairline wherever the physics or the agent was changed.
class ReturnChart {
    constructor(canvas, tooltip, maxReturn = 1000, windowSize = 200) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.tooltip = tooltip;
        this.maxReturn = maxReturn;
        this.windowSize = windowSize;
        this.history = [];
        this.events = [];
        this.hoverEpisode = null;
        this.margin = { left: 40, right: 44, top: 22, bottom: 24 };
        this.colors = {
            series: '#2a78d6',
            episode: 'rgba(42, 120, 214, 0.35)',
            grid: '#e1e0d9',
            event: '#b5b3ab',
            text: '#52514e',
            muted: '#898781'
        };

        canvas.addEventListener('pointermove', e => this.onPointer(e));
        canvas.addEventListener('pointerleave', () => {
            this.hoverEpisode = null;
            this.tooltip.hidden = true;
            this.draw();
        });
    }

    update(history, events) {
        this.history = history;
        this.events = events;
        this.draw();
    }

    // Episodes shown: the last windowSize, but at least 20 wide so early points don't stretch
    range() {
        const last = this.history.length ? this.history[this.history.length - 1].episode : 0;
        const lo = Math.max(0, last - this.windowSize);
        return { lo, hi: Math.max(lo + 20, last) };
    }

    layout() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
        if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
            this.canvas.width = Math.round(w * dpr);
            this.canvas.height = Math.round(h * dpr);
        }
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const m = this.margin;
        const { lo, hi } = this.range();
        return {
            w, h, lo, hi,
            x: ep => m.left + (ep - lo) / (hi - lo) * (w - m.left - m.right),
            y: r => m.top + (1 - r / this.maxReturn) * (h - m.top - m.bottom)
        };
    }

    draw() {
        const ctx = this.ctx;
        const { w, h, lo, hi, x, y } = this.layout();
        const m = this.margin;
        ctx.clearRect(0, 0, w, h);
        ctx.font = '11px system-ui, -apple-system, sans-serif';
        ctx.lineWidth = 1;

        // Gridlines and y ticks
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let r = 0; r <= this.maxReturn; r += 250) {
            const yy = Math.round(y(r)) + 0.5;
            ctx.strokeStyle = this.colors.grid;
            ctx.beginPath();
            ctx.moveTo(m.left, yy);
            ctx.lineTo(w - m.right, yy);
            ctx.stroke();
            ctx.fillStyle = this.colors.muted;
            ctx.fillText(r.toLocaleString(), m.left - 6, yy);
        }

        // X ticks on round episode numbers
        const span = hi - lo;
        const step = [5, 10, 20, 25, 50, 100].find(s => span / s <= 6) || 200;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = this.colors.muted;
        for (let ep = Math.ceil(lo / step) * step; ep <= hi; ep += step) {
            ctx.fillText(ep.toLocaleString(), x(ep), h - m.bottom + 6);
        }

        // Event hairlines, labeled newest first; a label that would overlap the
        // one to its right is skipped (the hairline stays)
        const visible = this.events.filter(e => e.x >= lo && e.x <= hi);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        let labelLimit = w;
        for (let i = visible.length - 1; i >= 0; i--) {
            const ex = Math.round(x(visible[i].x)) + 0.5;
            ctx.strokeStyle = this.colors.event;
            ctx.beginPath();
            ctx.moveTo(ex, m.top - 6);
            ctx.lineTo(ex, h - m.bottom);
            ctx.stroke();
            const width = ctx.measureText(visible[i].label).width;
            const lx = Math.min(ex + 3, w - width);
            if (lx + width + 8 <= labelLimit) {
                ctx.fillStyle = this.colors.text;
                ctx.fillText(visible[i].label, lx, m.top - 9);
                labelLimit = lx;
            }
        }

        const points = this.history.filter(p => p.episode >= lo);
        if (!points.length) return;

        // Each episode
        ctx.fillStyle = this.colors.episode;
        for (const p of points) {
            ctx.beginPath();
            ctx.arc(x(p.episode), y(p.ret), 2.5, 0, 2 * Math.PI);
            ctx.fill();
        }

        // 10-episode average
        ctx.strokeStyle = this.colors.series;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        points.forEach((p, i) => i ? ctx.lineTo(x(p.episode), y(p.avg)) : ctx.moveTo(x(p.episode), y(p.avg)));
        ctx.stroke();

        // End dot with a surface ring, and the latest average as a direct label
        const last = points[points.length - 1];
        ctx.beginPath();
        ctx.arc(x(last.episode), y(last.avg), 4, 0, 2 * Math.PI);
        ctx.fillStyle = this.colors.series;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#fff';
        ctx.stroke();
        ctx.fillStyle = this.colors.text;
        ctx.textBaseline = 'middle';
        ctx.fillText(Math.round(last.avg).toLocaleString(), x(last.episode) + 8, y(last.avg));

        // Crosshair
        if (this.hoverEpisode !== null) {
            const p = points.find(q => q.episode === this.hoverEpisode);
            if (p) {
                const hx = Math.round(x(p.episode)) + 0.5;
                ctx.strokeStyle = this.colors.text;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(hx, m.top);
                ctx.lineTo(hx, h - m.bottom);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(hx, y(p.avg), 4, 0, 2 * Math.PI);
                ctx.fillStyle = this.colors.series;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }
    }

    onPointer(e) {
        if (!this.history.length) return;
        const { lo, hi, x } = this.layout();
        const rect = this.canvas.getBoundingClientRect();
        const px = e.clientX - rect.left;
        // Snap to the nearest episode
        let best = null;
        for (const p of this.history) {
            if (p.episode < lo || p.episode > hi) continue;
            if (!best || Math.abs(x(p.episode) - px) < Math.abs(x(best.episode) - px)) best = p;
        }
        if (!best) return;
        this.hoverEpisode = best.episode;
        this.draw();

        const t = this.tooltip;
        t.replaceChildren();
        const value = document.createElement('strong');
        value.textContent = Math.round(best.ret).toLocaleString();
        const row1 = document.createElement('div');
        row1.append(value, document.createTextNode(` return, episode ${best.episode}`));
        const row2 = document.createElement('div');
        row2.textContent = `10-episode average ${Math.round(best.avg).toLocaleString()}`;
        const row3 = document.createElement('div');
        row3.className = 'muted';
        row3.textContent = `pole ${best.pole.toFixed(1)} m · force ${best.force} N · learning ${best.learning ? 'on' : 'off'}`;
        t.append(row1, row2, row3);
        t.hidden = false;
        const left = Math.min(Math.max(x(best.episode) + 12, 0), rect.width - t.offsetWidth);
        t.style.left = `${x(best.episode) + 12 + t.offsetWidth > rect.width ? x(best.episode) - 12 - t.offsetWidth : left}px`;
        t.style.top = `${this.margin.top}px`;
    }
}
