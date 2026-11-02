// The article's data figure: spikes per second of the whole model, with input on and then off, for three versions of the model (data/rest.json, tools/fig_rest.mjs).
const SERIES = [
    { key: 'original', name: 'Original rule', color: '#b7791f', dash: '' },
    { key: 'noSerotonin', name: 'No serotonin synapses', color: '#c2185b', dash: '6 3' },
    { key: 'adapting', name: 'No serotonin synapses, with adaptation (this page)', color: '#1f6fa8', dash: '' },
];
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent?.appendChild(e); return e; };
const nice = v => { const p = 10 ** Math.floor(Math.log10(v)), m = v / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p; };

function panel(root, data, k, tip) {
    const W = 460, H = 220, M = { l: 52, r: 12, t: 24, b: 34 }, input = data.inputs[k], n = data.seconds * 1000 / data.binMs;
    const top = nice(Math.max(...SERIES.flatMap(s => data.series[s.key][k])));
    const x = i => M.l + (W - M.l - M.r) * (i + 0.5) / n, y = v => H - M.b - (H - M.t - M.b) * v / top;
    const box = document.createElement('div'); box.className = 'fig-panel'; root.appendChild(box);
    const title = document.createElement('p'); title.className = 'fig-title'; title.textContent = input.name; box.appendChild(title);
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `Spikes per second over ${data.seconds} seconds: ${input.name}, on for ${input.on} s. The table below has the values.` }, box);
    el('rect', { x: M.l, y: M.t, width: (W - M.l - M.r) * input.on / data.seconds, height: H - M.t - M.b, fill: 'rgba(0,0,0,0.05)' }, svg);
    el('text', { x: M.l + 6, y: M.t + 14, class: 'fig-note' }, svg).textContent = 'input on';
    for (let i = 0; i <= 2; i++) {
        const v = top * i / 2;
        el('line', { x1: M.l, x2: W - M.r, y1: y(v), y2: y(v), stroke: 'rgba(0,0,0,0.1)' }, svg);
        el('text', { x: M.l - 6, y: y(v) + 4, 'text-anchor': 'end', class: 'fig-axis' }, svg).textContent = v.toLocaleString('en-US');
    }
    for (let s = 0; s <= data.seconds; s++) el('text', { x: M.l + (W - M.l - M.r) * s / data.seconds, y: H - M.b + 16, 'text-anchor': 'middle', class: 'fig-axis' }, svg).textContent = s;
    el('text', { x: (W + M.l - M.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'fig-axis' }, svg).textContent = 'seconds';
    for (const s of SERIES) el('polyline', { points: data.series[s.key][k].map((v, i) => `${x(i)},${y(v)}`).join(' '), fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-dasharray': s.dash, 'stroke-linejoin': 'round' }, svg);
    const cross = el('line', { y1: M.t, y2: H - M.b, stroke: 'rgba(0,0,0,0.35)', visibility: 'hidden' }, svg);
    const dots = SERIES.map(s => el('circle', { r: 4, fill: s.color, stroke: '#fff', 'stroke-width': 2, visibility: 'hidden' }, svg));
    const hit = el('rect', { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: 'transparent' }, svg);
    const move = e => {
        const r = svg.getBoundingClientRect(), i = Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) * W / r.width - M.l) / (W - M.l - M.r) * n - 0.5)));
        cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
        SERIES.forEach((s, j) => { dots[j].setAttribute('cx', x(i)); dots[j].setAttribute('cy', y(data.series[s.key][k][i])); dots[j].setAttribute('visibility', 'visible'); });
        tip.innerHTML = `<b>${((i + 0.5) * data.binMs / 1000).toFixed(2)} s</b>` + SERIES.map(s => `<span><i style="background:${s.color}"></i>${s.name}: ${data.series[s.key][k][i].toLocaleString('en-US')}</span>`).join('');
        tip.hidden = false;
        const host = root.getBoundingClientRect(), left = e.clientX - host.left;
        tip.style.left = `${Math.min(left + 12, host.width - tip.offsetWidth)}px`; tip.style.top = `${e.clientY - host.top + 12}px`;
    };
    hit.addEventListener('pointermove', move); hit.addEventListener('pointerdown', move);
    hit.addEventListener('pointerleave', () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); });
}

export async function restFigure(id) {
    const root = document.getElementById(id), data = await fetch('data/rest.json').then(r => r.json());
    const legend = root.querySelector('.fig-legend'), panels = root.querySelector('.fig-panels'), table = root.querySelector('tbody');
    legend.innerHTML = SERIES.map(s => `<span><svg width="26" height="8" aria-hidden="true"><line x1="0" x2="26" y1="4" y2="4" stroke="${s.color}" stroke-width="2" stroke-dasharray="${s.dash}"/></svg>${s.name}</span>`).join('');
    const tip = document.createElement('div'); tip.className = 'fig-tip'; tip.hidden = true; root.appendChild(tip);
    for (const k of Object.keys(data.inputs)) panel(panels, data, k, tip);
    // the same numbers as a table: during input, and in the last 0.5 s
    const during = (v, on) => { const b = on * 1000 / data.binMs, a = v.slice(Math.floor(b / 2), b); return a.reduce((x, y) => x + y, 0) / a.length; };
    const end = v => { const a = v.slice(-Math.round(500 / data.binMs)); return a.reduce((x, y) => x + y, 0) / a.length; };
    const f = v => (Math.round(v / 100) * 100).toLocaleString('en-US');
    table.innerHTML = Object.keys(data.inputs).flatMap(k => SERIES.map(s => `<tr><td>${data.inputs[k].name}</td><td class="word">${s.name}</td><td>${f(during(data.series[s.key][k], data.inputs[k].on))}</td><td>${f(end(data.series[s.key][k]))}</td></tr>`)).join('');
}
