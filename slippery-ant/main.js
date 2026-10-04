import { AntRenderer } from './render.js';
import { ReturnChart, fmtSteps } from './chart.js';

const MU_ICE = 0.02, MU_RUBBER = 2.0;
const $ = id => document.getElementById(id);
const fmtMu = mu => mu >= 0.995 ? mu.toFixed(1) : mu >= 0.0995 ? mu.toFixed(2) : mu.toFixed(3).replace(/0$/, '');
const floorName = mu => mu >= 1.5 ? 'rubber' : mu <= 0.03 ? 'ice' : 'μ ' + fmtMu(mu);

const state = { friction: MU_RUBBER, learning: true, speed: '1', step: 0, ready: false };
let renderer = null, lastFrame = null, chartDirty = false;
const recent = [];

const chart = new ReturnChart($('returns'), $('returns-tip'), {
    window: 600000,
    empty: 'Each episode lasts up to 1,000 steps (50 s at 1×); they appear here as they end',
    describe: p => `${fmtSteps(p.x)} steps · ${floorName(p.friction)} · learning ${p.learning ? 'on' : 'off'}`
});

const worker = new Worker('worker.js', { type: 'module' });
worker.onmessage = ({ data }) => {
    if (data.type === 'ready') onReady(data);
    else if (data.type === 'frame') lastFrame = data;
    else if (data.type === 'episode') onEpisode(data);
};
worker.onerror = e => {
    $('status').textContent = 'The simulation failed to start: ' + (e.message || 'unknown error');
};
worker.postMessage({ type: 'init', friction: state.friction });
requestAnimationFrame(() => chart.draw());

function onReady(info) {
    state.ready = true;
    state.pretrained = info.pretrainedSteps;
    renderer = new AntRenderer($('view'), info);
    renderer.setFriction(state.friction);
    document.body.classList.add('ready');
    $('status').textContent = '';
    new ResizeObserver(() => renderer.resize()).observe($('view'));
    requestAnimationFrame(frame);
}

function onEpisode(ep) {
    chart.add({ x: ep.step, y: ep.ret, friction: ep.friction, learning: ep.learning });
    chartDirty = true;
    recent.push(ep.ret);
    if (recent.length > 20) recent.shift();
    $('ro-last').textContent = Math.round(ep.ret).toLocaleString();
    $('ro-avg').textContent = Math.round(recent.reduce((a, b) => a + b, 0) / recent.length).toLocaleString();
}

let rateTime = 0, lastChart = 0;
function frame(now) {
    if (lastFrame && renderer) {
        renderer.update(lastFrame);
        const f = lastFrame;
        if (f.step !== state.step) {
            state.step = f.step;
            $('ro-steps').textContent = fmtSteps(f.step);
            $('ro-total').textContent = fmtSteps(f.step + (state.pretrained || 0));
            $('ro-ep').textContent = `${Math.round(f.epReturn).toLocaleString()} after ${f.epLength}`;
            $('ro-delta').textContent = state.learning ? f.meanDelta.toFixed(3) : '—';
            $('ro-value').textContent = f.value.toFixed(2);
            $('ro-push').textContent = state.learning ? `${(100 * f.push).toFixed(1)}% · ${(100 * f.criticPush).toFixed(1)}%` : '—';
        }
    }
    if (lastFrame && now - rateTime > 500) {
        rateTime = now;
        $('ro-rate').textContent = `${Math.round(lastFrame.rate).toLocaleString()}/s`;
    }
    if (chartDirty && now - lastChart > 200) {
        chart.draw();
        chartDirty = false;
        lastChart = now;
    }
    renderer?.render();
    requestAnimationFrame(frame);
}

// --- Controls (the panel and the buttons inside the article share these)

const sliderToMu = v => MU_ICE * (MU_RUBBER / MU_ICE) ** (v / 100);
const muToSlider = mu => 100 * Math.log(mu / MU_ICE) / Math.log(MU_RUBBER / MU_ICE);

function setFriction(mu, mark = true) {
    state.friction = mu;
    worker.postMessage({ type: 'friction', value: mu });
    renderer?.setFriction(mu);
    $('friction').value = muToSlider(mu);
    $('friction-value').textContent = fmtMu(mu);
    document.querySelectorAll('[data-floor]').forEach(b => b.setAttribute('aria-pressed', String(
        (b.dataset.floor === 'ice' && mu <= 0.03) || (b.dataset.floor === 'rubber' && mu >= 1.5))));
    $('hud-floor').textContent = floorName(mu);
    document.body.dataset.floor = mu <= 0.2 ? 'ice' : 'rubber';
    if (mark) markEvent(floorName(mu));
}

function setLearning(on) {
    state.learning = on;
    worker.postMessage({ type: 'learning', value: on });
    $('learning').checked = on;
    markEvent(on ? 'learning on' : 'learning off');
}

function setSpeed(s) {
    state.speed = s;
    worker.postMessage({ type: 'speed', value: s });
    document.querySelectorAll('[data-speed]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.speed === s)));
}

function resetAgent() {
    worker.postMessage({ type: 'reset-agent' });
    markEvent('agent reset');
}

function markEvent(label) {
    chart.mark(state.step, label);
    chartDirty = true;
}

$('friction').addEventListener('input', e => {
    const mu = sliderToMu(+e.target.value);
    state.friction = mu;
    worker.postMessage({ type: 'friction', value: mu });
    renderer?.setFriction(mu);
    $('friction-value').textContent = fmtMu(mu);
});
$('friction').addEventListener('change', e => setFriction(sliderToMu(+e.target.value)));
$('learning').addEventListener('change', e => setLearning(e.target.checked));
$('reset-agent').addEventListener('click', resetAgent);
document.querySelectorAll('[data-floor]').forEach(b => b.addEventListener('click', () => setFriction(b.dataset.floor === 'ice' ? MU_ICE : MU_RUBBER)));
document.querySelectorAll('[data-speed]').forEach(b => b.addEventListener('click', () => setSpeed(b.dataset.speed)));

// Buttons inside the article: data-do="ice rubber max learning-off ..."
document.querySelectorAll('[data-do]').forEach(b => b.addEventListener('click', () => {
    for (const action of b.dataset.do.split(' ')) {
        if (action === 'ice') setFriction(MU_ICE);
        if (action === 'rubber') setFriction(MU_RUBBER);
        if (action === 'max') setSpeed('max');
        if (action === 'realtime') setSpeed('1');
        if (action === 'learning-off' && state.learning) setLearning(false);
        if (action === 'learning-on' && !state.learning) setLearning(true);
        if (action === 'reset') resetAgent();
    }
    if (window.matchMedia('(max-width: 1099px)').matches) $('demo').scrollIntoView({ behavior: 'smooth' });
}));

setFriction(MU_RUBBER, false);
setSpeed('1');

// --- Recorded runs in the article (exported from tools/runs by tools/export-run.mjs)
document.querySelectorAll('canvas[data-run]').forEach(async canvas => {
    const run = await fetch(`data/${canvas.dataset.run}.json`).then(r => r.ok ? r.json() : null).catch(() => null);
    if (!run) { canvas.closest('figure').hidden = true; return; }
    const tip = canvas.parentElement.querySelector('.tooltip');
    const c = new ReturnChart(canvas, tip, {
        describe: p => `${fmtSteps(p.x)} steps · ${floorName(p.friction)}${run.frozen ? ' · learning off' : ''}`
    });
    c.setData(run.points.map(([x, y, f]) => ({ x, y, friction: f })), run.events.map(([x, label]) => ({ x, label })));
    new ResizeObserver(() => c.draw()).observe(canvas);
});

// Math
if (window.renderMathInElement) {
    window.renderMathInElement(document.body, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }] });
}
