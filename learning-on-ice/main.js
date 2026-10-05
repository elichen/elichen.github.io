import { BodyRenderer } from './render.js';
import { ReturnChart, fmtSteps } from './chart.js';
import { WORLDS } from './worlds.js';

const $ = id => document.getElementById(id);
const worldName = w => (WORLDS[w]?.label || w).toLowerCase();

const state = { world: 'normal', learning: true, speed: '1', step: 0, ready: false };
let renderer = null, lastFrame = null, chartDirty = false;
const recent = [];

const chart = new ReturnChart($('returns'), $('returns-tip'), {
    window: 600000,
    empty: 'Each episode lasts up to 1,000 steps (15 s at 1×); they appear here as they end',
    describe: p => `${fmtSteps(p.x)} steps · ${worldName(p.world)} · learning ${p.learning ? 'on' : 'off'}`
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
worker.postMessage({ type: 'init', world: state.world });
requestAnimationFrame(() => chart.draw());

function onReady(info) {
    state.ready = true;
    state.pretrained = info.pretrainedSteps;
    renderer = new BodyRenderer($('view'), info);
    renderer.setWorld(WORLDS[state.world]);
    document.body.classList.add('ready');
    $('status').textContent = '';
    new ResizeObserver(() => renderer.resize()).observe($('view'));
    requestAnimationFrame(frame);
}

function onEpisode(ep) {
    chart.add({ x: ep.step, y: ep.ret, world: ep.world, learning: ep.learning });
    chartDirty = true;
    recent.push(ep.ret);
    if (recent.length > 20) recent.shift();
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
            // Actor and critic together: mean |δz|/v as a percentage of each weight's cap
            $('ro-push').textContent = state.learning ? `${(50 * (f.push + f.criticPush)).toFixed(1)}%` : 'off';
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

function setWorld(name, mark = true) {
    state.world = name;
    worker.postMessage({ type: 'world', value: name });
    renderer?.setWorld(WORLDS[name]);
    document.querySelectorAll('[data-world]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.world === name)));
    $('hud-floor').textContent = WORLDS[name].hud;
    if (mark) markEvent(worldName(name));
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

$('learning').addEventListener('change', e => setLearning(e.target.checked));
$('reset-agent').addEventListener('click', resetAgent);
document.querySelectorAll('[data-world]').forEach(b => b.addEventListener('click', () => setWorld(b.dataset.world)));
document.querySelectorAll('[data-speed]').forEach(b => b.addEventListener('click', () => setSpeed(b.dataset.speed)));

// Buttons inside the article: data-do="ice rubber max learning-off ..."
document.querySelectorAll('[data-do]').forEach(b => b.addEventListener('click', () => {
    for (const action of b.dataset.do.split(' ')) {
        if (action in WORLDS) setWorld(action);
        if (action === 'max') setSpeed('max');
        if (action === 'realtime') setSpeed('1');
        if (action === 'learning-off' && state.learning) setLearning(false);
        if (action === 'learning-on' && !state.learning) setLearning(true);
        if (action === 'reset') resetAgent();
    }
    // Bring the view back if it has scrolled away
    const r = $('view').getBoundingClientRect();
    if (r.bottom < 120 || r.top > innerHeight - 120) $('demo').scrollIntoView({ behavior: 'smooth' });
}));

setWorld('normal', false);
setSpeed('1');

// --- Recorded runs in the article (exported from tools/runs by tools/export-run.mjs)
document.querySelectorAll('canvas[data-run]').forEach(async canvas => {
    const run = await fetch(`data/${canvas.dataset.run}.json`).then(r => r.ok ? r.json() : null).catch(() => null);
    if (!run) { canvas.closest('figure').hidden = true; return; }
    const tip = canvas.parentElement.querySelector('.tooltip');
    const c = new ReturnChart(canvas, tip, {
        describe: p => `${fmtSteps(p.x)} steps · ${worldName(p.world)}${run.frozen ? ' · learning off' : ''}`
    });
    c.setData(run.points.map(([x, y, w]) => ({ x, y, world: w })), run.events.map(([x, label]) => ({ x, label })));
    new ResizeObserver(() => c.draw()).observe(canvas);
});

// Math
if (window.renderMathInElement) {
    window.renderMathInElement(document.body, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }] });
}
