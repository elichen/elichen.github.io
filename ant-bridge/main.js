import { Sim } from './sim.js';
import { View } from './render.js';

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const view = new View(canvas);
const settings = { angle: 40, traffic: 200, speed: 1 };
let sim = null, said = {}, noteTimer = 0;

function restart() {
    sim = new Sim({ angle: settings.angle, traffic: settings.traffic, seed: (Math.random() * 1e9) | 0 });
    // let the column reach the fork first, so the twig is not empty on the first frame
    while (sim.t < 3) sim.step();
    view.setFork(sim);
    said = {};
    noteTimer = 0;
    $('note').hidden = true;
    $('moved').textContent = '–';
}

function resize() {
    const r = canvas.getBoundingClientRect();
    view.resize(r.width, r.height);
}

function fmtTime(t) {
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

// a short line at the top of the view at a few moments, so a first-time viewer knows what to look for
function note(key, text) {
    if (said[key]) return;
    said[key] = true;
    const el = $('note');
    el.textContent = text;
    el.hidden = false;
    el.classList.remove('fade');
    noteTimer = 7;
}

function narrate(m) {
    if (sim.traffic === 0) {
        if (m.ants > 0) note('stopped', 'No traffic: with nobody walking over them, the ants in the bridge let go.');
        else if (said.stopped) note('gone', 'The bridge is gone. Turn the traffic back up and they build it again.');
        return;
    }
    said.stopped = said.gone = false;
    if (m.joins > 0) note('first', 'An ant stretched across the gap was walked over and locked in place.');
    if (m.d > 2.5 && m.ants >= 4) note('moving', 'The bridge is sliding away from the fork, into the gap. No ant is steering it.');
}

// drop the resolution on slow GPUs: if frames stay slow for two seconds, render fewer pixels
let slow = 0, pixelRatio = Math.min(window.devicePixelRatio, 2);
function adapt(dt) {
    slow = dt > 1 / 40 ? slow + dt : Math.max(0, slow - dt);
    if (slow > 2 && pixelRatio > 1) {
        pixelRatio = Math.max(1, pixelRatio - 0.25);
        view.renderer.setPixelRatio(pixelRatio);
        resize();
        slow = 0;
    }
}

let last = performance.now(), uiTimer = 0;
function frame(now) {
    const real = Math.min(0.1, (now - last) / 1000);
    last = now;
    // run as many fixed simulation steps as the speed asks for, within a time budget
    const t0 = performance.now();
    let todo = real * settings.speed;
    while (todo >= sim.p.dt * 0.5 && performance.now() - t0 < 24) { sim.step(); todo -= sim.p.dt; }
    view.update(real, sim);
    view.render(real);
    adapt(real);
    uiTimer -= real;
    if (uiTimer <= 0) {
        uiTimer = 0.25;
        const m = sim.measure();
        $('ants').textContent = m.ants;
        if (Number.isFinite(m.d) && m.ants) $('moved').textContent = `${m.d.toFixed(1)} cm`;
        else if (!m.ants) $('moved').textContent = '–';
        $('saved').textContent = m.saved > 0.3 ? `${m.saved.toFixed(1)} cm` : '–';
        $('clock').textContent = fmtTime(sim.t);
        narrate(m);
    }
    if (noteTimer > 0) {
        noteTimer -= real;
        if (noteTimer <= 0.6) $('note').classList.add('fade');
    }
    requestAnimationFrame(frame);
}

for (const b of document.querySelectorAll('[data-angle]')) {
    b.addEventListener('click', () => {
        settings.angle = +b.dataset.angle;
        document.querySelectorAll('[data-angle]').forEach(x => x.setAttribute('aria-pressed', x === b));
        restart();
    });
}
for (const b of document.querySelectorAll('[data-speed]')) {
    b.addEventListener('click', () => {
        settings.speed = +b.dataset.speed;
        document.querySelectorAll('[data-speed]').forEach(x => x.setAttribute('aria-pressed', x === b));
    });
}
$('traffic').addEventListener('input', (e) => {
    settings.traffic = +e.target.value;
    $('traffic-value').textContent = settings.traffic;
    sim.traffic = settings.traffic;
});
$('restart').addEventListener('click', restart);
window.addEventListener('resize', resize);

await view.ready;
restart();
resize();
$('status').hidden = true;
window.antBridge = {
    get sim() { return sim; }, view, settings,
    // for testing in a background tab, where requestAnimationFrame is paused
    advance(seconds, frameDt = 1 / 30) {
        const n = Math.round(seconds / sim.p.dt);
        for (let i = 0; i < n; i++) sim.step();
        for (let i = 0; i < 4; i++) { view.update(frameDt, sim); view.render(frameDt); }
        return sim.measure();
    },
};
requestAnimationFrame(frame);
