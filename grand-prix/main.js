import { loadAll, DT } from './sim.js';
import { Race } from './race.js';
import { View } from './render.js';
import { Controls } from './input.js';
import { Engine } from './audio.js';

// TUM racetrack-database names -> what people call them
const CIRCUITS = {
    Austin: ['Circuit of the Americas', 'Austin, United States'],
    Budapest: ['Hungaroring', 'Budapest, Hungary'],
    Catalunya: ['Circuit de Barcelona-Catalunya', 'Montmeló, Spain'],
    Hockenheim: ['Hockenheimring', 'Hockenheim, Germany'],
    IMS: ['Indianapolis Motor Speedway', 'the oval, Indianapolis, United States'],
    MexicoCity: ['Autódromo Hermanos Rodríguez', 'Mexico City, Mexico'],
    Montreal: ['Circuit Gilles Villeneuve', 'Montreal, Canada'],
    Monza: ['Monza', 'Monza, Italy'],
    MoscowRaceway: ['Moscow Raceway', 'Volokolamsk, Russia'],
    Norisring: ['Norisring', 'Nuremberg, Germany'],
    Nuerburgring: ['Nürburgring', 'Nürburg, Germany'],
    Oschersleben: ['Motorsport Arena Oschersleben', 'Oschersleben, Germany'],
    Sakhir: ['Bahrain International Circuit', 'Sakhir, Bahrain'],
    SaoPaulo: ['Interlagos', 'São Paulo, Brazil'],
    Sepang: ['Sepang International Circuit', 'Sepang, Malaysia'],
    Shanghai: ['Shanghai International Circuit', 'Shanghai, China'],
    Silverstone: ['Silverstone', 'Silverstone, United Kingdom'],
    Sochi: ['Sochi Autodrom', 'Sochi, Russia'],
    Spa: ['Spa-Francorchamps', 'Stavelot, Belgium'],
    Spielberg: ['Red Bull Ring', 'Spielberg, Austria'],
    Suzuka: ['Suzuka', 'Suzuka, Japan'],
    YasMarina: ['Yas Marina', 'Abu Dhabi, United Arab Emirates'],
};
const SHORT = { Austin: 'Austin', MexicoCity: 'Mexico City', MoscowRaceway: 'Moscow', Nuerburgring: 'Nürburgring', SaoPaulo: 'Interlagos', YasMarina: 'Yas Marina', Spielberg: 'Red Bull Ring', IMS: 'Indianapolis', Sakhir: 'Bahrain', Catalunya: 'Barcelona', Budapest: 'Hungaroring' };

// Colors sampled from one photo (Wikimedia Commons): css for the page, the rest
// for the cars and the circuit.
export const PALETTES = {
    // John Chapman's photo of a Lotus 72 in John Player Special colors beside a McLaren M19
    A: {
        css: { ink: '#181716', paper: '#f1f3db', muted: '#b8b39c', ai: '#f2d785', you: '#f1ca0e', glass: 'rgba(24, 23, 22, 0.78)' },
        ai: { paint: 0x181716, accent: 0xf2d785, helmet: 0xf2d785 },
        you: { paint: 0xf1ca0e, accent: 0x181716, helmet: 0xf3f0e6 },
        asphalt: '#55565a', grass: '#56703a', verge: 0x86a35a, line: 0xf2f2ec,
        kerbA: '#c8282c', kerbB: '#f2f0e8', wall: '#dedcd2', wallBand: '#181716',
        gantry: 0x2c2c2e, stand: 0x8d9198, roof: 0xd8d8d4, pit: 0xe8e6de,
        crowd: ['#f1ca0e', '#f2d785', '#d23a2a', '#f4f4f0', '#2a4f8a', '#181716', '#956448', '#3d7d3a'],
        trees: ['#3e5a2a', '#4b6a30', '#365224', '#57733a', '#2f4a22'],
        haze: 0xc9d3dc,
    },
};

const $ = id => document.getElementById(id);
const read = async (name, kind) => {
    const r = await fetch('model/' + name);
    if (!r.ok) throw new Error(`${name}: ${r.status}`);
    return kind === 'text' ? r.text() : r.arrayBuffer();
};

const fmt = t => {
    if (t == null || !isFinite(t)) return '–';
    const m = Math.floor(t / 60), s = t - 60 * m;
    return m ? `${m}:${s.toFixed(3).padStart(6, '0')}` : s.toFixed(3);
};

const state = { track: null, laps: 1, race: null, phase: 'picker', prev: null, acc: 0, lightsAt: 0 };
const view = new View($('view'), PALETTES.A);
const controls = new Controls($('touch'));
const engine = new Engine();
let data;

async function init() {
    data = await loadAll('', read);
    const centers = {};
    data.centers = centers;
    const list = $('tracks');
    const names = Object.keys(CIRCUITS).filter(n => data.tracks[n]);
    for (const name of names) {
        const t = data.tracks[name];
        const li = document.createElement('li');
        const b = document.createElement('button');
        b.type = 'button';
        b.dataset.track = name;
        b.setAttribute('aria-pressed', 'false');
        b.innerHTML = `${outline(t, 72, 54)}<span class="name">${SHORT[name] ?? CIRCUITS[name][0]}</span>`;
        b.addEventListener('click', () => select(name));
        li.append(b);
        list.append(li);
    }
    await select(new URLSearchParams(location.search).get('track') ?? 'Suzuka');
    $('start').disabled = false;
    $('start').textContent = 'Start the race';
    $('watch').disabled = false;
    requestAnimationFrame(loop);
}

// An SVG outline of the racing line, north up.
function outline(t, w, h, cls = '') {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < t.count; i++) {
        minX = Math.min(minX, t.x[i]); maxX = Math.max(maxX, t.x[i]);
        minY = Math.min(minY, t.y[i]); maxY = Math.max(maxY, t.y[i]);
    }
    const k = Math.min(w / (maxX - minX), h / (maxY - minY));
    const ox = (w - k * (maxX - minX)) / 2, oy = (h - k * (maxY - minY)) / 2;
    const P = i => `${(ox + k * (t.x[i] - minX)).toFixed(1)},${(oy + k * (maxY - t.y[i])).toFixed(1)}`;
    let d = `M${P(0)}`;
    for (let i = 3; i < t.count; i += 3) d += `L${P(i)}`;
    return `<svg viewBox="0 0 ${w} ${h}" class="${cls}" aria-hidden="true"><path d="${d}Z"/></svg>`;
}

async function select(name) {
    if (!data.tracks[name]) name = 'Suzuka';
    state.track = name;
    for (const b of document.querySelectorAll('.tracks button')) {
        b.setAttribute('aria-pressed', String(b.dataset.track === name));
        if (b.dataset.track === name) b.scrollIntoView({ block: 'nearest' });
    }
    const t = data.tracks[name];
    if (!data.centers[name]) data.centers[name] = new Float32Array(await read(`tracks/${name}.center.bin`, 'binary'));
    view.setCircuit(t, data.centers[name]);
    view.setLights(0);
    const [full, place] = CIRCUITS[name];
    $('go-title').textContent = full;
    const ai = data.meta.ai_laps?.[name];
    $('go-meta').textContent = `${place} · ${(t.length / 1000).toFixed(2)} km` + (ai ? ` · the AI laps it in ${fmt(ai)}` : '');
    const url = new URL(location.href);
    url.searchParams.set('track', name);
    history.replaceState(null, '', url);
}

// watch: the AI races alone and the cameras follow it
function startRace(watch = false) {
    const t = data.tracks[state.track];
    state.watch = watch;
    state.race = new Race(t, data.policy, state.laps, { watch });
    view.follow = watch ? 'ai' : 'you';
    view.cars.you.visible = !watch;
    setCam(watch ? 'tv' : 'chase');
    state.prev = null;
    state.acc = 0;
    state.phase = 'grid';
    state.lightsAt = performance.now();
    state.lightsOut = 5000 + 200 + Math.random() * 1800;
    view.chase.ready = false;
    view.setLights(0);
    $('picker').hidden = true;
    $('results').hidden = true;
    $('hud').hidden = false;
    $('lights').classList.remove('off');
    buildMinimap(t);
    state.map.you.style.display = watch ? 'none' : '';
    engine.start();
}

function buildMinimap(t) {
    const svg = $('minimap');
    svg.innerHTML = outline(t, 180, 150).replace(/^<svg[^>]*>|<\/svg>$/g, '');
    svg.setAttribute('viewBox', '0 0 180 150');
    const dot = (cls, color) => {
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('r', '5');
        c.setAttribute('fill', color);
        c.setAttribute('stroke', '#181716');
        c.setAttribute('stroke-width', '1.5');
        c.classList.add(cls);
        svg.append(c);
        return c;
    };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < t.count; i++) {
        minX = Math.min(minX, t.x[i]); maxX = Math.max(maxX, t.x[i]);
        minY = Math.min(minY, t.y[i]); maxY = Math.max(maxY, t.y[i]);
    }
    const k = Math.min(180 / (maxX - minX), 150 / (maxY - minY));
    const ox = (180 - k * (maxX - minX)) / 2, oy = (150 - k * (maxY - minY)) / 2;
    state.map = { k, ox, oy, minX, maxY, ai: dot('ai', getCss('--ai')), you: dot('you', getCss('--you')) };
}

const getCss = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function placeDot(dot, st) {
    const m = state.map;
    dot.setAttribute('cx', (m.ox + m.k * (st[0] - m.minX)).toFixed(1));
    dot.setAttribute('cy', (m.oy + m.k * (m.maxY - st[1])).toFixed(1));
}

// ---- main loop ---------------------------------------------------------------------
let last = performance.now(), clock = 0;
function loop(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    clock += dt;
    if (state.phase === 'picker') {
        view.overview(clock);
    } else {
        const race = state.race;
        if (state.phase === 'grid') {
            const t = now - state.lightsAt;
            const lit = Math.min(5, Math.floor(t / 1000));
            setLights(t >= state.lightsOut ? 0 : lit);
            if (t >= state.lightsOut) {
                state.phase = 'race';
                race.start();
                $('lights').classList.add('off');
                flash('Go!', 900);
            }
        }
        if (state.phase !== 'grid') {
            state.acc += dt;
            while (state.acc >= DT) {
                state.prev = { ai: Float64Array.from(race.ai.state), you: Float64Array.from(race.you.state) };
                race.step(controls.read(race.you.state[3], DT));
                state.acc -= DT;
                events(race);
            }
        }
        view.frame(race, state.prev, state.phase === 'grid' ? 1 : state.acc / DT, dt);
        hud(race);
        engine.update(race.you, race.ai, view.camera, view.cars, state.watch);
        if (race.phase === 'finished' && state.phase !== 'done') finish(race);
    }
    requestAnimationFrame(loop);
}

function setLights(n) {
    view.setLights(n);
    document.querySelectorAll('#lights span').forEach((s, k) => s.classList.toggle('on', k < n));
}

let messageTimer = 0;
function flash(text, ms = 1600) {
    const m = $('message');
    m.textContent = text;
    m.classList.add('show');
    clearTimeout(messageTimer);
    messageTimer = setTimeout(() => m.classList.remove('show'), ms);
}

function events(race) {
    const you = race.you, ai = race.ai;
    if (state.watch) {
        if (ai.out && !ai.out.told) { ai.out.told = true; flash(`The AI ${ai.out.reason}`); }
        if (ai.lapTimes.length !== (ai.lapsTold ?? 0) && !ai.finish) { ai.lapsTold = ai.lapTimes.length; flash(`Lap ${ai.lapsTold}: ${fmt(ai.lapTimes.at(-1))}`, 2200); }
        return;
    }
    if (you.out && !you.out.told) { you.out.told = true; flash(you.out.reason === 'into the wall' ? 'Into the wall' : you.out.reason === 'spun' ? 'Spun' : 'Stopped'); }
    if (ai.out && !ai.out.told) { ai.out.told = true; flash(`The AI ${ai.out.reason}`); }
    if (you.onGrass && !you.grassTold) { you.grassTold = true; flash('Off the track', 900); }
    if (!you.onGrass) you.grassTold = false;
    for (const car of [you, ai]) {
        if (car.lapTimes.length !== (car.lapsTold ?? 0)) {
            car.lapsTold = car.lapTimes.length;
            if (car === you && !car.finish) flash(`Lap ${car.lapTimes.length}: ${fmt(car.lapTimes.at(-1))}`, 2200);
        }
    }
    if (you.finish && !you.finishTold) {
        you.finishTold = true;
        flash(race.winner === 'you' ? 'You win!' : 'Finished', 2500);
    }
}

function hud(race) {
    const you = race.you, ai = race.ai;
    const car = state.watch ? ai : you;  // the dash shows the car you follow
    $('speed').textContent = Math.round(car.state[3] * 3.6);
    $('gear').textContent = gearFor(car.state[3]);
    $('throttle').style.setProperty('--v', car.state[7].toFixed(3));
    $('brake').style.setProperty('--v', car.state[8].toFixed(3));
    const lapNo = Math.min(race.laps, car.lapTimes.length + 1);
    $('laptime').textContent = car.finish ? `Finished ${fmt(car.finish)}` : `Lap ${lapNo}/${race.laps}  ${fmt(Math.max(0, car.time - car.lapStart))}`;
    // tower
    const gap = race.gap();
    const rows = [
        { car: ai, name: 'AI', color: getCss('--ai') },
        { car: you, name: 'You', color: getCss('--you') },
    ].filter(r => !(state.watch && r.car === you))
     .sort((a, b) => (b.car.finish ? 1e9 - b.car.finish : b.car.progress) - (a.car.finish ? 1e9 - a.car.finish : a.car.progress));
    $('tower').innerHTML = rows.map((r, k) => {
        let g = '';
        if (k === 0) g = r.car.finish ? fmt(r.car.finish) : state.watch ? fmt(r.car.time) : 'Leader';
        else {
            const behind = r.car === you ? gap : -gap;
            g = r.car.finish ? `+${(r.car.finish - rows[0].car.finish).toFixed(3)}` : `+${Math.max(0, behind).toFixed(3)}`;
        }
        const lap = r.car.finish ? 'Finished' : `Lap ${Math.min(race.laps, r.car.lapTimes.length + 1)}/${race.laps}`;
        return `<li><span class="pos">${k + 1}</span><span class="bar" style="background:${r.color}"></span><span>${r.name}</span><span class="gap">${g}</span><span class="lap">${lap}</span></li>`;
    }).join('');
    if (state.map) { placeDot(state.map.ai, ai.state); placeDot(state.map.you, you.state); }
}

// 8 gears, shifting at 11,800 rpm; only for the display and the engine sound
const GEAR_TOP = [0, 22, 32, 41, 50, 60, 70, 80, 200];
export function gearFor(u) {
    for (let g = 1; g <= 8; g++) if (u < GEAR_TOP[g]) return g;
    return 8;
}

function finish(race) {
    state.phase = 'done';
    const you = race.you, ai = race.ai;
    if (state.watch) {
        $('results-title').textContent = `The AI: ${fmt(ai.finish)}`;
        $('results-table').innerHTML = ai.lapTimes.map((t, k) => `<tr><td>Lap ${k + 1}</td><td>${fmt(t)}</td></tr>`).join('');
        setTimeout(() => { $('results').hidden = false; $('again').focus(); }, 1200);
        return;
    }
    const won = race.winner === 'you';
    $('results-title').textContent = won ? 'You beat the AI' : 'The AI wins';
    const best = car => Math.min(...car.lapTimes);
    const rows = [[ai, 'AI'], [you, 'You']].sort((a, b) => a[0].finish - b[0].finish);
    $('results-table').innerHTML = rows.map(([car, name], k) => `
        <tr><td>${k + 1}. ${name}</td><td>${k ? '+' + (car.finish - rows[0][0].finish).toFixed(3) : fmt(car.finish)}</td></tr>
        <tr class="sub"><td colspan="2">${car.lapTimes.length > 1 ? `best lap ${fmt(best(car))}` : `lap ${fmt(car.lapTimes[0])}`}</td></tr>`).join('');
    setTimeout(() => { $('results').hidden = false; $('again').focus(); }, 1200);
}

function toPicker() {
    state.phase = 'picker';
    state.race = null;
    $('hud').hidden = true;
    $('results').hidden = true;
    $('picker').hidden = false;
    engine.stop();
    setLights(0);
}

// ---- wiring ---------------------------------------------------------------------------
$('start').addEventListener('click', () => startRace(false));
$('watch').addEventListener('click', () => startRace(true));
$('again').addEventListener('click', () => startRace(state.watch));
$('back').addEventListener('click', toPicker);
$('quit').addEventListener('click', toPicker);
document.querySelectorAll('[data-laps]').forEach(b => b.addEventListener('click', () => {
    state.laps = Number(b.dataset.laps);
    document.querySelectorAll('[data-laps]').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
}));
const setCam = mode => {
    view.mode = mode;
    view.tv = null;
    document.querySelectorAll('[data-cam]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.cam === mode)));
};
document.querySelectorAll('[data-cam]').forEach(b => b.addEventListener('click', () => setCam(b.dataset.cam)));
addEventListener('keydown', e => {
    if (e.code === 'KeyC' && state.phase !== 'picker') {
        const modes = ['chase', 'tv', 'onboard'];
        setCam(modes[(modes.indexOf(view.mode) + 1) % 3]);
    }
    if (e.code === 'Escape') {
        if ($('about').classList.contains('open')) closeAbout();
        else if (state.phase !== 'picker') toPicker();
    }
    if (e.code === 'Enter' && state.phase === 'picker' && !$('start').disabled && document.activeElement?.tagName !== 'BUTTON') startRace(false);
});
$('sound').addEventListener('click', () => {
    const on = engine.toggle();
    $('sound').setAttribute('aria-pressed', String(on));
});

const about = $('about');
const closeAbout = () => { about.classList.remove('open'); about.inert = true; };
$('about-open').addEventListener('click', async () => {
    if (!$('about-prose').innerHTML) $('about-prose').innerHTML = await (await fetch('about.html')).text();
    about.inert = false;
    about.classList.add('open');
    $('about-close').focus();
});
$('about-close').addEventListener('click', closeAbout);

window.grandPrix = { state, view, controls, get data() { return data; } };
init().catch(err => {
    $('start').textContent = 'Could not load';
    $('go-meta').textContent = String(err);
    console.error(err);
});
