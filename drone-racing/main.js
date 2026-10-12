// Page: loads the policy and courses, runs the flight at 62.5 Hz policy ticks,
// and draws the field, the radio, telemetry and sound each frame. Dragging the view
// moves the camera anywhere on the field.
import { TICK, MISS, GROUND, BOUNDS, INVERSION, STRIKE } from './sim.js';
import { Flight } from './flight.js';
import { loadPolicy } from './policy.js';
import { pilotSpot } from './layout.js';
import { View } from './render.js';
import { Transmitter, Telemetry } from './hud.js';
import { MotorSound } from './audio.js';
import { CameraInput } from './controls.js';

const COURSES = ['uzh_swift_race_v1', 'uzh_big_s_v2', 'uzh_kidney_v1', 'uzh_figure_eight_v1', 'uzh_multistory_slalom_v1'];
const CRASH = { [STRIKE]: 'hit a ring', [MISS]: 'missed a ring', [GROUND]: 'hit the ground',
    [BOUNDS]: 'left the field', [INVERSION]: 'flipped over' };
const VIEWS = ['pilot', 'chase', 'fpv', 'free'];
const touch = matchMedia('(hover: none) and (pointer: coarse)').matches;
const VIEW_NOTES = touch ? {
    pilot: 'Where the pilot stands. Drag to move the camera.',
    chase: 'Behind the drone. Drag to circle it, pinch to zoom.',
    fpv: 'The drone’s own camera, tilted up 25°.',
    free: 'Drag to orbit. Two fingers move and zoom.',
} : {
    pilot: 'Where the pilot stands. Drag to move the camera anywhere.',
    chase: 'Following the drone. Drag to swing around it, scroll to zoom.',
    fpv: 'The drone’s own camera, tilted up 25° as racers mount them.',
    free: 'Drag to orbit, right-drag to move, scroll to zoom, WASD and Q/E to fly.',
};
const $ = id => document.getElementById(id);

const status = $('status');
const say = text => {
    if (status.textContent !== text) status.textContent = text;
    status.hidden = !text;
};

let track, policy, view, radio, telemetry, sound, flight, layout, input;
let simTime = 0, last = performance.now(), slow = false;
const prev = { p: new Float64Array(3), q: new Float64Array(4) };
const pose = { p: [0, 0, 0], q: [1, 0, 0, 0], v: [0, 0, 0], thrust: new Float64Array(4), hover: 2.147 };
const laps = { count: 0, last: null, best: null, start: null };
let crashText = '';

async function init() {
    say('Loading the policy and the airfield…');
    [track, policy] = await Promise.all([fetch('model/track.json').then(r => r.json()), loadPolicy('model/')]);
    view = new View($('view'));
    radio = new Transmitter($('radio'));
    telemetry = new Telemetry($('hud'), [['speed', 'Speed'], ['height', 'Height'], ['rpm', 'Motor RPM'],
        ['lap', 'Lap'], ['last', 'Last lap'], ['best', 'Best lap']]);
    sound = new MotorSound();
    pose.hover = track.plant.hover_per_rotor;

    const list = $('courses');
    for (const id of COURSES) {
        const c = track.courses.find(x => x.id === id);
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = c.name;
        b.dataset.course = id;
        b.addEventListener('click', () => setCourse(id));
        list.appendChild(b);
    }
    document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
    $('sound').addEventListener('click', toggleSound);
    $('slow').addEventListener('click', () => {
        slow = !slow;
        $('slow').setAttribute('aria-pressed', slow);
    });
    setupAbout();
    setupFullscreen();
    // Grabbing the view switches to the free camera from where it is; in chase, dragging
    // and zooming swing around the drone instead
    input = new CameraInput($('view'), {
        begin(kind) {
            if (view.mode === 'free' || (view.mode === 'chase' && (kind === 'orbit' || kind === 'zoom'))) return;
            setView('free', true);
        },
        orbit: (dx, dy) => view.orbit(dx, dy),
        pan: (dx, dy) => view.pan(dx, dy),
        zoom: f => view.zoom(f),
    });
    addEventListener('keydown', e => {
        if (e.target.closest?.('input, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
        if (e.key === 'v' || e.key === 'V') setView(VIEWS[(VIEWS.indexOf(view.mode) + 1) % VIEWS.length]);
        if ((e.key === 'f' || e.key === 'F') && document.fullscreenEnabled) toggleFullscreen();
    });
    setCourse(COURSES[0]);
    setView('pilot');
    await view.loadField('field/');
    say('');
    requestAnimationFrame(frame);
}

function setCourse(id) {
    const data = track.courses.find(c => c.id === id);
    flight = new Flight(track, data, { policy });
    layout = pilotSpot(flight);
    view.setCourse(flight, layout);
    Object.assign(laps, { count: 0, last: null, best: null, start: null });
    savePrev();
    document.querySelectorAll('[data-course]').forEach(b => b.setAttribute('aria-pressed', b.dataset.course === id));
    $('course-note').textContent = `${data.name}: ${flight.course.count} gates` +
        (data.open ? ', flown as a loop from the finish back to the first gate' : '');
}

// keepCamera: the free camera starts from the current view (when the viewer grabs it)
function setView(mode, keepCamera = false) {
    view.setMode(mode, keepCamera);
    document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', b.dataset.view === mode));
    $('camera-note').textContent = VIEW_NOTES[mode];
}

function toggleSound() {
    if (sound.running) sound.stop(); else sound.start();
    $('sound').setAttribute('aria-pressed', sound.running);
}

// "How it works" slides in over the left of the view; the race keeps running
function setupAbout() {
    const panel = $('about'), opener = $('about-open');
    const show = open => {
        panel.classList.toggle('open', open);
        panel.inert = !open;
        opener.setAttribute('aria-expanded', open);
        if (open) panel.querySelector('h2').focus({ preventScroll: true });
        else if (panel.contains(document.activeElement)) opener.focus();
    };
    panel.querySelector('h2').tabIndex = -1;
    opener.addEventListener('click', () => show(!panel.classList.contains('open')));
    $('about-close').addEventListener('click', () => show(false));
    addEventListener('keydown', e => { if (e.key === 'Escape' && panel.classList.contains('open')) show(false); });
}

function setupFullscreen() {
    const button = $('fullscreen');
    if (!document.fullscreenEnabled) return;
    button.hidden = false;
    button.addEventListener('click', toggleFullscreen);
    document.addEventListener('fullscreenchange', () => button.setAttribute('aria-pressed', !!document.fullscreenElement));
}

function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(() => {});
}

function savePrev() {
    prev.p.set(flight.plant.p);
    prev.q.set(flight.plant.q);
}

function handle(e) {
    if (e.type === 'handoff') laps.start = null;
    if (e.type === 'gate' && e.gate === 0 && laps.start === null) laps.start = e.time;
    if (e.type === 'lap') {
        laps.count++;
        laps.last = e.lapTime;
        laps.best = Math.min(laps.best ?? Infinity, e.lapTime);
        laps.start = e.time;
    }
    if (e.type === 'crash') {
        crashText = `Crashed: ${CRASH[e.reason] ?? 'crashed'}`;
        sound.crash();
    }
}

// Advance the simulation by `seconds` (also used by tests, since hidden tabs pause frames)
function advance(seconds) {
    simTime += seconds;
    let n = 0;
    while (simTime >= TICK && n++ < 20) {
        savePrev();
        flight.tick();
        for (const e of flight.events.splice(0)) handle(e);
        simTime -= TICK;
    }
    if (simTime > TICK) simTime = 0;   // fell behind (tab was hidden): drop the backlog
}

const fmt = s => s == null ? '–' : `${s.toFixed(2)} s`;
const PHASES = { pad: 'Disarmed', spool: 'Arming', climb: 'Taking off', stage: 'Flying to the start',
    launch: 'Launching', race: '', crashed: '' };

function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    update(dt);
    requestAnimationFrame(frame);
}

function update(dt) {
    advance(slow ? dt * 0.25 : dt);
    if (view.mode === 'free') view.fly(input.move, dt);
    const a = Math.min(simTime / TICK, 1), plant = flight.plant;
    for (let i = 0; i < 3; i++) pose.p[i] = prev.p[i] + (plant.p[i] - prev.p[i]) * a;
    slerp(prev.q, plant.q, a, pose.q);
    pose.v = plant.v;
    pose.thrust.set(plant.thrust);
    view.draw(pose, slow ? dt * 0.25 : dt, dt);

    // Radio: throttle shown linear in thrust, as on a real transmitter
    const s = flight.stick;
    radio.set(flight.cmd.thrust / plant.maxThrust, -s[3], s[2], s[1]);
    const racing = flight.phase === 'race';
    const clock = laps.start != null && racing ? flight.time - laps.start : 0;
    radio.screen(racing ? 'AI PILOT' : flight.phase === 'pad' || flight.phase === 'crashed' ? 'DISARMED' : 'AUTO',
        racing ? `${clock.toFixed(1)} s` : '');
    say(flight.phase === 'crashed' ? crashText : PHASES[flight.phase]);

    const speed = Math.hypot(...plant.v);
    telemetry.set('speed', `${(speed * 3.6).toFixed(0)} km/h`);
    telemetry.set('height', `${Math.max(plant.p[2] - 0.03, 0).toFixed(1)} m`);
    const rpm = plant.thrust.reduce((t, x) => t + Math.sqrt(Math.max(x, 0) / 1.682e-8), 0) / 4;
    telemetry.set('rpm', `${(Math.round(rpm / 100) * 100).toLocaleString('en-US')}`);
    telemetry.set('lap', racing && laps.start != null ? `${laps.count + 1} · ${clock.toFixed(1)} s` : '–');
    telemetry.set('last', fmt(laps.last));
    telemetry.set('best', fmt(laps.best));

    const cam = view.camera;
    const fwd = [0, 1, 2].map(i => -cam.matrixWorld.elements[8 + i]);
    const right = [0, 1, 2].map(i => cam.matrixWorld.elements[i]);
    sound.update(plant.thrust, { position: pose.p, velocity: plant.v },
        { position: cam.position.toArray(), forward: fwd, right }, view.mode === 'fpv');
}

function slerp(a, b, t, out) {
    let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    const s = d < 0 ? -1 : 1;
    d = Math.abs(d);
    let k0 = 1 - t, k1 = t;
    if (d < 0.9995) {
        const th = Math.acos(d), sn = Math.sin(th);
        k0 = Math.sin((1 - t) * th) / sn; k1 = Math.sin(t * th) / sn;
    }
    for (let i = 0; i < 4; i++) out[i] = k0 * a[i] + s * k1 * b[i];
    const l = Math.hypot(...out);
    for (let i = 0; i < 4; i++) out[i] /= l;
    return out;
}

// For testing in a background tab, where frames don't run: advance(seconds) then draw
window.droneRacing = {
    run(seconds, fps = 60) { for (let t = 0; t < seconds; t += 1 / fps) update(1 / fps); },
    get flight() { return flight; }, get view() { return view; }, setCourse, setView };
init().catch(err => { console.error(err); say(`Could not start: ${err.message}`); });
