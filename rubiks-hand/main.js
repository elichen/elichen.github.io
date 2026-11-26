// Interactive Shadow Hand Rubik's Cube Solver — MuJoCo WASM + Three.js + Pure-JS PPO + Kociemba.
import * as THREE from 'three';
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.js';
import { RubikHandEnv, CubeSolverController, SUBSTEPS, makeRandom } from './rubik-env.js';
import { FACE_INFO, QPOS_CENTERS, SOLVED_FACELETS, wrapPi } from './cube-state.js';
import { Policy } from './policy.js';
import { RubikSceneRenderer, STICKER_HEX } from './render.js';

const MODEL_FILES = 'https://huggingface.co/elichen77/personal-site-models/resolve/9dc168fec06a922fbc17fc97638cd252c0cac9b0/rubiks-hand/';

const $ = id => document.getElementById(id);
const statusEl = $('status');
const random = makeRandom();

const CSS_COLORS = {
    U: '#f7f7f3', R: '#d92b2b', F: '#169c4b',
    D: '#f2c618', L: '#f06a14', B: '#1d5fd6', '?': '#888',
};

async function loadRobotFiles() {
    const { files: list } = await fetch('robot/manifest.json').then(r => r.json());
    const files = {};
    await Promise.all(list.map(async f => {
        const r = await fetch('robot/' + f);
        files[f] = f.endsWith('.xml') ? await r.text() : new Uint8Array(await r.arrayBuffer());
    }));
    return files;
}

async function loadPolicy(dir) {
    const [meta, bin] = await Promise.all([
        fetch(`${dir}/policy.json`).then(r => r.json()),
        fetch(`${dir}/policy.bin`).then(r => r.arrayBuffer()),
    ]);
    return new Policy(meta, new Float32Array(bin));
}

let env, reorientPol, twistPol, ctrl, view;
let substep = 0, simAccum = 0, lastFrame = 0;
let speed = 1, paused = false;
let solveElapsed = 0, solvedCubes = 0, autoNextTimer = null;
let drag = null;

// Draw live 2D unfolded cube net (U / L F R B / D)
function drawNet(facelets, activeFaceLetter) {
    const canvas = $('cube-net');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // 4x3 faces, each 3x3 stickers
    const cell = 24, gap = 3, faceGap = 8;
    const faceSize = 3 * cell + 2 * gap;
    const offX = Math.floor((w - (4 * faceSize + 3 * faceGap)) / 2);
    const offY = Math.floor((h - (3 * faceSize + 2 * faceGap)) / 2);

    // Order in facelet string: U(0), R(1), F(2), D(3), L(4), B(5)
    const placements = [
        { idx: 0, letter: 'U', gx: 1, gy: 0 },
        { idx: 4, letter: 'L', gx: 0, gy: 1 },
        { idx: 2, letter: 'F', gx: 1, gy: 1 },
        { idx: 1, letter: 'R', gx: 2, gy: 1 },
        { idx: 5, letter: 'B', gx: 3, gy: 1 },
        { idx: 3, letter: 'D', gx: 1, gy: 2 },
    ];

    for (const p of placements) {
        const fx = offX + p.gx * (faceSize + faceGap);
        const fy = offY + p.gy * (faceSize + faceGap);
        ctx.fillStyle = 'rgba(28, 25, 21, 0.08)';
        ctx.beginPath();
        ctx.roundRect(fx - 2, fy - 2, faceSize + 4, faceSize + 4, 5);
        ctx.fill();
        if (p.letter === activeFaceLetter) {
            ctx.strokeStyle = '#1c1915';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.roundRect(fx - 4, fy - 4, faceSize + 8, faceSize + 8, 6);
            ctx.stroke();
        }
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                const ch = facelets[p.idx * 9 + r * 3 + c] || '?';
                const x = fx + c * (cell + gap);
                const y = fy + r * (cell + gap);
                ctx.fillStyle = CSS_COLORS[ch] || '#888';
                ctx.beginPath();
                ctx.roundRect(x, y, cell, cell, 4);
                ctx.fill();
                ctx.strokeStyle = 'rgba(28, 25, 21, 0.34)';
                ctx.lineWidth = 1.2;
                ctx.stroke();
            }
        }
    }

    const solvedEl = $('net-solved');
    const isSolved = facelets === SOLVED_FACELETS;
    solvedEl.textContent = isSolved ? 'SOLVED' : 'SCRAMBLED';
    solvedEl.classList.toggle('ok', isSolved);
}

function renderPlanStrip() {
    const strip = $('plan-strip');
    strip.innerHTML = '<span class="plan-label">Kociemba</span>';
    const done = ctrl.completedMoves;
    const todo = ctrl.plan;
    if (done.length === 0 && todo.length === 0) {
        const chip = document.createElement('span');
        chip.className = 'mv-chip empty';
        chip.textContent = 'Cube Solved';
        strip.appendChild(chip);
        return;
    }
    for (const mv of done) {
        const chip = document.createElement('span');
        chip.className = 'mv-chip done';
        chip.textContent = mv;
        strip.appendChild(chip);
    }
    todo.forEach((mv, i) => {
        const chip = document.createElement('span');
        chip.className = 'mv-chip' + (i === 0 ? ' active' : '');
        chip.textContent = mv;
        strip.appendChild(chip);
    });
}

function updateHUD() {
    const info = FACE_INFO[ctrl.targetFaceId];
    const swatch = $('phase-swatch');
    const label = $('phase-label');
    const angleEl = $('phase-angle');

    if (ctrl.phase === 'solved') {
        swatch.style.background = '#1b8a4b';
        label.textContent = 'SOLVED';
        angleEl.textContent = '0.0°';
    } else if (ctrl.phase === 'reorient') {
        swatch.style.background = CSS_COLORS[info.letter];
        label.textContent = `REORIENT · ${info.letter} → TOP`;
        const q = env.cubeQuat();
        const a = info.axis;
        // dotZ from world rotated axis
        const nw = [
            (1 - 2*(q[2]*q[2]+q[3]*q[3]))*a[0] + 2*(q[1]*q[2]-q[0]*q[3])*a[1] + 2*(q[1]*q[3]+q[0]*q[2])*a[2],
            2*(q[1]*q[2]+q[0]*q[3])*a[0] + (1 - 2*(q[1]*q[1]+q[3]*q[3]))*a[1] + 2*(q[2]*q[3]-q[0]*q[1])*a[2],
            2*(q[1]*q[3]-q[0]*q[2])*a[0] + 2*(q[2]*q[3]+q[0]*q[1])*a[1] + (1 - 2*(q[1]*q[1]+q[2]*q[2]))*a[2],
        ];
        const deg = Math.acos(Math.max(-1, Math.min(1, nw[2]))) * 180 / Math.PI;
        angleEl.textContent = `${deg.toFixed(0)}°`;
    } else if (ctrl.phase === 'twist') {
        swatch.style.background = CSS_COLORS[info.letter];
        const mv = ctrl.plan[0] || info.letter;
        label.textContent = `TWIST · ${mv} (${ctrl.twistDir > 0 ? '+90°' : '−90°'})`;
        const cur = env.data.qpos[QPOS_CENTERS + ctrl.targetFaceId];
        const remDeg = Math.abs(wrapPi(ctrl.goalAngle - cur)) * 180 / Math.PI;
        angleEl.textContent = `${remDeg.toFixed(0)}° left`;
    } else {
        label.textContent = 'IDLE';
        angleEl.textContent = '–';
    }

    const total = ctrl.completedMoves.length + ctrl.plan.length;
    $('m-moves').textContent = `${ctrl.completedMoves.length} / ${total}`;
    $('m-time').textContent = `${solveElapsed.toFixed(1)}s`;
    $('m-solved').textContent = `${solvedCubes}`;

    drawNet(env.facelets(), ctrl.phase === 'solved' ? null : info.letter);
}

let toastTimer;
function toast(msg) {
    const el = $('toast');
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 2600);
}

function startScramble(nMoves) {
    clearTimeout(autoNextTimer);
    autoNextTimer = null;
    const moves = env.scramble(nMoves);
    ctrl.resetPlan();
    solveElapsed = 0;
    document.querySelectorAll('.scr-btn').forEach(b => b.classList.toggle('active', Number(b.dataset.scr) === nMoves));
    renderPlanStrip();
    updateHUD();
    toast(`Scrambled (${nMoves} moves) → Kociemba plan: ${ctrl.plan.length} quarter turns`);
}

function queueManualMove(mv) {
    clearTimeout(autoNextTimer);
    autoNextTimer = null;
    if (ctrl.phase === 'solved' || ctrl.phase === 'idle') {
        ctrl.completedMoves = [];
        ctrl.plan = [mv];
        ctrl._beginNextMove();
    } else {
        // Insert right after the currently executing move so it runs immediately next
        ctrl.plan.splice(1, 0, mv);
    }
    renderPlanStrip();
    updateHUD();
    toast(`Queued move ${mv}`);
}

function advance(seconds) {
    const sub = env.model.opt.timestep;
    simAccum = Math.min(simAccum + seconds, 60 * sub);
    let stepsDone = 0;
    while (simAccum >= sub && stepsDone < 60) {
        simAccum -= sub;
        stepsDone++;
        if (substep % SUBSTEPS === 0) {
            env.act(ctrl.action(reorientPol, twistPol));
        }
        env.substep();
        substep++;
        if (substep % SUBSTEPS === 0) {
            if (!drag && ctrl.phase !== 'solved') solveElapsed += env.dt;
            if (!drag) {
                const ev = ctrl.update();
                if (ev) {
                    if (ev.event === 'solved') {
                        solvedCubes++;
                        toast(`Solved in ${ctrl.completedMoves.length} turns (${solveElapsed.toFixed(1)}s)!`);
                        renderPlanStrip();
                        autoNextTimer = setTimeout(() => {
                            if (!paused) startScramble(8);
                        }, 3200);
                    } else if (ev.event === 'dropped') {
                        toast('Dropped cube — resetting with fresh scramble');
                        startScramble(8);
                    } else if (ev.event === 'twisted' || ev.event === 'replan') {
                        renderPlanStrip();
                    }
                }
            }
        }
    }
}

// Mouse grab / poke interaction on the physical cube
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function rayAt(e) {
    const r = view.canvas.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(pointer, view.camera);
    return raycaster.ray;
}

function setupPointer() {
    const canvas = view.canvas;
    canvas.addEventListener('wheel', e => { if (!e.ctrlKey) e.stopImmediatePropagation(); }, { capture: true });

    canvas.addEventListener('pointerdown', e => {
        const ray = rayAt(e);
        const hits = raycaster.intersectObjects(view.cubeHitMeshes, false);
        if (hits.length === 0) return;
        e.stopImmediatePropagation();
        try { canvas.setPointerCapture(e.pointerId); } catch {}
        const pos = new THREE.Vector3(...env.cubePos());
        const camDir = new THREE.Vector3();
        view.camera.getWorldDirection(camDir);
        const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camDir, pos);
        drag = { plane, goal: pos.clone() };
        env.pull = pullCube;
        canvas.classList.add('grabbing');
    }, { capture: true });

    canvas.addEventListener('pointermove', e => {
        const ray = rayAt(e);
        if (drag) {
            const p = new THREE.Vector3();
            if (ray.intersectPlane(drag.plane, p)) {
                drag.goal.set(
                    Math.max(0.22, Math.min(0.42, p.x)),
                    Math.max(-0.10, Math.min(0.10, p.y)),
                    Math.max(0.02, Math.min(0.14, p.z)),
                );
            }
        } else {
            const over = raycaster.intersectObjects(view.cubeHitMeshes, false).length > 0;
            canvas.classList.toggle('can-grab', over);
        }
    });

    const release = () => {
        if (!drag) return;
        env.pull = null;
        for (let i = 0; i < 6; i++) env.data.xfrc_applied[6 * env.cubeBody + i] = 0;
        drag = null;
        canvas.classList.remove('grabbing');
        // Dynamically re-plan from live physical cube state after user disturbance
        ctrl.replan();
        renderPlanStrip();
    };
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', release);
}

function pullCube() {
    if (!drag) return;
    const p = env.cubePos();
    const v = env.data.qvel;
    const off = 6 * env.cubeBody;
    const kp = 45.0, kd = 2.2;
    env.data.xfrc_applied[off + 0] = kp * (drag.goal.x - p[0]) - kd * v[24];
    env.data.xfrc_applied[off + 1] = kp * (drag.goal.y - p[1]) - kd * v[25];
    env.data.xfrc_applied[off + 2] = kp * (drag.goal.z - p[2]) - kd * v[26] + 0.68;
}

function wireUI() {
    $('btn-scr-5').addEventListener('click', () => startScramble(5));
    $('btn-scr-12').addEventListener('click', () => startScramble(12));
    $('btn-scr-20').addEventListener('click', () => startScramble(20));

    $('btn-toggle-manual').addEventListener('click', () => {
        const drawer = $('manual-drawer');
        drawer.hidden = !drawer.hidden;
        $('btn-toggle-manual').classList.toggle('active', !drawer.hidden);
    });

    document.querySelectorAll('.mv-btn').forEach(btn => {
        btn.addEventListener('click', () => queueManualMove(btn.dataset.mv));
    });

    $('btn-pause').addEventListener('click', () => {
        paused = !paused;
        $('btn-pause').textContent = paused ? 'Resume' : 'Pause';
    });

    document.querySelectorAll('.speed-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            speed = Number(btn.dataset.speed);
            document.querySelectorAll('.speed-btn').forEach(b => b.classList.toggle('active', b === btn));
        });
    });

    document.querySelectorAll('.cam-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            view.setPreset(btn.dataset.cam);
            document.querySelectorAll('.cam-btn').forEach(b => b.classList.toggle('active', b === btn));
        });
    });
}

let hudTick = 0;
function frame(now) {
    const dt = Math.min(0.05, (now - (lastFrame || now)) / 1000);
    lastFrame = now;
    if (!paused) advance(dt * speed);
    view.draw(dt, ctrl);
    if (++hudTick % 3 === 0) updateHUD();
    requestAnimationFrame(frame);
}

async function init() {
    try {
        const [mujoco, files, rPol, tPol] = await Promise.all([
            loadMujoco(),
            loadRobotFiles(),
            loadPolicy(MODEL_FILES + 'model/reorient'),
            loadPolicy(MODEL_FILES + 'model/twist'),
        ]);
        env = new RubikHandEnv(mujoco, files, random);
        reorientPol = rPol;
        twistPol = tPol;
        ctrl = new CubeSolverController(env);
        view = new RubikSceneRenderer($('view'), env);

        // Load verified MuJoCo WASM benchmark metrics
        fetch('model/metrics.json').then(r => r.ok ? r.json() : null).then(m => {
            if (!m) return;
            for (const key of ['reorient', 'twist', 'solve5', 'solve12', 'solve20']) {
                if (m[key]) {
                    const rEl = $(`stat-${key}`), tEl = $(`stat-${key}-t`);
                    if (rEl) rEl.textContent = m[key].rate;
                    if (tEl) tEl.textContent = m[key].time;
                }
            }
        }).catch(() => {});

        wireUI();
        setupPointer();
        startScramble(8);
        statusEl.hidden = true;
        window.rubikDemo = { env, ctrl, view, advance, startScramble, queueManualMove };
        requestAnimationFrame(frame);
    } catch (e) {
        statusEl.textContent = `Could not initialize simulation: ${e.message}`;
        throw e;
    }
}

init();
