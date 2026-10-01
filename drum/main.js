import { DrumView } from './render.js';
import { strike as playStrike, tone } from './audio.js';
import { EXAMPLES, GWW, fromStroke, problem, iconPath, resample } from './shapes.js';
import { toUnitArea, align, overlap } from './compare.js';

const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const view = new DrumView($('view'));
const twins = new DrumView($('twins'), { elevation: 64, spacing: 1.2 });

const state = {
  drum: null,          // current drum: outline, nodes, tri, lam, shapes
  example: 'star',
  K: +$('k').value,
  heard: null,         // last listener output
  shown: null,         // guess outline currently drawn (for morphing)
  picked: null,        // isolated mode
  drumId: 0,
  twins: null,
};
const setStatus = (t) => { $('status').textContent = t; };
const pitch = () => 65 * 2 ** ((+$('pitch').value / 100) * 2);
const mallet = () => +$('mallet').value / 100;

// --- worker ------------------------------------------------------------------

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'ready') {
    useOutline(EXAMPLES[state.example].points);
    worker.postMessage({ type: 'gww', outlines: GWW, n: 12 });
  } else if (m.type === 'drum' && m.id === state.drumId) {
    state.drum = m.drum;
    view.setDrums([m.drum]);
    // the sand starts forming a figure straight away (no sound until you ask)
    state.picked = Math.min(3, m.drum.lam.length - 1);
    view.showMode(0, state.picked);
    drawFigures();
    if (!state.introDone) intro();
    else { drawNotes(); hear(); }
    setStatus(`Found the drum’s lowest ${m.drum.lam.length} notes in ${Math.round(m.ms)} ms. The sand is gathering where note ${state.picked + 1} leaves the head still. Click the drum to strike it.`);
  } else if (m.type === 'drum' && String(m.id).startsWith('quiz-')) {
    quizDrum(m);
  } else if (m.type === 'heard') {
    if (m.id === 'twins') drawTwinGuess(m);
    else if (String(m.id).startsWith('quiz-')) quizHeard(m);
    else if (m.id === state.drumId && m.K === state.K) showGuess(m);
  } else if (m.type === 'gww') {
    state.twins = m.drums;
    twins.setDrums(m.drums);
    drawTwinTable(m.drums);
    drawTwinNotes(m.drums);
    worker.postMessage({ type: 'hear', id: 'twins', lam: m.drums[0].lam, K: 40 });
  } else if (m.type === 'error') {
    setStatus(`Couldn’t make that drum: ${m.message}`);
  }
};

function useOutline(points) {
  state.drumId++;
  state.shown = null;
  setStatus('Finding the drum’s notes…');
  worker.postMessage({ type: 'drum', id: state.drumId, outline: Array.from(points) });
}

// On the first drum, let the network hear the notes one at a time, so its
// drawing visibly sharpens.
function intro() {
  state.introDone = true;
  const final = state.K;
  let k = 3;
  const step = () => {
    state.K = k;
    $('k').value = k;
    $('k-out').textContent = k;
    drawNotes();
    hear();
    if (k++ < final) setTimeout(step, window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 260);
  };
  step();
}

function hear() {
  if (!state.drum) return;
  worker.postMessage({ type: 'hear', id: state.drumId, lam: state.drum.lam, K: state.K });
}

// --- the notes ---------------------------------------------------------------

function drawNotes() {
  const svg = $('notes'), lam = state.drum.lam;
  svg.replaceChildren();
  const ratios = Array.from(lam, (v) => Math.sqrt(v / lam[0]));
  const top = ratios[ratios.length - 1];
  const x = (r) => 8 + ((r - 1) / (top - 1)) * 304;
  ratios.forEach((r, k) => {
    const heard = k < state.K;
    const line = document.createElementNS(NS, 'line');
    line.setAttribute('x1', x(r)); line.setAttribute('x2', x(r));
    line.setAttribute('y1', 70); line.setAttribute('y2', 70 - (heard ? 58 : 34) * (1 - k / 90));
    line.setAttribute('class', `bar${heard ? '' : ' unheard'}${state.picked === k ? ' picked' : ''}`);
    const hit = document.createElementNS(NS, 'rect');
    hit.setAttribute('x', x(r) - 3); hit.setAttribute('y', 6); hit.setAttribute('width', 6); hit.setAttribute('height', 66);
    hit.setAttribute('class', 'hit');
    const title = document.createElementNS(NS, 'title');
    title.textContent = `Note ${k + 1}: ${r.toFixed(3)} × the lowest`;
    hit.append(title);
    hit.setAttribute('tabindex', '0');
    hit.setAttribute('role', 'button');
    hit.setAttribute('aria-label', `Play note ${k + 1} alone`);
    hit.addEventListener('click', () => pickMode(k));
    hit.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickMode(k); }
    });
    svg.append(line, hit);
  });
  for (const [r, label] of [[1, 'lowest'], [top, `${top.toFixed(1)}×`]]) {
    const t = document.createElementNS(NS, 'text');
    t.setAttribute('x', x(r)); t.setAttribute('y', 86);
    t.setAttribute('text-anchor', r === 1 ? 'start' : 'end');
    t.setAttribute('class', 'axis');
    t.textContent = label;
    svg.append(t);
  }
}

function pickMode(k) {
  if (state.picked === k) {
    state.picked = null;
    view.showMode(0, null);
  } else {
    state.picked = k;
    view.showMode(0, k);
    tone(Math.sqrt(state.drum.lam[k] / state.drum.lam[0]), { f0: pitch() });
    setStatus(`Note ${k + 1} on its own. The sand moves to where this note leaves the drumhead still.`);
  }
  drawNotes();
  drawFigures();
}

// Buttons under the drum for the first few notes' sand figures.
function drawFigures() {
  const box = $('figures');
  if (!box.children.length) {
    for (let k = 0; k < 10; k++) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'figure-note';
      b.textContent = k + 1;
      b.setAttribute('aria-label', `Note ${k + 1} alone, as a sand figure`);
      b.addEventListener('click', () => pickMode(k));
      box.append(b);
    }
  }
  [...box.children].forEach((b, k) => b.setAttribute('aria-pressed', String(state.picked === k)));
}

// --- the network's guess ------------------------------------------------------

const pathOf = (p) => {
  let d = '';
  for (let i = 0; i < p.length; i += 2) d += `${i ? 'L' : 'M'}${p[i].toFixed(4)} ${(-p[i + 1]).toFixed(4)}`;
  return d + 'Z';
};

function showGuess(m) {
  state.heard = m;
  const truth = toUnitArea(resample(state.drum.outline, 64));
  const ranked = m.outlines.map((o, i) => ({ o: toUnitArea(align(o, truth)), c: m.confidence[i] }))
    .sort((a, b) => b.c - a.c);
  const target = ranked[0].o;
  $('overlap').textContent = `${Math.round(overlap(truth, target) * 100)}%`;
  // morph from what is on screen now
  const from = state.shown && state.shown.length === target.length ? state.shown : target;
  const svg = $('guess');
  svg.replaceChildren();
  const t = document.createElementNS(NS, 'path');
  t.setAttribute('class', 'true');
  t.setAttribute('d', pathOf(truth));
  const g = document.createElementNS(NS, 'path');
  g.setAttribute('class', 'drawn');
  svg.append(t, g);
  const t0 = performance.now();
  const step = (now) => {
    const u = Math.min(1, (now - t0) / 350), e = 1 - (1 - u) ** 3;
    const cur = Float64Array.from(target, (v, i) => from[i] + (v - from[i]) * e);
    g.setAttribute('d', pathOf(cur));
    state.shown = cur;
    if (u < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
  // the other shapes it considered
  const box = $('candidates');
  box.replaceChildren();
  ranked.forEach((r, i) => {
    const fig = document.createElement('div');
    fig.className = `candidate${i === 0 ? ' top' : ''}`;
    fig.innerHTML = `<svg viewBox="-1.3 -1.3 2.6 2.6" aria-hidden="true"><path d="${pathOf(r.o)}"/></svg>${Math.round(r.c * 100)}%`;
    box.append(fig);
  });
}

// --- striking and drawing ----------------------------------------------------

function strikeAt(v, clientX, clientY, drums) {
  const hit = v.pick(clientX, clientY);
  if (!hit) return false;
  const w = v.weightsAt(hit.drum, hit.tri);
  playStrike(drums[hit.drum].lam, w, { f0: pitch(), mallet: mallet() });
  v.strike(hit.drum, w, hit.x, hit.y);
  return true;
}

let press = null;
const canvas = $('view');
canvas.addEventListener('pointerdown', (e) => {
  if (!e.isPrimary || e.button !== 0) return;
  press = { x: e.clientX, y: e.clientY, id: e.pointerId, drawing: false, pts: [] };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', (e) => {
  if (!press || e.pointerId !== press.id) return;
  if (!press.drawing && Math.hypot(e.clientX - press.x, e.clientY - press.y) > 8) {
    press.drawing = true;
    const p0 = view.pickPlane(press.x, press.y);
    if (p0) press.pts.push(p0);
  }
  if (press.drawing) {
    const p = view.pickPlane(e.clientX, e.clientY);
    if (p) press.pts.push(p);
    view.setStroke(press.pts);
  }
});
canvas.addEventListener('pointerup', (e) => {
  if (!press || e.pointerId !== press.id) return;
  const p = press;
  press = null;
  if (!p.drawing) {
    if (state.drum && strikeAt(view, e.clientX, e.clientY, [state.drum])) struck();
    return;
  }
  view.setStroke([]);
  if (p.pts.length < 12) return;
  const flat = closeLoop(p.pts.map(([x, , z]) => [x, -z])).flat();
  const outline = fromStroke(Float64Array.from(flat));
  const why = problem(outline);
  if (why) { setStatus(why); return; }
  state.example = null;
  markExample();
  useOutline(outline);
});

// A hand-drawn loop usually overshoots its start or stops short of it, and
// often begins with a little lead-in. Cut it where its end comes closest to
// its beginning, so the rim has no kink.
function closeLoop(pts) {
  const n = pts.length;
  let bi = 0, bj = n - 1, bd = Infinity;
  for (let i = 0; i < Math.floor(n * 0.3); i++) {
    for (let j = Math.floor(n * 0.7); j < n; j++) {
      const d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
      if (d < bd) { bd = d; bi = i; bj = j; }
    }
  }
  return pts.slice(bi, bj + 1);
}

// Keyboard: Enter or Space on the drum strikes it a little off centre.
canvas.addEventListener('keydown', (e) => {
  if ((e.key !== 'Enter' && e.key !== ' ') || !state.drum) return;
  e.preventDefault();
  const d = state.drum, node = strikeNode(d);
  const w = d.shapes.map((s) => s[node]);
  playStrike(d.lam, w, { f0: pitch(), mallet: mallet() });
  view.strike(0, w, d.nodes[2 * node], d.nodes[2 * node + 1]);
  struck();
});

function struck() {
  state.picked = null;
  drawNotes();
  drawFigures();
  setStatus('That was every note at once, and the sand jumps wherever the head moves. Pick a sand figure to see one note on its own.');
}

$('twins').addEventListener('click', (e) => {
  if (!state.twins) return;
  if (strikeAt(twins, e.clientX, e.clientY, state.twins)) {
    for (const o of $('twin-notes').children) o.setAttribute('aria-pressed', 'false');
  }
});

// --- can you hear it? ----------------------------------------------------------

const quiz = { shapes: null, round: 0, choices: null, drum: null, netPick: null, you: [0, 0], net: [0, 0], answered: true, revealed: 0 };

fetch(new URL('./model/quiz.json', import.meta.url)).then((r) => r.json()).then((q) => { quiz.shapes = q; });

function rotate(p, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return Float64Array.from(p, (v, i) => (i % 2 ? s * p[i - 1] + c * v : c * v - s * p[i + 1]));
}

function newRound() {
  if (!quiz.shapes) return;
  quiz.round++;
  quiz.answered = false;
  quiz.drum = null;
  quiz.netPick = null;
  const pool = quiz.shapes, picks = [];
  while (picks.length < 4) {
    const s = pool[Math.floor(Math.random() * pool.length)];
    if (!picks.some((p) => p.cat === s.cat)) picks.push(s);
  }
  quiz.answer = Math.floor(Math.random() * 4);
  quiz.choices = picks.map((s) => ({ ...s, shown: rotate(Float64Array.from(s.outline), Math.random() * 2 * Math.PI) }));
  const box = $('quiz-choices');
  box.replaceChildren();
  quiz.choices.forEach((ch, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'choice';
    b.setAttribute('aria-label', `Drum ${i + 1}`);
    b.innerHTML = `<svg viewBox="-1.25 -1.25 2.5 2.5" aria-hidden="true"><path d="${pathOf(toUnitArea(ch.shown))}"/></svg>`;
    b.addEventListener('click', () => answer(i));
    box.append(b);
  });
  $('quiz-result').textContent = 'Listening…';
  $('quiz-next').hidden = true;
  worker.postMessage({ type: 'drum', id: `quiz-${quiz.round}`, outline: Array.from(quiz.choices[quiz.answer].outline) });
}

// A point a little off the middle, where the lowest note moves at about 60%
// of its most: most notes speak there.
function strikeNode(d) {
  const sh = d.shapes[0];
  let best = d.nRim;
  for (let i = d.nRim; i < sh.length; i++) if (Math.abs(sh[i]) > Math.abs(sh[best])) best = i;
  const target = 0.6 * Math.abs(sh[best]);
  let node = best;
  for (let i = d.nRim; i < sh.length; i++) {
    if (Math.abs(Math.abs(sh[i]) - target) < Math.abs(Math.abs(sh[node]) - target)) node = i;
  }
  return node;
}

function playQuiz() {
  const d = quiz.drum;
  if (!d) return;
  const node = strikeNode(d);
  playStrike(d.lam, d.shapes.map((s) => s[node]), { f0: pitch(), mallet: mallet() });
}

function quizDrum(m) {
  if (m.id !== `quiz-${quiz.round}`) return;
  quiz.drum = m.drum;
  playQuiz();
  $('quiz-result').textContent = 'Which drum was that? Play it again if you like.';
  $('quiz-play').textContent = 'Play it again';
  worker.postMessage({ type: 'hear', id: `quiz-${quiz.round}`, lam: m.drum.lam, K: 40 });
}

function quizHeard(m) {
  if (m.id !== `quiz-${quiz.round}`) return;
  const i = m.confidence.indexOf(Math.max(...m.confidence));
  const scores = quiz.choices.map((ch) => {
    const truth = toUnitArea(resample(Float64Array.from(ch.outline), 64));
    return overlap(truth, toUnitArea(align(m.outlines[i], truth)));
  });
  quiz.netPick = scores.indexOf(Math.max(...scores));
  if (quiz.answered) reveal(quiz.lastPick);
}

function answer(i) {
  if (quiz.answered || !quiz.drum) return;
  quiz.answered = true;
  quiz.lastPick = i;
  quiz.you[1]++;
  if (i === quiz.answer) quiz.you[0]++;
  reveal(i);
}

function reveal(i) {
  // the network's answer may still be on its way; quizHeard calls back
  if (quiz.netPick === null) { $('quiz-result').textContent = 'Waiting for the network…'; return; }
  if (quiz.revealed === quiz.round) return;
  quiz.revealed = quiz.round;
  quiz.net[1]++;
  if (quiz.netPick === quiz.answer) quiz.net[0]++;
  [...$('quiz-choices').children].forEach((b, k) => {
    b.disabled = true;
    b.classList.add(k === quiz.answer ? 'right' : 'wrong');
    if (k === i) b.insertAdjacentHTML('beforeend', '<span class="tag">Your pick</span>');
    if (k === quiz.netPick) b.insertAdjacentHTML('beforeend', '<span class="tag">Network’s pick</span>');
  });
  const you = i === quiz.answer, net = quiz.netPick === quiz.answer;
  $('quiz-result').textContent = you && net ? 'You both got it.'
    : you ? 'You got it; the network didn’t.'
      : net ? 'The network got this one; you didn’t.'
        : 'Neither of you got this one.';
  $('score-you').textContent = `${quiz.you[0]} of ${quiz.you[1]}`;
  $('score-net').textContent = `${quiz.net[0]} of ${quiz.net[1]}`;
  $('quiz-next').hidden = false;
}

$('quiz-play').addEventListener('click', () => {
  if (!quiz.drum || quiz.choices === null) newRound();
  else playQuiz();
});
$('quiz-next').addEventListener('click', newRound);

// --- the isospectral pair ------------------------------------------------------

function drawTwinTable(drums) {
  const body = $('twins-table').querySelector('tbody');
  body.replaceChildren();
  const [a, b] = drums.map((d) => d.lam);
  let worst = 0;
  for (let k = 0; k < a.length; k++) worst = Math.max(worst, Math.abs(a[k] / b[k] - 1));
  for (let k = 0; k < 10; k++) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${k + 1}</td><td>${Math.sqrt(a[k] / a[0]).toFixed(6)}</td><td>${Math.sqrt(b[k] / b[0]).toFixed(6)}</td>`;
    body.append(tr);
  }
  const digits = Math.max(1, Math.floor(-Math.log10(worst + 1e-16)));
  $('twins-status').textContent = `This page’s solver gives both drums the same ${a.length} lowest notes, ` +
    `agreeing to ${digits} decimal places. Click either drum to hear it.`;
}

// One note on both drums at once: same pitch, different sand patterns.
function drawTwinNotes(drums) {
  const box = $('twin-notes');
  box.replaceChildren();
  for (let k = 0; k < 8; k++) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'twin-note';
    b.textContent = k + 1;
    b.setAttribute('aria-label', `Note ${k + 1}`);
    b.addEventListener('click', () => {
      const on = b.getAttribute('aria-pressed') !== 'true';
      for (const o of box.children) o.setAttribute('aria-pressed', String(on && o === b));
      drums.forEach((_, i) => twins.showMode(i, on ? k : null));
      if (on) tone(Math.sqrt(drums[0].lam[k] / drums[0].lam[0]), { f0: pitch() });
    });
    box.append(b);
  }
}

function drawTwinGuess(m) {
  const i = m.confidence.indexOf(Math.max(...m.confidence));
  const g = toUnitArea(m.outlines[i]);
  $('twins-guess').innerHTML = `<path d="${pathOf(g)}"/>`;
}

// --- controls --------------------------------------------------------------

function markExample() {
  for (const b of $('examples').children) b.setAttribute('aria-pressed', String(b.dataset.name === state.example));
}
for (const [name, ex] of Object.entries(EXAMPLES)) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'example';
  b.dataset.name = name;
  b.title = ex.label;
  b.setAttribute('aria-label', ex.label);
  b.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${iconPath(ex.points)}"/></svg>`;
  b.addEventListener('click', () => { state.example = name; markExample(); useOutline(ex.points); });
  $('examples').append(b);
}
markExample();

$('k').addEventListener('input', () => {
  state.K = +$('k').value;
  $('k-out').textContent = state.K;
  if (state.drum) { drawNotes(); hear(); }
});

worker.postMessage({ type: 'init', model: new URL('./model/listener.json', import.meta.url).href });

function loop() {
  view.frame();
  twins.frame();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

window.drum = { state, view, twins };
