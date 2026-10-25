// Two Hours of Atari: BBF snapshots play on the wasm Arcade Learning Environment.
import { Game } from './game.js';
import { Policy } from './policy.js';

const tf = window.tf;

// ALE's sound goes through SDL's Emscripten audio driver, which asks for a 64-sample ScriptProcessor buffer. Browsers
// need at least 256, so make a larger node and run SDL's callback on 64-sample slices of it. About 3% of slices come
// back as garbage (values like 1e34); those repeat the last good slice. The TIA's levels are low (peaks ~0.03).
const SOUND_GAIN = 6;
const createScriptProcessor = AudioContext.prototype.createScriptProcessor;
AudioContext.prototype.createScriptProcessor = function (size, inputs, outputs) {
  if (size >= 256) return createScriptProcessor.call(this, size, inputs, outputs);
  const node = createScriptProcessor.call(this, 256, inputs, outputs);
  const last = [];
  Object.defineProperty(node, 'onaudioprocess', {
    set(fn) {
      node.addEventListener('audioprocess', e => {
        const out = e.outputBuffer, ch = Array.from({ length: out.numberOfChannels }, (_, c) => out.getChannelData(c));
        for (let j = 0; j < 256 / size; j++) {
          const slices = ch.map(d => d.subarray(j * size, (j + 1) * size));
          fn({ outputBuffer: { numberOfChannels: ch.length, getChannelData: c => slices[c] } });
          slices.forEach((d, c) => {
            if (d.every(v => Math.abs(v) <= 1)) { for (let i = 0; i < size; i++) d[i] *= SOUND_GAIN; last[c] = d.slice(); }
            else d.set(last[c] ?? new Float32Array(size));
          });
        }
      });
    },
  });
  return node;
};
const ACTION_NAMES = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
  'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'];
const ARROWS = { UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→', UPRIGHT: '↗', UPLEFT: '↖', DOWNRIGHT: '↘', DOWNLEFT: '↙' };
// Weights live on Hugging Face, pinned to a commit (model/ is the local copy, not on Pages).
const MODEL = 'https://huggingface.co/eli-aifund/bbf-atari100k-tfjs/resolve/6bfc00ab3fda2a999e9a6794f7955e1814b00595/';
const MOVES_PER_MINUTE = 60 * 60 / 4;  // 4 frames per move at 60 frames a second

const $ = id => document.getElementById(id);
const app = { manifest: null, game: null, policy: null, entry: null, frames: [], busy: false, token: 0, scores: [] };
window.atari = app;

function moveLabel(id) {
  const name = ACTION_NAMES[id];
  if (name === 'NOOP') return 'wait';
  const fire = name.endsWith('FIRE') && name !== 'FIRE';
  const dir = fire ? name.slice(0, -4) : name;
  if (name === 'FIRE') return 'fire';
  return (ARROWS[dir] || dir) + (fire ? ' + fire' : '');
}

function practiceText(step) {
  const min = Math.round(step / MOVES_PER_MINUTE);
  const t = min >= 60 ? `${Math.floor(min / 60)} h ${String(min % 60).padStart(2, '0')} min` : `${min} min`;
  return `${t} · ${step.toLocaleString()} moves`;
}

async function selectBackend() {
  for (const name of ['webgpu', 'webgl', 'cpu']) {
    try {
      if (await tf.setBackend(name)) { await tf.ready(); return name; }
    } catch (e) { console.warn(`Backend ${name} unavailable:`, e); }
  }
  throw new Error('No TensorFlow.js backend is available');
}

function setLoading(text, frac) {
  $('loading').hidden = text == null;
  if (text != null) $('loading-text').textContent = text;
  $('progress').style.width = `${Math.round((frac ?? 0) * 100)}%`;
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = +res.headers.get('content-length') || 0;
  if (!res.body || !total) return res.arrayBuffer();
  const reader = res.body.getReader(), out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got); got += value.length;
    onProgress(got / total);
  }
  return out.buffer;
}

// ---------- Tabs and practice slider ----------

function buildTabs() {
  const tabs = $('tabs');
  tabs.innerHTML = '';
  for (const g of app.manifest.games) {
    const b = document.createElement('button');
    b.role = 'tab';
    b.innerHTML = `<span class="cap"></span><span class="name">${g.title}${g.snapshots.length ? '' : '<small>training</small>'}</span>`;
    b.disabled = !g.snapshots.length;
    b.setAttribute('aria-selected', 'false');
    b.onclick = () => selectGame(g);
    g.tab = b;
    tabs.append(b);
  }
}

async function selectGame(entry, step) {
  for (const g of app.manifest.games) g.tab.setAttribute('aria-selected', String(g === entry));
  app.entry = entry;
  const slider = $('practice');
  slider.max = entry.snapshots.length - 1;
  // opens on the snapshot with the best average score (the last one if none are scored yet)
  const best = entry.snapshots.reduce((b, s, i) => (s.eval ?? -Infinity) >= (entry.snapshots[b].eval ?? -Infinity) ? i : b, entry.snapshots.length - 1);
  const i = step == null ? best : entry.snapshots.findIndex(s => s.step === step);
  slider.value = Math.max(0, i);
  await loadSnapshot();
}

async function loadSnapshot() {
  const entry = app.entry, snap = entry.snapshots[+$('practice').value];
  $('practice-out').textContent = practiceText(snap.step);
  drawCurve();
  $('practice-note').textContent = snap.eval != null
    ? `The average score of this network is ${Math.round(snap.eval).toLocaleString()} points in ${snap.evalGames} games.`
    : '';
  const token = ++app.token;
  app.frames = [];
  setLoading(`Loading the network after ${practiceText(snap.step)}…`, 0);
  const dir = `${MODEL}${entry.id}/${String(snap.step).padStart(6, '0')}`;
  const [meta, buf] = await Promise.all([
    fetch(`${dir}/policy.json`).then(r => r.json()),
    fetchWithProgress(`${dir}/policy.bin`, f => token === app.token && setLoading(`Loading the network after ${practiceText(snap.step)}…`, f)),
  ]);
  if (token !== app.token) return;
  app.policy?.dispose();
  app.policy = new Policy(tf, meta, buf);
  if (!app.game || app.game.rom !== entry.id) {
    // SDL opens audio only once per wasm module, so each game on screen gets a module of its own (~6 ms)
    app.game?.dispose();
    app.game = null;
    app.screenALE?.SDL2?.audioContext?.close();
    app.screenALE = await newALE();
    if (token !== app.token) return;
    app.game = new Game(app.screenALE, `/roms/${entry.id}.bin`, { seed: 1, sound: true });
    applySound();
    app.game.rom = entry.id;
    app.game.onFrame = () => app.frames.push(app.game.ale.getScreenRGB().slice());
    buildQBars();
  } else {
    app.game.reset();
  }
  app.scores = [];
  updateScores();
  setLoading(null);
}

// ---------- Learning curve ----------

function drawCurve() {
  const e = app.entry, svg = $('curve'), sel = +$('practice').value;
  const pts = [{ step: 0, eval: e.random }, ...e.snapshots].filter(s => s.eval != null);
  const W = 520, H = 110, L = 40, B = 18, T = 6;
  const xmax = 100000, ymax = Math.max(e.human, ...pts.map(p => p.eval)) * 1.1;
  const x = s => L + (W - L - 4) * s / xmax, y = v => T + (H - B - T) * (1 - v / ymax);
  const k = v => v >= 1000 ? `${Math.round(v / 1000)}k` : `${v}`;
  let h = `<line class="axis" x1="${L}" y1="${H - B}" x2="${W - 4}" y2="${H - B}"/>`;
  for (const m of [0, 30, 60, 90]) h += `<text x="${x(m * MOVES_PER_MINUTE)}" y="${H - 4}" text-anchor="middle">${m ? `${m} min` : '0'}</text>`;
  h += `<line class="human" x1="${L}" x2="${W - 4}" y1="${y(e.human)}" y2="${y(e.human)}"/>`;
  h += `<text x="${W - 4}" y="${y(e.human) - 4}" text-anchor="end">human tester, ${Math.round(e.human).toLocaleString()}</text>`;
  for (const v of [0, ymax / 2]) h += `<text x="${L - 6}" y="${y(v) + 3}" text-anchor="end">${k(Math.round(v / 1000) * 1000)}</text>`;
  h += `<polyline class="line" points="${pts.map(p => `${x(p.step)},${y(p.eval)}`).join(' ')}"/>`;
  e.snapshots.forEach((s, i) => {
    if (s.eval == null) return;
    h += `<circle class="pt${i === sel ? ' on' : ''}" data-i="${i}" cx="${x(s.step)}" cy="${y(s.eval)}" r="${i === sel ? 5 : 3.5}"><title>${practiceText(s.step)}: ${Math.round(s.eval).toLocaleString()}</title></circle>`;
  });
  svg.innerHTML = h;
  svg.querySelectorAll('.pt').forEach(c => c.onclick = () => { $('practice').value = c.dataset.i; loadSnapshot(); });
}

// ---------- Readouts ----------

function buildQBars() {
  const box = $('qbars');
  box.innerHTML = '';
  app.qrows = app.game.actions.map(a => {
    const name = document.createElement('span'), bar = document.createElement('span'), fill = document.createElement('span');
    name.className = 'name'; name.textContent = moveLabel(a);
    bar.className = 'bar'; bar.append(fill);
    box.append(name, bar);
    return { name, bar, fill };
  });
  const frames = $('frames');
  if (!frames.children.length) {
    app.frameCanvases = [0, 1, 2, 3].map(() => {
      const c = document.createElement('canvas'); c.width = c.height = 84; frames.append(c); return c;
    });
  }
}

function drawQ(q, chosen) {
  let lo = Infinity, hi = -Infinity;
  for (const v of q) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const span = Math.max(hi - lo, 1e-6);
  app.qrows.forEach((r, a) => {
    r.fill.style.width = `${8 + 92 * (q[a] - lo) / span}%`;
    r.name.classList.toggle('chosen', a === chosen);
    r.bar.classList.toggle('chosen', a === chosen);
  });
}

function drawFrames() {
  const s = app.game.state;
  app.frameCanvases.forEach((c, k) => {
    const ctx = c.getContext('2d'), img = ctx.createImageData(84, 84);
    for (let i = 0; i < 84 * 84; i++) {
      const v = s[i * 4 + k];
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  });
}

function updateScores() {
  $('score').textContent = Math.round(app.game?.score ?? 0).toLocaleString();
  const n = app.scores.length;
  const mean = n ? app.scores.reduce((a, b) => a + b, 0) / n : null;
  $('mean').textContent = mean == null ? '–' : Math.round(mean).toLocaleString();
  $('mean-label').textContent = n ? `Average of ${n} game${n > 1 ? 's' : ''}` : 'Average';
}

// ---------- Loop: one move per 4 displayed frames, at 60 frames a second ----------

async function decide() {
  app.busy = true;
  const game = app.game, policy = app.policy, token = app.token;
  const qt = policy.q(game.state);
  const q = await qt.data();
  qt.dispose();
  if (token === app.token && game === app.game) {
    const a = game.chooseAction(q);
    drawQ(q, a);
    const { ended } = game.step(a);
    if (ended) app.scores.push(game.lastScore);
    drawFrames();
    updateScores();
  }
  app.busy = false;
}

let last = 0, debt = 0;
function frame(t) {
  requestAnimationFrame(frame);
  if (!app.policy || !app.game || !$('loading').hidden) { last = t; return; }
  debt = Math.min(debt + (t - last) / (1000 / 60), 8);
  last = t;
  const ctx = app.ctx;
  while (debt >= 1 && app.frames.length) {
    const rgb = app.frames.shift();
    const img = app.img;
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
      img.data[j] = rgb[i]; img.data[j + 1] = rgb[i + 1]; img.data[j + 2] = rgb[i + 2]; img.data[j + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    debt -= 1;
  }
  if (app.frames.length < 4 && !app.busy) decide();
}

const newALE = () => window.createALEModule({ locateFile: f => `vendor/${f}`, printErr: () => {} });

async function main() {
  setLoading('Loading the emulator…', 0);
  const [backend, manifest, ALE] = await Promise.all([
    selectBackend(),
    fetch('data/manifest.json').then(r => r.json()),
    newALE(),
  ]);
  app.backend = backend;
  app.manifest = manifest;
  app.ALE = ALE;
  app.ctx = $('screen').getContext('2d');
  app.img = app.ctx.createImageData(160, 210);
  buildTabs();
  renderResults();
  buildMenu();
  buildSound();
  $('practice').addEventListener('input', () => {
    $('practice-out').textContent = practiceText(app.entry.snapshots[+$('practice').value].step);
  });
  $('practice').addEventListener('change', loadSnapshot);
  const first = manifest.games.find(g => g.snapshots.length);
  await selectGame(first);
  requestAnimationFrame(frame);
}

// Plays n games without drawing, batched on the GPU, with the loaded network; resolves to their scores.
app.evaluate = async (n = 10, seed = 1000) => {
  const games = Array.from({ length: n }, (_, i) => new Game(app.ALE, `/roms/${app.entry.id}.bin`, { seed: seed + i }));
  const scores = new Array(n).fill(null), batch = new Uint8Array(n * 84 * 84 * 4);
  let live = games.map((_, i) => i);
  while (live.length) {
    live.forEach((g, k) => batch.set(games[g].state, k * 28224));
    const qt = app.policy.q(batch.subarray(0, live.length * 28224)), q = await qt.data();
    qt.dispose();
    const A = app.policy.numActions;
    live = live.filter((g, k) => {
      const game = games[g];
      if (game.step(game.chooseAction(q.subarray(k * A, (k + 1) * A))).ended) { scores[g] = game.lastScore; return false; }
      return true;
    });
  }
  games.forEach(g => g.dispose());
  return scores;
};

function renderResults() {
  const n = v => Math.round(v).toLocaleString();
  const rows = app.manifest.games.map(g => `<tr><td>${g.title}</td>
      ${g.final ? `<td class="ours">${n(g.final.score)}</td>` : '<td class="dim">training</td>'}
      <td>${n(g.paper.mean)} <span class="dim">(${n(g.paper.min)}–${n(g.paper.max)})</span></td>
      <td>${n(g.human)}</td><td>${n(g.random)}</td></tr>`).join('');
  $('results-table').innerHTML = `<table class="results"><thead><tr><th>Game</th><th>Our run<br>1 seed</th>
    <th>Paper, 14 seeds<br>mean (range)</th><th>Human</th><th>Random</th></tr></thead><tbody>${rows}</tbody></table>
    <p>Each score is the average of 100 games after 100,000 moves of practice. The paper's scores are for BBF with two updates per move, the same setting as our run.</p>`;
}

// ---------- Sound: on by default; browsers start audio only after a click or tap ----------

// The light shows whether sound is really playing: until the first click or key press, browsers keep it paused.
function applySound() {
  const ctx = app.screenALE?.SDL2?.audioContext, button = $('sound');
  const light = () => button.classList.toggle('live', ctx?.state === 'running');
  if (ctx) ctx.onstatechange = light;
  if (button.getAttribute('aria-pressed') === 'true') ctx?.resume().catch(() => {});
  else ctx?.suspend();
  light();
}

function buildSound() {
  const button = $('sound');
  // the button goes by the light: dark means "turn sound on", even when it is the browser that holds sound back
  button.onclick = () => { button.setAttribute('aria-pressed', String(!button.classList.contains('live'))); applySound(); };
  const elsewhere = e => { if (!button.contains(e.target)) applySound(); };
  addEventListener('pointerdown', elsewhere);
  addEventListener('keydown', elsewhere);
}

// ---------- Operator menu: the explanation ----------

function buildMenu() {
  const dialog = $('menu'), items = [...$('menu-list').querySelectorAll('button')];
  const show = item => {
    items.forEach(b => b.setAttribute('aria-current', String(b === item)));
    dialog.querySelectorAll('.menu-body section').forEach(sec => sec.hidden = sec.id !== item.dataset.section);
  };
  items.forEach(b => {
    if (b.dataset.close != null) b.onclick = () => dialog.close();
    else { b.onclick = () => show(b); b.onfocus = () => show(b); }
  });
  $('menu-open').onclick = () => { dialog.showModal(); items[0].focus(); show(items[0]); };
  dialog.addEventListener('click', e => { if (e.target === dialog) dialog.close(); });
  $('menu-list').addEventListener('keydown', e => {
    const k = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
    if (!k) return;
    e.preventDefault();
    items[(items.indexOf(document.activeElement) + k + items.length) % items.length].focus();
  });
}

main().catch(e => { console.error(e); setLoading(`Something went wrong: ${e.message}`); });
