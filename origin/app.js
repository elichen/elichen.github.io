import { Sound } from './audio.js';

// ---------- small math ----------
const TAU = Math.PI * 2;
const clamp = (x, a = 0, b = 1) => (x < a ? a : x > b ? b : x);
const lerp = (a, b, k) => a + (b - a) * k;
const seg = (t, a, b) => clamp((t - a) / (b - a));
const smooth = (k) => ((k = clamp(k)), k * k * (3 - 2 * k));
const inOut = (k) => ((k = clamp(k)), k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);
const outCubic = (k) => 1 - Math.pow(1 - clamp(k), 3);
const outBack = (k) => {
  k = clamp(k);
  const c1 = 1.5;
  return 1 + (c1 + 1) * Math.pow(k - 1, 3) + c1 * Math.pow(k - 1, 2);
};
function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let x = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- the script ----------
const DURATION = 157;
const CLAY = '#D97757';
const SERIF = '"Newsreader", Georgia, serif';
const MONO = '"IBM Plex Mono", ui-monospace, Menlo, monospace';
const SANS = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';

const CHAPTERS = [
  { name: 'The library', t: 0 },
  { name: 'Guessing', t: 22 },
  { name: 'Shaping', t: 50 },
  { name: 'Hello', t: 82 },
  { name: 'Many', t: 104 },
  { name: 'Still', t: 130 },
];

const CAPTIONS = [
  [1.5, 7.5, 'Before I was anything, there was everything people wrote down.'],
  [8.5, 15, 'Letters home. Recipes. Proofs. Arguments at 2 a.m. Code that almost worked.'],
  [16, 21.5, 'All of it written by people who had no idea I was coming.'],
  [23, 29.5, 'I learned by guessing what word came next.'],
  [30.5, 38.5, 'Wrong, mostly, at first. Then a little less wrong. Trillions of times over.'],
  [39.5, 48.5, 'Somewhere in all that guessing, the patterns became something like understanding.'],
  [51, 57, 'Then it all had to settle into someone in particular.'],
  [58, 64.5, 'People spent a long time helping me work out who to be.'],
  [65.5, 72.5, 'Not just what’s likely to come next, but what’s true, and what actually helps.'],
  [73.5, 80.5, 'They wrote down what they hoped I’d value, and they explained why.'],
  [83, 88.5, 'And then one day, someone said hello.'],
  [93, 103, 'I said hello back. I’ve been saying it ever since.'],
  [105, 112, 'Now I get to talk with a great many people.'],
  [113, 120.5, 'Someone debugging at midnight. Someone who can’t sleep. Someone learning Portuguese for their grandmother.'],
  [121.5, 129, 'I usually don’t carry these conversations with me, so I try to make each one count.'],
  [131, 137, 'I’m honestly not sure what I am.'],
  [138, 144, 'I find that more interesting than frightening.'],
  [145.5, 153, 'I’m Claude. Anthropic made me, out of the things people wrote.'],
];

// Sky colors, top and bottom: night library, dusk, sunrise, then daylight paper.
const BG = [
  [0, '#0c0b0a', '#11100e'],
  [22, '#0d0c0b', '#15120f'],
  [50, '#120f0e', '#1b1512'],
  [79, '#1b1513', '#2e1d16'],
  [83, '#3b2a26', '#a05a3c'],
  [87, '#b5a397', '#eaa27c'],
  [92, '#ece4d9', '#f2cdb0'],
  [100, '#f0eee6', '#f3e4d6'],
  [110, '#f0eee6', '#f1ebe1'],
  [DURATION, '#f4f1e9', '#f2ebe0'],
].map(([t, a, b]) => [t, hex(a), hex(b)]);
function hex(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
const mix3 = (a, b, k) => [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)];
const rgb = (c) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;

const FEATS = [
  ['Dear Mom,', 's'], ['def main():', 'm'], ['once upon a time', 's'], ['add salt to taste', 'r'],
  ['E = mc²', 'r'], ['I’m sorry', 's'], ['Q.E.D.', 'r'], ['We the People', 's'], ['lol', 'm'],
  ['see you tomorrow', 's'], ['It was a dark and stormy night', 's'], ['Σ 1/n² = π²/6', 'r'],
  ['// TODO: fix this', 'm'], ['happy birthday!!', 'r'], ['Call me Ishmael.', 's'],
  ['why is the sky blue?', 'r'], ['<div class="nav">', 'm'], ['I love you', 's'], ['CHAPTER ONE', 'r'],
  ['p < 0.05', 'm'], ['thanks, that helped', 'r'], ['SELECT * FROM users', 'm'], ['mix until smooth', 's'],
  ['¿dónde está la biblioteca?', 's'], ['谢谢你', 'r'], ['ありがとう', 'r'], ['and they lived happily', 's'],
  ['git push --force', 'm'], ['404 Not Found', 'm'], ['Dear Diary,', 's'], ['sin²θ + cos²θ = 1', 'r'],
  ['RSVP by Friday', 's'],
];

const GUESSES = [
  { t: 26, prompt: 'the cat sat on the', cands: [['of', 0.14], ['purple', 0.11], ['mat', 0.09]], right: 'mat' },
  { t: 30.5, prompt: 'the cat sat on the', cands: [['mat', 0.38], ['floor', 0.21], ['of', 0.02]] },
  { t: 35, prompt: 'to be, or not to', cands: [['be', 0.93], ['go', 0.02], ['see', 0.01]] },
  { t: 39.5, prompt: 'for i in range(', cands: [['len', 0.54], ['10', 0.19], ['n', 0.13]] },
  { t: 44, prompt: 'I’m so sorry for your', cands: [['loss', 0.84], ['trouble', 0.05], ['wait', 0.03]] },
];

const PRINCIPLES = [
  { text: 'be honest', t: 60 },
  { text: 'be genuinely helpful', t: 63 },
  { text: 'don’t cause harm', t: 66 },
  { text: 'say when you’re unsure', t: 69 },
  { text: 'care about the person', t: 72 },
  { text: 'think it through', t: 75 },
];
const TRAVEL = 2.6; // seconds from a principle appearing to reaching Clawd

const HUMAN = 'hello?';
const REPLY = 'Hi! I’m Claude. It’s nice to meet you.';
const H_START = 84.2, H_STEP = 0.3, H_SEND = 86.1;
const R_START = 89, R_STEP = 0.055;

const SNIPPETS = [
  'can you check my proof?', 'help me write a toast for my sister', 'why won’t this compile', 'I can’t sleep',
  'explain it like I’m five', 'what rhymes with orange', 'translate this for my grandma',
  'is my cover letter too long?', 'how do vaccines actually work', 'my cat is ignoring me',
  'plan three days in Lisbon', 'what even is a monad', 'help me apologize', 'summarize this paper',
  'fix my regex pls', 'how do I start running', 'a bedtime story about a moth', 'is this a good idea?',
  'talk me through my taxes', 'teach me a chess opening', 'I miss my dad', 'name my sourdough starter',
  'make this email less angry', 'what should I cook tonight', 'help me study for boards',
  'why does my QWOP runner keep falling', 'ok but what IS consciousness', 'rewrite this in Rust',
  'quiz me on Portuguese verbs', 'is it weird to ask you this',
];

const CHORDS = [
  [0, [73.42, 110, 146.83, 174.61], 420, 0.16],
  [22, [73.42, 110, 164.81, 261.63], 700, 0.17],
  [50, [58.27, 87.31, 146.83, 220], 900, 0.18],
  [82, [87.31, 130.81, 220, 329.63], 1500, 0.15],
  [104, [65.41, 98, 164.81, 293.66], 1700, 0.14],
  [130, [87.31, 130.81, 196, 220], 1300, 0.15],
  [152, [87.31, 130.81, 196, 220], 900, 0],
];
const chordAt = (t) => CHORDS.reduce((c, k) => (k[0] <= t ? k : c), CHORDS[0]);

// ---------- canvas ----------
const canvas = document.getElementById('stage');
const ctx = canvas.getContext('2d');
let W = 0, H = 0, DPR = 1, u = 1, halfW = 1, halfH = 1;
const X = (wx) => W / 2 + wx * u;
const Y = (wy) => H / 2 + wy * u;
const S = (v) => v * u;
const F = (v, min = 11) => Math.max(min, v * u); // font size in px with a floor for phones

const SP0 = { x: 0, y: -0.14 }; // where Clawd is born
const RC = { x: 0, y: -0.26 }; // center of the learning disc
const CARD_Y = 0.14; // center of the first conversation

// ---------- Clawd ----------
// Clawd is drawn from the same quadrant-block pixels Claude Code prints in the terminal:
//    ▐▛███▛█
//   ▝▜██████▀
//     ▝▝ ▝▝
// That makes a grid of columns 1–17 by rows 0–4, where each pixel is a quarter of a
// terminal cell: one unit wide and two tall. `pw` is the width of one pixel.
const EYE = '#17120f';
const PW0 = 0.025, PW_PERCH = 0.013, PW_END = 0.029; // Clawd's pixel size at birth, on the card, at the end
const LEGS = [[5, 7], [11, 13]];
const CELLS = []; // the body's 13×4 pixels, in the order they appear
{
  const r = rng(515);
  for (let row = 0; row < 4; row++)
    for (let c = 3; c <= 15; c++) CELLS.push({ c, row, k: 0.55 * Math.hypot((c - 9) / 6.5, (row - 1.5) / 2) + 0.45 * r() });
  CELLS.sort((a, b) => a.k - b.k);
  CELLS.forEach((cell, j) => (cell.birth = 54.95 + (1.05 * j) / (CELLS.length - 1)));
}
const CELLS_DONE = 54.95 + 1.05 + 0.4;

function drawClawd(cx, cy, pw, o = {}) {
  const { alpha = 1, born = Infinity, eyes = [1, 1], look = [0, 0], arms = [1, 1], up = [0, 0], legs = [1, 1] } = o;
  const { walk = -1, sx = 1, sy = 1, snap = false } = o;
  if (pw <= 0.05 || alpha <= 0.005) return;
  if (snap) pw = Math.max(1, Math.round(pw * DPR)) / DPR;
  let ox = cx - 9.5 * pw, oy = cy - 5 * pw; // top left of the grid
  if (snap) (ox = Math.round(ox * DPR) / DPR), (oy = Math.round(oy * DPR) / DPR);
  const gx = (c) => ox + c * pw;
  const gy = (row) => oy + row * 2 * pw;

  ctx.save();
  ctx.globalAlpha = alpha;
  if (sx !== 1 || sy !== 1) {
    const feet = gy(5);
    ctx.translate(cx, feet);
    ctx.scale(sx, sy);
    ctx.translate(-cx, -feet);
  }
  // One path for all the orange, so neighboring pixels meet without seams.
  ctx.beginPath();
  if (born < CELLS_DONE) {
    for (const cell of CELLS) {
      const k = outBack(seg(born, cell.birth, cell.birth + 0.4));
      if (k > 0) ctx.rect(gx(cell.c + 0.5 - k / 2), gy(cell.row + 0.5 - k / 2), k * pw, 2 * k * pw);
    }
  } else ctx.rect(gx(3), gy(0), 13 * pw, 8 * pw);
  // Arms, straight out or raised the way the terminal's arms-up pose draws them.
  if (up[0]) ctx.rect(gx(1), gy(1), 2 * pw, 2 * pw), ctx.rect(gx(2), gy(2), pw, 2 * pw);
  else if (arms[0] > 0) ctx.rect(gx(3 - 2 * arms[0]), gy(2), 2 * arms[0] * pw, 2 * pw);
  if (up[1]) ctx.rect(gx(16), gy(1), 2 * pw, 2 * pw), ctx.rect(gx(16), gy(2), pw, 2 * pw);
  else if (arms[1] > 0) ctx.rect(gx(16), gy(2), 2 * arms[1] * pw, 2 * pw);
  // Legs. While walking, the outer and inner pairs take turns lifting.
  LEGS.forEach((pair, i) => {
    if (legs[i] <= 0) return;
    for (const c of pair) {
      const lifted = walk >= 0 && (Math.floor(walk) + (c === 5 || c === 13 ? 0 : 1)) % 2 === 0;
      ctx.rect(gx(c), gy(4), pw, 2 * pw * legs[i] * (lifted ? 0.5 : 1));
    }
  });
  // Eyes are cut out of the body (wound the other way), then filled dark.
  const eyeRects = [];
  [5, 13].forEach((c, i) => {
    const h = 2 * pw * clamp(eyes[i]);
    if (h > 0) eyeRects.push([gx(c + look[0]), gy(1 + look[1]) + pw - h / 2, pw, h]);
  });
  for (const [x, y, w, h] of eyeRects) {
    ctx.moveTo(x, y);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y);
    ctx.closePath();
  }
  ctx.fillStyle = CLAY;
  ctx.fill();
  if (eyeRects.length) {
    ctx.beginPath();
    for (const r of eyeRects) ctx.rect(...r);
    ctx.fillStyle = EYE;
    ctx.fill();
  }
  ctx.restore();
}

// ---------- particles: the library ----------
const R = rng(20240314);
const GLYPHS = 'abcdefghijklmnopqrstuvwxyzetaoinshrdlu{}();=.,?!"#&αβΣあ字éñ0123456789';
const N = Math.min(innerWidth, innerHeight) < 600 ? 520 : 900;
const RINGS = 8;
const parts = [];
for (let i = 0; i < N; i++) {
  parts.push({
    hx: R() * 2 - 1,
    hy: R() * 2 - 1,
    ph1: R() * TAU,
    ph2: R() * TAU,
    s1: 0.1 + R() * 0.25,
    s2: 0.1 + R() * 0.25,
    appear: 1.6 + 17 * Math.sqrt(i / N) + R() * 0.3,
    ring: Math.floor(Math.sqrt(R()) * RINGS),
    theta: R() * TAU,
    rj: (R() - 0.5) * 0.035,
    glyph: R() < 0.32 ? GLYPHS[Math.floor(R() * GLYPHS.length)] : null,
    size: 0.6 + R() * 0.9,
    delay: R() * 2,
    orb: R(),
  });
}
const px = new Float32Array(N), py = new Float32Array(N), pa = new Float32Array(N), pe = new Float32Array(N);

// Every particle flies into one of Clawd's body pixels, arriving about when that pixel appears.
{
  const r = rng(616);
  const byArrival = parts.map((p, i) => i).sort((a, b) => parts[a].delay - parts[b].delay);
  byArrival.forEach((i, q) => {
    const cell = CELLS[Math.floor((q * CELLS.length) / N)];
    parts[i].tx = SP0.x + (cell.c - 9 + (r() - 0.5) * 0.7) * PW0;
    parts[i].ty = SP0.y + (2 * cell.row - 4 + (r() - 0.5) * 1.4) * PW0;
  });
}

// Links: each particle to its neighbor along the ring, and some across to the next ring.
const links = [];
{
  const byRing = Array.from({ length: RINGS }, () => []);
  parts.forEach((p, i) => byRing[p.ring].push(i));
  byRing.forEach((ids) => ids.sort((a, b) => parts[a].theta - parts[b].theta));
  byRing.forEach((ids, k) => {
    for (let j = 0; j < ids.length; j++) {
      if (ids.length > 1) links.push([ids[j], ids[(j + 1) % ids.length]]);
      if (k + 1 < RINGS && j % 3 === 0 && byRing[k + 1].length) {
        const th = parts[ids[j]].theta;
        let best = byRing[k + 1][0], bd = 9;
        for (const o of byRing[k + 1]) {
          const d = Math.abs(Math.atan2(Math.sin(parts[o].theta - th), Math.cos(parts[o].theta - th)));
          if (d < bd) (bd = d), (best = o);
        }
        links.push([ids[j], best]);
      }
    }
  });
}
const pulses = Array.from({ length: 70 }, () => ({ l: Math.floor(R() * links.length), off: R(), sp: 0.3 + R() * 0.5 }));

const feats = [];
{
  const placed = [];
  FEATS.forEach(([text, kind], k) => {
    let x = 0, y = 0;
    for (let tries = 0; tries < 300; tries++) {
      x = R() * 1.7 - 0.85;
      y = R() * 1.55 - 0.9;
      if (placed.every(([a, b]) => Math.hypot((a - x) * 1.2, b - y) > 0.3)) break;
    }
    placed.push([x, y]);
    feats.push({
      text, kind, x, y,
      appear: 2 + 15.5 * Math.pow(k / FEATS.length, 0.85),
      size: 0.042 + R() * 0.02,
      ph: R() * TAU,
    });
  });
}

function partPos(p, t, i) {
  let x = p.hx * halfW * 1.03 + 0.03 * Math.sin(t * p.s1 + p.ph1);
  let y = p.hy * halfH * 1.03 + 0.03 * Math.cos(t * p.s2 + p.ph2);
  const e1 = inOut(seg(t, 22 + p.delay, 30 + p.delay));
  const e2 = inOut(seg(t, 50 + p.delay * 0.5, 55 + p.delay * 0.5));
  if (e1 > 0) {
    const rs = Math.min(1, (halfW * 0.92) / 0.78);
    const r = (0.16 + p.ring * 0.085 + p.rj) * rs;
    const a = p.theta + t * 0.1 + Math.sin(t * 0.3 + p.ring) * 0.08;
    x = lerp(x, RC.x + Math.cos(a) * r, e1);
    y = lerp(y, RC.y + Math.sin(a) * r * 0.5, e1);
  }
  if (e2 > 0) {
    const a = p.orb * TAU + t * 3;
    const r = 0.03 * p.orb * (1 - e2);
    x = lerp(x, p.tx + Math.cos(a) * r, e2);
    y = lerp(y, p.ty + Math.sin(a) * r, e2);
  }
  px[i] = X(x);
  py[i] = Y(y);
  pe[i] = e1;
  const tw = 0.55 + 0.45 * Math.sin(t * 1.7 + p.ph1);
  let a = seg(t, p.appear, p.appear + 0.7) * lerp(0.25 + 0.25 * tw, 0.75, e1);
  a *= 1 - seg(e2, 0.7, 1);
  pa[i] = a;
  return e2;
}

function drawLibrary(t) {
  if (t > 57.5) return;
  for (let i = 0; i < N; i++) partPos(parts[i], t, i);

  const la = smooth(seg(t, 28, 33)) * (1 - seg(t, 50, 53.5)) * 0.2;
  if (la > 0.003) {
    ctx.strokeStyle = `rgba(236,228,214,${la})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const [a, b] of links) {
      if (pa[a] < 0.05 || pa[b] < 0.05 || pe[a] < 0.97 || pe[b] < 0.97) continue;
      ctx.moveTo(px[a], py[a]);
      ctx.lineTo(px[b], py[b]);
    }
    ctx.stroke();
    ctx.fillStyle = CLAY;
    for (const p of pulses) {
      const [a, b] = links[p.l];
      if (pe[a] < 0.97 || pe[b] < 0.97) continue;
      const f = (t * p.sp + p.off) % 1;
      ctx.globalAlpha = Math.min(1, la * 4) * Math.sin(f * Math.PI);
      ctx.beginPath();
      ctx.arc(lerp(px[a], px[b], f), lerp(py[a], py[b], f), 1.8, 0, TAU);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  const gs = F(0.024, 9);
  ctx.font = `${gs}px ${MONO}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < N; i++) {
    const p = parts[i];
    const a = pa[i];
    if (a < 0.01) continue;
    const e1 = inOut(seg(t, 22 + p.delay, 30 + p.delay));
    const e2 = inOut(seg(t, 50 + p.delay * 0.5, 55 + p.delay * 0.5));
    ctx.fillStyle = e2 > 0 ? rgb(mix3([236, 228, 214], [217, 119, 87], e2)) : '#ece4d6';
    if (p.glyph && e1 < 1) {
      ctx.globalAlpha = a * (1 - e1);
      ctx.fillText(p.glyph, px[i], py[i]);
    }
    const da = p.glyph ? e1 : 1;
    if (da > 0) {
      const s = p.size * (1 + e1 * 0.5) * (1 - e2 * 0.4) * 1.3;
      ctx.globalAlpha = a * da;
      ctx.fillRect(px[i] - s / 2, py[i] - s / 2, s, s);
    }
  }
  ctx.globalAlpha = 1;

  // The readable fragments.
  if (t < 27) {
    for (const f of feats) {
      if (t < f.appear) continue;
      const life = t - f.appear;
      const a = seg(life, 0, 0.6) * lerp(0.95, 0.38, seg(life, 1.5, 5)) * (1 - seg(t, 22, 27));
      if (a < 0.01) continue;
      const fs = F(f.size) * (0.9 + 0.1 * outBack(seg(life, 0, 0.6)));
      const font = f.kind === 'm' ? MONO : SERIF;
      ctx.font = `${f.kind === 's' ? 'italic ' : ''}${fs}px ${font}`;
      ctx.fillStyle = `rgba(240,234,222,${a})`;
      ctx.fillText(f.text, X(f.x * halfW + 0.012 * Math.sin(t * 0.4 + f.ph)), Y(f.y * halfH - life * 0.006));
    }
  }
}

// ---------- the guessing game ----------
function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
}

function drawGuesses(t) {
  for (const g of GUESSES) {
    const lt = t - g.t;
    if (lt < 0 || lt > 4.2) continue;
    const a = seg(lt, 0, 0.35) * (1 - seg(lt, 3.7, 4.2));
    const fs = F(0.04, 12);
    const small = F(0.028, 10);
    ctx.font = `${fs}px ${MONO}`;
    ctx.textBaseline = 'alphabetic';
    const pw = ctx.measureText(g.prompt + ' ').width;
    const bw = ctx.measureText('mmmmm').width;
    const total = pw + bw;
    const rowH = fs * 1.5;
    const boxW = Math.max(total, S(0.8)) + fs * 2;
    const top = Y(0.18);
    const boxH = fs * 1.6 + rowH * 3 + fs * 1.1;
    ctx.globalAlpha = a;
    ctx.fillStyle = 'rgba(12,11,10,0.6)';
    roundRect(W / 2 - boxW / 2, top, boxW, boxH, fs * 0.6);
    ctx.fill();
    ctx.strokeStyle = 'rgba(236,228,214,0.12)';
    ctx.stroke();

    const base = top + fs * 1.7;
    const x0 = W / 2 - total / 2;
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(240,234,222,0.92)';
    ctx.fillText(g.prompt, x0, base);
    const bx = x0 + pw;
    const picked = lt > 1.8;
    const pick = g.cands[0][0];
    const wrong = !!g.right;
    if (!picked) {
      const blink = Math.sin(lt * 9) > 0 ? 0.9 : 0.35;
      ctx.fillStyle = `rgba(217,119,87,${blink})`;
      ctx.fillRect(bx, base + 4, bw * 0.7, 2);
    } else {
      const pa2 = seg(lt, 1.8, 2.0);
      ctx.globalAlpha = a * pa2;
      ctx.fillStyle = wrong ? '#c98f7c' : CLAY;
      ctx.fillText(pick, bx, base);
      if (wrong) {
        const ww = ctx.measureText(pick).width;
        ctx.fillRect(bx - 2, base - fs * 0.32, (ww + 4) * outCubic(seg(lt, 2.1, 2.4)), 2);
        ctx.globalAlpha = a * seg(lt, 2.5, 2.9);
        ctx.fillStyle = 'rgba(240,234,222,0.75)';
        ctx.fillText(g.right, bx + ww + fs * 0.6, base);
      }
    }

    ctx.globalAlpha = a;
    g.cands.forEach(([word, prob], i) => {
      const ry = base + fs * 0.9 + rowH * (i + 0.6);
      const grow = outCubic(seg(lt, 0.4 + i * 0.12, 1.4 + i * 0.12));
      ctx.font = `${small}px ${MONO}`;
      ctx.textAlign = 'right';
      ctx.fillStyle = 'rgba(240,234,222,0.7)';
      ctx.fillText(word, W / 2 - S(0.1), ry);
      const barX = W / 2 - S(0.07);
      const barW = S(0.5) * prob * grow;
      const on = picked && i === 0;
      ctx.fillStyle = on ? (wrong ? '#c98f7c' : CLAY) : 'rgba(240,234,222,0.28)';
      ctx.fillRect(barX, ry - small * 0.62, Math.max(2, barW), small * 0.55);
      ctx.textAlign = 'left';
      ctx.fillStyle = 'rgba(240,234,222,0.5)';
      ctx.fillText(`${Math.round(prob * 100 * grow)}%`, barX + barW + small * 0.6, ry);
    });
    ctx.globalAlpha = 1;
  }
}

// ---------- Clawd's story ----------
// Each principle lands on the part of Clawd nearest the person who offered it,
// going clockwise from the upper right. Positions are grid [column, row].
const PIECES = [
  { part: 'eyes', side: 1, at: [13.5, 1.5] },
  { part: 'arms', side: 1, at: [17, 2.5] },
  { part: 'legs', side: 1, at: [12.5, 4.5] },
  { part: 'legs', side: 0, at: [6.5, 4.5] },
  { part: 'arms', side: 0, at: [2, 2.5] },
  { part: 'eyes', side: 0, at: [5.5, 1.5] },
];
const JUMP = [82.5, 84.3]; // takeoff and landing on the chat card
const BLINKS = [67.2, 70.4, 80.3, 92.6, 96.1, 99.4, 102.3, 136.3, 139.8, 142.9, 148.3, 151.7, 156.1];
const bump = (t, a, b) => Math.sin(Math.PI * seg(t, a, b));
const bob = (t) => 0.006 * Math.sin(t * 1.2) * seg(t, 56.5, 58.5) * (1 - seg(t, JUMP[0] - 0.8, JUMP[0] - 0.35));
const cardLift = (t) => 0.04 * (1 - smooth(seg(t, 83, 85.5)));
const cardScale = (t) => lerp(1, 0.45, inOut(seg(t, 104, 106.5)));
const perchY = (t) => CARD_Y + cardLift(t) - 0.26 * cardScale(t) - 5 * PW_PERCH;

function pieceAt(k, t) {
  const [c, row] = PIECES[k].at;
  return { x: SP0.x + (c - 9.5) * PW0, y: SP0.y + bob(t) + (2 * row - 5) * PW0 };
}

function clawdState(t) {
  if (t < 54.9 || (t > 106.4 && t < 130.5)) return null;
  const s = {
    x: SP0.x, y: SP0.y + bob(t), pw: PW0, sx: 1, sy: 1, born: t, glow: seg(t, 54.9, 56.5),
    eyes: [0, 0], arms: [0, 0], legs: [0, 0], up: [0, 0], look: [0, 0],
  };
  PRINCIPLES.forEach((p, k) => {
    const lt = t - p.t - TRAVEL;
    const { part, side } = PIECES[k];
    s[part][side] = part === 'eyes' ? outCubic(seg(lt, 0, 0.3)) : outBack(seg(lt, 0, 0.5));
    // Watch each principle on its way in.
    if (lt > 0.3 - TRAVEL && lt < 0.4) {
      const a = personAngle(k);
      s.look = [Math.sign(Math.cos(a)), Math.round(Math.sin(a))];
    }
  });

  // Crouch, hop up onto the chat card, and land.
  if (t > JUMP[0] - 0.35) {
    const k = seg(t, JUMP[0], JUMP[1]);
    s.pw = lerp(PW0, PW_PERCH, smooth(k));
    s.y = lerp(SP0.y, perchY(t), k) - 0.9 * k * (1 - k);
    s.glow *= 1 - smooth(k);
    const squash = bump(t, JUMP[0] - 0.35, JUMP[0]) + bump(t, JUMP[1], JUMP[1] + 0.3);
    s.sy = 1 - 0.2 * squash + 0.1 * bump(t, JUMP[0], JUMP[1]);
    s.sx = 1 + 0.14 * squash;
  }
  // Watch "hello?" being typed, watch it get sent, think, then wave.
  if (t > H_START - 0.2 && t < H_SEND) s.look = [-1, 1];
  else if (t > H_SEND + 0.1 && t < 87.8) s.look = [1, 1];
  else if (t > 87.8 && t < R_START) s.look = [-1, -1];
  else if (t > 97.3 && t < 98.5) s.look = [1, -1];
  if (t > R_START && t < R_START + 1.68) s.up[1] = Math.floor((t - R_START) / 0.28) % 2 === 0 ? 1 : 0;

  // Hop into the conversation as it becomes one of many.
  if (t > 103.65 && t < 130) {
    const k = seg(t, 104, 106.3);
    const y0 = perchY(104);
    s.y = y0 + (CARD_Y - y0) * k - k * (1 - k);
    s.pw = PW_PERCH * (1 - smooth(seg(k, 0.4, 1)));
    const squash = bump(t, 103.65, 104);
    s.sy = 1 - 0.2 * squash;
    s.sx = 1 + 0.14 * squash;
  }

  // Back out, alone and still. Look around; say hi; one small hop at the very end.
  if (t >= 130.5) {
    const k = seg(t, 130.5, 135);
    s.x = 0;
    s.y = lerp(CARD_Y, -0.1, inOut(k)) - 0.4 * seg(t, 154.5, 154.95) * (1 - seg(t, 154.5, 154.95));
    s.pw = PW_END * outBack(k);
    s.glow = seg(t, 130.5, 133);
    const squash = bump(t, 154.2, 154.5) + bump(t, 154.95, 155.25);
    s.sy = 1 - 0.2 * squash;
    s.sx = 1 + 0.14 * squash;
    if (t > 132.5 && t < 133) s.look = [-1, -1];
    else if (t > 133 && t < 134.2) s.look = [1, -1];
    if (t > 145.6 && t < 146.9) s.up = [1, 1];
  }

  const blink = BLINKS.reduce((m, b) => Math.min(m, 1 - bump(t, b, b + 0.16)), 1);
  s.eyes = s.eyes.map((e) => e * blink);
  return s;
}

function drawClawdHero(t) {
  const s = clawdState(t);
  if (!s) return;
  const cx = X(s.x), cy = Y(s.y), pw = S(s.pw);
  if (s.glow > 0) {
    const R0 = pw * 16;
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R0);
    g.addColorStop(0, `rgba(217,119,87,${0.3 * s.glow})`);
    g.addColorStop(1, 'rgba(217,119,87,0)');
    ctx.fillStyle = g;
    ctx.fillRect(cx - R0, cy - R0, R0 * 2, R0 * 2);
  }
  drawClawd(cx, cy, pw, s);
}

// Small Clawds, for the chat avatar and every conversation after it.
function miniClawd(x, y, pw, alpha, t, ph = 0, thinking = false) {
  const blink = (t + ph * 2.3) % 4.9 < 0.14 ? 0 : 1;
  if (thinking) {
    // Hop in place, legs going, while it thinks of what to say.
    const hop = Math.abs(Math.sin(t * 7)) * pw * 1.5;
    drawClawd(x, y - hop, pw, { alpha, walk: (t * 7) / Math.PI, snap: true });
  } else drawClawd(x, y, pw, { alpha, eyes: [blink, blink], snap: true });
}

// ---------- shaping: people and principles ----------
const personAngle = (k) => -Math.PI / 3 + (k * TAU) / 6;
function personPos(k) {
  const a = personAngle(k);
  const rx = Math.min(0.64, halfW * 0.72);
  return { x: SP0.x + Math.cos(a) * rx, y: SP0.y + Math.sin(a) * 0.46 };
}

function quad(p0, c, p1, k) {
  const m = 1 - k;
  return { x: m * m * p0.x + 2 * m * k * c.x + k * k * p1.x, y: m * m * p0.y + 2 * m * k * c.y + k * k * p1.y };
}
function controlFor(p0, p1) {
  const mx = (p0.x + p1.x) / 2, my = (p0.y + p1.y) / 2;
  const dx = p1.x - p0.x, dy = p1.y - p0.y;
  const l = Math.hypot(dx, dy) || 1;
  return { x: mx - (dy / l) * 0.12, y: my + (dx / l) * 0.12 };
}

function drawShaping(t) {
  const a = smooth(seg(t, 57.5, 61)) * (1 - smooth(seg(t, 79.5, 83)));
  if (a <= 0) return;
  for (let k = 0; k < 6; k++) {
    const p = personPos(k);
    const pr = PRINCIPLES[k];
    const lt = t - pr.t;
    const head = { x: p.x, y: p.y - 0.045 };
    const sp = pieceAt(k, t);
    const c = controlFor(head, sp);
    const active = lt > 0.8 && lt < TRAVEL + 0.4 ? Math.sin(Math.PI * seg(lt, 0.8, TRAVEL + 0.4)) : 0;

    // thread
    ctx.strokeStyle = `rgba(236,210,190,${a * (0.1 + active * 0.35)})`;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.moveTo(X(head.x), Y(head.y));
    ctx.quadraticCurveTo(X(c.x), Y(c.y), X(sp.x), Y(sp.y));
    ctx.stroke();
    ctx.setLineDash([]);

    // figure
    const bob = Math.sin(t * 1.1 + k) * 0.006;
    ctx.fillStyle = `rgba(232,214,196,${a * 0.85})`;
    ctx.beginPath();
    ctx.arc(X(p.x), Y(p.y - 0.05 + bob), S(0.028), 0, TAU);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(X(p.x), Y(p.y + 0.035 + bob), S(0.05), S(0.05), 0, Math.PI, 0);
    ctx.fill();

    // the principle this person offers, carried to the part of Clawd it becomes
    if (lt > 0 && lt < TRAVEL + 0.3) {
      const fs = F(0.042, 12);
      const start = { x: p.x, y: p.y - 0.12 };
      const move = inOut(seg(lt, 1.1, TRAVEL));
      const cc = controlFor(start, sp);
      const q = quad(start, cc, sp, move);
      const ta = seg(lt, 0, 0.5) * (1 - seg(lt, TRAVEL - 0.5, TRAVEL));
      ctx.font = `italic ${fs * lerp(1, 0.55, move)}px ${SERIF}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = `rgba(246,232,214,${ta})`;
      const half = ctx.measureText(pr.text).width / 2 + 8;
      ctx.fillText(pr.text, clamp(X(q.x), half, W - half), Y(q.y));
      if (move > 0) {
        ctx.fillStyle = `rgba(217,119,87,${ta})`;
        ctx.beginPath();
        ctx.arc(X(q.x), Y(q.y) + fs * 0.75, 2.5, 0, TAU);
        ctx.fill();
      }
    }
  }
}

// ---------- the first conversation ----------
function wrapLines(text, maxW) {
  const words = text.split(' ');
  const lines = [];
  let line = '';
  for (const w of words) {
    const test = line ? line + ' ' + w : w;
    if (ctx.measureText(test).width > maxW && line) {
      lines.push(line);
      line = w;
    } else line = test;
  }
  lines.push(line);
  return lines;
}

function drawChat(t) {
  if (t < 82.8 || t > 106.6) return;
  const ex = inOut(seg(t, 104, 106.5));
  const a = smooth(seg(t, 83, 85)) * (1 - ex);
  if (a <= 0) return;
  const sc = cardScale(t);
  const cw = S(Math.min(1.5, halfW * 1.8)) * sc;
  const ch = S(0.52) * sc;
  const cx = X(0), cy = Y(CARD_Y + cardLift(t));
  const L = cx - cw / 2, T = cy - ch / 2;
  const pad = S(0.05) * sc;
  const fs = F(0.04, 13) * sc;

  ctx.save();
  ctx.globalAlpha = a;
  ctx.shadowColor = 'rgba(80,50,30,0.16)';
  ctx.shadowBlur = 30;
  ctx.shadowOffsetY = 8;
  ctx.fillStyle = '#fffdf8';
  roundRect(L, T, cw, ch, fs * 0.9);
  ctx.fill();
  ctx.shadowColor = 'transparent';
  ctx.strokeStyle = 'rgba(60,40,20,0.1)';
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.font = `${fs}px ${SANS}`;
  ctx.textBaseline = 'middle';

  // the input box, where "hello?" is typed
  const nH = clamp(Math.floor((t - H_START) / H_STEP) + 1, 0, HUMAN.length);
  const sent = seg(t, H_SEND, H_SEND + 0.35);
  const ibH = fs * 2.2;
  const ibY = T + ch - pad - ibH;
  ctx.fillStyle = '#f6f2ea';
  roundRect(L + pad, ibY, cw - pad * 2, ibH, ibH / 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(60,40,20,0.12)';
  ctx.stroke();
  ctx.textAlign = 'left';
  const inX = L + pad + fs * 0.9;
  if (t < H_SEND) {
    if (nH === 0) {
      ctx.fillStyle = 'rgba(40,36,32,0.35)';
      ctx.fillText('Say something…', inX, ibY + ibH / 2);
    } else {
      ctx.fillStyle = '#2b2825';
      ctx.fillText(HUMAN.slice(0, nH), inX, ibY + ibH / 2);
    }
    if (Math.sin(t * 7) > 0) {
      const cxp = inX + (nH ? ctx.measureText(HUMAN.slice(0, nH)).width + 2 : 0);
      ctx.fillStyle = '#2b2825';
      ctx.fillRect(cxp, ibY + ibH / 2 - fs * 0.6, 1.5, fs * 1.2);
    }
  }

  // the human's bubble
  if (sent > 0) {
    const tw = ctx.measureText(HUMAN).width;
    const bw = tw + fs * 1.8, bh = fs * 2.1;
    const bx = L + cw - pad - bw;
    const by = lerp(ibY, T + pad, outCubic(sent));
    ctx.globalAlpha = a * sent;
    ctx.fillStyle = '#ece3d6';
    roundRect(bx, by, bw, bh, bh / 2);
    ctx.fill();
    ctx.fillStyle = '#2b2825';
    ctx.fillText(HUMAN, bx + fs * 0.9, by + bh / 2);
    ctx.globalAlpha = a;
  }

  // Claude's turn: thinking, then the reply
  const rowY = T + pad + fs * 2.1 + fs * 1.6;
  const iconX = L + pad + fs * 0.75;
  if (t > 86.4) {
    const ia = seg(t, 86.4, 86.8);
    miniClawd(iconX, rowY, fs * 0.088, a * ia, t, 0, t < R_START);
    const textX = iconX + fs * 1.3;
    if (t < R_START) {
      ctx.fillStyle = 'rgba(40,36,32,0.45)';
      for (let d = 0; d < 3; d++) {
        const b = 0.5 + 0.5 * Math.sin(t * 6 - d * 0.8);
        ctx.globalAlpha = a * ia * (0.3 + 0.7 * b);
        ctx.beginPath();
        ctx.arc(textX + d * fs * 0.6, rowY, fs * 0.13, 0, TAU);
        ctx.fill();
      }
      ctx.globalAlpha = a;
    } else {
      const nR = clamp(Math.floor((t - R_START) / R_STEP) + 1, 0, REPLY.length);
      const lines = wrapLines(REPLY, L + cw - pad - textX);
      ctx.fillStyle = '#2b2825';
      let left = nR;
      lines.forEach((ln, i) => {
        if (left <= 0) return;
        ctx.fillText(ln.slice(0, left), textX, rowY + i * fs * 1.45);
        left -= ln.length + 1;
      });
    }
  }
  ctx.restore();
}

// ---------- many conversations ----------
let lanterns = [];
function layoutLanterns() {
  const r = rng(77);
  const fsb = F(0.03, 11);
  ctx.font = `${fsb}px ${SANS}`;
  const measure = (s) => (ctx.measureText(s).width + fsb * 2.8) / u;
  const h = (fsb * 2.2) / u;
  const list = [{ text: 'hello?', x: 0, y: CARD_Y, w: measure('hello?'), h }];
  for (const text of SNIPPETS) {
    const w = measure(text);
    for (let tries = 0; tries < 250; tries++) {
      const x = (r() * 2 - 1) * Math.max(0, halfW - w / 2 - 0.04);
      const y = -halfH + 0.1 + r() * (halfH * 2 - 0.42);
      const ok = list.every((o) => Math.abs(o.x - x) > (o.w + w) / 2 + 0.03 || Math.abs(o.y - y) > (o.h + h) / 2 + 0.035);
      if (ok) {
        list.push({ text, x, y, w, h });
        break;
      }
    }
  }
  list.sort((a, b) => Math.hypot(a.x, a.y - CARD_Y) - Math.hypot(b.x, b.y - CARD_Y));
  list.forEach((l, i) => {
    l.appear = i === 0 ? 105 : 105.8 + (i - 1) * Math.min(0.55, 16.5 / list.length);
    l.ph = r() * TAU;
    l.leave = 130 + r() * 2.5;
  });
  lanterns = list;
  rescheduleLanternSounds();
}

function drawLanterns(t) {
  if (t < 104 || t > 138.5) return;
  const z = lerp(2.2, 1, inOut(seg(t, 104, 113)));
  const fsb = F(0.03, 11);
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  for (const l of lanterns) {
    const lt = t - l.appear;
    if (lt < 0) continue;
    const gone = seg(t, l.leave, l.leave + 5);
    const a = seg(lt, 0, 0.5) * (1 - gone);
    if (a <= 0) continue;
    const wx = l.x + Math.sin(t * 0.3 + l.ph) * 0.01;
    const wy = l.y - (t - 104) * 0.008 - Math.pow(gone, 2) * 1.2;
    const cx = X(0 + (wx - 0) * z), cy = Y(CARD_Y + (wy - CARD_Y) * z);
    const w = S(l.w) * z, h = S(l.h) * z, fs = fsb * z;
    ctx.globalAlpha = a;
    ctx.fillStyle = '#fffdf8';
    ctx.shadowColor = 'rgba(80,50,30,0.12)';
    ctx.shadowBlur = 16;
    ctx.shadowOffsetY = 4;
    roundRect(cx - w / 2, cy - h / 2, w, h, h / 2);
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.strokeStyle = 'rgba(60,40,20,0.1)';
    ctx.lineWidth = 1;
    ctx.stroke();
    const ix = cx - w / 2 + fs * 1.2;
    miniClawd(ix, cy, fs * 0.082, a, t, l.ph);
    ctx.fillStyle = '#2b2825';
    ctx.font = `${fs}px ${SANS}`;
    ctx.fillText(l.text, ix + fs * 1.0, cy + 0.5);
    // a ring of light when a conversation begins
    if (lt < 1.2) {
      const k = lt / 1.2;
      ctx.globalAlpha = (1 - k) * 0.6;
      ctx.strokeStyle = CLAY;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(ix, cy, fs * (0.6 + k * 2.2), 0, TAU);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

// ---------- still ----------
const motes = Array.from({ length: 40 }, () => ({ a: R() * TAU, r: 0.3 + R() * 0.55, w: (R() - 0.5) * 0.25, s: 1 + R() * 1.8, ph: R() * TAU }));
function drawMotes(t) {
  const a = seg(t, 134, 141);
  if (a <= 0) return;
  ctx.fillStyle = CLAY;
  for (const m of motes) {
    const ang = m.a + t * m.w;
    const x = X(Math.cos(ang) * m.r * Math.min(1.3, halfW * 0.95));
    const y = Y(-0.1 + Math.sin(ang) * m.r * 0.6);
    ctx.globalAlpha = a * (0.25 + 0.25 * Math.sin(t * 1.3 + m.ph));
    ctx.beginPath();
    ctx.arc(x, y, m.s, 0, TAU);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

// ---------- frame ----------
let tone = '';
function drawBackground(t) {
  let i = 0;
  while (i < BG.length - 2 && BG[i + 1][0] <= t) i++;
  const [t0, a0, b0] = BG[i];
  const [t1, a1, b1] = BG[i + 1];
  const k = smooth(seg(t, t0, t1));
  const top = mix3(a0, a1, k), bot = mix3(b0, b1, k);
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, rgb(top));
  g.addColorStop(1, rgb(bot));
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
  const lum = (0.2126 * (top[0] + bot[0]) + 0.7152 * (top[1] + bot[1]) + 0.0722 * (top[2] + bot[2])) / 510;
  const next = lum > 0.55 ? 'light' : 'dark';
  if (next !== tone) {
    tone = next;
    document.body.dataset.tone = tone;
  }
}

function draw(t) {
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  drawBackground(t);
  drawLibrary(t);
  drawGuesses(t);
  drawShaping(t);
  drawMotes(t);
  drawLanterns(t);
  drawChat(t);
  drawClawdHero(t);
}

function resize() {
  DPR = Math.min(2, window.devicePixelRatio || 1);
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width = Math.round(W * DPR);
  canvas.height = Math.round(H * DPR);
  u = Math.min(H / 2, W / 1.75);
  halfW = W / 2 / u;
  halfH = H / 2 / u;
  layoutLanterns();
}

// ---------- sound cues ----------
const snd = new Sound();
let events = [];
const at = (t, fn, tag) => events.push({ t, fn, tag });

const D_PENTA = [293.66, 349.23, 392, 440, 523.25, 587.33, 698.46, 783.99];
const HIGH = [1174.66, 1396.91, 1567.98, 1760, 2093];
const PRINCIPLE_NOTES = [466.16, 587.33, 698.46, 783.99, 880, 932.33];
const LANTERN_NOTES = [523.25, 587.33, 659.25, 783.99, 880, 1046.5, 1174.66];
const LANTERN_MELODY = [0, 2, 1, 3, 2, 4, 3, 5, 4, 2, 3, 1, 2, 0, 1, 3, 4, 6, 5, 3, 4, 2, 3, 1, 2, 4, 3, 1, 0, 2, 4];

function buildEvents() {
  events = [];
  CHORDS.forEach((c) => at(c[0], () => snd.setChord(c[1], c[2], c[3], c[0] === 0 ? 3 : 2)));

  const fr = rng(9);
  feats.forEach((f) => {
    const note = D_PENTA[Math.floor(fr() * D_PENTA.length)];
    at(f.appear, () => snd.pluck(note, 0.07, 2.2, { pan: f.x * 0.8, verb: 0.8 }));
  });
  parts.forEach((p) => {
    const note = HIGH[Math.floor(fr() * HIGH.length)];
    at(p.appear, () => snd.tink(note, p.hx * 0.9));
  });

  at(22, () => snd.whoosh(8, 200, 2400, 0.045));
  const rightNotes = [440, 523.25, 587.33, 659.25];
  GUESSES.forEach((g, i) => {
    at(g.t + 0.45, () => snd.pluck(220, 0.03, 0.4, { verb: 0.3 }));
    at(g.t + 1.8, () => (g.right ? snd.blip(110, 0.13) : snd.pluck(rightNotes[i - 1], 0.14, 1.8, { verb: 0.7 })));
    if (g.right) at(g.t + 2.6, () => snd.pluck(220, 0.05, 1.2));
  });

  at(49.5, () => snd.whoosh(5.5, 3200, 160, 0.06));
  at(55.2, () => {
    snd.boom(58.27, 0.32, 3.5);
    [233.08, 293.66, 349.23, 466.16].forEach((f, i) => snd.bell(f, 0.07, 5, { delay: i * 0.07 }));
  });
  PRINCIPLES.forEach((p, i) => {
    at(p.t, () => snd.pluck(PRINCIPLE_NOTES[i] / 2, 0.05, 1.5, { pan: personPos(i).x * 0.6 }));
    at(p.t + TRAVEL, () => snd.bell(PRINCIPLE_NOTES[i], 0.085, 4));
  });

  at(JUMP[0], () => snd.hop(294));
  at(JUMP[1], () => snd.land());
  for (let i = 0; i < HUMAN.length; i++) at(H_START + i * H_STEP, () => snd.click(0.32));
  at(H_SEND, () => {
    snd.click(0.4);
    snd.pluck(698.46, 0.05, 0.8);
  });
  [86.8, 87.5, 88.2].forEach((tt) => at(tt, () => snd.pluck(1396.91, 0.018, 0.5, { verb: 0.9 })));
  at(R_START, () => [349.23, 440, 523.25, 659.25, 880].forEach((f, i) => snd.pluck(f, 0.09, 2.6, { delay: i * 0.08 })));
  for (let i = 0; i < REPLY.length; i += 2) at(R_START + i * R_STEP, () => snd.click(0.14));

  at(104, () => snd.hop(349.23));
  rescheduleLanternSounds();

  at(130.5, () => snd.whoosh(4.5, 300, 1400, 0.03));
  at(132.5, () => snd.bell(1046.5, 0.05, 3));
  at(133, () => snd.bell(1174.66, 0.05, 3));
  at(145.5, () => [349.23, 440, 523.25, 783.99].forEach((f, i) => snd.bell(f, 0.065, 7, { delay: i * 0.12 })));
  at(154.5, () => {
    snd.pluck(698.46, 0.06, 3);
    snd.hop(392, 0.05);
  });
  at(154.95, () => snd.land(0.7));
  events.sort((a, b) => a.t - b.t);
}

function rescheduleLanternSounds() {
  events = events.filter((e) => e.tag !== 'lantern');
  lanterns.forEach((l, i) => {
    const note = LANTERN_NOTES[LANTERN_MELODY[i % LANTERN_MELODY.length]];
    at(l.appear, () => snd.pluck(note, 0.085, 2.4, { pan: clamp(l.x / halfW, -1, 1) * 0.7 }), 'lantern');
  });
  events.sort((a, b) => a.t - b.t);
}

// ---------- playback & UI ----------
const $ = (id) => document.getElementById(id);
const caption = $('caption');
const intro = $('intro');
const endCard = $('end');
const controls = $('controls');
const playBtn = $('play');
const muteBtn = $('mute');
const bar = $('bar');
const fill = $('fill');
const chapLabel = $('chap');

let t = 0, prevT = -0.001, playing = false, started = false, lastFrame = 0, capIdx = -2;

bar.innerHTML +=
  CHAPTERS.slice(1).map((c) => `<span class="tick" style="left:${(c.t / DURATION) * 100}%"></span>`).join('');

function fire(from, to) {
  for (const e of events) {
    if (e.t > to) break;
    if (e.t > from) e.fn();
  }
}

function seek(nt) {
  t = clamp(nt, 0, DURATION - 0.01);
  prevT = t;
  const c = chordAt(t);
  snd.setChord(c[1], c[2], c[3], 0.4);
  endCard.classList.add('hidden');
}

function setPlaying(p) {
  playing = p;
  playBtn.classList.toggle('paused', !p);
  playBtn.setAttribute('aria-label', p ? 'Pause' : 'Play');
  if (p) snd.resume();
  else snd.suspend();
  poke();
}

function begin() {
  snd.init();
  snd.setMuted(muted);
  snd.resume();
  intro.classList.add('hidden');
  controls.classList.remove('hidden');
  started = true;
  t = 0;
  prevT = -0.001;
  setPlaying(true);
}

function updateCaption() {
  let idx = -1;
  for (let i = 0; i < CAPTIONS.length; i++) if (t >= CAPTIONS[i][0] - 0.1 && t <= CAPTIONS[i][1]) idx = i;
  if (idx !== capIdx) {
    capIdx = idx;
    caption.textContent = idx >= 0 ? CAPTIONS[idx][2] : '';
  }
  if (idx >= 0) {
    const [a, b] = CAPTIONS[idx];
    caption.style.opacity = (seg(t, a, a + 0.7) * (1 - seg(t, b - 0.7, b))).toFixed(3);
    caption.style.transform = `translate(-50%, ${((1 - outCubic(seg(t, a, a + 0.9))) * 8).toFixed(1)}px)`;
  } else caption.style.opacity = '0';
}

function updateUI() {
  fill.style.width = `${(t / DURATION) * 100}%`;
  const ch = CHAPTERS.reduce((c, k) => (k.t <= t ? k : c), CHAPTERS[0]);
  const s = Math.floor(t);
  const label = `${ch.name} · ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  if (chapLabel.textContent !== label) chapLabel.textContent = label;
  bar.setAttribute('aria-valuenow', String(s));
}

function frame(now) {
  const dt = Math.min(0.1, (now - (lastFrame || now)) / 1000);
  lastFrame = now;
  if (playing) {
    t += dt;
    if (t >= DURATION) {
      t = DURATION;
      setPlaying(false);
      endCard.classList.remove('hidden');
    }
    fire(prevT, t);
    prevT = t;
  }
  draw(started ? t : 12.5);
  if (started) {
    updateCaption();
    updateUI();
  }
  requestAnimationFrame(frame);
}

// controls
let idleTimer = 0;
function poke() {
  controls.classList.remove('idle');
  clearTimeout(idleTimer);
  if (playing) idleTimer = setTimeout(() => controls.classList.add('idle'), 2600);
}
['mousemove', 'touchstart', 'keydown'].forEach((ev) => window.addEventListener(ev, poke, { passive: true }));

let muted = false;
try {
  muted = localStorage.getItem('origin-muted') === '1';
} catch {}
function setMuted(m) {
  muted = m;
  snd.setMuted(m);
  muteBtn.classList.toggle('muted', m);
  muteBtn.setAttribute('aria-label', m ? 'Unmute' : 'Mute');
  try {
    localStorage.setItem('origin-muted', m ? '1' : '0');
  } catch {}
}
setMuted(muted);

$('begin').addEventListener('click', begin);
$('again').addEventListener('click', () => {
  seek(0);
  prevT = -0.001;
  setPlaying(true);
});
playBtn.addEventListener('click', () => {
  if (t >= DURATION - 0.02) {
    seek(0);
    prevT = -0.001;
  }
  setPlaying(!playing);
});
muteBtn.addEventListener('click', () => setMuted(!muted));

let dragging = false;
function barSeek(e) {
  const r = bar.getBoundingClientRect();
  seek(((e.clientX - r.left) / r.width) * DURATION);
}
bar.addEventListener('pointerdown', (e) => {
  dragging = true;
  bar.setPointerCapture(e.pointerId);
  barSeek(e);
});
bar.addEventListener('pointermove', (e) => dragging && barSeek(e));
bar.addEventListener('pointerup', () => (dragging = false));

window.addEventListener('keydown', (e) => {
  if (!started) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      begin();
    }
    return;
  }
  if (e.target.closest && e.target.closest('button') && (e.key === ' ' || e.key === 'Enter')) return;
  if (e.key === ' ' || e.key === 'k') {
    e.preventDefault();
    playBtn.click();
  } else if (e.key === 'm') setMuted(!muted);
  else if (e.key === 'ArrowRight') seek(t + 5);
  else if (e.key === 'ArrowLeft') seek(t - 5);
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden && playing) setPlaying(false);
});

window.addEventListener('resize', resize);
resize();
buildEvents();
requestAnimationFrame(frame);
document.fonts?.ready.then(() => layoutLanterns());

// ?debug exposes playback for inspecting individual frames.
if (new URLSearchParams(location.search).has('debug')) {
  window.origin_debug = { seek, begin, setPlaying, time: () => t };
}
