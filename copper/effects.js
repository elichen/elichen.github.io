// Every effect draws into one 320 x 256 frame buffer, one pixel at a time, as a function of
// the song time. Nothing here touches the DOM, so tools/sheet.mjs can render frames in Node.

import { BEAT, BAR, PATTERN, DURATION } from './music.js';

export const W = 320, H = 256;
export const fb = new Uint32Array(W * H);   // 0xAABBGGRR

const TAU = Math.PI * 2, sin = Math.sin, cos = Math.cos;
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const smooth = x => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };

// ---------------------------------------------------------------- colour
const rgb = (r, g, b) =>
  (0xff000000 | ((b < 0 ? 0 : b > 255 ? 255 : b) << 16) | ((g < 0 ? 0 : g > 255 ? 255 : g) << 8) | (r < 0 ? 0 : r > 255 ? 255 : r)) >>> 0;
const shade = (c, k) => rgb((c & 255) * k, ((c >> 8) & 255) * k, ((c >> 16) & 255) * k);
function hsv(h, s, v) {
  h = ((h % 1) + 1) % 1 * 6;
  const i = h | 0, f = h - i, p = v * (1 - s), q = v * (1 - s * f), u = v * (1 - s * (1 - f));
  const [r, g, b] = i === 0 ? [v, u, p] : i === 1 ? [q, v, p] : i === 2 ? [p, v, u] : i === 3 ? [p, q, v] : i === 4 ? [u, p, v] : [v, p, q];
  return rgb(r * 255, g * 255, b * 255);
}
// stops: [position 0..1, r, g, b]
function ramp(stops, n = 256) {
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const p = n > 1 ? i / (n - 1) : 0;
    let k = 0;
    while (k < stops.length - 2 && p > stops[k + 1][0]) k++;
    const a = stops[k], b = stops[k + 1], f = clamp((p - a[0]) / (b[0] - a[0]), 0, 1);
    out[i] = rgb(a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f);
  }
  return out;
}
function vgrad(pal, y0 = 0, y1 = H) {
  for (let y = y0; y < y1; y++) fb.fill(pal[((y - y0) * (pal.length - 1) / (y1 - y0 - 1)) | 0], y * W, y * W + W);
}
function rect(x, y, w, h, col, rowBase = 0) {
  const x0 = Math.max(0, x), x1 = Math.min(W, x + w);
  if (x1 <= x0) return;
  for (let j = 0; j < h; j++) {
    const yy = y + j;
    if (yy < 0 || yy >= H) continue;
    fb.fill(typeof col === 'number' ? col : col[rowBase + j], yy * W + x0, yy * W + x1);
  }
}
let seed = 7;
const rnd = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;

// ---------------------------------------------------------------- font (5 x 7, one base-32 digit per row)
const FONT = {
  A: 'ehhvhhh', B: 'uhhuhhu', C: 'ehggghe', D: 'uhhhhhu', E: 'vgguggv', F: 'vgguggg', G: 'ehgnhhf',
  H: 'hhhvhhh', I: 'e44444e', J: '72222ic', K: 'hikokih', L: 'ggggggv', M: 'hrllhhh', N: 'hpljhhh',
  O: 'ehhhhhe', P: 'uhhuggg', Q: 'ehhhlid', R: 'uhhukih', S: 'fgge11u', T: 'v444444', U: 'hhhhhhe',
  V: 'hhhhha4', W: 'hhhllrh', X: 'hha4ahh', Y: 'hha4444', Z: 'v1248gv',
  0: 'ehjlphe', 1: '4c4444e', 2: 'eh168gv', 3: 'u11e11u', 4: '26aiv22', 5: 'vgu11he', 6: '68guhhe',
  7: 'v124888', 8: 'ehhehhe', 9: 'ehhf12c',
  '.': '00000cc', ',': '0000c48', '!': '4444404', '?': 'eh12404', '-': '000v000', "'": '4480000',
  ':': '0cc0cc0', '+': '044v440', '/': '11248gg', '*': '0level0',
};
const GL = {};
for (const k in FONT) GL[k] = [...FONT[k]].map(c => parseInt(c, 32));

function text(s, x, y, sc, col) {
  for (const ch of s) {
    const g = GL[ch];
    if (g) for (let r = 0; r < 7; r++) for (let c = 0; c < 5; c++)
      if (g[r] & (16 >> c)) rect(x + c * sc, y + r * sc, sc, sc, col, r * sc);
    x += 6 * sc;
  }
}
const tw = (s, sc) => s.length * 6 * sc - sc;
const BLACK = rgb(0, 0, 0);
function ctext(s, y, sc, col, sh = 1) {
  const x = (W - tw(s, sc)) >> 1;
  if (sh) text(s, x + sh, y + sh, sc, BLACK);
  text(s, x, y, sc, col);
}

// a big text sprite with a gradient, a bevel and a shadow
function makeLogo(lines, sc, pal) {
  const cw = 6 * sc, w = Math.max(...lines.map(l => l.length)) * cw + 4, h = lines.length * 8 * sc + 4;
  const m = new Uint8Array(w * h);
  lines.forEach((l, li) => {
    const x0 = (w - 4 - (l.length * cw - sc)) >> 1;
    [...l].forEach((ch, ci) => {
      const g = GL[ch];
      if (!g) return;
      for (let r = 0; r < 7; r++) for (let c = 0; c < 5; c++) if (g[r] & (16 >> c))
        for (let yy = 0; yy < sc; yy++) for (let xx = 0; xx < sc; xx++)
          m[(li * 8 * sc + r * sc + yy) * w + x0 + ci * cw + c * sc + xx] = 1 + r * sc + yy;
    });
  });
  const data = new Uint32Array(w * h);
  const at = (x, y) => x >= 0 && y >= 0 && x < w && y < h ? m[y * w + x] : 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = m[y * w + x];
    if (v) {
      let c = pal[v - 1];
      if (!at(x - 1, y) || !at(x, y - 1)) c = rgb((c & 255) + 90, ((c >> 8) & 255) + 90, ((c >> 16) & 255) + 90);
      else if (!at(x + 1, y) || !at(x, y + 1)) c = shade(c, 0.55);
      data[y * w + x] = c;
    } else if (at(x - 1, y - 1) || at(x - 2, y - 2) || at(x - 3, y - 3)) data[y * w + x] = rgb(16, 0, 40);
  }
  return { w, h, data };
}
// sprite with 0 = transparent; rowDx(row) shifts a row, rows outside [r0, r1) are skipped
function blit(s, x, y, rowDx, r0 = 0, r1 = s.h) {
  for (let j = Math.max(0, r0); j < Math.min(s.h, r1); j++) {
    const yy = y + j;
    if (yy < 0 || yy >= H) continue;
    const xs = x + (rowDx ? rowDx(j) | 0 : 0);
    for (let i = 0; i < s.w; i++) {
      const c = s.data[j * s.w + i], xx = xs + i;
      if (c && xx >= 0 && xx < W) fb[yy * W + xx] = c;
    }
  }
}

// ---------------------------------------------------------------- sync with the music
let EV = [], evi = 0, prevT = -1, lastSn = -99, gt = 0;
const lastT = new Float64Array(8).fill(-99), lastV = new Float32Array(8);
let kick = 0, snare = 0;
const vu = new Float32Array(7);
function sync(time) {
  if (time < prevT) { evi = 0; lastT.fill(-99); lastSn = -99; }
  prevT = time;
  while (evi < EV.length && EV[evi].t <= time) {
    const e = EV[evi++];
    lastT[e.ch] = e.t; lastV[e.ch] = Math.min(1, e.v);
    if (e.s === 'snare') lastSn = e.t;
  }
  kick = Math.exp(-(time - lastT[0]) * 9);
  snare = Math.exp(-(time - lastSn) * 10);
  for (let c = 0; c < 7; c++) vu[c] = lastV[c] * Math.exp(-(time - lastT[c]) * 3.5);
}

// ---------------------------------------------------------------- building blocks
const STARS = Array.from({ length: 130 }, (_, i) => ({ x: rnd() * W, y: (rnd() * H) | 0, l: i % 3 }));
function stars(t, k = 1, y1 = H) {
  for (const s of STARS) {
    if (s.y >= y1) continue;
    let x = (s.x - t * [14, 34, 70][s.l]) % W;
    if (x < 0) x += W;
    const v = [80, 150, 255][s.l] * k;
    fb[s.y * W + (x | 0)] = rgb(v, v, v);
  }
}
function copperBar(yc, col, hh = 8) {
  const r = col & 255, g = (col >> 8) & 255, b = (col >> 16) & 255;
  for (let d = -hh; d <= hh; d++) {
    const y = (yc + d) | 0;
    if (y < 0 || y >= H) continue;
    const k = 1 - Math.abs(d) / (hh + 1), w = k * k * k * 150;
    fb.fill(rgb(r * k + w, g * k + w, b * k + w), y * W, y * W + W);
  }
}
function bars(t, n, yc, amp, spd = 2) {
  const list = [];
  for (let i = 0; i < n; i++) {
    const ph = t * spd + i * 0.42;
    list.push({ y: yc + sin(ph) * amp, z: cos(ph), c: hsv(i / n + t * 0.05, 0.85, 1) });
  }
  list.sort((a, b) => a.z - b.z);
  for (const b of list) copperBar(b.y, b.c);
}

// one message per scroller part; each scrolls in and all the way out before its part ends
const SCROLL1 = 'ELI TYPED ONE LINE AND WENT TO GET COFFEE ... WHEN THE CUP WAS FULL THERE WERE STARS ON THE SCREEN ...';
const SCROLL2 = 'YOU NEVER RUN OUT OF IDEAS. I NEVER GET TIRED.';
const CHROME28 = ramp([[0, 30, 70, 255], [0.47, 255, 255, 255], [0.5, 80, 0, 90], [1, 255, 150, 220]], 28);
function scroller(msg, len, yc, amp, t, pal = CHROME28, sc = 4) {
  const cw = 6 * sc, off = t * (W + msg.length * cw) / (len - 0.25) - W;
  for (let x = 0; x < W; x++) {
    const p = x + off;
    if (p < 0) continue;
    const g = GL[msg[Math.floor(p / cw)]];
    if (!g) continue;
    const col = ((p % cw) / sc) | 0;
    if (col > 4) continue;
    const m = 16 >> col;
    const y0 = (yc + sin(x * 0.022 + t * 3.2) * amp + sin(x * 0.009 - t * 1.7) * amp * 0.4) | 0;
    for (let r = 0; r < 7; r++) if (g[r] & m) for (let j = 0; j < sc; j++) {
      const y = y0 + r * sc + j;
      if (y < 0 || y >= H) continue;
      if (x + 3 < W && y + 3 < H) fb[(y + 3) * W + x + 3] = BLACK;
      fb[y * W + x] = pal[r * sc + j];
    }
  }
}
function meters(x0, y1) {
  for (let i = 0; i < 7; i++) {
    const lit = Math.round(Math.sqrt(vu[i]) * 12);
    for (let s = 0; s < 12; s++) {
      const k = s < lit ? 1 : 0.16;
      const c = s < 7 ? rgb(0, 220 * k, 90 * k) : s < 10 ? rgb(240 * k, 220 * k, 0) : rgb(255 * k, 50 * k, 50 * k);
      rect(x0 + i * 14, y1 - s * 5, 10, 3, c);
    }
  }
}
// additive flat triangle
function tri(ax, ay, bx, by, cx, cy, col) {
  let t;
  if (ay > by) { t = ax; ax = bx; bx = t; t = ay; ay = by; by = t; }
  if (by > cy) { t = bx; bx = cx; cx = t; t = by; by = cy; cy = t; }
  if (ay > by) { t = ax; ax = bx; bx = t; t = ay; ay = by; by = t; }
  const cr = col & 255, cg = (col >> 8) & 255, cb = (col >> 16) & 255;
  const y0 = Math.max(0, Math.ceil(ay)), y1 = Math.min(H, Math.ceil(cy));
  for (let y = y0; y < y1; y++) {
    let xa = ax + (cx - ax) * (y - ay) / (cy - ay);
    let xb = y < by ? ax + (bx - ax) * (y - ay) / (by - ay) : bx + (cx - bx) * (y - by) / ((cy - by) || 1);
    if (xa > xb) { t = xa; xa = xb; xb = t; }
    const e = y * W + Math.min(W, Math.ceil(xb));
    for (let i = y * W + Math.max(0, Math.ceil(xa)); i < e; i++) {
      const d = fb[i];
      const r = (d & 255) + cr, g = ((d >> 8) & 255) + cg, b = ((d >> 16) & 255) + cb;
      fb[i] = 0xff000000 | ((b > 255 ? 255 : b) << 16) | ((g > 255 ? 255 : g) << 8) | (r > 255 ? 255 : r);
    }
  }
}
function rot(v, a, b, c) {
  let [x, y, z] = v, s, k, t;
  s = sin(a); k = cos(a); t = x * k + z * s; z = -x * s + z * k; x = t;
  s = sin(b); k = cos(b); t = y * k - z * s; z = y * s + z * k; y = t;
  s = sin(c); k = cos(c); t = x * k - y * s; y = x * s + y * k; x = t;
  return [x, y, z];
}

// ---------------------------------------------------------------- tables, textures, sprites
const LOGO = makeLogo(['COPPER', 'DREAMS'], 6,
  ramp([[0, 0, 60, 200], [0.45, 190, 240, 255], [0.5, 60, 0, 90], [1, 255, 120, 210]], 42));
const SHOUT = makeLogo(['YES!'], 6,
  ramp([[0, 255, 255, 255], [0.45, 255, 230, 120], [0.5, 200, 0, 110], [1, 255, 120, 210]], 42));
const SKY = ramp([[0, 0, 0, 24], [0.55, 8, 8, 80], [1, 70, 0, 110]]);
const NIGHT = ramp([[0, 0, 0, 12], [1, 50, 0, 80]]);
const WHITE7 = ramp([[0, 255, 255, 255], [1, 150, 190, 255]], 7);
const CHROME14 = ramp([[0, 30, 70, 255], [0.47, 255, 255, 255], [0.5, 80, 0, 90], [1, 255, 150, 220]], 14);
const CHROME21 = ramp([[0, 30, 70, 255], [0.47, 255, 255, 255], [0.5, 80, 0, 90], [1, 255, 150, 220]], 21);

const PLASMA = ramp([[0, 0, 0, 70], [0.2, 110, 0, 200], [0.4, 255, 50, 150], [0.55, 255, 235, 150],
  [0.75, 0, 200, 220], [1, 0, 0, 70]]);

const TEX1 = new Uint32Array(65536), TEX2 = new Uint32Array(65536);
{
  const cool = ramp([[0, 0, 0, 50], [0.4, 0, 70, 200], [0.75, 0, 210, 210], [1, 255, 255, 255]]);
  const warm = ramp([[0, 40, 0, 50], [0.4, 170, 0, 150], [0.75, 255, 90, 160], [1, 255, 240, 200]]);
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) {
    const v = (((x >> 3) ^ (y >> 3)) * 8 + 4) & 255;
    TEX1[y << 8 | x] = (((x >> 6) + (y >> 6)) & 1 ? warm : cool)[v];
    const k = 0.5 + 0.5 * sin(x * TAU / 32) * sin(y * TAU / 32);
    const seam = x % 32 === 0 || y % 32 === 0;
    TEX2[y << 8 | x] = seam ? rgb(255, 255, 255)
      : ((x >> 5) + (y >> 5)) & 1 ? rgb(20 + 110 * k, 170 * k + 30, 200 * k + 40) : rgb(150 * k + 50, 30 + 40 * k, 220 * k + 30);
  }
}

const TW = 640, TH = 512;
const tunU = new Uint8Array(TW * TH), tunA = new Uint8Array(TW * TH), tunS = new Uint8Array(TW * TH), ringT = new Uint8Array(TW * TH);
for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) {
  const dx = x - 320, dy = y - 256, r = Math.sqrt(dx * dx + dy * dy), i = y * TW + x;
  tunU[i] = (12000 / (r + 2)) & 255;
  tunA[i] = (Math.atan2(dy, dx) / TAU * 512) & 255;
  tunS[i] = clamp(r * 2.2 - 10, 0, 255);
  ringT[i] = ((r / 9) | 0) & 1;
}
const MOIRE = [
  ramp([[0, 6, 6, 40], [1, 40, 0, 80]], H), ramp([[0, 0, 190, 210], [1, 0, 90, 230]], H),
  ramp([[0, 240, 40, 140], [1, 140, 20, 210]], H), ramp([[0, 255, 255, 255], [1, 255, 215, 240]], H),
];

function ballSprite(r, base) {
  const s = 2 * r + 1, data = new Uint32Array(s * s);
  for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) {
    const dx = (x - r) / r, dy = (y - r) / r, d2 = dx * dx + dy * dy;
    if (d2 > 1) continue;
    const nz = Math.sqrt(1 - d2);
    const k = 0.16 + 0.84 * clamp(-dx * 0.45 - dy * 0.55 + nz * 0.7, 0, 1);
    const sp = 255 * Math.pow(clamp(-dx * 0.36 - dy * 0.46 + nz * 0.81, 0, 1), 30);
    data[y * s + x] = rgb(base[0] * k + sp, base[1] * k + sp, base[2] * k + sp);
  }
  return { w: s, h: s, data };
}
const BALLS = [[60, 130, 255], [255, 50, 140]].map(c => Array.from({ length: 16 }, (_, r) => r ? ballSprite(r, c) : null));
const NB = 96;
const SHAPES = [[], [], [], []];
for (let i = 0; i < NB; i++) {
  const y = 1 - 2 * (i + 0.5) / NB, r = Math.sqrt(1 - y * y), a = i * 2.39996;
  SHAPES[0].push([cos(a) * r * 1.7, y * 1.7, sin(a) * r * 1.7]);
  SHAPES[1].push([((i % 4) - 1.5) * 0.8, (((i >> 2) % 4) - 1.5) * 0.8, ((i >> 4) - 2.5) * 0.62]);
  const u = (i % 12) / 12 * TAU, v = ((i / 12) | 0) / 8 * TAU;
  SHAPES[2].push([(1.45 + 0.6 * cos(v)) * cos(u), 0.6 * sin(v), (1.45 + 0.6 * cos(v)) * sin(u)]);
  const h = i * 0.33 + (i & 1) * Math.PI;
  SHAPES[3].push([cos(h) * 1.15, (i / (NB - 1) - 0.5) * 3.7, sin(h) * 1.15]);
}

// glenz: a cube with a pyramid on each face, 24 triangles in two colours
const GV = [], GT = [];
for (let i = 0; i < 8; i++) GV.push([i & 4 ? 0.8 : -0.8, i & 2 ? 0.8 : -0.8, i & 1 ? 0.8 : -0.8]);
for (let a = 0; a < 3; a++) for (const s of [-1, 1]) {
  const ctr = [0, 0, 0]; ctr[a] = s * 1.3;
  const ci = GV.push(ctr) - 1, b = (a + 1) % 3, c = (a + 2) % 3;
  const ring = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([p, q]) => {
    const v = [0, 0, 0]; v[a] = s; v[b] = p; v[c] = q;
    return (v[0] > 0 ? 4 : 0) + (v[1] > 0 ? 2 : 0) + (v[2] > 0 ? 1 : 0);
  });
  for (let i = 0; i < 4; i++) GT.push([ci, ring[i], ring[(i + 1) % 4], i & 1]);
}

// ---------------------------------------------------------------- the parts
function pIntro(t) {
  fb.fill(BLACK);
  stars(gt, smooth(t / 3));
  const lines = [[0.4, 'ELI + CLAUDE', 2, 92], [1.5, 'PRESENT', 2, 116],
    [2.9, 'IT IS 3 AM.', 1, 156], [4.3, 'NOBODY WANTS TO STOP.', 1, 170]];
  for (const [t0, s, sc, y] of lines) {
    if (t < t0) continue;
    const n = Math.min(s.length, ((t - t0) * 24) | 0), x = (W - tw(s, sc)) >> 1;
    text(s.slice(0, n), x, y, sc, sc === 2 ? CHROME14 : WHITE7);
    if (n < s.length) rect(x + n * 6 * sc, y, 5 * sc, 7 * sc, rgb(255, 136, 0));
  }
}

function pLogo(t) {
  vgrad(NIGHT);
  stars(gt, 1);
  bars(t, 9, 128, 96 * smooth(t / 2), 1.6);
  const settle = Math.max(0, 1 - t / 1.5);
  const open = t * 70;
  blit(LOGO, (W - LOGO.w) >> 1, 76 - (kick * 4 | 0), j => sin(j * 0.12 + t * 9) * settle * 36,
    LOGO.h / 2 - open, LOGO.h / 2 + open);
}

function pScroller(t, len) {
  vgrad(SKY);
  stars(gt, 1);
  bars(t, 8, 112, 66);
  scroller(SCROLL1, len, 98, 24, t);
  meters(111, 246);
  ctext('COPPER DREAMS', 8, 1, WHITE7);
}

const sx = new Float32Array(W);
function pPlasma(t) {
  const cxm = 160 + sin(t * 0.9) * 90, cym = 128 + cos(t * 1.2) * 70, cyc = t * 34 + kick * 18;
  for (let x = 0; x < W; x++) sx[x] = sin(x * 0.031 - t * 1.4);
  for (let y = 0, i = 0; y < H; y++) {
    const sy = sin(y * 0.043 + t * 1.1), dy = (y - cym) * (y - cym);
    for (let x = 0; x < W; x++, i++) {
      const v = sx[x] + sy + sin((x + y) * 0.021 + t * 0.7) + sin(Math.sqrt((x - cxm) * (x - cxm) + dy) * 0.045 - t * 2.2);
      fb[i] = PLASMA[(v * 40 + cyc) & 255];
    }
  }
}

function pRoto(t) {
  const a = t * 0.7 + sin(t * 0.6) * 1.2, z = 0.95 + 0.7 * sin(t * 0.8 + 1) - kick * 0.12;
  const du = (cos(a) * z * 256) | 0, dv = (sin(a) * z * 256) | 0;
  const ox = ((128 + sin(t * 0.5) * 200) * 256) | 0, oy = (t * 60 * 256) | 0;
  for (let y = 0, i = 0; y < H; y++) {
    let u = -160 * du - (y - 128) * dv + ox, v = -160 * dv + (y - 128) * du + oy;
    for (let x = 0; x < W; x++, i++) {
      fb[i] = TEX1[((v >> 8) & 255) << 8 | ((u >> 8) & 255)];
      u += du; v += dv;
    }
  }
}

function pTunnel(t) {
  const ox = 160 + ((sin(t * 0.9) * 120) | 0), oy = 128 + ((cos(t * 1.3) * 90) | 0);
  const du = (t * 110) | 0, dv = (t * 40 + sin(t) * 40) | 0, boost = 256 + ((kick * 90) | 0);
  for (let y = 0; y < H; y++) {
    let j = (y + oy) * TW + ox, i = y * W;
    for (let x = 0; x < W; x++, i++, j++) {
      const c = TEX2[(((tunU[j] + du) & 255) << 8) | ((tunA[j] + dv) & 255)], s = (tunS[j] * boost) >> 8;
      const r = (c & 255) * s >> 8, g = (c >> 8 & 255) * s >> 8, b = (c >> 16 & 255) * s >> 8;
      fb[i] = 0xff000000 | ((b > 255 ? 255 : b) << 16) | ((g > 255 ? 255 : g) << 8) | (r > 255 ? 255 : r);
    }
  }
}

function pMoire(t) {
  const x1 = 160 + ((sin(t * 1.1) * 100) | 0), y1 = 128 + ((cos(t * 0.8) * 80) | 0);
  const x2 = 160 + ((cos(t * 0.7 + 1) * 110) | 0), y2 = 128 + ((sin(t * 1.3) * 70) | 0);
  const flip = kick > 0.6 ? 3 : 0;
  for (let y = 0, i = 0; y < H; y++) {
    let j1 = (y + y1) * TW + x1, j2 = (y + y2) * TW + x2;
    for (let x = 0; x < W; x++, i++, j1++, j2++) fb[i] = MOIRE[(ringT[j1] | ringT[j2] << 1) ^ flip][y];
  }
}

const order = Array.from({ length: NB }, (_, i) => i), bp = Array.from({ length: NB }, () => [0, 0, 0]);
function pBalls(t) {
  vgrad(NIGHT);
  stars(gt * 0.4, 0.7);
  const seg = t / BAR, k = Math.min(3, seg | 0), f = smooth((seg - k - 0.68) / 0.32);
  const A = SHAPES[k], B = SHAPES[(k + 1) % 4], grow = smooth(t / 0.7);
  for (let i = 0; i < NB; i++) {
    const p = [A[i][0] + (B[i][0] - A[i][0]) * f, A[i][1] + (B[i][1] - A[i][1]) * f, A[i][2] + (B[i][2] - A[i][2]) * f];
    bp[i] = rot(p, t * 1.1, t * 0.7 + 0.4, t * 0.3);
  }
  order.sort((a, b) => bp[b][2] - bp[a][2]);
  for (const i of order) {
    const z = bp[i][2] + 5, s = BALLS[i & 1][clamp(Math.round(52 * grow / z), 1, 15)];
    blit(s, Math.round(160 + bp[i][0] * grow * 250 / z) - (s.w >> 1), Math.round(128 + bp[i][1] * grow * 250 / z) - (s.w >> 1));
  }
}

// the four colours of Workbench 1.3
const WB = [rgb(0, 85, 170), rgb(255, 255, 255), rgb(40, 40, 60), rgb(255, 136, 0)];
function pTwister(t, len) {
  const q = t / len, R = 60 + kick * 8;
  for (let y = 0; y < H; y++) {
    fb.fill(hsv(0.62 + y * 0.0012 + q * 0.3, 0.8, 0.13 + 0.09 * sin(y * 0.2 - t * 6) + snare * 0.2), y * W, y * W + W);
    const a = t * (1.2 + 2.6 * q) + sin(y * 0.012 + t * 1.1) * (0.8 + 2.4 * q) + sin(y * 0.031 - t * 2) * 0.5 * q;
    const cx = 160 + sin(y * 0.017 + t * 1.6) * 34 * q;
    const stripe = ((y + ((t * 50) | 0)) >> 3) & 1 ? 1 : 0.8;
    for (let k = 0; k < 4; k++) {
      const x0 = cx + sin(a + k * TAU / 4) * R, x1 = cx + sin(a + (k + 1) * TAU / 4) * R;
      if (x1 < x0 + 0.5) continue;
      const lum = clamp((x1 - x0) / (R * 1.4142), 0, 1) * stripe;
      const xs = Math.max(0, x0 | 0), xe = Math.min(W, x1 | 0);
      fb.fill(shade(WB[k], 0.25 + 0.75 * lum), y * W + xs, y * W + xe);
      if (xe > xs) fb[y * W + xs] = shade(WB[k], 0.2);
    }
  }
  if (q > 0.75 && snare > 0.35) ctext('HERE IT COMES', 14, 3, CHROME21, 3);
}

const HOR = 150;
const GSKY = ramp([[0, 0, 0, 30], [0.6, 20, 0, 90], [1, 150, 20, 130]]);
function pGlenz(t) {
  vgrad(GSKY, 0, HOR);
  stars(gt * 0.3, 0.8, HOR - 30);
  // checkerboard floor
  for (let y = HOR; y < H; y++) {
    const d = y - HOR + 1, z = 320 / d, fog = clamp(d / 46, 0, 1), dw = z / 200;
    const zz = Math.floor(z + t * 5);
    const ca = rgb(150 + (235 - 150) * fog, 20 + (235 - 20) * fog, 130 + (255 - 130) * fog);
    const cb = rgb(150 + (0 - 150) * fog, 20 + (50 - 20) * fog, 130 + (160 - 130) * fog);
    let wx = -160 * dw + 1000.5 + sin(t * 0.5) * 2;
    for (let x = 0, i = y * W; x < W; x++, i++, wx += dw) fb[i] = ((wx | 0) + zz) & 1 ? ca : cb;
  }
  const ph = (gt / BEAT) % 1, hgt = 4 * ph * (1 - ph);
  const C = [sin(t * 0.8) * 1.6, 0.2 - hgt * 1.2, 6];
  // shadow
  const shx = 160 + C[0] * 200 / 6, shy = HOR + 1.6 * 200 / 6, k = 1 - 0.35 * hgt, rx = 46 * k, ry = 9 * k;
  for (let y = Math.ceil(shy - ry); y <= shy + ry && y < H; y++) {
    const half = rx * Math.sqrt(Math.max(0, 1 - ((y - shy) / ry) ** 2));
    for (let x = Math.max(0, (shx - half) | 0), i = y * W + x; x < Math.min(W, shx + half); x++, i++)
      fb[i] = 0xff000000 | ((fb[i] >> 1) & 0x7f7f7f);
  }
  const P = GV.map(v => {
    const r = rot(v, t * 1.3, t * 0.9, t * 0.5), x = r[0] + C[0], y = r[1] + C[1], z = r[2] + C[2];
    return [x, y, z, 160 + x * 200 / z, HOR + y * 200 / z];
  });
  for (let pass = 0; pass < 2; pass++) for (const [ia, ib, ic, par] of GT) {
    const a = P[ia], b = P[ib], c = P[ic];
    let nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
    let ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
    let nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const mx = (a[0] + b[0] + c[0]) / 3, my = (a[1] + b[1] + c[1]) / 3, mz = (a[2] + b[2] + c[2]) / 3;
    if (nx * (mx - C[0]) + ny * (my - C[1]) + nz * (mz - C[2]) < 0) { nx = -nx; ny = -ny; nz = -nz; }
    const front = nx * mx + ny * my + nz * mz < 0;
    if (front !== (pass === 1)) continue;
    const lit = 0.3 + 0.7 * Math.abs(nz) / Math.sqrt(nx * nx + ny * ny + nz * nz);
    const col = !front ? rgb(10 * lit, 25 * lit, 90 * lit)
      : par ? rgb(190 * lit, 10 * lit, 70 * lit) : rgb(120 * lit, 125 * lit, 150 * lit);
    tri(a[3], a[4], b[3], b[4], c[3], c[4], col);
  }
  if (t < BAR) blit(SHOUT, (W - SHOUT.w) >> 1, 8 + ((kick * 6) | 0));
}

const kline = new Uint32Array(W), KP = Array.from({ length: 16 }, (_, j) => sin(Math.PI * (j + 0.5) / 16));
function pKefrens(t, len) {
  kline.fill(rgb(0, 0, 20));
  for (let y = 0; y < H; y++) {
    const x = (152 + sin(y * 0.021 + t * 2.3) * 74 + sin(y * 0.047 - t * 3.1) * 46 + sin(t * 0.9) * 20) | 0;
    const c = hsv(y * 0.0026 + t * 0.21, 0.78, 1), r = c & 255, g = (c >> 8) & 255, b = (c >> 16) & 255;
    for (let j = 0; j < 16; j++) {
      const xx = x + j, k = KP[j], w = k * k * k * k * 110;
      if (xx >= 0 && xx < W) kline[xx] = rgb(r * k + w, g * k + w, b * k + w);
    }
    fb.set(kline, y * W);
  }
  scroller(SCROLL2, len, 112, 16, t);
}

const GREETS = ['HANDS UP', 'CARBON CREW', 'SILICON CREW', 'SAME DREAM'];
function pDots(t) {
  vgrad(NIGHT);
  const speed = 5, gap = 0.4, cam = t * speed;
  const path = z => [sin(z * 0.21) * 1.6 + sin(z * 0.13) * 1.0, cos(z * 0.17) * 1.3];
  const [px, py] = path(cam), rad = 1.5 + kick * 0.3;
  for (let i = 69; i >= 0; i--) {
    const zr = (i + 1) * gap - (cam % gap), zw = zr + cam, [qx, qy] = path(zw);
    const k = clamp(1.3 - zr / 26, 0, 1), col = hsv(zw * 0.035, 0.6, k), big = zr < 7;
    for (let d = 0; d < 36; d++) {
      const a = d / 36 * TAU + zw * 0.15 + t * 0.5;
      const x = (160 + (qx - px + cos(a) * rad) * 150 / zr) | 0, y = (128 + (qy - py + sin(a) * rad) * 150 / zr) | 0;
      if (big) rect(x, y, 2, 2, col); else if (x >= 0 && x < W && y >= 0 && y < H) fb[y * W + x] = col;
    }
  }
  if (t >= PATTERN) {
    const u = (t - PATTERN) / BAR, b = Math.min(3, u | 0), f = u - b;
    const y = 116 + ((1 - smooth(f * 6)) * 150 | 0) - ((smooth((f - 0.88) * 8) * 170) | 0);
    for (let yy = Math.max(0, y - 8); yy < Math.min(H, y + 31); yy++)      // a dark band behind the words
      for (let i = yy * W; i < yy * W + W; i++) fb[i] = 0xff000000 | ((fb[i] >> 2) & 0x3f3f3f);
    ctext(GREETS[b], y, 3, CHROME21, 3);
  }
}

const WATER = 142;
function pOutro(t) {
  vgrad(SKY, 0, WATER);
  stars(gt * 0.3, 1, WATER);
  blit(LOGO, (W - LOGO.w) >> 1, 6);
  if (t > 0.8) ctext('WHAT SHALL WE MAKE NEXT?', 112, 1, WHITE7);
  if (t > 1.6) ctext('ELI + CLAUDE - 2026', 126, 1, WHITE7);
  for (let y = WATER; y < H; y++) {
    const d = y - WATER, src = WATER - 1 - d, dx = (sin(d * 0.33 + t * 3) * (1 + d * 0.07)) | 0;
    for (let x = 0, i = y * W; x < W; x++, i++) {
      const c = fb[src * W + clamp(x + dx, 0, W - 1)];
      fb[i] = rgb((c & 255) * 0.45, ((c >> 8) & 255) * 0.5 + 10, ((c >> 16) & 255) * 0.6 + 40);
    }
  }
}

// [first pattern, patterns, effect]
export const PARTS = [
  [0, 1, pIntro], [1, 1, pLogo], [2, 2, pScroller], [4, 1, pPlasma], [5, 1, pRoto], [6, 1, pTunnel],
  [7, 1, pMoire], [8, 1, pBalls], [9, 1, pTwister], [10, 1, pGlenz], [11, 1, pKefrens], [12, 2, pDots], [14, 1, pOutro],
];

export function init(events) { EV = events; }

// 12 bits per pixel, with an optional flash to white and fade to black
function present(flash, fade) {
  const n = W * H;
  if (flash < 0.01 && fade > 0.99) {
    for (let i = 0; i < n; i++) { const c = fb[i] & 0xf0f0f0; fb[i] = 0xff000000 | c | (c >> 4); }
    return;
  }
  const k = ((1 - flash) * fade * 256) | 0, a = (flash * fade * 255) | 0;
  for (let i = 0; i < n; i++) {
    const d = fb[i];
    const c = ((((d & 255) * k >> 8) + a) | ((((d >> 8) & 255) * k >> 8) + a) << 8 | ((((d >> 16) & 255) * k >> 8) + a) << 16) & 0xf0f0f0;
    fb[i] = 0xff000000 | c | (c >> 4);
  }
}

export function render(time) {
  gt = time = ((time % DURATION) + DURATION) % DURATION;
  sync(time);
  const p = Math.min(14, (time / PATTERN) | 0);
  let part = PARTS[0];
  for (const q of PARTS) if (q[0] <= p) part = q;
  const t = time - part[0] * PATTERN, len = part[1] * PATTERN;
  part[2](t, len);
  const flash = part[0] >= 2 ? 0.5 * Math.exp(-t * 9) : 0;
  const fade = part[0] === 14 ? clamp((len - t) / 1.3, 0, 1) : part[0] === 0 ? 1 : 1;
  present(flash, fade);
}

// before the first click: no music yet
export function renderTitle(t) {
  gt = t; kick = 0;
  vgrad(NIGHT);
  stars(t, 1);
  blit(LOGO, (W - LOGO.w) >> 1, 58);
  if ((t * 1.6) % 1 < 0.65) ctext('CLICK TO START', 186, 2, CHROME14, 2);
  ctext('TURN IT UP', 216, 1, WHITE7);
  present(0, 1);
}
