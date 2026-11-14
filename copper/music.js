// A small tracker. Every instrument is synthesized here (no sample files), the song is
// written with a few helpers, and the whole tune is mixed once into a stereo buffer.
// The demo then reads the audio clock, so pictures and sound cannot drift apart.

export const BPM = 138, SPEED = 6, BARS = 60, CHANNELS = 7;
export const TICK = 2.5 / BPM, ROW = TICK * SPEED, BEAT = ROW * 4, BAR = ROW * 16;
export const PATTERN = BAR * 4, DURATION = BAR * BARS;

const TAU = Math.PI * 2;
const TR = 16744;   // tonal samples: 64 samples per cycle at C4
const DR = 22050;   // drums

function noise(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2147483648) - 1;
}

// 8-bit samples, as Paula played them
function q8(d) {
  for (let i = 0; i < d.length; i++) d[i] = Math.round(Math.max(-1, Math.min(1, d[i])) * 127) / 127;
  return d;
}

function makeInstruments() {
  const I = {};

  { // lead: a pulse whose width sweeps, plus a slightly detuned saw; the whole sample loops
    const n = 8192, d = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const ph = (i % 64) / 64, pw = 0.5 + 0.36 * Math.sin(TAU * i / n);
      const pulse = (ph < pw ? 1 : -1) - (2 * pw - 1);
      const ph2 = ((i * 129 / 128) % 64) / 64;
      d[i] = 0.4 * pulse + 0.3 * (2 * ph2 - 1);
    }
    I.lead = { data: q8(d), rate: TR, ref: 261.63, loop: 0, vol: 0.30, decay: 0.991, vib: 1 };
  }
  { // arp: one cycle of a 25% pulse
    const d = new Float32Array(64);
    for (let i = 0; i < 64; i++) d[i] = i < 16 ? 0.9 : -0.3;
    I.arp = { data: q8(d), rate: TR, ref: 261.63, loop: 0, vol: 0.26, decay: 0.86 };
  }
  { // bass: saw + square through a low-pass filter that closes; the steady end loops
    const n = 256 * 40, d = new Float32Array(n);
    let y1 = 0, y2 = 0;
    for (let i = 0; i < n; i++) {
      const ph = (i % 256) / 256, t = i / TR;
      const x = 0.7 * (2 * ph - 1) + 0.5 * (ph < 0.5 ? 1 : -1);
      const a = 1 - Math.exp(-TAU * (480 + 3200 * Math.exp(-t * 14)) / TR);
      y1 += a * (x - y1); y2 += a * (y1 - y2);
      d[i] = y2 * 1.1;
    }
    I.bass = { data: q8(d), rate: TR, ref: 65.41, loop: n - 2048, vol: 0.52, decay: 0.975 };
  }
  { // kick: a sine that falls from 185 Hz to 45 Hz
    const n = (0.3 * DR) | 0, d = new Float32Array(n), r = noise(1);
    let ph = 0;
    for (let i = 0; i < n; i++) {
      const t = i / DR;
      ph += TAU * (45 + 140 * Math.exp(-t * 28)) / DR;
      d[i] = Math.tanh(2.2 * (Math.sin(ph) * Math.exp(-t * 8) + (t < 0.003 ? r() * 0.4 : 0)));
    }
    I.kick = { data: q8(d), rate: DR, loop: -1, vol: 0.95, decay: 1 };
  }
  { // snare
    const n = (0.22 * DR) | 0, d = new Float32Array(n), r = noise(2);
    let p = 0;
    for (let i = 0; i < n; i++) {
      const t = i / DR, x = r(), hp = x - 0.5 * p; p = x;
      d[i] = hp * 0.6 * Math.exp(-t * 16) + Math.sin(TAU * 190 * t) * 0.5 * Math.exp(-t * 24);
    }
    I.snare = { data: q8(d), rate: DR, loop: -1, vol: 0.55, decay: 1 };
  }
  const hat = (len, fall, seed) => {
    const n = (len * DR) | 0, d = new Float32Array(n), r = noise(seed);
    let p = 0;
    for (let i = 0; i < n; i++) { const x = r(), hp = (x - p) * 0.5; p = x; d[i] = hp * Math.exp(-i / DR * fall); }
    return q8(d);
  };
  I.hat = { data: hat(0.05, 90, 3), rate: DR, loop: -1, vol: 0.2, decay: 1 };
  I.ohat = { data: hat(0.25, 14, 4), rate: DR, loop: -1, vol: 0.2, decay: 1 };
  I.crash = { data: hat(1.6, 2.8, 5), rate: DR, loop: -1, vol: 0.3, decay: 1 };
  return I;
}

const SEMI = { c: 0, d: 2, e: 4, f: 5, g: 7, a: 9, b: 11 };
function midi(s) {
  const m = /^([a-g])(#?)(\d)$/.exec(s);
  if (!m) throw new Error('bad note ' + s);
  return SEMI[m[1]] + (m[2] ? 1 : 0) + 12 * (+m[3] + 1);
}

// Channels: 0 kick, 1 snare + hats, 2 bass, 3 arpeggio, 4 lead, 5 lead echo, 6 crash
function buildSong() {
  const rows = BARS * 16;
  const T = Array.from({ length: CHANNELS }, () => new Array(rows));
  const put = (c, r, cell) => { T[c][((r % rows) + rows) % rows] = cell; };

  // bass root, then the three chord notes for the arpeggio
  const CH = { Am: [33, 69, 72, 76], F: [29, 65, 69, 72], C: [36, 64, 67, 72], G: [31, 62, 67, 71], E: [28, 64, 68, 71] };
  const A1 = ['Am', 'F', 'C', 'G'], A2 = ['Am', 'F', 'G', 'E'], B1 = ['F', 'G', 'Am', 'Am'], B2 = ['F', 'G', 'C', 'E'];
  const prog = [A1, A2, A1, A2, A1, A2, B1, B2, A1, A2, A1, A2, B1, B2, A1];

  const bass = (bar, name, pat = 'L.LHL.LHL.LHL.LH', v = 1) => {
    const r = CH[name][0];
    for (let i = 0; i < 16; i++) {
      if (pat[i] === 'L') put(2, bar * 16 + i, { s: 'bass', n: r, v });
      else if (pat[i] === 'H') put(2, bar * 16 + i, { s: 'bass', n: r + 12, v: v * 0.8 });
    }
  };
  const arp = (bar, name, pat, v = 1, oct = 0) => {
    const [, a, b, c] = CH[name];
    for (let i = 0; i < 16; i++) {
      if (pat[i] === '.') continue;
      put(3, bar * 16 + i, { s: 'arp', n: a + oct, arp: [0, b - a, c - a], v: pat[i] === 'X' ? v : v * 0.6 });
    }
  };
  const lead = (bar, str, v = 1) => {
    const tk = str.split(' ');
    if (tk.length !== 16) throw new Error('lead bar needs 16 steps: ' + str);
    tk.forEach((s, i) => {
      if (s === '.') return;
      const r = bar * 16 + i, n = midi(s);
      put(4, r, { s: 'lead', n, v });
      put(5, r + 3, { s: 'lead', n, v: v * 0.4 });   // the echo is a second voice, three rows late
    });
  };
  const K = (r, v = 1) => put(0, r, { s: 'kick', v });
  const S = (r, v = 1) => put(1, r, { s: 'snare', v });
  const Hc = (r, v = 1) => put(1, r, { s: 'hat', v });
  const Ho = (r, v = 1) => put(1, r, { s: 'ohat', v });
  const drums = (bar, style) => {
    const r = bar * 16;
    if (style === 'hat') { for (let i = 0; i < 16; i += 2) Hc(r + i, i % 4 ? 1 : 0.6); return; }
    for (let i = 0; i < 16; i += 4) K(r + i);
    if (style === 'kick') { for (let i = 0; i < 16; i += 4) { Hc(r + i, 0.5); Ho(r + i + 2); } return; }
    for (let i = 0; i < 16; i++) {
      if (i % 8 === 4) S(r + i);
      else if (i % 4 === 2) Ho(r + i);
      else if (i % 2 === 0) Hc(r + i, 0.7);
      else if (style === 'busy') Hc(r + i, 0.4);
    }
  };
  const fill = bar => { const r = bar * 16; S(r + 12, 0.8); S(r + 13, 0.6); S(r + 14, 0.9); S(r + 15, 1); K(r + 14, 0.8); };

  for (let p = 0; p < 15; p++) for (let b = 0; b < 4; b++) {
    const bar = p * 4 + b, name = prog[p][b], r = bar * 16;
    if (p === 0) {
      arp(bar, name, 'X.x.X.x.X.x.X.x.', 0.9 + 0.2 * b);
      if (b >= 2) drums(bar, 'hat');
    } else if (p === 1) {
      arp(bar, name, 'X.xxX.x.X.xxX.x.', 0.9); bass(bar, name); drums(bar, 'kick');
      if (b === 3) for (let i = 8; i < 16; i++) S(r + i, 0.35 + 0.08 * (i - 8));
    } else if (p === 8) {             // breakdown
      arp(bar, name, 'X...x...X...x.x.', 1.3, 12); bass(bar, name, 'L...............', 0.9);
    } else if (p === 9) {             // build
      arp(bar, name, 'X.x.X.x.X.x.X.x.', 0.9); bass(bar, name, 'L.L.L.L.L.L.L.L.');
      if (b < 3) for (let i = 0; i < 16; i += 4) K(r + i);
      const step = b < 2 ? 4 : b === 2 ? 2 : 1;
      for (let i = 0; i < 16; i += step) S(r + i, b === 3 ? 0.4 + 0.04 * i : 0.55 + 0.1 * b);
    } else if (p === 14) {            // outro
      arp(bar, name, 'X..x..x.X..x..x.'); bass(bar, name); drums(bar, b < 2 ? 'full' : 'hat');
    } else {
      const bSection = p === 6 || p === 7 || p === 12 || p === 13;
      arp(bar, name, bSection ? 'X.xxX.xxX.xxX.xx' : 'X..x..x.X..x..x.', 1, p >= 10 ? 12 : 0);
      bass(bar, name); drums(bar, p >= 4 ? 'busy' : 'full');
      if (b === 3 && p % 2 === 1) fill(bar);
    }
  }
  for (const bar of [8, 16, 24, 32, 40, 48, 56]) put(6, bar * 16, { s: 'crash' });

  const LA = [
    'e5 . . e5 . . d5 . c5 . . d5 . . e5 .',
    'f5 . . f5 . . e5 . c5 . . a4 . . c5 .',
    'g5 . . g5 . . e5 . c5 . . e5 . . g5 .',
    'd5 . . d5 . . b4 . g4 . . b4 . . d5 .',
    'a5 . . g5 . . e5 . a5 . . g5 . . e5 .',
    'a5 . . g5 . . f5 . e5 . . f5 . . c5 .',
    'b5 . . a5 . . g5 . d5 . . g5 . . b5 .',
    'b5 . . . g#5 . . . e5 . g#5 . b5 . e6 .',
  ];
  const LB = [
    'a5 . . . . . c6 . . . a5 . f5 . a5 .',
    'b5 . . . . . d6 . . . b5 . g5 . b5 .',
    'c6 . . . . . . . . . . . b5 . c6 .',
    'e6 . d6 . c6 . b5 . a5 . b5 . c6 . e6 .',
    'f6 . . . . . e6 . . . c6 . a5 . c6 .',
    'g6 . . . . . d6 . . . b5 . g5 . b5 .',
    'e6 . . . c6 . . . g5 . . . c6 . e6 .',
    'e6 . . . . . . . g#5 . b5 . e6 . g#6 .',
  ];
  const LBREAK = [
    'e5 . . . . . . . . . . . d5 . c5 .',
    'a4 . . . . . . . . . . . c5 . f5 .',
    'g5 . . . . . . . . . . . e5 . c5 .',
    'd5 . . . . . . . g5 . . . b5 . d6 .',
  ];
  const LBUILD = [
    'a4 c5 e5 a5 c5 e5 a5 c6 a4 c5 e5 a5 c5 e5 a5 c6',
    'a4 c5 f5 a5 c5 f5 a5 c6 a4 c5 f5 a5 c5 f5 a5 c6',
    'b4 d5 g5 b5 d5 g5 b5 d6 b4 d5 g5 b5 d5 g5 b5 d6',
    'b4 e5 g#5 b5 e5 g#5 b5 e6 e5 g#5 b5 e6 g#5 b5 e6 g#6',
  ];
  for (let i = 0; i < 8; i++) {
    lead(8 + i, LA[i]); lead(16 + i, LA[i]); lead(24 + i, LB[i]);
    lead(40 + i, LA[i]); lead(48 + i, LB[i]);
  }
  for (let i = 0; i < 4; i++) { lead(32 + i, LBREAK[i], 0.9); lead(36 + i, LBUILD[i], 0.75); lead(56 + i, LA[i]); }
  put(4, 8, { off: 1 }); put(5, 11, { off: 1 });   // the last note rings into the intro, then stops
  return T;
}

// Returns { L, R, events }. events: every note start as { t, ch, s, v }, for the pictures.
export function renderSong(rate = 44100) {
  const I = makeInstruments(), T = buildSong();
  const total = Math.round(DURATION * rate);
  const L = new Float32Array(total), R = new Float32Array(total);
  const pans = [0.5, 0.58, 0.5, 0.3, 0.64, 0.28, 0.5];
  const chans = pans.map(p => ({
    pl: Math.cos(p * Math.PI / 2), pr: Math.sin(p * Math.PI / 2),
    smp: null, pos: 0, vol: 0, g: 0, note: 0, age: 0, arp: null,
  }));
  const events = [];
  const nTicks = BARS * 16 * SPEED, spt = rate * TICK;

  for (let k = 0; k < nTicks; k++) {
    if (k % SPEED === 0) {
      const row = k / SPEED;
      for (let c = 0; c < CHANNELS; c++) {
        const cell = T[c][row], ch = chans[c];
        if (!cell) continue;
        if (cell.off) { ch.smp = null; continue; }
        const v = cell.v ?? 1;
        ch.smp = I[cell.s]; ch.pos = 0; ch.note = cell.n; ch.age = 0; ch.arp = cell.arp || null;
        ch.vol = v * ch.smp.vol;
        events.push({ t: k * TICK, ch: c, s: cell.s, v });
      }
    }
    const a = Math.round(k * spt), b = Math.min(total, Math.round((k + 1) * spt));
    for (const ch of chans) {
      const s = ch.smp;
      if (!s) { ch.g = 0; continue; }
      let step = s.rate / rate;
      if (s.ref) {
        let n = ch.note;
        if (ch.arp) n += ch.arp[ch.age % ch.arp.length];
        if (s.vib && ch.age > 16) n += 0.2 * Math.sin((ch.age - 16) * 0.68);
        step *= 440 * Math.pow(2, (n - 69) / 12) / s.ref;
      }
      const d = s.data, len = d.length, ls = s.loop, tg = ch.vol, pl = ch.pl, pr = ch.pr;
      let pos = ch.pos, g = ch.g;
      for (let i = a; i < b; i++) {
        if (pos >= len) {
          if (ls < 0) { ch.smp = null; break; }
          pos = ls + (pos - len) % (len - ls);
        }
        g += (tg - g) * 0.02;
        const x = d[pos | 0] * g;
        L[i] += x * pl; R[i] += x * pr;
        pos += step;
      }
      ch.pos = pos; ch.g = g; ch.vol *= s.decay; ch.age++;
    }
  }

  // output filter (the A500 had one) and a soft limiter
  const a = 1 - Math.exp(-TAU * 9000 / rate);
  let yl = 0, yr = 0, peak = 0;
  for (let i = 0; i < total; i++) {
    yl += a * (L[i] - yl); yr += a * (R[i] - yr);
    L[i] = yl; R[i] = yr;
    peak = Math.max(peak, Math.abs(yl), Math.abs(yr));
  }
  const drive = 1.7 / peak;
  for (let i = 0; i < total; i++) {
    L[i] = Math.tanh(L[i] * drive) * 0.9;
    R[i] = Math.tanh(R[i] * drive) * 0.9;
  }
  return { L, R, rate, events };
}
