// Renders frames in Node and writes a contact sheet (BMP), plus the tune's levels.
//   node tools/sheet.mjs out.bmp [seconds into each pattern = 3.2]      (title screen is the last tile)
import { writeFileSync } from 'node:fs';
import { renderSong, PATTERN } from '../music.js';
import { W, H, fb, init, render, renderTitle } from '../effects.js';

const out = process.argv[2] || 'sheet.bmp', at = +(process.argv[3] ?? 3.2);
const song = renderSong(44100);
init(song.events);
let peak = 0, sq = 0;
for (const v of song.L) { peak = Math.max(peak, Math.abs(v)); sq += v * v; }
console.log(`tune: ${(song.L.length / song.rate).toFixed(1)} s, ${song.events.length} notes, peak ${peak.toFixed(2)}, rms ${Math.sqrt(sq / song.L.length).toFixed(2)}`);

const COLS = 4, ROWS = 4, SW = COLS * W, SH = ROWS * H;
const px = Buffer.alloc(SW * SH * 3);
let ms = 0;
for (let k = 0; k < 16; k++) {
  const t0 = performance.now();
  if (k < 15) render(k * PATTERN + at); else renderTitle(1);
  ms = Math.max(ms, performance.now() - t0);
  const ox = (k % COLS) * W, oy = ((k / COLS) | 0) * H;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = fb[y * W + x], o = ((SH - 1 - oy - y) * SW + ox + x) * 3;
    px[o] = (c >> 16) & 255; px[o + 1] = (c >> 8) & 255; px[o + 2] = c & 255;
  }
}
const head = Buffer.alloc(54);
head.write('BM'); head.writeUInt32LE(54 + px.length, 2); head.writeUInt32LE(54, 10);
head.writeUInt32LE(40, 14); head.writeInt32LE(SW, 18); head.writeInt32LE(SH, 22);
head.writeUInt16LE(1, 26); head.writeUInt16LE(24, 28); head.writeUInt32LE(px.length, 34);
writeFileSync(out, Buffer.concat([head, px]));
console.log(`${out}: 15 patterns at +${at} s and the title; slowest frame ${ms.toFixed(1)} ms`);
