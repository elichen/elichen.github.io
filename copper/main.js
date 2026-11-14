// Page: canvas, audio and keys. The pictures follow the audio clock.
import { renderSong, DURATION, PATTERN } from './music.js';
import { W, H, fb, init, render, renderTitle, PARTS } from './effects.js';

const canvas = document.getElementById('screen'), hint = document.getElementById('hint');
const g2 = canvas.getContext('2d');
const image = new ImageData(new Uint8ClampedArray(fb.buffer), W, H);

// whole device pixels per Amiga pixel, so every pixel is the same size
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const k = Math.max(1, Math.floor(Math.min(innerWidth * dpr / W, innerHeight * dpr / H)));
  const fit = Math.min(innerWidth / W, innerHeight / H);
  const s = k / dpr <= fit ? k / dpr : fit;
  canvas.style.width = W * s + 'px'; canvas.style.height = H * s + 'px';
}
addEventListener('resize', resize); resize();

const song = renderSong(44100);
init(song.events);

let ctx = null, gain = null, src = null, t0 = 0, muted = false;
const now = () => (((ctx.currentTime - t0) % DURATION) + DURATION) % DURATION;

function play(at) {
  if (src) src.stop();
  src = ctx.createBufferSource();
  src.buffer = ctx.song; src.loop = true;
  src.connect(gain); src.start(0, at);
  t0 = ctx.currentTime - at;
}
function start() {
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  ctx.song = ctx.createBuffer(2, song.L.length, song.rate);
  ctx.song.copyToChannel(song.L, 0); ctx.song.copyToChannel(song.R, 1);
  gain = ctx.createGain(); gain.connect(ctx.destination);
  play(0);
  hint.classList.add('on');
  setTimeout(() => hint.classList.remove('on'), 6000);
}
function toggle() {
  if (!ctx) return start();
  if (ctx.state === 'running') ctx.suspend(); else ctx.resume();
}
function skip(dir) {
  if (!ctx) return;
  const p = now() / PATTERN;
  let i = 0;
  PARTS.forEach((q, k) => { if (q[0] <= p) i = k; });
  // back: to the start of this part, or to the part before if this one has just started
  if (dir < 0 && p - PARTS[i][0] > 0.25) dir = 0;
  i = (i + dir + PARTS.length) % PARTS.length;
  play(PARTS[i][0] * PATTERN);
}

addEventListener('pointerdown', e => { if (e.target.tagName !== 'A') toggle(); });
addEventListener('keydown', e => {
  if (e.code === 'Space' || e.code === 'Enter') { e.preventDefault(); toggle(); }
  else if (e.code === 'ArrowRight') skip(1);
  else if (e.code === 'ArrowLeft') skip(-1);
  else if (e.code === 'KeyM' && gain) { muted = !muted; gain.gain.value = muted ? 0 : 1; }
  else if (e.code === 'KeyF') { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen?.(); }
});

function frame(ms) {
  if (ctx) render(now()); else renderTitle(ms / 1000);
  g2.putImageData(image, 0, 0);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// console: copper.show(seconds) draws one frame of the demo without the music
window.copper = { song, now: () => ctx ? now() : -1, show(t) { render(t); g2.putImageData(image, 0, 0); } };
