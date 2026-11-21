// Crisp HiDPI canvas charts for Grokking, Live.
// Styled in Distill's clean white scientific aesthetic.

const P = 59;
const NFREQ = 29;
const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
const SANS = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';

export const COLORS = {
  train: '#2b303a',
  test: '#c0392b',
  restricted: '#1f5aa6',
  excluded: '#b86b14',
  wnorm: '#6b5b8a',
  mutedBar: '#d9dde3',
  activeBar: '#1f5aa6',
  otherBar: '#7b8794',
  grid: 'rgba(0, 0, 0, 0.075)',
  axis: 'rgba(0, 0, 0, 0.22)',
  text2: 'rgba(0, 0, 0, 0.60)',
  text3: 'rgba(0, 0, 0, 0.42)',
  cursor: 'rgba(31, 90, 166, 0.75)',
  wavePalette: ['#1f5aa6', '#c0392b', '#1b8a5a', '#b86b14', '#6b5b8a'],
};

function setupCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(10, Math.round(rect.width));
  const h = Math.max(10, Math.round(rect.height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function niceMaxEpoch(maxEp) {
  if (maxEp <= 500) return 500;
  if (maxEp <= 1000) return 1000;
  if (maxEp <= 2000) return 2000;
  if (maxEp <= 3000) return 3000;
  return Math.ceil(maxEp / 1000) * 1000;
}

// ---------- 1. Main Training Dynamics Chart ----------
// mode: 'acc' (Train/Test accuracy + Weight norm) or 'loss' (Train/Test/Restricted/Excluded log-loss)
export function drawTrainingChart(canvas, history, { mode = 'acc', inspectedEpoch = null, wdChanges = [] }) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  const pad = { l: 46, r: mode === 'acc' ? 46 : 16, t: 22, b: 28 };
  const pw = Math.max(10, w - pad.l - pad.r);
  const ph = Math.max(10, h - pad.t - pad.b);

  const lastEp = history.length ? history[history.length - 1].epoch : 0;
  const maxEp = Math.max(3000, niceMaxEpoch(lastEp));
  const xOf = (ep) => pad.l + (ep / maxEp) * pw;

  // Store layout for pointer scrubbing
  canvas._plot = { pad, pw, ph, maxEp };

  // Horizontal grid + Y labels
  ctx.font = `11px ${MONO}`;
  ctx.textBaseline = 'middle';

  let yOf;
  if (mode === 'acc') {
    yOf = (v) => pad.t + ph * (1 - Math.max(0, Math.min(1, v)));
    for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
      const y = yOf(frac);
      ctx.strokeStyle = frac === 0 ? COLORS.axis : COLORS.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + pw, y);
      ctx.stroke();

      ctx.fillStyle = COLORS.text3;
      ctx.textAlign = 'right';
      ctx.fillText(`${Math.round(frac * 100)}%`, pad.l - 7, y);
    }
  } else {
    // Log scale from 1e-4 up to 30 (or 100 when high-loss runs like wd=0 exceed 25)
    let maxLoss = 20;
    for (const pt of history) {
      if (pt.teLoss > maxLoss) maxLoss = pt.teLoss;
      if (pt.excLoss > maxLoss) maxLoss = pt.excLoss;
    }
    const topVal = maxLoss > 25 ? 100 : 30;
    const logMin = -4;
    const logMax = Math.log10(topVal);
    yOf = (v) => {
      const lv = Math.log10(Math.max(1e-4, v));
      const t = (lv - logMin) / (logMax - logMin);
      return pad.t + ph * (1 - Math.max(0, Math.min(1, t)));
    };
    const ticks = [
      { v: 1e-4, label: '10⁻⁴' },
      { v: 1e-2, label: '10⁻²' },
      { v: 1e-1, label: '0.1' },
      { v: 1, label: '1' },
      { v: 10, label: '10' },
    ];
    if (topVal >= 100) ticks.push({ v: 100, label: '100' });
    for (const tk of ticks) {
      const y = yOf(tk.v);
      ctx.strokeStyle = tk.v === 1e-4 ? COLORS.axis : COLORS.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + pw, y);
      ctx.stroke();

      ctx.fillStyle = COLORS.text3;
      ctx.textAlign = 'right';
      ctx.fillText(tk.label, pad.l - 7, y);
    }
  }

  // X ticks
  const xStep = maxEp <= 1000 ? 250 : maxEp <= 3000 ? 500 : 1000;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let ep = 0; ep <= maxEp; ep += xStep) {
    const x = xOf(ep);
    ctx.strokeStyle = COLORS.grid;
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + ph);
    ctx.stroke();
    ctx.fillStyle = COLORS.text3;
    ctx.fillText(ep === 0 ? '0' : `${ep}`, x, pad.t + ph + 7);
  }

  // Mark live weight-decay change events
  for (const ev of wdChanges) {
    if (ev.epoch <= maxEp) {
      const x = xOf(ev.epoch);
      ctx.save();
      ctx.strokeStyle = 'rgba(0,0,0,0.28)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, pad.t + ph);
      ctx.stroke();
      ctx.fillStyle = COLORS.text2;
      ctx.font = `10px ${MONO}`;
      ctx.textAlign = 'left';
      ctx.fillText(`wd=${ev.wd.toFixed(1)}`, x + 4, pad.t + 4);
      ctx.restore();
    }
  }

  if (history.length >= 2) {
    const drawLine = (accessor, color, width = 2, dash = []) => {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i < history.length; i++) {
        const x = xOf(history[i].epoch);
        const y = yOf(accessor(history[i]));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    };

    if (mode === 'acc') {
      // Right axis for weight norm
      let maxW = 45;
      for (const pt of history) if (pt.wnorm > maxW) maxW = Math.ceil(pt.wnorm / 10) * 10;
      const yW = (v) => pad.t + ph * (1 - Math.max(0, Math.min(1, v / maxW)));

      // Right axis labels
      ctx.fillStyle = COLORS.wnorm;
      ctx.font = `10px ${MONO}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      for (const frac of [0, 0.5, 1]) {
        const val = Math.round(maxW * frac);
        ctx.fillText(`${val}`, pad.l + pw + 7, yW(val));
      }
      ctx.save();
      ctx.fillStyle = COLORS.wnorm;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText('‖W‖₂', pad.l + pw + 7, pad.t - 7);
      ctx.restore();

      // Weight norm line
      ctx.save();
      ctx.strokeStyle = COLORS.wnorm;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      for (let i = 0; i < history.length; i++) {
        const x = xOf(history[i].epoch);
        const y = yW(history[i].wnorm);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();

      drawLine((p) => p.trAcc, COLORS.train, 2);
      drawLine((p) => p.teAcc, COLORS.test, 2.4);
    } else {
      drawLine((p) => p.excLoss, COLORS.excluded, 1.8, [4, 3]);
      drawLine((p) => p.resLoss, COLORS.restricted, 2.2);
      drawLine((p) => p.trLoss, COLORS.train, 1.8);
      drawLine((p) => p.teLoss, COLORS.test, 2.2);
    }
  }

  // Inspected / scrubbed epoch vertical cursor
  if (inspectedEpoch !== null && history.length > 0) {
    const x = xOf(inspectedEpoch);
    ctx.save();
    ctx.strokeStyle = COLORS.cursor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + ph);
    ctx.stroke();
    ctx.fillStyle = COLORS.cursor;
    ctx.beginPath();
    ctx.arc(x, pad.t + 3, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

// ---------- 2. Interactive Fourier Spectrum of W_E ----------
export function drawSpectrumChart(canvas, power, { top5K = [], keepMask = (1 << NFREQ) - 1, selectedFreq = 1, hoveredK = 0 }) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  const pad = { l: 38, r: 10, t: 14, b: 28 };
  const pw = Math.max(10, w - pad.l - pad.r);
  const ph = Math.max(10, h - pad.t - pad.b);

  let maxP = 0.25;
  if (power) {
    for (let i = 0; i < NFREQ; i++) if (power[i] > maxP) maxP = power[i];
  }
  maxP = Math.min(1, Math.ceil(maxP * 10) / 10);

  // Horizontal grid
  ctx.font = `10px ${MONO}`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (const frac of [0, 0.5, 1]) {
    const y = pad.t + ph * (1 - frac);
    ctx.strokeStyle = frac === 0 ? COLORS.axis : COLORS.grid;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + pw, y);
    ctx.stroke();
    ctx.fillStyle = COLORS.text3;
    ctx.fillText(`${Math.round(maxP * frac * 100)}%`, pad.l - 6, y);
  }

  const slotW = pw / NFREQ;
  const barW = Math.max(4, slotW - 3);
  canvas._bars = { pad, pw, ph, slotW };

  const top5Set = new Set(top5K);

  for (let i = 0; i < NFREQ; i++) {
    const k = i + 1;
    const val = power ? power[i] : 0;
    const bh = Math.max(1.5, (val / maxP) * ph);
    const x = pad.l + i * slotW + (slotW - barW) * 0.5;
    const y = pad.t + ph - bh;
    const kept = (keepMask & (1 << i)) !== 0;
    const isTop = top5Set.has(k);
    const isSel = k === selectedFreq;
    const isHov = k === hoveredK;

    if (isHov) {
      ctx.fillStyle = 'rgba(31, 90, 166, 0.08)';
      ctx.fillRect(pad.l + i * slotW, pad.t, slotW, ph);
    }

    if (!kept) {
      // Muted frequency: pale outline + diagonal strike
      ctx.fillStyle = 'rgba(192, 57, 43, 0.12)';
      ctx.fillRect(x, y, barW, bh);
      ctx.strokeStyle = 'rgba(192, 57, 43, 0.55)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, barW - 1, Math.max(1, bh - 1));
    } else if (isTop) {
      const rank = top5K.indexOf(k);
      ctx.fillStyle = COLORS.wavePalette[rank % COLORS.wavePalette.length];
      ctx.fillRect(x, y, barW, bh);
    } else {
      ctx.fillStyle = COLORS.otherBar;
      ctx.globalAlpha = 0.45;
      ctx.fillRect(x, y, barW, bh);
      ctx.globalAlpha = 1;
    }

    // Indicator dot for currently viewed circle frequency
    if (isSel) {
      ctx.fillStyle = COLORS.text2;
      ctx.beginPath();
      ctx.arc(x + barW * 0.5, pad.t + ph + 21, 2.3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Frequency label (every 2 or top5)
    if (k === 1 || k % 5 === 0 || isTop || isHov) {
      ctx.fillStyle = !kept ? COLORS.test : isTop ? '#111' : COLORS.text3;
      ctx.font = `${isTop ? '600 ' : ''}10px ${MONO}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`${k}`, x + barW * 0.5, pad.t + ph + 6);
    }
  }
}

// ---------- 3. 2D Clock Circle Embedding Plot ----------
export function drawCircleChart(canvas, coords, k, { highlightA = -1, highlightB = -1, highlightC = -1 } = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  const cx = w * 0.5;
  const cy = h * 0.5;
  const radius = Math.min(w, h) * 0.37;

  // Faint reference unit circle and crosshair
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx - radius - 8, cy);
  ctx.lineTo(cx + radius + 8, cy);
  ctx.moveTo(cx, cy - radius - 8);
  ctx.lineTo(cx, cy + radius + 8);
  ctx.stroke();

  if (!coords || coords.length < P * 2) return;

  const px = (t) => cx + coords[t * 2] * radius;
  const py = (t) => cy - coords[t * 2 + 1] * radius;

  // Connect tokens in cyclic frequency order: step s -> t = (s * inv_k) mod P so
  // when frequency k crystallizes, the connecting polygon becomes a clean 59-gon!
  // Compute modular inverse of k mod P
  let invK = 1;
  for (let m = 1; m < P; m++) {
    if ((m * k) % P === 1) { invK = m; break; }
  }
  ctx.strokeStyle = 'rgba(31, 90, 166, 0.22)';
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  for (let step = 0; step <= P; step++) {
    const t = ((step % P) * invK) % P;
    const x = px(t);
    const y = py(t);
    if (step === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw all 59 token dots colored by their modular angle (k * t mod P) / P
  for (let t = 0; t < P; t++) {
    const x = px(t);
    const y = py(t);
    const hue = (((k * t) % P) / P) * 320 + 190; // rich cyclic palette
    const isHi = t === highlightA || t === highlightB || t === highlightC;
    const r = isHi ? 5.2 : 3.2;

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = isHi ? '#111111' : `hsl(${hue % 360}, 62%, 44%)`;
    ctx.fill();

    if (isHi) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = t === highlightC ? COLORS.test : COLORS.restricted;
      ctx.stroke();
    }
  }

  // Collision-aware labels: place highlighted {c*, a, b} first (nudging inward if adjacent),
  // then skip background landmarks that would collide with any placed label.
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const placed = []; // { x, y }
  const collides = (x, y, minDist) =>
    placed.some((p) => Math.hypot(p.x - x, p.y - y) < minDist);

  const labelPos = (t, offset) => {
    const x = px(t);
    const y = py(t);
    const dx = x - cx;
    const dy = y - cy;
    const len = Math.hypot(dx, dy) || 1;
    return { x: x + (dx / len) * offset, y: y + (dy / len) * offset };
  };

  // Deduplicated priority list for highlighted tokens
  const hiTokens = [];
  for (const t of [highlightC, highlightA, highlightB]) {
    if (t >= 0 && !hiTokens.includes(t)) hiTokens.push(t);
  }

  for (const t of hiTokens) {
    const tag = t === highlightC ? `${t}*` : `${t}`;
    // Try outward first; if colliding with another highlighted label, try inward then further outward
    const candidates = [labelPos(t, 14), labelPos(t, -14), labelPos(t, 24)];
    let best = candidates[0];
    for (const c of candidates) {
      if (!collides(c.x, c.y, 20)) {
        best = c;
        break;
      }
    }
    placed.push(best);
    ctx.fillStyle = t === highlightC ? COLORS.test : '#000';
    ctx.font = `700 10.5px ${MONO}`;
    ctx.fillText(tag, best.x, best.y);
  }

  ctx.fillStyle = COLORS.text2;
  ctx.font = `10px ${MONO}`;
  for (let step = 0; step < P; step += 6) {
    const t = (step * invK) % P;
    if (hiTokens.includes(t)) continue;
    const pos = labelPos(t, 12);
    if (collides(pos.x, pos.y, 20)) continue;
    placed.push(pos);
    ctx.fillText(`${t}`, pos.x, pos.y);
  }
}

// ---------- 4. Constructive Wave Interference Probe Chart ----------
export function drawProbeChart(canvas, probe, targetC, keepMask) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  if (!probe) return;

  const pad = { l: 54, r: 14, t: 10, b: 24 };
  const pw = Math.max(10, w - pad.l - pad.r);
  const ph = Math.max(10, h - pad.t - pad.b);

  // Top 62% = 5 top Fourier waves + 1 residual ('other') lane; Bottom 38% = full centered output logits
  const wavesH = ph * 0.62;
  const sumTop = pad.t + wavesH + 10;
  const sumH = Math.max(30, ph - wavesH - 10);
  const slotW = pw / P;
  const xCenter = (c) => pad.l + (c + 0.5) * slotW;

  // Vertical guide band at the true answer c*
  const tx = pad.l + targetC * slotW;
  ctx.fillStyle = 'rgba(192, 57, 43, 0.08)';
  ctx.fillRect(tx, pad.t, slotW, ph);
  ctx.strokeStyle = 'rgba(192, 57, 43, 0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(tx + slotW * 0.5, pad.t);
  ctx.lineTo(tx + slotW * 0.5, pad.t + ph);
  ctx.stroke();

  // Shared wave scale across the 5 top waves and the residual ('other') lane
  let waveMax = 1e-3;
  for (const wv of probe.waves) {
    for (let c = 0; c < P; c++) {
      const a = Math.abs(wv[c]);
      if (a > waveMax) waveMax = a;
    }
  }
  if (probe.resid) {
    for (let c = 0; c < P; c++) {
      const a = Math.abs(probe.resid[c]);
      if (a > waveMax) waveMax = a;
    }
  }

  // Draw 5 individual Fourier waves + 6th 'other' (remaining 24 frequencies) lane
  const laneH = wavesH / 6;
  for (let f = 0; f < 6; f++) {
    const isResid = f === 5;
    const k = isResid ? 0 : probe.top5K[f];
    const kept = isResid ? true : (keepMask & (1 << (k - 1))) !== 0;
    const midY = pad.t + (f + 0.5) * laneH;
    const amp = laneH * 0.43;

    // Lane baseline
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, midY);
    ctx.lineTo(pad.l + pw, midY);
    ctx.stroke();

    // Label on left
    ctx.fillStyle = isResid ? COLORS.text3 : kept ? COLORS.wavePalette[f] : COLORS.text3;
    ctx.font = `10.5px ${MONO}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(isResid ? 'other' : `k=${k}${kept ? '' : ' ×'}`, pad.l - 8, midY);

    // Wave polyline
    const wv = isResid ? probe.resid : probe.waves[f];
    if (!wv) continue;
    ctx.strokeStyle = isResid ? 'rgba(43, 48, 58, 0.42)' : kept ? COLORS.wavePalette[f] : COLORS.mutedBar;
    ctx.lineWidth = isResid ? 1.15 : kept ? 1.5 : 1;
    ctx.beginPath();
    for (let c = 0; c < P; c++) {
      const x = xCenter(c);
      const y = midY - (wv[c] / waveMax) * amp;
      if (c === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Dot at true answer c*
    const ty = midY - (wv[targetC] / waveMax) * amp;
    ctx.fillStyle = isResid ? 'rgba(43, 48, 58, 0.55)' : kept ? COLORS.wavePalette[f] : COLORS.mutedBar;
    ctx.beginPath();
    ctx.arc(xCenter(targetC), ty, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  // Divider + label for bottom full logits
  ctx.fillStyle = COLORS.text2;
  ctx.font = `600 10px ${SANS}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('Logits', pad.l - 8, sumTop + sumH * 0.45);
  ctx.fillStyle = COLORS.text3;
  ctx.font = `10px ${MONO}`;
  ctx.fillText('(all)', pad.l - 8, sumTop + sumH * 0.72);

  // Combined centered logits bars
  let meanL = 0;
  for (let c = 0; c < P; c++) meanL += probe.logits[c];
  meanL /= P;
  let maxL = 1e-3;
  let predC = 0;
  for (let c = 0; c < P; c++) {
    if (probe.logits[c] > probe.logits[predC]) predC = c;
    const a = Math.abs(probe.logits[c] - meanL);
    if (a > maxL) maxL = a;
  }

  const zeroY = sumTop + sumH * 0.72;
  const posScale = sumH * 0.68 / maxL;
  const negScale = sumH * 0.25 / maxL;

  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, zeroY);
  ctx.lineTo(pad.l + pw, zeroY);
  ctx.stroke();

  const barW = Math.max(2, slotW - 1.5);
  for (let c = 0; c < P; c++) {
    const v = probe.logits[c] - meanL;
    const x = pad.l + c * slotW + (slotW - barW) * 0.5;
    const bh = v >= 0 ? v * posScale : -v * negScale;
    const y = v >= 0 ? zeroY - bh : zeroY;
    if (c === predC) {
      ctx.fillStyle = predC === targetC ? COLORS.test : '#b86b14';
    } else if (c === targetC) {
      ctx.fillStyle = 'rgba(192, 57, 43, 0.45)';
    } else {
      ctx.fillStyle = 'rgba(43, 48, 58, 0.35)';
    }
    ctx.fillRect(x, y, barW, Math.max(1, bh));
  }

  // X-axis residue labels
  ctx.font = `10px ${MONO}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let c = 0; c < P; c++) {
    if (c === 0 || c % 10 === 0 || c === targetC || c === predC) {
      ctx.fillStyle = c === targetC ? COLORS.test : c === predC ? '#b86b14' : COLORS.text3;
      ctx.font = `${(c === targetC || c === predC) ? '700 ' : ''}10px ${MONO}`;
      ctx.fillText(`${c}`, xCenter(c), pad.t + ph + 5);
    }
  }
}

// ---------- 5. Static Article Comparison Chart (data/experiments.json) ----------
export function drawComparisonChart(canvas, expData) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  if (!expData || !expData.runs) return;

  const pad = { l: 46, r: 16, t: 14, b: 28 };
  const pw = Math.max(10, w - pad.l - pad.r);
  const ph = Math.max(10, h - pad.t - pad.b);
  const maxEp = 3000;
  const xOf = (ep) => pad.l + (ep / maxEp) * pw;
  const yOf = (v) => pad.t + ph * (1 - Math.max(0, Math.min(1, v)));

  // Grid
  ctx.font = `11px ${MONO}`;
  ctx.textBaseline = 'middle';
  for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
    const y = yOf(frac);
    ctx.strokeStyle = frac === 0 ? COLORS.axis : COLORS.grid;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + pw, y);
    ctx.stroke();
    ctx.fillStyle = COLORS.text3;
    ctx.textAlign = 'right';
    ctx.fillText(`${Math.round(frac * 100)}%`, pad.l - 7, y);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let ep = 0; ep <= maxEp; ep += 500) {
    const x = xOf(ep);
    ctx.strokeStyle = COLORS.grid;
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + ph);
    ctx.stroke();
    ctx.fillStyle = COLORS.text3;
    ctx.fillText(`${ep}`, x, pad.t + ph + 7);
  }

  // Delayed decay vertical marker at ep=800
  const x800 = xOf(800);
  ctx.save();
  ctx.strokeStyle = 'rgba(31, 90, 166, 0.35)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(x800, pad.t);
  ctx.lineTo(x800, pad.t + ph);
  ctx.stroke();
  ctx.fillStyle = COLORS.restricted;
  ctx.font = `10px ${MONO}`;
  ctx.textAlign = 'left';
  ctx.fillText('wd 0→2.2', x800 + 4, pad.t + 6);
  ctx.restore();

  const specs = [
    { key: 'default', color: COLORS.test, width: 2.4, dash: [] },
    { key: 'delayedDecay', color: COLORS.restricted, width: 2.0, dash: [] },
    { key: 'noDecay', color: COLORS.train, width: 1.8, dash: [5, 3] },
    { key: 'lowData', color: COLORS.excluded, width: 1.8, dash: [2, 3] },
  ];

  for (const sp of specs) {
    const run = expData.runs[sp.key];
    if (!run) continue;
    ctx.save();
    ctx.strokeStyle = sp.color;
    ctx.lineWidth = sp.width;
    ctx.lineJoin = 'round';
    ctx.setLineDash(sp.dash);
    ctx.beginPath();
    for (let i = 0; i < run.series.length; i++) {
      const pt = run.series[i];
      const x = xOf(pt.ep);
      const y = yOf(pt.teAcc);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }
}
