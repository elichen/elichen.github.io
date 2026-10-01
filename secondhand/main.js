(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const css = getComputedStyle(document.documentElement);
  const C = (name) => css.getPropertyValue(name).trim();
  const COLOR = {
    surface: C("--surface"), rule: C("--rule"), ditto: C("--ditto"), haze: C("--haze"),
    pupil: C("--pupil"), tick: C("--tick"), muted: C("--muted")
  };
  const PROMPT = $("prompt-text").textContent;

  const pad = $("pad"), sheet = $("sheet");
  const padCtx = pad.getContext("2d"), sheetCtx = sheet.getContext("2d");

  let spacing = 0.2, vocab = null, calib = null, hands = [];
  let ready = false;

  // visitor's strokes, in pad CSS pixels
  let strokes = [], drawing = null, idleTimer = 0;
  // what the network is primed with: { kind, line: {pts, lift}, xh (px), label }
  let prime = null, handIndex = -1;

  // sheet state
  let job = 0, seed = 1, lines = [], renderXh = 22, playing = false;

  const worker = new Worker("worker.js?h=1");
  worker.onerror = (e) => {
    console.error("worker:", e.message);
    $("sheet-status").textContent = "The network stopped with an error.";
  };

  // ---------- canvas sizing ----------
  function fit(canvas, ctx, height) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.height = height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return w;
  }

  // squared paper with darker baselines every `every` squares
  function grid(ctx, w, h, size, baseline0, every) {
    ctx.fillStyle = COLOR.surface;
    ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLOR.rule;
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    const off = baseline0 % size;
    for (let y = off; y < h; y += size) { ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(w, Math.round(y) + 0.5); }
    for (let x = (w % size) / 2; x < w; x += size) { ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, h); }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    for (let y = baseline0; y < h; y += size * every) { ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(w, Math.round(y) + 0.5); }
    ctx.stroke();
  }

  // ---------- the pad ----------
  let padW = 0, padH = 168;
  const padGrid = () => padH / 7;
  const padBaseline = () => padGrid() * 4.6;

  function drawPad() {
    padH = pad.clientWidth < 600 ? 140 : 168;
    padW = fit(pad, padCtx, padH);
    grid(padCtx, padW, padH, padGrid(), padBaseline(), 100);
    padCtx.lineCap = "round"; padCtx.lineJoin = "round";
    if (prime && prime.kind === "hand" && !strokes.length) {
      drawLine(padCtx, prime.line.pts, prime.line.lift, 18, padBaseline(), padGrid() * 0.95, COLOR.pupil, false);
    }
    padCtx.strokeStyle = COLOR.pupil;
    padCtx.lineWidth = 2.2;
    for (const s of strokes.concat(drawing ? [drawing] : [])) {
      padCtx.beginPath();
      s.forEach(([x, y], i) => (i ? padCtx.lineTo(x, y) : padCtx.moveTo(x, y)));
      if (s.length === 1) padCtx.lineTo(s[0][0] + 0.1, s[0][1]);
      padCtx.stroke();
    }
    $("pad-empty").hidden = strokes.length > 0 || !!drawing || (prime && prime.kind === "hand");
    $("undo").disabled = !strokes.length;
    $("clear").disabled = !strokes.length;
  }

  function padPoint(e) {
    const r = pad.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  pad.addEventListener("pointerdown", (e) => {
    if (e.button && e.button !== 0) return;
    e.preventDefault();
    try { pad.setPointerCapture(e.pointerId); } catch (_) { /* synthetic or stale pointer */ }
    clearTimeout(idleTimer);
    drawing = [padPoint(e)];
    drawPad();
  });
  pad.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ev of evs.length ? evs : [e]) drawing.push(padPoint(ev));
    drawPad();
  });
  const endStroke = () => {
    if (!drawing) return;
    strokes.push(drawing);
    drawing = null;
    drawPad();
    clearTimeout(idleTimer);
    idleTimer = setTimeout(usePad, 1100);
  };
  pad.addEventListener("pointerup", endStroke);
  pad.addEventListener("pointercancel", endStroke);

  $("undo").addEventListener("click", () => { strokes.pop(); drawPad(); if (strokes.length) usePad(); });
  $("clear").addEventListener("click", () => { strokes = []; setStatus(""); drawPad(); });

  function setStatus(text) { $("pad-status").textContent = text; }

  // Prime with the visitor's own line, once it looks like a whole line.
  function usePad() {
    if (!strokes.length || !calib) return;
    let x0 = Infinity, x1 = -Infinity;
    for (const s of strokes) for (const [x] of s) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); }
    if (x1 - x0 < padW * 0.35) {
      setStatus(`Keep going: write all of “${PROMPT}”.`);
      return;
    }
    const line = Ink.normalize(strokes, calib, spacing);
    if (!line || line.xh < 4) { setStatus("That line is too small to read. Try writing a little larger."); return; }
    prime = { kind: "pad", line, xh: line.xh, label: "your hand" };
    handIndex = -1;
    markHands();
    setStatus("Got it. Writing in your hand…");
    write(true);
  }

  // ---------- borrowed hands ----------
  function drawLine(ctx, pts, lift, x0, base, xh, color, ditto) {
    const path = new Path2D();
    let start = true;
    for (let i = 0; i < pts.length; i++) {
      const x = x0 + pts[i][0] * xh, y = base + pts[i][1] * xh;
      if (start) { path.moveTo(x, y); path.lineTo(x + 0.01, y); } else path.lineTo(x, y);
      start = lift[i] === 1;
    }
    strokeInk(ctx, path, xh, color, ditto);
  }

  function strokeInk(ctx, path, xh, color, ditto) {
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    if (ditto) {
      // spirit-duplicator bleed: a soft violet halo under a fine core
      ctx.strokeStyle = COLOR.haze; ctx.globalAlpha = 0.6; ctx.lineWidth = Math.max(2.5, xh * 0.24);
      ctx.stroke(path);
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = color; ctx.lineWidth = Math.max(1.3, xh * (ditto ? 0.1 : 0.09));
    ctx.stroke(path);
  }

  function buildHands() {
    const box = $("hands");
    box.textContent = "";
    hands.forEach((h, i) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "hand";
      b.setAttribute("aria-pressed", "false");
      b.setAttribute("aria-label", `Borrowed hand ${i + 1}`);
      const c = document.createElement("canvas");
      b.appendChild(c);
      box.appendChild(b);
      const ctx = c.getContext("2d");
      const w = c.clientWidth || 132, hgt = c.clientHeight || 34, dpr = Math.min(window.devicePixelRatio || 1, 2);
      c.width = w * dpr; c.height = hgt * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      let wx = 0;
      for (const p of h.pts) wx = Math.max(wx, p[0]);
      const xh = Math.min((w - 12) / wx, hgt / 3.1);
      drawLine(ctx, h.pts, h.lift, 6, hgt * 0.68, xh, COLOR.pupil, false);
      b.addEventListener("click", () => useHand(i));
    });
  }

  function markHands() {
    [...$("hands").children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === handIndex)));
  }

  function useHand(i) {
    handIndex = i;
    strokes = [];
    prime = { kind: "hand", line: hands[i], xh: 22, label: `hand ${i + 1}` };
    markHands();
    setStatus(`Borrowed hand ${i + 1}. Write on the pad to use your own instead.`);
    drawPad();
    write(true);
  }

  // ---------- text ----------
  function cleanText(raw) {
    const t = raw
      .replace(/[‘’‛′]/g, "'").replace(/[“”„″]/g, '"').replace(/[–—]/g, "-").replace(/…/g, "...")
      .replace(/\s+/g, " ").trim();
    const dropped = new Set();
    const kept = [...t].filter((ch) => {
      if (ch === " " || vocab.has(ch)) return true;
      dropped.add(ch); return false;
    }).join("").replace(/\s+/g, " ").trim();
    $("text-note").textContent = dropped.size
      ? `Left out ${[...dropped].join(" ")}: the network never saw anyone write ${dropped.size > 1 ? "them" : "it"}.`
      : "";
    return kept;
  }

  // Greedy wrap, then re-wrap narrower so the last line isn't left with a word or two.
  function wrap(text, perLine) {
    const first = wrapGreedy(text, perLine);
    if (first.length < 2) return first;
    const even = Math.ceil(text.length / first.length) + 2;
    const balanced = wrapGreedy(text, Math.min(perLine, even));
    return balanced.length === first.length ? balanced : first;
  }

  function wrapGreedy(text, perLine) {
    const words = text.split(" ");
    const out = [];
    let cur = "";
    for (const w of words) {
      if (!cur) cur = w;
      else if ((cur + " " + w).length <= perLine) cur += " " + w;
      else { out.push(cur); cur = w; }
    }
    if (cur) out.push(cur);
    // very long words: hard split
    return out.flatMap((l) => (l.length <= perLine + 6 ? [l] : l.match(new RegExp(`.{1,${perLine}}`, "g"))));
  }

  // ---------- writing ----------
  const MARGIN = 22;

  function charsPerLine() {
    let wx = 0;
    for (const p of prime.line.pts) wx = Math.max(wx, p[0]);
    const perChar = wx / PROMPT.length;                 // x-heights per character, in this hand
    const room = (sheet.clientWidth - 2 * MARGIN) / renderXh;
    return Math.max(8, Math.min(42, Math.floor(room / perChar)));
  }

  function write(newSeed) {
    if (!ready || !prime) return;
    const text = cleanText($("text").value);
    if (!text) return;
    if (newSeed) seed = (Math.random() * 1e9) | 0;
    renderXh = Math.max(14, Math.min(20, prime.xh * 0.75));
    const parts = wrap(text, charsPerLine());
    job++;
    let offset = 0;
    lines = parts.map((t) => {
      const l = { text: t, offset, pts: [], mix: [], drawn: 0, done: false, x0: null };
      offset += t.length + 1;
      return l;
    });
    stripText = lines.map((l) => l.text).join(" ");
    stripIndex = -2;
    worker.postMessage({
      type: "write", id: job, prime: Ink.toMoves(prime.line, spacing), primeText: PROMPT,
      lines: parts, bias: Number($("neat").value) / 10, seed
    });
    $("sheet-status").textContent = "";
    layoutSheet();
    if (!playing) { playing = true; requestAnimationFrame(tick); }
  }

  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") {
      spacing = m.spacing;
      vocab = new Set(m.vocab);
      ready = true;
      if (!prime) {
        if (hands.length) useHand(0);
      } else write(true);
      return;
    }
    if (m.type === "error") { console.error(m.message); $("sheet-status").textContent = "The network stopped with an error."; return; }
    if (m.id !== job) return;
    if (m.type === "points") {
      const l = lines[m.line];
      const stride = 6 * m.mixK + 1;
      for (let i = 0, j = 0; i < m.pts.length; i += 4, j += stride) {
        l.pts.push([m.pts[i], m.pts[i + 1], m.pts[i + 2], m.pts[i + 3]]);
        l.mix.push(m.mix.subarray(j, j + stride));
      }
    } else if (m.type === "lineDone") {
      lines[m.line].done = true;
    }
  };

  // ---------- the sheet ----------
  let sheetW = 0;
  const rowH = () => renderXh * 3.6;

  function layoutSheet() {
    const h = Math.max(220, Math.ceil(lines.length * rowH() + renderXh * 1.6));
    sheetW = fit(sheet, sheetCtx, h);
    drawSheet();
  }

  function baselineOf(i) { return renderXh * 2.7 + i * rowH(); }

  function drawSheet() {
    const h = sheet.clientHeight;
    grid(sheetCtx, sheetW, h, rowH() / 4, baselineOf(0), 4);
    let pen = null;
    lines.forEach((l, i) => {
      if (!l.drawn) return;
      const lastPrimeY = prime.line.pts[prime.line.pts.length - 1][1];
      if (l.x0 === null) l.x0 = l.pts[0][0];
      const k0 = 0;
      // shrink a line that came out wider than the sheet
      let maxX = 0;
      for (let k = k0; k < l.drawn; k++) maxX = Math.max(maxX, l.pts[k][0] - l.x0);
      const room = (sheetW - 2 * MARGIN) / renderXh;
      const xh = renderXh * Math.min(1, room / Math.max(maxX, 1e-6));
      const path = new Path2D();
      const base = baselineOf(i);
      for (let k = k0; k < l.drawn; k++) {
        const p = l.pts[k];
        const x = MARGIN + (p[0] - l.x0) * xh, y = base + (p[1] + lastPrimeY) * xh;
        if (k === k0 || p[2] === 1) { path.moveTo(x, y); path.lineTo(x + 0.01, y); } else path.lineTo(x, y);
        if (k === l.drawn - 1 && !(l.done && l.drawn === l.pts.length)) pen = { line: l, k, xh, mix: l.mix[l.drawn] };
      }
      strokeInk(sheetCtx, path, renderXh, COLOR.ditto, true);
    });
    if ($("guesses").checked) drawLoupe(pen);
    if (pen) {
      const p = pen.line.pts[pen.k], i = lines.indexOf(pen.line);
      const lastPrimeY = prime.line.pts[prime.line.pts.length - 1][1];
      const x = MARGIN + (p[0] - pen.line.x0) * pen.xh, y = baselineOf(i) + (p[1] + lastPrimeY) * pen.xh;
      sheetCtx.fillStyle = COLOR.tick;
      sheetCtx.beginPath(); sheetCtx.arc(x, y, 3, 0, Math.PI * 2); sheetCtx.fill();
    }
  }

  // Close-up of the pen tip: the last few moves, and the network's guesses for
  // the next one, one ellipse (two standard deviations) per likely component.
  const loupe = $("loupe"), loupeCtx = loupe.getContext("2d");
  const LOUPE = 150, ZOOM = 20;          // pixels per resampling step in the close-up
  (function fitLoupe() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    loupe.width = LOUPE * dpr; loupe.height = LOUPE * dpr;
    loupeCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  })();

  function drawLoupe(pen) {
    const ctx = loupeCtx, c = LOUPE / 2;
    ctx.fillStyle = COLOR.surface;
    ctx.fillRect(0, 0, LOUPE, LOUPE);
    if (!pen || !pen.mix) { $("loupe-cap").textContent = "Next move"; return; }
    const l = pen.line, tip = l.pts[pen.k], unit = ZOOM / spacing;   // px per x-height
    const X = (p) => c + (p[0] - tip[0]) * unit, Y = (p) => c + (p[1] - tip[1]) * unit;
    // grid at the sheet's square size
    ctx.strokeStyle = COLOR.rule; ctx.globalAlpha = 0.45; ctx.lineWidth = 1;
    const sq = 0.9 * unit;
    ctx.beginPath();
    for (let g = ((c - tip[0] * unit) % sq + sq) % sq; g < LOUPE; g += sq) { ctx.moveTo(g, 0); ctx.lineTo(g, LOUPE); }
    for (let g = ((c - tip[1] * unit) % sq + sq) % sq; g < LOUPE; g += sq) { ctx.moveTo(0, g); ctx.lineTo(LOUPE, g); }
    ctx.stroke(); ctx.globalAlpha = 1;
    // recent ink
    const path = new Path2D();
    const from = Math.max(0, pen.k - 14);
    for (let k = from; k <= pen.k; k++) {
      const p = l.pts[k];
      if (k === from || p[2] === 1) path.moveTo(X(p), Y(p)); else path.lineTo(X(p), Y(p));
    }
    strokeInk(ctx, path, 40, COLOR.ditto, true);
    // guesses
    const mix = pen.mix, K = (mix.length - 1) / 6;
    ctx.strokeStyle = COLOR.tick; ctx.fillStyle = COLOR.tick;
    for (let j = K - 1; j >= 0; j--) {
      const w = mix[6 * j];
      if (w < 0.02) continue;
      const cx = c + mix[6 * j + 1] * ZOOM, cy = c + mix[6 * j + 2] * ZOOM;
      const sx = mix[6 * j + 3], sy = mix[6 * j + 4], r = mix[6 * j + 5];
      const a = sx * sx, b = r * sx * sy, d = sy * sy;
      const mean = (a + d) / 2, dev = Math.sqrt(((a - d) / 2) ** 2 + b * b);
      const r1 = Math.sqrt(mean + dev) * 2 * ZOOM, r2 = Math.sqrt(Math.max(mean - dev, 1e-9)) * 2 * ZOOM;
      ctx.globalAlpha = 0.25 + 0.6 * w; ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.ellipse(cx, cy, Math.max(r1, 2), Math.max(r2, 2), 0.5 * Math.atan2(2 * b, a - d), 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 0.12 * w; ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = COLOR.tick;
    ctx.beginPath(); ctx.arc(c, c, 3, 0, Math.PI * 2); ctx.fill();
    $("loupe-cap").textContent = `Next move · lift ${Math.round(mix[mix.length - 1] * 100)}%`;
  }

  // attention strip: which letter the network is reading
  let stripText = "", stripIndex = -2;
  function drawStrip(idx) {
    if (idx === stripIndex) return;
    stripIndex = idx;
    const el = $("strip");
    el.textContent = "";
    const a = document.createElement("span"); a.className = "done"; a.textContent = stripText.slice(0, Math.max(0, idx));
    el.appendChild(a);
    if (idx >= 0 && idx < stripText.length) {
      const b = document.createElement("span"); b.className = "now"; b.textContent = stripText[idx];
      el.appendChild(b);
    }
    el.appendChild(document.createTextNode(stripText.slice(Math.max(0, idx + 1))));
  }

  function tick() {
    let active = lines.find((l) => !(l.done && l.drawn >= l.pts.length));
    if (active) {
      const queued = active.pts.length - active.drawn;
      active.drawn = Math.min(active.pts.length, active.drawn + Math.max(4, Math.ceil(queued / 24)));
      if (active.drawn) drawStrip(active.offset + Math.min(active.pts[active.drawn - 1][3], active.text.length - 1));
      drawSheet();
      requestAnimationFrame(tick);
    } else {
      drawStrip(stripText.length);
      drawSheet();
      playing = false;
    }
  }

  // ---------- controls ----------
  $("compose").addEventListener("submit", (e) => { e.preventDefault(); write(true); });
  $("again").addEventListener("click", () => write(true));
  $("neat").addEventListener("change", () => write(false));
  $("guesses").addEventListener("change", () => { $("loupe-box").hidden = !$("guesses").checked; drawSheet(); });
  $("save").addEventListener("click", () => {
    sheet.toBlob((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "second-hand.png";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
  });

  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { drawPad(); layoutSheet(); }, 120);
  });

  // ---------- start ----------
  drawPad();
  layoutSheet();
  Promise.all([
    fetch("model/calib.json").then((r) => r.json()),
    fetch("model/hands.json").then((r) => r.json())
  ]).then(([c, h]) => {
    calib = c;
    hands = h;
    buildHands();
    if (ready && !prime && hands.length) useHand(0);
  });
})();
