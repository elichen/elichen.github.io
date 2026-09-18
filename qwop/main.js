// Learning to QWOP — runs the ragdoll physics and the four trained "brains".
(() => {
  const P = QwopPhysics;
  const canvas = document.getElementById('track');
  const ctx = canvas.getContext('2d');
  const banner = document.getElementById('banner');

  const BRAINS = [
    { id: 'human', color: '#e3342a', dot: '#ffffff' },
    { id: 'teacher', color: '#8d939c' },
    { id: 'bc', color: '#e0a81c' },
    { id: 'dagger', color: '#2f9a5f' },
    { id: 'rl', color: '#e3342a' },
  ];
  const RESET_NOISE = 0.03;
  const keys = { q: false, w: false, o: false, p: false };

  // ---------------------------------------------------------- controllers --
  function makeMlp(layers) {
    const bufs = layers.map((l) => new Float64Array(l.b.length));
    const obs = new Float64Array(P.NOBS);
    return (r) => {
      P.observe(r.s, r.acc, r.prev, obs);
      let x = obs;
      for (let l = 0; l < layers.length; l++) {
        const { W, b } = layers[l], out = bufs[l], last = l === layers.length - 1;
        for (let i = 0; i < out.length; i++) {
          const w = W[i];
          let sum = b[i];
          for (let j = 0; j < w.length; j++) sum += w[j] * x[j];
          out[i] = last ? sum : Math.tanh(sum);
        }
        x = out;
      }
      let best = 0;
      for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
      return best;
    };
  }

  // Scripted finite-state machine (mirror of train/teacher.py).
  function makeTeacher(t) {
    const K = t.TH.length;
    return (r) => {
      const m = r.mem;
      const split = r.s[1 * 6 + 2] - r.s[2 * 6 + 2];
      m.count += 1;
      const p = m.phase;
      if (t.SG[p] * (split - t.TH[p]) > 0 || m.count >= t.D[p]) {
        m.phase = (p + 1) % K;
        m.count = 0;
      }
      let act = t.A[m.phase];
      const ta = r.s[2];
      if (ta > t.PHI[0]) act = t.A[K];
      if (ta < -t.PHI[1]) act = t.A[K + 1];
      return act;
    };
  }

  function humanAction() {
    const hip = keys.q === keys.w ? 0 : keys.q ? 1 : 2;
    const knee = keys.o === keys.p ? 0 : keys.o ? 1 : 2;
    return hip * 3 + knee;
  }

  // --------------------------------------------------------------- state --
  let controllers = { human: humanAction };
  let runners = [];
  let focusId = 'rl';
  let ghosts = true;
  let subCount = 0;
  let raceTime = 0;
  let restartAt = null;
  let waitingForHuman = false;
  let camX = 0;
  let lastFrame = null;
  let accumulator = 0;

  function newRunner(brain) {
    const w = P.create();
    const q = P.POSE0.map((v, j) => (j < 6 ? v + RESET_NOISE * (Math.random() * 2 - 1) : v));
    P.setPose(w.s, 0.0, P.POSE0_TORSO + RESET_NOISE * 0.5 * (Math.random() * 2 - 1), q);
    return { ...brain, s: w.s, acc: w.acc, prev: 0, action: 0, fell: false, finished: false,
      time: 0, mem: { phase: 0, count: 0 } };
  }

  function resetRace() {
    runners = BRAINS.filter((b) => controllers[b.id] && (b.id !== 'human' || focusId === 'human'))
      .map(newRunner);
    subCount = 0;
    raceTime = 0;
    restartAt = null;
    waitingForHuman = false;
    camX = 0;
    banner.hidden = true;
  }

  function focus() {
    return runners.find((r) => r.id === focusId) || runners[0];
  }

  function showBanner(title, sub) {
    banner.querySelector('.banner-title').textContent = title;
    banner.querySelector('.banner-sub').textContent = sub;
    banner.hidden = false;
  }

  function substepAll() {
    const decide = subCount % P.NSUB === 0;
    for (const r of runners) {
      if (r.finished) continue;
      if (decide && !r.fell) {
        r.action = controllers[r.id](r);
        r.prev = r.action;
      }
      P.setTargets(r.fell ? 0 : r.action);
      P.substep(r.s, r.acc);
      if (!r.fell && (subCount + 1) % P.NSUB === 0) {
        r.time = raceTime + P.H;
        if (P.fatalTouch(r.s)) r.fell = true;
        else if (r.s[0] >= P.GOAL_X) r.finished = true;
        if (r.fell || r.finished) r.dist = Math.min(r.s[0], P.GOAL_X);
      }
    }
    subCount += 1;
    raceTime += P.H;
    const f = focus();
    if (restartAt === null && !waitingForHuman && (f.fell || f.finished)) {
      const d = f.dist.toFixed(1);
      if (f.finished) showBanner(`100 m in ${f.time.toFixed(2)} s`, focusId === 'human' ? 'press space to run again' : '');
      else showBanner(`fell at ${d} m`, focusId === 'human' ? 'press space to try again' : '');
      if (focusId === 'human') waitingForHuman = true;
      else restartAt = raceTime + (f.finished ? 3.5 : 1.8);
    }
    if (restartAt !== null && raceTime >= restartAt) resetRace();
  }

  // ------------------------------------------------------------- drawing --
  // Everything is laid out on the original game's 640x400 stage; `u` scales it.
  const FONT = '"Mundo Sans Std", "Gill Sans", "Gill Sans MT", "Trebuchet MS", "Lucida Grande", sans-serif';
  const TRACK_TOP = 313, TRACK_BOTTOM = 382, GROUND = 377;
  const KEY_RECTS = { q: [22, 20], w: [75, 20], o: [510, 20], p: [564, 20] };
  const KEY_SIZE = 46;
  let W = 640, Hpx = 400, u = 1, scale = 146, groundY = GROUND;
  let backdrop = null, trackTex = null, dirtTex = null;
  let bestDist = 0;

  function makeSpeckle(w, h, base, dots, seed) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const g = c.getContext('2d');
    g.fillStyle = base;
    g.fillRect(0, 0, w, h);
    let s = seed;
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    for (const [color, n, size] of dots) {
      g.fillStyle = color;
      for (let i = 0; i < n; i++) g.fillRect(rnd() * w, rnd() * h, size * (0.5 + rnd()), size * (0.5 + rnd()));
    }
    return c;
  }

  function buildBackdrop() {
    const dpr = window.devicePixelRatio || 1;
    backdrop = document.createElement('canvas');
    backdrop.width = canvas.width;
    backdrop.height = canvas.height;
    const g = backdrop.getContext('2d');
    g.setTransform(dpr * u, 0, 0, dpr * u, 0, 0);
    // night sky falling to black, then the out-of-focus stadium: blue hoarding, green infield
    const sky = g.createLinearGradient(0, 0, 0, TRACK_TOP);
    [[0, '#5f748c'], [0.2, '#4b5c70'], [0.45, '#2a323e'], [0.62, '#12151b'], [0.665, '#0f1319'],
      [0.7, '#184f9f'], [0.735, '#2b7fe3'], [0.765, '#1f66c4'], [0.79, '#25706a'], [0.815, '#3a8430'],
      [0.9, '#569a35'], [0.975, '#74ad40'], [0.983, '#cfc3ab'], [1, '#e9e0d0']].forEach(([t, c]) => sky.addColorStop(t, c));
    g.fillStyle = sky;
    g.fillRect(0, 0, 640, TRACK_TOP);
    // track with soft lane lines that thicken toward the camera
    const track = g.createLinearGradient(0, TRACK_TOP, 0, TRACK_BOTTOM);
    track.addColorStop(0, '#c9451f');
    track.addColorStop(1, '#de5626');
    g.fillStyle = track;
    g.fillRect(0, TRACK_TOP, 640, TRACK_BOTTOM - TRACK_TOP);
    for (const [y, t] of [[319, 2.2], [332, 3.2], [350, 4.6]]) {
      const lane = g.createLinearGradient(0, y - t, 0, y + t);
      lane.addColorStop(0, 'rgba(255,255,255,0)');
      lane.addColorStop(0.35, 'rgba(255,250,244,0.92)');
      lane.addColorStop(0.65, 'rgba(255,250,244,0.92)');
      lane.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = lane;
      g.fillRect(0, y - t, 640, t * 2);
    }
    trackTex = makeSpeckle(512, 64, 'rgba(0,0,0,0)', [['rgba(90,20,0,0.10)', 500, 2], ['rgba(255,200,160,0.08)', 400, 2]], 7);
    dirtTex = makeSpeckle(512, 32, '#33241a', [['#4a3626', 500, 3], ['#21160f', 500, 3], ['#6a5138', 120, 2]], 11);
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    W = canvas.clientWidth;
    Hpx = W * 0.625;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(Hpx * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    u = W / 640;
    scale = 146 * u;
    groundY = GROUND * u;
    buildBackdrop();
  }

  const sx = (x) => (x - camX) * scale + W * 0.45;
  const sy = (y) => groundY - y * scale;

  function shade(hex, k) {
    const n = parseInt(hex.slice(1), 16);
    const f = (v) => Math.max(0, Math.min(255, Math.round(k <= 1 ? v * k : v + (255 - v) * (k - 1))));
    return `rgb(${f(n >> 16)}, ${f((n >> 8) & 255)}, ${f(n & 255)})`;
  }

  // Draw in a body's local frame (metres, y up).
  function inBody(r, body, fn) {
    const o = body * 6;
    ctx.save();
    ctx.translate(sx(r.s[o]), sy(r.s[o + 1]));
    ctx.rotate(-r.s[o + 2]);
    ctx.scale(scale, -scale);
    fn();
    ctx.restore();
  }

  // Tapered, rounded limb from (x0,y0) half-width w0 to (x1,y1) half-width w1, lit from the front.
  function limb(x0, y0, w0, x1, y1, w1, color, dark = 0.5) {
    const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy), nx = -dy / len, ny = dx / len;
    const wm = Math.max(w0, w1);
    const g = ctx.createLinearGradient(x0 - nx * wm, y0 - ny * wm, x0 + nx * wm, y0 + ny * wm);
    g.addColorStop(0, shade(color, 1.22));
    g.addColorStop(0.45, color);
    g.addColorStop(1, shade(color, dark));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(x0 + nx * w0, y0 + ny * w0);
    ctx.lineTo(x1 + nx * w1, y1 + ny * w1);
    ctx.arc(x1, y1, w1, Math.atan2(ny, nx), Math.atan2(-ny, -nx), true);
    ctx.lineTo(x0 - nx * w0, y0 - ny * w0);
    ctx.arc(x0, y0, w0, Math.atan2(-ny, -nx), Math.atan2(ny, nx), true);
    ctx.fill();
  }

  function drawArm(r, body, skin) {
    inBody(r, body, () => {
      limb(0, 0.275, 0.05, 0, 0.0, 0.04, skin);
      limb(0, 0.0, 0.04, 0.235, -0.085, 0.03, skin);
      limb(0.235, -0.085, 0.036, 0.275, -0.095, 0.034, skin);
    });
  }

  function drawLeg(r, thigh, calf, foot, skin, k) {
    inBody(r, thigh, () => limb(0, 0.225, 0.088, 0, -0.225, 0.058, skin));
    inBody(r, calf, () => {
      limb(0, 0.225, 0.06, 0, -0.17, 0.038, skin);
      limb(0, -0.125, 0.04, 0, -0.225, 0.036, k === 1 ? '#f6f6f6' : '#b9b9b9', 0.82);
    });
    inBody(r, foot, () => {
      const g = ctx.createLinearGradient(0, 0.06, 0, -0.045);
      g.addColorStop(0, shade('#ffffff', k));
      g.addColorStop(1, shade('#b9bcc4', k));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.moveTo(-0.165, -0.04);
      ctx.quadraticCurveTo(-0.185, 0.03, -0.12, 0.065);
      ctx.lineTo(-0.02, 0.07);
      ctx.quadraticCurveTo(0.09, 0.03, 0.17, -0.005);
      ctx.quadraticCurveTo(0.19, -0.04, 0.15, -0.04);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = shade('#8b8f99', k);
      ctx.fillRect(-0.165, -0.04, 0.325, 0.012);
    });
  }

  function drawRunner(r, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    const skin = '#8a5634', farSkin = '#5e3a23', kit = r.color;
    drawArm(r, 8, farSkin);
    drawLeg(r, 2, 4, 6, farSkin, 0.72);
    inBody(r, 0, () => {
      limb(0.01, 0.30, 0.045, 0.02, 0.40, 0.04, skin);                // neck
      ctx.fillStyle = '#1a1310';                                       // flat-top hair
      ctx.beginPath();
      ctx.ellipse(-0.005, 0.475, 0.102, 0.122, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillRect(-0.085, 0.53, 0.165, 0.062);
      const face = ctx.createLinearGradient(-0.08, 0.5, 0.11, 0.4);
      face.addColorStop(0, shade(skin, 0.62));
      face.addColorStop(1, shade(skin, 1.2));
      ctx.fillStyle = face;
      ctx.beginPath();
      ctx.ellipse(0.022, 0.447, 0.086, 0.103, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();                                                 // nose
      ctx.moveTo(0.098, 0.475);
      ctx.lineTo(0.135, 0.43);
      ctx.lineTo(0.095, 0.42);
      ctx.fill();
      ctx.fillStyle = '#120c09';
      ctx.fillRect(0.058, 0.47, 0.022, 0.012);
      limb(0, 0.255, 0.105, 0, 0.27, 0.085, skin);                     // shoulders
      limb(0, 0.2, 0.122, 0, -0.14, 0.098, kit);                       // singlet
      limb(0, -0.14, 0.1, 0, -0.30, 0.095, kit);                       // briefs
      ctx.fillStyle = 'rgba(255,255,255,0.92)';
      ctx.beginPath();
      ctx.moveTo(-0.02, -0.17);
      ctx.lineTo(0.085, -0.19);
      ctx.lineTo(0.07, -0.30);
      ctx.lineTo(-0.035, -0.27);
      ctx.fill();
    });
    drawLeg(r, 1, 3, 5, skin, 1);
    drawArm(r, 7, skin);
    ctx.restore();
  }

  function marker(m) {
    const px = sx(m);
    if (px < -60 * u || px > W + 60 * u) return;
    ctx.fillStyle = 'rgba(255,252,246,0.96)';
    ctx.beginPath();
    ctx.moveTo(px - 3.5 * u, TRACK_TOP * u);
    ctx.lineTo(px + 3.5 * u, TRACK_TOP * u);
    ctx.lineTo(px + 15 * u, TRACK_BOTTOM * u);
    ctx.lineTo(px - 15 * u, TRACK_BOTTOM * u);
    ctx.fill();
    if (m > 0) {
      ctx.font = `bold ${15 * u}px ${FONT}`;
      ctx.textAlign = 'left';
      ctx.fillText(m === P.GOAL_X ? 'FINISH' : `${m}m`, px + 12 * u, (TRACK_TOP + 27) * u);
    }
  }

  function drawWorld() {
    ctx.drawImage(backdrop, 0, 0, W, Hpx);
    // scrolling grain so speed reads even between markers
    const scroll = (tex, y, h) => {
      const tw = tex.width * u * 0.5, off = -((camX * scale) % tw + tw) % tw;
      for (let x = off; x < W; x += tw) ctx.drawImage(tex, x, y * u, tw, h * u);
    };
    scroll(trackTex, TRACK_TOP, TRACK_BOTTOM - TRACK_TOP);
    scroll(dirtTex, TRACK_BOTTOM, 400 - TRACK_BOTTOM);
    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.fillRect(0, TRACK_BOTTOM * u, W, 1.5 * u);
    for (let m = 0; m <= P.GOAL_X; m += 10) marker(m);
  }

  function drawKey(k, down) {
    const [x, y] = KEY_RECTS[k], s = KEY_SIZE, r = 6;
    ctx.save();
    ctx.scale(u, u);
    ctx.translate(x, y + (down ? 2 : 0));
    const rim = ctx.createLinearGradient(0, 0, s, s);
    rim.addColorStop(0, down ? '#8f8a80' : '#f1eee7');
    rim.addColorStop(1, down ? '#3d3a35' : '#6b675f');
    ctx.fillStyle = rim;
    ctx.shadowColor = 'rgba(0,0,0,0.55)';
    ctx.shadowBlur = down ? 2 : 6;
    ctx.shadowOffsetY = down ? 1 : 3;
    ctx.beginPath();
    ctx.roundRect(0, 0, s, s, r);
    ctx.fill();
    ctx.shadowColor = 'transparent';
    const face = ctx.createLinearGradient(0, 5, 0, s - 7);
    face.addColorStop(0, down ? '#a39e93' : '#d9d5cc');
    face.addColorStop(1, down ? '#8a857b' : '#aaa69c');
    ctx.fillStyle = face;
    ctx.beginPath();
    ctx.roundRect(5, 4, s - 10, s - 11, 4);
    ctx.fill();
    ctx.fillStyle = down ? '#000' : '#15130f';
    ctx.font = `bold 28px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(k.toUpperCase(), s / 2, s / 2 - 1);
    ctx.restore();
  }

  function drawHud(f) {
    const hip = Math.floor(f.action / 3), knee = f.action % 3, live = !f.fell && !f.finished;
    drawKey('q', live && hip === 1);
    drawKey('w', live && hip === 2);
    drawKey('o', live && knee === 1);
    drawKey('p', live && knee === 2);

    ctx.save();
    ctx.scale(u, u);
    ctx.fillStyle = '#fff';
    ctx.shadowColor = 'rgba(0,0,0,0.6)';
    ctx.shadowBlur = 3;
    ctx.shadowOffsetY = 1;
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'center';
    ctx.font = `bold 13px ${FONT}`;
    ctx.fillText('THIGHS', 72, 86);
    ctx.fillText('CALVES', 561, 86);
    ctx.font = `bold 27px ${FONT}`;
    ctx.fillText(`${(f.dist ?? Math.min(f.s[0], P.GOAL_X)).toFixed(1)} metres`, 320, 46);
    ctx.font = `13px ${FONT}`;
    ctx.fillStyle = 'rgba(255,255,255,0.78)';
    ctx.fillText(`${f.time.toFixed(1)} s  ·  ${Math.max(0, f.s[3]).toFixed(1)} m/s`, 320, 66);
    ctx.fillText('LEARNING TO QWOP  ·  after Foddy.net 2008', 320, 13);
    ctx.textAlign = 'left';
    ctx.font = `bold 12px ${FONT}`;
    ctx.fillStyle = '#fff';
    ctx.fillText(`Best: ${Math.floor(bestDist)}m`, 36, 13);

    // race progress, tucked into the dirt strip
    ctx.shadowColor = 'transparent';
    ctx.fillStyle = 'rgba(255,255,255,0.28)';
    ctx.fillRect(20, 391, 600, 1.5);
    for (const r of runners) {
      if (!ghosts && r.id !== focusId) continue;
      ctx.globalAlpha = r.fell ? 0.4 : 1;
      ctx.fillStyle = r.dot || r.color;
      ctx.strokeStyle = r.dot ? r.color : 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(20 + 600 * Math.max(0, Math.min(1, r.s[0] / P.GOAL_X)), 391.7, r.id === focusId ? 4.2 : 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  function render() {
    const f = focus();
    camX += (f.s[0] - camX) * 0.15;
    if (!f.fell) bestDist = Math.max(bestDist, Math.min(f.s[0], P.GOAL_X));
    drawWorld();
    if (ghosts) for (const r of runners) if (r !== f) drawRunner(r, 0.38);
    drawRunner(f, 1);
    drawHud(f);
  }

  function frame(now) {
    if (lastFrame === null) lastFrame = now;
    accumulator += Math.min(0.1, (now - lastFrame) / 1000);
    lastFrame = now;
    while (accumulator >= P.H) {
      substepAll();
      accumulator -= P.H;
    }
    render();
    requestAnimationFrame(frame);
  }

  // --------------------------------------------------------------- input --
  function setKey(k, down) {
    if (!(k in keys)) return;
    keys[k] = down;
  }

  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k in keys) { setKey(k, true); if (focusId !== 'human') selectBrain('human'); e.preventDefault(); }
    if ((k === ' ' || k === 'r') && focusId === 'human') { resetRace(); e.preventDefault(); }
  });
  window.addEventListener('keyup', (e) => setKey(e.key.toLowerCase(), false));
  window.addEventListener('blur', () => { for (const k in keys) keys[k] = false; });

  const pointerKeys = new Map();
  function keyAt(e) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / u, y = (e.clientY - rect.top) / u;
    for (const [k, [kx, ky]] of Object.entries(KEY_RECTS)) {
      if (x >= kx - 4 && x <= kx + KEY_SIZE + 4 && y >= ky - 4 && y <= ky + KEY_SIZE + 4) return k;
    }
    return null;
  }
  canvas.addEventListener('pointerdown', (e) => {
    const k = keyAt(e);
    if (!k) return;
    e.preventDefault();
    if (focusId !== 'human') selectBrain('human');
    canvas.setPointerCapture(e.pointerId);
    pointerKeys.set(e.pointerId, k);
    setKey(k, true);
  });
  const releasePointer = (e) => {
    const k = pointerKeys.get(e.pointerId);
    if (!k) return;
    pointerKeys.delete(e.pointerId);
    if (![...pointerKeys.values()].includes(k)) setKey(k, false);
  };
  canvas.addEventListener('pointerup', releasePointer);
  canvas.addEventListener('pointercancel', releasePointer);

  function selectBrain(id) {
    if (!controllers[id]) return;
    focusId = id;
    bestDist = 0;
    document.querySelectorAll('.brain').forEach((el) => {
      el.setAttribute('aria-pressed', el.dataset.brain === id ? 'true' : 'false');
    });
    document.getElementById('key-hint').textContent = id === 'human'
      ? 'You are running. Space restarts.'
      : 'Keys pressed by the network. Press Q, W, O or P to take over.';
    resetRace();
  }

  document.querySelectorAll('.brain').forEach((el) => {
    el.addEventListener('click', () => selectBrain(el.dataset.brain));
  });
  banner.addEventListener('click', () => { if (focusId === 'human') resetRace(); });
  document.getElementById('ghosts').addEventListener('change', (e) => { ghosts = e.target.checked; });
  window.addEventListener('resize', resize);

  function fillStats(stats) {
    for (const [id, m] of Object.entries(stats || {})) {
      const el = document.querySelector(`.brain[data-brain="${id}"] .brain-stat`);
      if (!el) continue;
      el.textContent = m.finish_rate >= 0.005
        ? `${Math.round(m.finish_rate * 100)}% finish · ${m.time_100m.toFixed(1)} s`
        : `${m.speed.toFixed(1)} m/s · ${Math.round(m.fall_rate * 100)}% fall in 100 s`;
    }
  }

  // Debug hook: step the race without waiting for animation frames.
  window.qwopAdvance = (seconds) => {
    for (let i = 0; i < Math.round(seconds / P.H); i++) substepAll();
    camX = focus().s[0];
    render();
    const f = focus();
    return { id: f.id, x: f.s[0], t: f.time, fell: f.fell, finished: f.finished };
  };

  resize();
  fetch('models.json').then((r) => r.json()).then((m) => {
    controllers = {
      human: humanAction,
      teacher: makeTeacher(m.teacher),
      bc: makeMlp(m.bc),
      dagger: makeMlp(m.dagger),
      rl: makeMlp(m.rl),
    };
    fillStats(m.stats);
    selectBrain('rl');
    requestAnimationFrame(frame);
  }).catch((err) => {
    console.error(err);
    focusId = 'human';
    selectBrain('human');
    requestAnimationFrame(frame);
  });
})();
