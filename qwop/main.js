// Learning to QWOP — runs the ragdoll physics and the four trained "brains".
(() => {
  const P = QwopPhysics;
  const canvas = document.getElementById('track');
  const ctx = canvas.getContext('2d');
  const banner = document.getElementById('banner');
  const hudDist = document.getElementById('hud-dist');
  const hudTime = document.getElementById('hud-time');
  const hudSpeed = document.getElementById('hud-speed');
  const keycaps = {};
  document.querySelectorAll('.keycap').forEach((el) => { keycaps[el.dataset.key] = el; });

  const BRAINS = [
    { id: 'human', color: '#1d2a44' },
    { id: 'teacher', color: '#8a8f98' },
    { id: 'bc', color: '#d9a21b' },
    { id: 'dagger', color: '#2f8f6b' },
    { id: 'rl', color: '#d8432b' },
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
      }
    }
    subCount += 1;
    raceTime += P.H;
    const f = focus();
    if (restartAt === null && !waitingForHuman && (f.fell || f.finished)) {
      const d = Math.min(f.s[0], P.GOAL_X).toFixed(1);
      if (f.finished) showBanner(`100 m in ${f.time.toFixed(2)} s`, focusId === 'human' ? 'press space to run again' : '');
      else showBanner(`fell at ${d} m`, focusId === 'human' ? 'press space to try again' : '');
      if (focusId === 'human') waitingForHuman = true;
      else restartAt = raceTime + (f.finished ? 3.5 : 1.8);
    }
    if (restartAt !== null && raceTime >= restartAt) resetRace();
  }

  // ------------------------------------------------------------- drawing --
  let W = 960, Hpx = 420, scale = 110, groundY = 330;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    W = canvas.clientWidth;
    Hpx = canvas.clientHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(Hpx * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    scale = Math.max(62, Math.min(118, W / 8));
    groundY = Hpx - 84;
  }

  const sx = (x) => (x - camX) * scale + W * 0.38;
  const sy = (y) => groundY - y * scale;

  function capsule(r, body, x0, y0, x1, y1, rad, color) {
    const o = body * 6, c = Math.cos(r.s[o + 2]), s = Math.sin(r.s[o + 2]);
    ctx.strokeStyle = color;
    ctx.lineWidth = rad * 2 * scale;
    ctx.beginPath();
    ctx.moveTo(sx(r.s[o] + c * x0 - s * y0), sy(r.s[o + 1] + s * x0 + c * y0));
    ctx.lineTo(sx(r.s[o] + c * x1 - s * y1), sy(r.s[o + 1] + s * x1 + c * y1));
    ctx.stroke();
  }

  function shade(hex, k) {
    const n = parseInt(hex.slice(1), 16);
    const f = (v) => Math.round(v * k);
    return `rgb(${f(n >> 16)}, ${f((n >> 8) & 255)}, ${f(n & 255)})`;
  }

  function drawRunner(r, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineCap = 'round';
    const skin = '#e9c9a4', far = shade(r.color, 0.62), farSkin = '#c9a883';
    // far side (left limbs)
    capsule(r, 8, 0, -0.275, 0, 0.275, 0.045, farSkin);
    capsule(r, 2, 0, -0.225, 0, 0.225, 0.07, far);
    capsule(r, 4, 0, -0.225, 0, 0.225, 0.055, farSkin);
    capsule(r, 6, -0.13, 0, 0.13, 0, 0.04, '#23262d');
    // torso + head
    capsule(r, 0, 0, -0.30, 0, 0.30, 0.10, r.color);
    const c = Math.cos(r.s[2]), s = Math.sin(r.s[2]);
    const hx = r.s[0] - s * 0.45, hy = r.s[1] + c * 0.45;
    ctx.fillStyle = skin;
    ctx.beginPath();
    ctx.arc(sx(hx), sy(hy), 0.12 * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = r.color;
    ctx.lineWidth = 0.035 * scale;
    ctx.beginPath();  // headband
    ctx.arc(sx(hx), sy(hy), 0.12 * scale, -r.s[2] - Math.PI * 0.92, -r.s[2] - Math.PI * 0.08);
    ctx.stroke();
    // near side (right limbs)
    capsule(r, 1, 0, -0.225, 0, 0.225, 0.07, shade(r.color, 0.85));
    capsule(r, 3, 0, -0.225, 0, 0.225, 0.055, skin);
    capsule(r, 5, -0.13, 0, 0.13, 0, 0.04, '#23262d');
    capsule(r, 7, 0, -0.275, 0, 0.275, 0.045, skin);
    ctx.restore();
  }

  function drawWorld() {
    const css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue('--ink').trim();
    ctx.fillStyle = css.getPropertyValue('--sky').trim();
    ctx.fillRect(0, 0, W, Hpx);

    // far hills, parallax
    const hill = (par, base, amp, color) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(0, groundY);
      for (let px = 0; px <= W; px += 8) {
        const u = (px / scale + camX * par) * 0.35;
        ctx.lineTo(px, groundY - base - amp * (Math.sin(u) + 0.5 * Math.sin(u * 2.3 + 1.7)));
      }
      ctx.lineTo(W, groundY);
      ctx.fill();
    };
    hill(0.12, 70, 22, css.getPropertyValue('--hill-far').trim());
    hill(0.3, 34, 14, css.getPropertyValue('--hill-near').trim());

    // track
    ctx.fillStyle = css.getPropertyValue('--track').trim();
    ctx.fillRect(0, groundY, W, Hpx - groundY);
    ctx.fillStyle = css.getPropertyValue('--track-line').trim();
    ctx.fillRect(0, groundY, W, 3);
    ctx.fillRect(0, groundY + 40, W, 2);

    const x0 = Math.floor(camX - W * 0.4 / scale) - 1, x1 = Math.ceil(camX + W / scale);
    ctx.font = '600 15px "Barlow Condensed", "Arial Narrow", sans-serif';
    ctx.textAlign = 'center';
    for (let m = x0; m <= x1; m++) {
      if (m < 0 || m > 100) continue;
      const px = sx(m), major = m % 5 === 0;
      ctx.fillStyle = css.getPropertyValue('--track-line').trim();
      ctx.fillRect(px - 1, groundY + 3, 2, major ? 16 : 7);
      if (major) ctx.fillText(`${m}`, px, groundY + 34);
    }
    // finish line
    const fx = sx(P.GOAL_X);
    if (fx > -40 && fx < W + 40) {
      for (let i = 0; i < 6; i++) for (let k = 0; k < 2; k++) {
        ctx.fillStyle = (i + k) % 2 ? ink : '#fff';
        ctx.fillRect(fx + k * 7, groundY + i * 7, 7, 7);
      }
      ctx.fillStyle = ink;
      ctx.fillRect(fx - 1, groundY - 150, 3, 150);
      ctx.font = '700 22px "Barlow Condensed", "Arial Narrow", sans-serif';
      ctx.fillText('FINISH', fx, groundY - 158);
    }
  }

  function drawMinimap() {
    const x = 18, y = 18, w = W - 36;
    ctx.fillStyle = 'rgba(29, 42, 68, 0.16)';
    ctx.fillRect(x, y + 5, w, 3);
    ctx.fillStyle = 'rgba(29, 42, 68, 0.55)';
    ctx.font = '600 11px "Barlow Condensed", "Arial Narrow", sans-serif';
    for (let m = 0; m <= 100; m += 25) {
      ctx.fillRect(x + (w * m) / 100 - 0.5, y + 2, 1, 9);
      ctx.textAlign = m === 0 ? 'left' : m === 100 ? 'right' : 'center';
      ctx.fillText(`${m} m`, x + (w * m) / 100, y + 24);
    }
    for (const r of runners) {
      if (!ghosts && r.id !== focusId) continue;
      const px = x + w * Math.max(0, Math.min(1, r.s[0] / 100));
      ctx.fillStyle = r.color;
      ctx.globalAlpha = r.fell ? 0.35 : 1;
      ctx.beginPath();
      ctx.arc(px, y + 6.5, r.id === focusId ? 6 : 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  function render() {
    const f = focus();
    camX += (f.s[0] - camX) * 0.12;
    drawWorld();
    if (ghosts) for (const r of runners) if (r !== f) drawRunner(r, 0.3);
    drawRunner(f, 1);
    drawMinimap();

    hudDist.textContent = Math.max(0, Math.min(f.s[0], P.GOAL_X)).toFixed(1);
    hudTime.textContent = f.time.toFixed(1);
    hudSpeed.textContent = f.s[3].toFixed(1);
    const hip = Math.floor(f.action / 3), knee = f.action % 3, live = !f.fell && !f.finished;
    keycaps.q.classList.toggle('down', live && hip === 1);
    keycaps.w.classList.toggle('down', live && hip === 2);
    keycaps.o.classList.toggle('down', live && knee === 1);
    keycaps.p.classList.toggle('down', live && knee === 2);
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

  Object.entries(keycaps).forEach(([k, el]) => {
    const down = (e) => { e.preventDefault(); if (focusId !== 'human') selectBrain('human'); setKey(k, true); };
    const up = (e) => { e.preventDefault(); setKey(k, false); };
    el.addEventListener('pointerdown', down);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointerleave', up);
    el.addEventListener('pointercancel', up);
  });

  function selectBrain(id) {
    if (!controllers[id]) return;
    focusId = id;
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
