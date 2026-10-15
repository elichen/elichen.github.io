// Double-precision CPU version of one Sim step (sim.js), for checking the shaders.
// Same order of operations; a full cell list with no per-cell cap.

export const RC = 2 ** (1 / 6);

function minImage(d, L) { return d - L * Math.round(d / L); }

function pair(r2) {           // |F| / r for WCA between molecules
  const ir2 = 1 / Math.max(r2, 0.36), ir6 = ir2 * ir2 * ir2;
  return (48 * ir6 - 24) * ir6 * ir2;
}

function shifted(dx, dy, delta) {   // force on the first body, d = x1 - x2
  const reach = delta + RC, r2 = dx * dx + dy * dy;
  if (r2 >= reach * reach) return [0, 0];
  const r = Math.sqrt(r2), is = 1 / Math.max(r - delta, 0.6), is6 = is ** 6;
  const k = (48 * is6 - 24) * is6 * is / Math.max(r, 1e-6);
  return [dx * k, dy * k];
}

export function wcaEnergy(r) { return r < RC ? 4 * (r ** -12 - r ** -6) + 1 : 0; }

function cells(pos, n, L) {
  const nx = Math.floor(L[0] / RC), ny = Math.floor(L[1] / RC);
  const head = new Int32Array(nx * ny).fill(-1), next = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const cx = Math.min(nx - 1, Math.floor(pos[2 * i] * nx / L[0]));
    const cy = Math.min(ny - 1, Math.floor(pos[2 * i + 1] * ny / L[1]));
    const c = cy * nx + cx;
    next[i] = head[c]; head[c] = i;
  }
  return { nx, ny, head, next };
}

// Calls fn(i, j, dx, dy, r2) once per molecule pair closer than RC.
export function forEachPair(pos, n, L, fn) {
  const { nx, ny, head, next } = cells(pos, n, L);
  for (let cy = 0; cy < ny; cy++) for (let cx = 0; cx < nx; cx++) {
    for (let i = head[cy * nx + cx]; i >= 0; i = next[i]) {
      for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
        const c = ((cy + oy + ny) % ny) * nx + (cx + ox + nx) % nx;
        for (let j = head[c]; j >= 0; j = next[j]) {
          if (j <= i) continue;
          const dx = minImage(pos[2 * i] - pos[2 * j], L[0]), dy = minImage(pos[2 * i + 1] - pos[2 * j + 1], L[1]);
          const r2 = dx * dx + dy * dy;
          if (r2 < RC * RC) fn(i, j, dx, dy, r2);
        }
      }
    }
  }
}

export function potentialEnergy(st, L) {
  let u = 0;
  forEachPair(st.pos, st.n, L, (i, j, dx, dy, r2) => { u += wcaEnergy(Math.sqrt(r2)); });
  const bodyPair = (dx, dy, delta) => wcaEnergy(Math.hypot(dx, dy) - delta);
  for (const b of st.beads) {
    for (let i = 0; i < st.n; i++) {
      u += bodyPair(minImage(st.pos[2 * i] - b.x[0], L[0]), minImage(st.pos[2 * i + 1] - b.x[1], L[1]), b.a - 0.5);
    }
  }
  for (let p = 0; p < st.beads.length; p++) for (let q = p + 1; q < st.beads.length; q++) {
    const B = st.beads[p], O = st.beads[q];
    u += bodyPair(minImage(B.x[0] - O.x[0], L[0]), minImage(B.x[1] - O.x[1], L[1]), B.a + O.a - 1);
  }
  return u;
}

// One step; st = { n, pos, vel, beads: [{x, v, wrap, a, m, ext}], lambda, p (sum of molecule velocities) },
// returns the next state.
export function step(st, cfg) {
  const { L, dt } = cfg, n = st.n;
  const f = new Float64Array(2 * n);
  forEachPair(st.pos, n, L, (i, j, dx, dy, r2) => {
    const k = pair(r2);
    f[2 * i] += dx * k; f[2 * i + 1] += dy * k;
    f[2 * j] -= dx * k; f[2 * j + 1] -= dy * k;
  });
  const fluid = st.beads.map(() => [0, 0]);
  let ex = 0, ey = 0;
  st.beads.forEach(b => { ex += b.ext?.[0] ?? 0; ey += b.ext?.[1] ?? 0; });
  for (let i = 0; i < n; i++) {
    st.beads.forEach((b, k) => {
      const [fx, fy] = shifted(minImage(st.pos[2 * i] - b.x[0], L[0]), minImage(st.pos[2 * i + 1] - b.x[1], L[1]), b.a - 0.5);
      f[2 * i] += fx; f[2 * i + 1] += fy;
      fluid[k][0] -= fx; fluid[k][1] -= fy;
    });
    f[2 * i] -= ex / n; f[2 * i + 1] -= ey / n;
  }
  const pos = new Float64Array(2 * n), vel = new Float64Array(2 * n);
  let ke = 0, px = 0, py = 0;
  const u = [st.p[0] / n, st.p[1] / n];
  for (let i = 0; i < 2 * n; i++) {
    const v0 = u[i % 2] + (st.vel[i] - u[i % 2]) * st.lambda, v1 = v0 + f[i] * dt;
    const Li = L[i % 2];
    let x1 = st.pos[i] + v1 * dt;
    x1 -= Li * Math.floor(x1 / Li);
    if (x1 >= Li) x1 = 0;
    pos[i] = x1; vel[i] = v1;
    ke += ((v0 + v1) / 2) ** 2;
    if (i % 2) py += v1; else px += v1;
  }
  const beads = st.beads.map((B, k) => {
    let Fx = fluid[k][0], Fy = fluid[k][1];
    st.beads.forEach((O, c) => {
      if (c === k) return;
      const [fx, fy] = shifted(minImage(B.x[0] - O.x[0], L[0]), minImage(B.x[1] - O.x[1], L[1]), B.a + O.a - 1);
      Fx += fx; Fy += fy;
    });
    const v = [B.v[0] + Fx * dt / B.m, B.v[1] + Fy * dt / B.m];
    const x = [B.x[0] + v[0] * dt, B.x[1] + v[1] * dt];
    const wrap = [...B.wrap];
    for (const d of [0, 1]) {
      const fl = Math.floor(x[d] / L[d]);
      x[d] -= L[d] * fl; wrap[d] += fl;
      if (x[d] >= L[d]) { x[d] = 0; wrap[d]++; }
    }
    return { x, v, wrap, a: B.a, m: B.m, ext: [0, 0], f: fluid[k] };
  });
  const T = ke / (2 * n);
  const lambda = cfg.thermostat
    ? Math.sqrt(Math.min(1.21, Math.max(0.81, 1 + dt / cfg.tauT * (cfg.T / Math.max(T, 1e-3) - 1))))
    : 1;
  return { n, pos, vel, beads, lambda, T, p: [px, py] };
}
