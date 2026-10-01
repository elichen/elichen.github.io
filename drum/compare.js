// Comparing outlines: the network's guess against a drum, after the turn,
// mirror and starting point that best line them up (the notes carry none of
// those), scored by how much of their area they share.

// Both outlines as complex samples, centred and scaled to unit area.
export function toUnitArea(p) {
  const n = p.length / 2;
  let a = 0, cx = 0, cy = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n, cr = p[2 * i] * p[2 * j + 1] - p[2 * j] * p[2 * i + 1];
    a += cr; cx += (p[2 * i] + p[2 * j]) * cr; cy += (p[2 * i + 1] + p[2 * j + 1]) * cr;
  }
  a /= 2; cx /= 6 * a; cy /= 6 * a;
  const s = 1 / Math.sqrt(Math.abs(a));
  return Float64Array.from(p, (v, i) => (v - (i % 2 ? cy : cx)) * s);
}

// Turn, mirror and re-start the guess so it lines up best with the truth
// (the notes carry no orientation, so this is fair game).
export function align(guess, truth) {
  const n = guess.length / 2;
  let best = { m: -1 };
  for (let v = 0; v < 4; v++) {
    const g = new Float64Array(2 * n);
    for (let k = 0; k < n; k++) {
      const src = v & 1 ? (n - k) % n : k;
      g[2 * k] = guess[2 * src];
      g[2 * k + 1] = v & 2 ? -guess[2 * src + 1] : guess[2 * src + 1];
    }
    for (let s = 0; s < n; s++) {
      let re = 0, im = 0;   // <truth, g shifted by s>
      for (let k = 0; k < n; k++) {
        const j = (k - s + n) % n;
        const tx = truth[2 * k], ty = truth[2 * k + 1], gx = g[2 * j], gy = g[2 * j + 1];
        re += tx * gx + ty * gy;
        im += ty * gx - tx * gy;
      }
      const m = Math.hypot(re, im);
      if (m > best.m) best = { m, g, s, c: re / m, si: im / m };
    }
  }
  const { g, s, c, si } = best, out = new Float64Array(2 * n);
  for (let k = 0; k < n; k++) {
    const j = (k - s + n) % n;
    out[2 * k] = c * g[2 * j] - si * g[2 * j + 1];
    out[2 * k + 1] = si * g[2 * j] + c * g[2 * j + 1];
  }
  return out;
}

function insidePoly(p, x, y) {
  let c = false;
  for (let i = 0, n = p.length / 2, j = n - 1; i < n; j = i++) {
    const ax = p[2 * j], ay = p[2 * j + 1], bx = p[2 * i], by = p[2 * i + 1];
    if ((ay > y) !== (by > y) && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) c = !c;
  }
  return c;
}

export function overlap(a, b) {
  const G = 120, R = 1.4;
  let both = 0, either = 0;
  for (let i = 0; i < G; i++) {
    for (let j = 0; j < G; j++) {
      const x = -R + (2 * R * (i + 0.5)) / G, y = -R + (2 * R * (j + 0.5)) / G;
      const A = insidePoly(a, x, y), B = insidePoly(b, x, y);
      if (A && B) both++;
      if (A || B) either++;
    }
  }
  return both / Math.max(1, either);
}
