import { insideOutline, makeOutline, sampleMode } from "./physics.mjs";

export const PRESETS = ["circle", "square", "petal", "drop"];
export const NAMES = {
  circle: "Round study",
  square: "Soft square",
  petal: "Five petals",
  drop: "Water drop",
  custom: "Your creation",
};
export const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
export const clone = (value) => JSON.parse(JSON.stringify(value));

export function createInstrument(preset = "circle") {
  return {
    preset,
    points: makeOutline(preset, 16),
    width: 100,
    tension: 50,
    decay: 2.5,
    mallet: 55,
  };
}

/** Validate shared and persisted data before it reaches the numerical solver. */
export function validateInstrument(value) {
  if (
    !value ||
    typeof value !== "object" ||
    !Array.isArray(value.points) ||
    value.points.length !== 16
  )
    return null;
  if (
    !value.points.every(
      (p) => p && Number.isFinite(p.x) && Number.isFinite(p.y),
    )
  )
    return null;
  const points = value.points.map((p, i) => {
    const angle = (i * Math.PI) / 8;
    const radius = clamp(Math.hypot(p.x, p.y), 0.28, 0.86);
    return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
  });
  const bounded = (key, min, max, fallback) =>
    Number.isFinite(value[key]) ? clamp(value[key], min, max) : fallback;
  return {
    preset: [...PRESETS, "custom"].includes(value.preset)
      ? value.preset
      : "custom",
    points,
    width: bounded("width", 70, 115, 100),
    tension: bounded("tension", 15, 100, 50),
    decay: bounded("decay", 0.4, 6, 2.5),
    mallet: bounded("mallet", 0, 100, 55),
  };
}

/** Smooth radial interpolation keeps the outline simple, closed and star-shaped. */
export function instrumentOutline(instrument, steps = 128) {
  const radii = instrument.points.map((p) => Math.hypot(p.x, p.y));
  const count = radii.length;
  return Array.from({ length: steps }, (_, i) => {
    const u = (i / steps) * count,
      j = Math.floor(u),
      t = u - j;
    const p0 = radii[(j - 1 + count) % count],
      p1 = radii[j % count];
    const p2 = radii[(j + 1) % count],
      p3 = radii[(j + 2) % count];
    const radius = clamp(
      0.5 *
        (2 * p1 +
          (-p0 + p2) * t +
          (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
          (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t),
      0.25,
      0.86,
    );
    const angle = (i / steps) * Math.PI * 2;
    return {
      x: clamp(
        (Math.cos(angle) * radius * instrument.width) / 100,
        -0.975,
        0.975,
      ),
      y: Math.sin(angle) * radius,
    };
  });
}

export function frequenciesFor(solution, instrument) {
  const waveSpeed = 320 * Math.sqrt(instrument.tension / 50);
  return solution.modes.map(
    (mode) => (Math.sqrt(mode.eigenvalue) / (2 * Math.PI)) * waveSpeed,
  );
}

/** A smooth spatial impulse and a fixed virtual contact pickup. */
export function strikeOptions(
  solution,
  instrument,
  x = 0.12,
  y = 0.08,
  selectedMode = null,
) {
  if (
    !Number.isFinite(x) ||
    !Number.isFinite(y) ||
    !insideOutline(solution.outline, x, y)
  ) {
    x = 0;
    y = 0;
  }
  const frequencies = frequenciesFor(solution, instrument);
  const radius = 0.015 + (1 - instrument.mallet / 100) * 0.11;
  const samples = [{ x, y, weight: 1 }];
  for (let i = 0; i < 8; i++) {
    const angle = (i * Math.PI) / 4;
    samples.push({
      x: x + Math.cos(angle) * radius,
      y: y + Math.sin(angle) * radius,
      weight: 0.35,
    });
  }
  const totalWeight = samples.reduce((sum, p) => sum + p.weight, 0);
  const displacement = solution.modes.map((mode, i) => {
    if (selectedMode !== null && selectedMode !== i) return 0;
    const excitation =
      samples.reduce(
        (sum, p) => sum + p.weight * sampleMode(solution, i, p.x, p.y),
        0,
      ) / totalWeight;
    return excitation / (mode.mass * frequencies[i] * Math.PI * 2);
  });
  // Both factors change sign when an eigenvector does, leaving the sound invariant.
  // This pickup stays inside even the minimum radius at the narrowest width.
  const amplitudes = displacement.map(
    (value, i) => value * sampleMode(solution, i, -0.11, 0.12),
  );
  return {
    frequencies,
    amplitudes,
    displacement,
    decay: instrument.decay,
    brightness: 0.25 + (instrument.mallet / 100) * 0.65,
    gain: 0.75,
  };
}

/** Turn a freehand contour into a bounded radial membrane, preserving its silhouette. */
export function drawnInstrument(path, previous) {
  if (
    !Array.isArray(path) ||
    path.length < 8 ||
    !path.every((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y))
  )
    return null;
  let xmin = Infinity,
    xmax = -Infinity,
    ymin = Infinity,
    ymax = -Infinity;
  for (const p of path) {
    xmin = Math.min(xmin, p.x);
    xmax = Math.max(xmax, p.x);
    ymin = Math.min(ymin, p.y);
    ymax = Math.max(ymax, p.y);
  }
  if (xmax - xmin < 0.25 || ymax - ymin < 0.25) return null;
  const center = { x: (xmin + xmax) / 2, y: (ymin + ymax) / 2 };
  const scale = 1.55 / Math.max(xmax - xmin, ymax - ymin);
  const samples = path.map((p) => ({
    angle:
      (Math.atan2(p.y - center.y, p.x - center.x) + Math.PI * 2) %
      (Math.PI * 2),
    radius: Math.hypot(p.x - center.x, p.y - center.y) * scale,
  }));
  const points = Array.from({ length: 16 }, (_, i) => {
    const angle = (i * Math.PI) / 8;
    let nearest = Infinity,
      radius = 0.6;
    for (const p of samples) {
      const distance = Math.abs(
        Math.atan2(Math.sin(p.angle - angle), Math.cos(p.angle - angle)),
      );
      if (distance < nearest) {
        nearest = distance;
        radius = p.radius;
      }
    }
    radius = clamp(radius, 0.28, 0.86);
    return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
  });
  return { ...clone(previous), preset: "custom", width: 100, points };
}
