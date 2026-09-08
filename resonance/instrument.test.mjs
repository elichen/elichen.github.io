import test from "node:test";
import assert from "node:assert/strict";
import { solveMembrane, insideOutline } from "./physics.mjs";
import {
  PRESETS,
  createInstrument,
  validateInstrument,
  instrumentOutline,
  frequenciesFor,
  strikeOptions,
  drawnInstrument,
} from "./instrument.mjs";

const near = (a, b, tolerance = 1e-10) =>
  assert.ok(Math.abs(a - b) <= tolerance, `${a} ≠ ${b}`);
const instrument = createInstrument();
const solution = solveMembrane(instrumentOutline(instrument), {
  resolution: 41,
  modeCount: 18,
});

test("shared instruments reject malformed input and bound geometry and tuning", () => {
  for (const invalid of [
    null,
    1,
    {},
    { points: [] },
    { points: Array(16).fill(null) },
  ]) {
    assert.equal(validateInstrument(invalid), null);
  }
  const nonfinite = structuredClone(instrument);
  nonfinite.points[7].x = NaN;
  assert.equal(validateInstrument(nonfinite), null);
  const extreme = {
    preset: "unknown",
    points: Array.from({ length: 16 }, (_, i) => ({ x: i % 2 ? 50 : 0, y: 0 })),
    width: 999,
    tension: -10,
    decay: 0,
    mallet: 999,
  };
  const before = structuredClone(extreme);
  const valid = validateInstrument(extreme);
  assert.deepEqual(
    extreme,
    before,
    "validation should not mutate the shared data",
  );
  assert.equal(valid.preset, "custom");
  assert.equal(valid.width, 115);
  assert.equal(valid.tension, 15);
  assert.equal(valid.decay, 0.4);
  assert.equal(valid.mallet, 100);
  valid.points.forEach((p, i) => {
    const radius = Math.hypot(p.x, p.y);
    near(radius, i % 2 ? 0.86 : 0.28);
    near(p.x, radius * Math.cos((i * Math.PI) / 8));
    near(p.y, radius * Math.sin((i * Math.PI) / 8));
  });
  const fallback = validateInstrument({
    ...instrument,
    width: NaN,
    tension: "40",
    decay: Infinity,
    mallet: null,
  });
  assert.equal(fallback.width, 100);
  assert.equal(fallback.tension, 50);
  assert.equal(fallback.decay, 2.5);
  assert.equal(fallback.mallet, 55);
  for (const preset of PRESETS)
    assert.equal(validateInstrument(createInstrument(preset)).preset, preset);
});

test("smooth outlines remain bounded, star-shaped, and physically solvable at control limits", () => {
  for (const width of [70, 115]) {
    const value = validateInstrument({
      ...instrument,
      width,
      points: instrument.points.map((p, i) => ({
        x: p.x * (i % 2 ? 10 : 0.01),
        y: p.y * (i % 2 ? 10 : 0.01),
      })),
    });
    const outline = instrumentOutline(value);
    assert.equal(outline.length, 128);
    assert.ok(insideOutline(outline, 0, 0));
    for (const p of outline) {
      assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y));
      assert.ok(Math.abs(p.x) <= 0.975 && Math.abs(p.y) <= 0.86);
    }
    // Positive oriented edges around the origin ensure a simple radial contour.
    outline.forEach((a, i) => {
      const b = outline[(i + 1) % outline.length];
      assert.ok(a.x * b.y - a.y * b.x > 0);
    });
    const result = solveMembrane(outline);
    assert.ok(
      result.modes.every(
        (m) => Number.isFinite(m.eigenvalue) && m.eigenvalue > 0,
      ),
    );
    assert.ok(result.residualMax < 2e-7);
  }
});

test("tension follows its square-root law and changing width changes physical pitch", () => {
  const base = frequenciesFor(solution, instrument);
  const higher = frequenciesFor(solution, { ...instrument, tension: 100 });
  const lower = frequenciesFor(solution, { ...instrument, tension: 25 });
  base.forEach((frequency, i) => {
    near(higher[i] / frequency, Math.sqrt(2));
    near(lower[i] / frequency, Math.sqrt(0.5));
  });
  const narrowInstrument = { ...instrument, width: 70 };
  const wideInstrument = { ...instrument, width: 115 };
  const narrow = solveMembrane(instrumentOutline(narrowInstrument), {
    resolution: 41,
  });
  const wide = solveMembrane(instrumentOutline(wideInstrument), {
    resolution: 41,
  });
  assert.ok(frequenciesFor(narrow, narrowInstrument)[0] > base[0] * 1.15);
  assert.ok(frequenciesFor(wide, wideInstrument)[0] < base[0] * 0.98);
  // Width belongs in the geometry solve, not a second pitch multiplier.
  assert.deepEqual(frequenciesFor(solution, narrowInstrument), base);
});

test("contact-pickup sound and reconstructed motion are invariant to eigenvector sign", () => {
  const flipped = {
    ...solution,
    modes: solution.modes.map((mode, i) => ({
      ...mode,
      values: Float32Array.from(
        mode.values,
        (value) => value * (i % 3 ? -1 : 1),
      ),
    })),
  };
  const original = strikeOptions(solution, instrument, 0.23, -0.11);
  const transformed = strikeOptions(flipped, instrument, 0.23, -0.11);
  original.amplitudes.forEach((amplitude, i) =>
    near(amplitude, transformed.amplitudes[i], 1e-15),
  );
  original.displacement.forEach((coefficient, i) => {
    near(coefficient, transformed.displacement[i] * (i % 3 ? -1 : 1), 1e-15);
    for (let j = 0; j < solution.coordinates.length; j += 97) {
      near(
        coefficient * solution.modes[i].values[j],
        transformed.displacement[i] * flipped.modes[i].values[j],
        1e-15,
      );
    }
  });
  const solo = strikeOptions(solution, instrument, 0.22, 0.13, 5);
  solo.amplitudes.forEach((amplitude, i) => {
    if (i !== 5) assert.ok(amplitude === 0);
  });
  assert.notEqual(solo.amplitudes[5], 0);
  const away = strikeOptions(solution, instrument, 2, 2);
  assert.deepEqual(away, strikeOptions(solution, instrument, 0, 0));
});

test("a soft mallet spreads its impact and reduces relative high-mode excitation", () => {
  const soft = strikeOptions(solution, { ...instrument, mallet: 0 });
  const hard = strikeOptions(solution, { ...instrument, mallet: 100 });
  const highToLowEnergy = (options) => {
    const low = options.displacement
      .slice(0, 3)
      .reduce((sum, x) => sum + x * x, 0);
    const high = options.displacement
      .slice(9)
      .reduce((sum, x) => sum + x * x, 0);
    return high / low;
  };
  assert.ok(highToLowEnergy(soft) < highToLowEnergy(hard) * 0.7);
  assert.ok(soft.brightness < hard.brightness);
  assert.deepEqual(
    soft.frequencies,
    hard.frequencies,
    "mallets should change excitation, not modal frequencies",
  );
});

test("the smallest valid membrane still has an audible pickup and a valid default strike", () => {
  const smallest = validateInstrument({
    ...instrument,
    width: 70,
    mallet: 100,
    points: instrument.points.map((p) => ({ x: p.x * 0.01, y: p.y * 0.01 })),
  });
  const result = solveMembrane(instrumentOutline(smallest), { resolution: 41 });
  assert.equal(insideOutline(result.outline, 0.12, 0.08), true);
  assert.equal(insideOutline(result.outline, -0.11, 0.12), true);
  const defaults = strikeOptions(result, smallest);
  const centered = strikeOptions(result, smallest, 0, 0);
  assert.deepEqual(defaults, strikeOptions(result, smallest, 0.12, 0.08));
  assert.deepEqual(strikeOptions(result, smallest, 0.22, 0.13), centered);
  assert.deepEqual(strikeOptions(result, smallest, NaN, Infinity), centered);
  assert.ok(Math.abs(defaults.amplitudes[0]) > 1e-5);
});

test("freehand conversion fits and centers a contour while preserving tuning and bounds", () => {
  const path = Array.from({ length: 100 }, (_, i) => {
    const angle = (i * Math.PI) / 50;
    return { x: 7 + Math.cos(angle) * 2, y: -5 + Math.sin(angle) };
  });
  const previous = {
    ...instrument,
    tension: 72,
    decay: 4.2,
    mallet: 18,
    width: 75,
  };
  const before = structuredClone(previous);
  const drawn = drawnInstrument(path, previous);
  assert.equal(drawn.preset, "custom");
  assert.equal(drawn.width, 100);
  assert.equal(drawn.tension, 72);
  assert.equal(drawn.decay, 4.2);
  assert.equal(drawn.mallet, 18);
  assert.deepEqual(previous, before);
  assert.equal(drawn.points.length, 16);
  drawn.points.forEach((p) =>
    assert.ok(
      Math.hypot(p.x, p.y) >= 0.28 - 1e-12 &&
        Math.hypot(p.x, p.y) <= 0.86 + 1e-12,
    ),
  );
  assert.ok(Math.abs(drawn.points[0].x) > Math.abs(drawn.points[4].y) * 1.8);
  const translated = drawnInstrument(
    path.map((p) => ({ x: p.x * 3 + 20, y: p.y * 3 - 30 })),
    previous,
  );
  drawn.points.forEach((p, i) => {
    near(p.x, translated.points[i].x);
    near(p.y, translated.points[i].y);
  });
  assert.ok(insideOutline(instrumentOutline(drawn), 0, 0));
  assert.equal(drawnInstrument([], previous), null);
  assert.equal(drawnInstrument(null, previous), null);
  assert.equal(
    drawnInstrument(Array(8).fill({ x: NaN, y: 0 }), previous),
    null,
  );
  assert.equal(
    drawnInstrument(
      path.map((p) => ({ x: p.x * 0.001, y: p.y * 0.001 })),
      previous,
    ),
    null,
  );
});
