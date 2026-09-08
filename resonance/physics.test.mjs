import test from "node:test";
import assert from "node:assert/strict";
import {
  makeOutline,
  solveMembrane,
  sampleMode,
  insideOutline,
} from "./physics.mjs";

const square = [
  { x: -0.8, y: -0.8 },
  { x: 0.8, y: -0.8 },
  { x: 0.8, y: 0.8 },
  { x: -0.8, y: 0.8 },
];

test("square spectrum matches analytic finite-difference eigenvalues, including multiplicities", () => {
  const solution = solveMembrane(square);
  const sideNodes = Math.sqrt(solution.coordinates.length);
  assert.ok(Number.isInteger(sideNodes));
  const expected = [];
  for (let m = 1; m <= sideNodes; m++) {
    for (let n = 1; n <= sideNodes; n++) {
      expected.push(
        (4 / solution.spacing ** 2) *
          (Math.sin((m * Math.PI) / (2 * (sideNodes + 1))) ** 2 +
            Math.sin((n * Math.PI) / (2 * (sideNodes + 1))) ** 2),
      );
    }
  }
  expected.sort((a, b) => a - b);
  solution.modes.forEach((mode, i) => {
    assert.ok(
      Math.abs(mode.eigenvalue / expected[i] - 1) < 1e-9,
      `mode ${i}: computed ${mode.eigenvalue}, expected ${expected[i]}`,
    );
  });
  // Continuous square membrane frequency ratios, within grid discretization.
  const continuous = [
    1,
    Math.sqrt(2.5),
    Math.sqrt(2.5),
    2,
    Math.sqrt(5),
    Math.sqrt(5),
  ];
  continuous.forEach((ratio, i) => {
    assert.ok(Math.abs(solution.modes[i].frequencyRatio / ratio - 1) < 0.025);
  });
});

test("actual returned Float32 modes satisfy the operator and remain mutually orthogonal", () => {
  const solution = solveMembrane(makeOutline("petal", 40));
  const { indices, resolution, spacing } = solution;
  const vectors = solution.modes.map((mode) => mode.values);
  const norms = vectors.map((v) =>
    Math.sqrt(v.reduce((sum, x) => sum + x * x, 0)),
  );
  solution.modes.forEach((mode, m) => {
    let residualSquared = 0;
    for (let cell = 0; cell < indices.length; cell++) {
      const i = indices[cell];
      if (i < 0) continue;
      let laplace = 4 * mode.values[i];
      for (const offset of [-1, 1, -resolution, resolution]) {
        const j = indices[cell + offset];
        if (j >= 0) laplace -= mode.values[j];
      }
      residualSquared +=
        (laplace / spacing ** 2 - mode.eigenvalue * mode.values[i]) ** 2;
    }
    const residual = Math.sqrt(residualSquared) / (mode.eigenvalue * norms[m]);
    assert.ok(residual < 3e-6, `mode ${m} relative residual ${residual}`);
    assert.ok(Math.abs(Math.max(...mode.values.map(Math.abs)) - 1) < 1e-6);
    assert.ok(Math.abs(mode.mass - norms[m] ** 2 * spacing ** 2) < 1e-12);
    for (let n = 0; n < m; n++) {
      const product = mode.values.reduce(
        (sum, x, k) => sum + x * vectors[n][k],
        0,
      );
      assert.ok(
        Math.abs(product / (norms[m] * norms[n])) < 1e-7,
        `modes ${m}, ${n}`,
      );
    }
  });
  assert.ok(solution.residualMax < 2e-7);
});

test("circular drum agrees with Bessel-zero frequency ratios and center strike selection rules", () => {
  const solution = solveMembrane(makeOutline("circle", 64), { resolution: 41 });
  // j_(m,n) / j_(0,1), including the two angular orientations for m > 0.
  // https://www.acs.psu.edu/drussell/demos/membranecircle/circle.html
  const besselRatios = [1, 1.59334, 1.59334, 2.13555, 2.13555, 2.29542];
  besselRatios.forEach((ratio, i) => {
    assert.ok(Math.abs(solution.modes[i].frequencyRatio / ratio - 1) < 0.02);
  });
  assert.ok(Math.abs(sampleMode(solution, 0, 0, 0)) > 0.99);
  for (let i = 1; i <= 4; i++)
    assert.ok(Math.abs(sampleMode(solution, i, 0, 0)) < 1e-5);
  assert.ok(Math.abs(sampleMode(solution, 5, 0, 0)) > 0.99);
});

test("editing the shape changes its spectrum and shrinking raises absolute pitch", () => {
  const outline = makeOutline("circle", 32);
  const original = solveMembrane(outline);
  const smaller = solveMembrane(
    outline.map((p) => ({ x: p.x * 0.7, y: p.y * 0.7 })),
  );
  const elongated = solveMembrane(
    outline.map((p) => ({ x: p.x * 0.48, y: p.y })),
  );
  assert.ok(smaller.fundamental / original.fundamental > 1.35);
  assert.ok(smaller.fundamental / original.fundamental < 1.5);
  assert.ok(
    Math.abs(
      elongated.modes[1].frequencyRatio - original.modes[1].frequencyRatio,
    ) > 0.1,
  );
  const edited = outline.map((p, i) => ({
    x: p.x * (i < 10 ? 0.6 : 1),
    y: p.y,
  }));
  const after = solveMembrane(edited);
  assert.notEqual(after.coordinates.length, original.coordinates.length);
  assert.ok(after.fundamental > original.fundamental);
  assert.ok(after.residualMax < 2e-7);
});

test("bilinear interpolation preserves grid values and fixed zero boundary", () => {
  const solution = solveMembrane(makeOutline("drop", 24));
  solution.coordinates.forEach(({ x, y }, i) => {
    assert.ok(
      Math.abs(sampleMode(solution, 0, x, y) - solution.modes[0].values[i]) <
        1e-6,
    );
  });
  for (const p of solution.outline)
    assert.equal(sampleMode(solution, 0, p.x, p.y), 0);
  for (const p of [
    { x: 1, y: 0 },
    { x: -0.99, y: 0 },
    { x: 0, y: -0.99 },
  ]) {
    assert.equal(sampleMode(solution, 0, p.x, p.y), 0);
  }
  assert.equal(sampleMode(solution, 99, 0, 0), 0);
  assert.equal(sampleMode(solution, 0, NaN, 0), 0);
  assert.equal(insideOutline(square, 0.8, 0), false);
  assert.equal(insideOutline(square, 0.799, 0), true);
});

test("solutions are deterministic, cloneable, and reject invalid geometry", () => {
  const a = solveMembrane(makeOutline("drop"), {
    resolution: 17,
    modeCount: 5,
  });
  const b = solveMembrane(makeOutline("drop"), {
    resolution: 17,
    modeCount: 5,
  });
  assert.deepEqual(a, b);
  const clone = structuredClone(a);
  assert.equal(sampleMode(clone, 0, 0, 0), sampleMode(a, 0, 0, 0));
  assert.throws(() => solveMembrane([]), TypeError);
  assert.throws(() => solveMembrane([{ x: NaN, y: 0 }, ...square]), TypeError);
  assert.throws(() => solveMembrane(square, { resolution: NaN }), TypeError);
  assert.throws(
    () => solveMembrane(square.map((p) => ({ x: p.x * 0.01, y: p.y * 0.01 }))),
    RangeError,
  );
});
