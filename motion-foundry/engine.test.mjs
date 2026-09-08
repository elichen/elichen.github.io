import test from "node:test";
import assert from "node:assert/strict";
import {
  TAU,
  mechanismAt,
  poseAt,
  sampleMechanism,
  presetDesign,
  presetTarget,
  validateDesign,
  resampleClosed,
  transformPoint,
} from "./kinematics.mjs";
import {
  MechanismSearch,
  fitDesignToTarget,
  measureDesign,
  compareCurves,
} from "./optimizer.mjs";

const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const near = (actual, expected, tolerance = 1e-9) =>
  assert.ok(
    Math.abs(actual - expected) < tolerance,
    `${actual} versus ${expected}`,
  );
const identity = { a: 1, b: 0, tx: 0, ty: 0 };
const transform = { a: 1.1, b: -0.6, tx: 2.4, ty: -1.2 };

test("analytic four-bar and slider closure preserves every rigid length for a complete crank turn", () => {
  const designs = [
    presetDesign("stride"),
    presetDesign("eight"),
    presetDesign("slider"),
  ];
  designs.push({
    ...presetDesign("stride"),
    params: { ...presetDesign("stride").params, branch: -1, traceOffset: -0.7 },
  });
  designs.push({
    ...presetDesign("slider"),
    params: { ...presetDesign("slider").params, branch: -1, railOffset: -0.8 },
  });
  for (const original of designs) {
    const design = validateDesign({ ...original, transform });
    assert.ok(design);
    const p = design.params,
      scale = Math.hypot(transform.a, transform.b);
    const start = mechanismAt(design, 0);
    for (let i = 0; i <= 512; i++) {
      const pose = mechanismAt(design, (i * TAU) / 512);
      assert.ok(pose.valid);
      const { O, A, B, G } = pose.joints;
      near(distance(O, A), p.crank * scale);
      near(distance(A, B), p.coupler * scale);
      near(
        distance(A, pose.tracer),
        Math.hypot(p.traceAlong, p.traceOffset) * p.coupler * scale,
      );
      near(distance(O, start.joints.O), 0);
      near(distance(G, start.joints.G), 0);
      if (design.family === "fourbar") {
        near(distance(B, G), p.rocker * scale);
        near(distance(O, G), scale);
      } else {
        const [left, right] = pose.guide;
        near(
          (B.x - left.x) * (right.y - left.y) -
            (B.y - left.y) * (right.x - left.x),
          0,
        );
      }
    }
    const end = mechanismAt(design, TAU);
    near(distance(start.tracer, end.tracer), 0);
    assert.deepEqual(poseAt(design, 0).P, start.tracer);
  }
});

test("full-cycle validation rejects toggles and impossible closure without altering input", () => {
  const good = presetDesign("stride");
  const before = structuredClone(good);
  assert.ok(validateDesign(good));
  assert.deepEqual(good, before);
  assert.equal(validateDesign(null), null);
  assert.equal(validateDesign({ ...good, family: "magic" }), null);
  assert.equal(
    validateDesign({ ...good, params: { ...good.params, crank: 1 } }),
    null,
  );
  assert.equal(
    validateDesign({
      ...good,
      params: { ...good.params, coupler: 0.1, rocker: 0.1 },
    }),
    null,
  );
  assert.equal(
    validateDesign({ ...good, params: { ...good.params, traceOffset: NaN } }),
    null,
  );
  assert.equal(
    validateDesign({ ...good, params: { ...good.params, branch: 0 } }),
    null,
  );
  assert.equal(
    validateDesign({ ...good, transform: { ...identity, a: 0 } }),
    null,
  );
  const slider = presetDesign("slider");
  assert.equal(
    validateDesign({
      ...slider,
      params: {
        ...slider.params,
        coupler: slider.params.crank + Math.abs(slider.params.railOffset),
      },
    }),
    null,
  );
  assert.deepEqual(sampleMechanism(null), []);
});

test("arc-length resampling includes closure and handles repeated endpoints and bad targets", () => {
  const square = [
    { x: 0, y: 0 },
    { x: 2, y: 0 },
    { x: 2, y: 2 },
    { x: 0, y: 2 },
  ];
  const sampled = resampleClosed([...square, square[0], square[0]], 8);
  assert.equal(sampled.length, 8);
  sampled.forEach((p, i) => near(distance(p, sampled[(i + 1) % 8]), 1));
  assert.deepEqual(resampleClosed(Array(8).fill({ x: 1, y: 1 })), []);
  assert.throws(() => new MechanismSearch([]), TypeError);
  assert.throws(
    () => new MechanismSearch(Array(8).fill({ x: 1, y: 1 })),
    RangeError,
  );
  assert.throws(
    () => new MechanismSearch([{ x: Infinity, y: 0 }, ...square]),
    TypeError,
  );
  assert.throws(
    () => new MechanismSearch(square, { family: "unknown" }),
    RangeError,
  );
});

test("similarity fit handles rotation, translation, scale, cyclic start, and reversed traversal", () => {
  const design = {
    family: "fourbar",
    params: {
      crank: 0.44,
      coupler: 0.98,
      rocker: 0.89,
      traceAlong: 1.3,
      traceOffset: 0.6,
      branch: 1,
    },
    transform: identity,
  };
  const points = sampleMechanism(design, 256).map((p) =>
    transformPoint(p, transform),
  );
  const shifted = points.slice(73).concat(points.slice(0, 73)).reverse();
  const fit = fitDesignToTarget(design, shifted);
  assert.ok(fit && validateDesign(fit.design));
  assert.ok(
    fit.error < 0.003,
    `coarse arc-length alignment error ${fit.error}`,
  );
  assert.ok(measureDesign(fit.design, shifted, { samples: 256 }).error < 0.001);
  const reflected = points.map((p) => ({ x: p.x, y: -p.y }));
  const mirrorFit = fitDesignToTarget(design, reflected);
  assert.ok(mirrorFit && validateDesign(mirrorFit.design));
  assert.ok(
    measureDesign(mirrorFit.design, reflected, { samples: 256 }).error < 0.001,
  );
  // Compare corresponding samples independently of the Procrustes formula.
  const rms =
    Math.sqrt(
      fit.curve.reduce(
        (sum, p, i) => sum + distance(p, fit.target[i]) ** 2,
        0,
      ) / fit.curve.length,
    ) / fit.match.diagonal;
  near(rms, fit.error, 1e-10);
});

test("editor measurement keeps a mechanism in place and exposes genuine deviations", () => {
  const design = presetDesign("slider");
  const target = sampleMechanism(design, 256);
  const fit = measureDesign(design, target);
  assert.ok(fit.error < 0.0002);
  const moved = { ...design, transform: { a: 1, b: 0, tx: 2, ty: -1 } };
  const measurement = measureDesign(moved, target);
  assert.ok(measurement.error > 0.5);
  assert.deepEqual(measurement.design.transform, moved.transform);
  assert.ok(fitDesignToTarget(moved, target).error < 0.0002);
  const oval = presetTarget("oval", 256);
  const shifted = oval.slice(49).concat(oval.slice(0, 49)).reverse();
  assert.ok(compareCurves(oval, shifted).error < 0.0005);
});

test("retrieval plus evolution improves and recovers a known synthetic linkage target", () => {
  const design = {
    family: "fourbar",
    params: {
      crank: 0.44,
      coupler: 0.98,
      rocker: 0.89,
      traceAlong: 1.3,
      traceOffset: 0.6,
      branch: 1,
    },
    transform: { a: 0.9, b: 0.4, tx: -0.6, ty: -0.9 },
  };
  const target = sampleMechanism(design, 256);
  const search = new MechanismSearch(target, {
    family: "fourbar",
    seed: 739,
    population: 40,
    maxGenerations: 120,
  });
  const initial = search.step(0);
  const final = search.step(120);
  assert.equal(final.done, true);
  assert.equal(final.generation, 120);
  assert.ok(final.best.error < initial.best.error * 0.3);
  assert.ok(final.best.error < 0.003);
  assert.ok(
    measureDesign(final.best.design, target, { samples: 256 }).error < 0.003,
  );
  for (let i = 1; i < final.history.length; i++)
    assert.ok(final.history[i].error <= final.history[i - 1].error);
  for (const candidate of final.candidates) {
    assert.ok(validateDesign(candidate.design));
    assert.ok(
      candidate.sizeRatio <= 5,
      `machine size ratio ${candidate.sizeRatio}`,
    );
    assert.equal(candidate.curve.length, candidate.target.length);
    assert.ok(
      candidate.curve.every(
        (p) => Number.isFinite(p.x) && Number.isFinite(p.y),
      ),
    );
  }
  assert.equal(search.step(5).evaluations, final.evaluations);
});

test("all target presets produce valid, verified mechanisms and bounded drawing extents", () => {
  for (const name of ["stride", "oval", "eight", "petal"]) {
    const target = presetTarget(name);
    const search = new MechanismSearch(target);
    const initial = search.step(0);
    const result = search.step(150);
    assert.ok(result.best.error < initial.best.error);
    assert.ok(result.best.error < 0.025, `${name}: ${result.best.error}`);
    const verified = measureDesign(result.best.design, target, {
      samples: 256,
    });
    assert.ok(
      Math.abs(verified.error - result.best.error) < 0.0002,
      `${name} coarse/fine discrepancy`,
    );
    // Check actual transformed joints as well as the conservative search bound.
    const points = [];
    for (let i = 0; i < 96; i++) {
      const pose = mechanismAt(result.best.design, (i * TAU) / 96);
      assert.ok(pose.valid);
      points.push(...Object.values(pose.joints), pose.tracer);
    }
    const width =
      Math.max(...points.map((p) => p.x)) - Math.min(...points.map((p) => p.x));
    const height =
      Math.max(...points.map((p) => p.y)) - Math.min(...points.map((p) => p.y));
    assert.ok(Math.hypot(width, height) / result.best.match.diagonal <= 5);
    assert.equal(new Set(result.candidates.map((c) => c.family)).size, 2);
  }
});

test("a seed gives reproducible results regardless of worker chunk size", () => {
  const target = presetTarget("eight");
  const options = {
    family: "slider",
    seed: 51,
    population: 20,
    maxGenerations: 20,
  };
  const a = new MechanismSearch(target, options);
  const b = new MechanismSearch(target, options);
  const whole = a.step(20);
  for (let i = 0; i < 5; i++) b.step(4);
  assert.deepEqual(b.step(0), whole);
});
