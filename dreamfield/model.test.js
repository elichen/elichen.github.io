const test = require("node:test");
const assert = require("node:assert/strict");
const {
  DreamfieldStyleModel,
  encodeStyleFeatures,
  extractStyleDataset,
  resampleStroke,
  stylizePath
} = require("./model.js");

function styleSample(t, target) {
  return {
    input: encodeStyleFeatures({
      t,
      tangent: { x: Math.cos(t * 1.7), y: Math.sin(t * 1.7) },
      curvature: Math.sin(t * 4) * 0.3,
      length: 1.4,
      closed: false,
      grainNormal: Math.sin(t * 19) * 0.7,
      grainTangent: Math.cos(t * 13) * 0.5
    }),
    target: new Float32Array(target)
  };
}

test("the style apprentice is a 690-parameter 12→18→18→6 MLP", () => {
  const model = new DreamfieldStyleModel({ seed: 7 });
  const encoded = encodeStyleFeatures({
    t: 0.5,
    tangent: { x: 1, y: 0 },
    curvature: 0,
    length: 1.5,
    closed: false,
    grainNormal: 0.2,
    grainTangent: -0.3
  });

  assert.equal(model.inputSize, 12);
  assert.equal(model.hiddenSize, 18);
  assert.equal(model.outputSize, 6);
  assert.equal(model.parameterCount, 690);
  assert.equal(encoded.length, 12);
  assert.ok(Math.abs(encoded[0]) < 1e-7);
  assert.ok(Math.abs(encoded[1]) < 1e-6);
  assert.ok(Math.abs(encoded[2] + 1) < 1e-6);
  assert.equal(encoded[5], 1);
  assert.equal(encoded[9], -1);
  assert.ok(Math.abs(encoded[10] - 0.2) < 1e-6);
  assert.ok(Math.abs(encoded[11] + 0.3) < 1e-6);
});

test("arc-length resampling preserves endpoints and evens out point spacing", () => {
  const source = [
    { x: 0, y: 0, pressure: 0.2 },
    { x: 0.1, y: 0, pressure: 0.3 },
    { x: 1, y: 0, pressure: 0.8 }
  ];
  const result = resampleStroke(source, 6);

  assert.equal(result.length, 6);
  assert.deepEqual(result[0], source[0]);
  assert.deepEqual(result[5], source[2]);
  for (let index = 1; index < result.length; index += 1) {
    assert.ok(Math.abs(result[index].x - result[index - 1].x - 0.2) < 1e-6);
  }
});

test("backpropagation agrees with finite differences for mixed tanh and sigmoid outputs", () => {
  const model = new DreamfieldStyleModel({ hiddenSize: 7, seed: 23 });
  const batch = [styleSample(0.37, [0.31, -0.22, 0.78, 0.91, 0.16, 0.42])];
  const checks = [["w3", 5], ["b3", 0], ["w2", 17], ["w1", 29]];
  const { gradients } = model.computeGradients(batch);
  const epsilon = 1e-3;

  for (const [name, index] of checks) {
    const value = model.params[name][index];
    model.params[name][index] = value + epsilon;
    const positive = model.lossForBatch(batch);
    model.params[name][index] = value - epsilon;
    const negative = model.lossForBatch(batch);
    model.params[name][index] = value;
    const numerical = (positive - negative) / (2 * epsilon);
    const analytical = gradients[name][index];
    const scale = Math.max(1e-4, Math.abs(numerical), Math.abs(analytical));
    assert.ok(
      Math.abs(numerical - analytical) / scale < 0.03,
      `${name}[${index}] numerical=${numerical} analytical=${analytical}`
    );
  }
});

test("training learns stroke residuals, brush weight, and palette instead of canvas pixels", () => {
  const model = new DreamfieldStyleModel({ hiddenSize: 8, seed: 101, learningRate: 0.01 });
  const samples = Array.from({ length: 24 }, (_, index) => {
    const t = index / 23;
    return styleSample(t, [
      0.3 * Math.sin(index),
      0.1 * Math.cos(index),
      0.82,
      0.95, 0.2, 0.1
    ]);
  });
  const initialLoss = model.lossForBatch(samples);
  for (let step = 0; step < 1200; step += 1) model.trainBatch(samples);
  const finalLoss = model.lossForBatch(samples);
  const prediction = model.predict(samples[9].input);

  assert.ok(finalLoss < initialLoss * 0.001, `loss ${initialLoss} -> ${finalLoss}`);
  assert.ok(prediction[2] > 0.76, `expected a heavy learned brush, got ${prediction[2]}`);
  assert.ok(prediction[3] > 0.9 && prediction[4] < 0.26 && prediction[5] < 0.18, `unexpected learned RGB ${prediction.slice(3)}`);
  for (const values of Object.values(model.params)) {
    for (const value of values) assert.ok(Number.isFinite(value));
  }
});

test("style extraction distinguishes a steady line from a wobbly hand and records its palette", () => {
  const straight = Array.from({ length: 70 }, (_, index) => ({
    x: -0.8 + index / 69 * 1.6,
    y: 0.1,
    pressure: 0.5
  }));
  const wobbly = Array.from({ length: 70 }, (_, index) => ({
    x: -0.8 + index / 69 * 1.6,
    y: 0.1 + Math.sin(index * 0.82) * 0.075,
    pressure: 0.35 + index / 69 * 0.4
  }));
  const calm = extractStyleDataset([{ points: straight, color: "#171813", width: 0.25 }]);
  const lively = extractStyleDataset([{ points: wobbly, color: "#2664ff", width: 0.9 }]);

  assert.ok(lively.metrics.wobble > calm.metrics.wobble + 0.015, `${calm.metrics.wobble} vs ${lively.metrics.wobble}`);
  assert.ok(lively.metrics.width > calm.metrics.width + 0.4);
  assert.deepEqual(lively.palette.map((entry) => entry.color), ["#2664FF"]);
  assert.ok(lively.samples.length >= 20);
  assert.equal(lively.samples[0].input.length, 12);
  assert.equal(lively.samples[0].target.length, 6);
});

test("generation follows a new subject scaffold and a new seed creates a new variation", () => {
  const source = Array.from({ length: 80 }, (_, index) => {
    const t = index / 79;
    return { x: -0.8 + 1.6 * t, y: Math.sin(t * 30) * 0.06, pressure: 0.7 };
  });
  const lesson = extractStyleDataset([{ points: source, color: "#2664ff", width: 0.8 }]);
  const model = new DreamfieldStyleModel({ seed: 9, learningRate: 0.009 });
  for (let step = 0; step < 1000; step += 1) {
    const batch = Array.from({ length: 32 }, (_, index) => lesson.samples[(step * 7 + index * 3) % lesson.samples.length]);
    model.trainBatch(batch);
  }

  const newVerticalSubject = Array.from({ length: 40 }, (_, index) => ({ x: 0, y: -0.75 + index / 39 * 1.5 }));
  const first = stylizePath(model, newVerticalSubject, { seed: 12 });
  const repeat = stylizePath(model, newVerticalSubject, { seed: 12 });
  const variation = stylizePath(model, newVerticalSubject, { seed: 13 });
  const yRange = Math.max(...first.map((point) => point.y)) - Math.min(...first.map((point) => point.y));
  const xRange = Math.max(...first.map((point) => point.x)) - Math.min(...first.map((point) => point.x));
  const seedDifference = Math.max(...first.map((point, index) => Math.hypot(point.x - variation[index].x, point.y - variation[index].y)));

  assert.deepEqual(first, repeat, "the same generation seed should be reproducible");
  assert.ok(yRange > 1.35, `generated path should retain the new vertical content, range=${yRange}`);
  assert.ok(xRange < 0.3, `generated path should not reconstruct the horizontal lesson, range=${xRange}`);
  assert.ok(seedDifference > 0.01, `fresh grain should make a visibly new variation, difference=${seedDifference}`);
  assert.ok(first.every((point) => point.color[2] > point.color[0]), "the learned cobalt palette should transfer");
});
