const test = require("node:test");
const assert = require("node:assert/strict");
const { DreamfieldMLP } = require("./model.js");

function sample(model, x, y, rgb) {
  return { x, y, rgb, features: model.encode(x, y) };
}

test("Fourier encoder has the declared shape and known center values", () => {
  const model = new DreamfieldMLP({ hiddenSize: 4, bands: 3, seed: 7 });
  const encoded = model.encode(0, 0);

  assert.equal(model.inputSize, 14);
  assert.equal(encoded.length, model.inputSize);
  assert.deepEqual(Array.from(encoded), [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
});
test("backpropagation agrees with finite differences", () => {
  const model = new DreamfieldMLP({ hiddenSize: 5, bands: 2, seed: 23 });
  const trainingSample = sample(model, 0.27, -0.41, [0.82, 0.13, 0.56]);
  const batch = [trainingSample];
  const checks = [
    ["w3", 3],
    ["b3", 0],
    ["w2", 7],
    ["w1", 11]
  ];

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
      Math.abs(numerical - analytical) / scale < 0.025,
      `${name}[${index}] numerical=${numerical} analytical=${analytical}`
    );
  }
});

test("Adam training overfits a single painted color", () => {
  const model = new DreamfieldMLP({ hiddenSize: 10, bands: 2, seed: 101, learningRate: 0.008 });
  const trainingSample = sample(model, -0.2, 0.35, [0.91, 0.12, 0.63]);
  const initialLoss = model.lossForBatch([trainingSample]);

  for (let step = 0; step < 500; step += 1) model.trainBatch([trainingSample]);

  const finalLoss = model.lossForBatch([trainingSample]);
  const prediction = model.predict(trainingSample.x, trainingSample.y);
  assert.ok(finalLoss < initialLoss * 0.001, `loss ${initialLoss} -> ${finalLoss}`);
  prediction.forEach((value) => assert.ok(Number.isFinite(value)));
});

test("the network learns four distinct corner examples", () => {
  const model = new DreamfieldMLP({ hiddenSize: 20, bands: 3, seed: 304, learningRate: 0.007 });
  const trainingSamples = [
    sample(model, -1, -1, [0.92, 0.08, 0.12]),
    sample(model, 1, -1, [0.08, 0.24, 0.94]),
    sample(model, -1, 1, [0.06, 0.84, 0.42]),
    sample(model, 1, 1, [0.94, 0.86, 0.12])
  ];

  const initialLoss = model.lossForBatch(trainingSamples);
  for (let step = 0; step < 1200; step += 1) model.trainBatch(trainingSamples);
  const finalLoss = model.lossForBatch(trainingSamples);

  assert.ok(finalLoss < 0.0002, `loss ${initialLoss} -> ${finalLoss}`);
  assert.ok(model.step === 1200);
  for (const values of Object.values(model.params)) {
    for (const value of values) assert.ok(Number.isFinite(value));
  }
});

test("the Doodle Apprentice separates bold ink from blank paper", () => {
  const model = new DreamfieldMLP({ hiddenSize: 28, bands: 4, seed: 811, learningRate: 0.0065 });
  const ink = [];
  const paper = [];
  const inkRgb = [0.086, 0.094, 0.137];
  const paperRgb = [0.929, 0.906, 0.839];

  for (let row = 0; row < 13; row += 1) {
    const y = row / 12 * 2 - 1;
    for (let column = 0; column < 17; column += 1) {
      const x = column / 16 * 2 - 1;
      const isCross = Math.abs(x) < 0.17 || Math.abs(y) < 0.2;
      (isCross ? ink : paper).push(sample(model, x, y, isCross ? inkRgb : paperRgb));
    }
  }

  for (let step = 0; step < 1500; step += 1) {
    const batch = [];
    for (let index = 0; index < 16; index += 1) {
      batch.push(ink[(step * 7 + index * 3) % ink.length]);
      batch.push(paper[(step * 11 + index * 5) % paper.length]);
    }
    model.trainBatch(batch);
  }

  const center = model.predict(0, 0);
  const corner = model.predict(0.92, 0.92);
  const centerBrightness = (center[0] + center[1] + center[2]) / 3;
  const cornerBrightness = (corner[0] + corner[1] + corner[2]) / 3;

  assert.equal(model.parameterCount, 1431);
  assert.ok(centerBrightness < 0.25, `expected dark ink at center, got ${centerBrightness}`);
  assert.ok(cornerBrightness > 0.75, `expected blank paper at corner, got ${cornerBrightness}`);
  assert.ok(cornerBrightness - centerBrightness > 0.58);
});
