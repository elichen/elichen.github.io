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
