const test = require("node:test");
const assert = require("node:assert/strict");
const { NeuralArtGenome, ACTIVATIONS } = require("./model.js");

function allParameters(genome) {
  return Object.values(genome.params).flatMap((values) => Array.from(values));
}

function outputAt(genome, x, y) {
  return Array.from(genome.forward(x, y, new Float32Array(4)));
}

test("the fixed CPPN topology contains exactly 256 neural parameters", () => {
  const genome = NeuralArtGenome.random(17);
  assert.equal(genome.parameterCount, 256);
  assert.equal(genome.activations1.length, 12);
  assert.equal(genome.activations2.length, 12);
  assert.equal(ACTIVATIONS.length, 4);
});

test("the same seed creates byte-identical genomes and outputs", () => {
  const first = NeuralArtGenome.random(90210);
  const second = NeuralArtGenome.random(90210);

  assert.deepEqual(first.serialize(), second.serialize());
  for (const [x, y] of [[0, 0], [-0.7, 0.2], [0.81, -0.63]]) {
    assert.deepEqual(outputAt(first, x, y), outputAt(second, x, y));
  }
});

test("zero-strength mutation is an exact clone and never changes its parent", () => {
  const parent = NeuralArtGenome.random(404);
  const before = parent.serialize();
  const child = parent.mutate({ seed: 405, strength: 0, generationBorn: 2 });

  assert.deepEqual(parent.serialize(), before);
  assert.deepEqual(allParameters(child), allParameters(parent));
  assert.deepEqual(Array.from(child.activations1), Array.from(parent.activations1));
  assert.deepEqual(child.style, parent.style);
  assert.notEqual(child.seed, parent.seed);
});

test("ordinary mutation changes genes without changing the parent", () => {
  const parent = NeuralArtGenome.random(811);
  const before = parent.serialize();
  const child = parent.mutate({ seed: 812, strength: 1.1, generationBorn: 2 });

  assert.deepEqual(parent.serialize(), before);
  assert.ok(child.differenceFrom(parent) > 0.001);
  assert.equal(child.lineageId, parent.lineageId);
  assert.equal(child.generationBorn, 2);
});

test("crossover receives numerical contributions from both parents", () => {
  const parentA = NeuralArtGenome.random(1201, { lineageId: "A" });
  const parentB = NeuralArtGenome.random(2302, { lineageId: "B" });
  const child = NeuralArtGenome.crossover(parentA, parentB, { seed: 3403, alpha: 0.5, generationBorn: 3 });

  assert.ok(child.differenceFrom(parentA) > 0.02);
  assert.ok(child.differenceFrom(parentB) > 0.02);
  assert.equal(child.lineageId, "A×B");
  assert.equal(child.generationBorn, 3);

  let mixedValues = 0;
  for (const name of Object.keys(child.params)) {
    for (let index = 0; index < child.params[name].length; index += 1) {
      const value = child.params[name][index];
      const minimum = Math.min(parentA.params[name][index], parentB.params[name][index]);
      const maximum = Math.max(parentA.params[name][index], parentB.params[name][index]);
      assert.ok(value >= minimum - 1e-6 && value <= maximum + 1e-6);
      if (value > minimum + 1e-5 && value < maximum - 1e-5) mixedValues += 1;
    }
  }
  assert.ok(mixedValues > 220, `expected broad crossover, got ${mixedValues} mixed values`);
});

test("serialization round-trips every gene and prediction", () => {
  const original = NeuralArtGenome.random(777).mutate({ seed: 778, strength: 1.4 });
  const restored = NeuralArtGenome.deserialize(JSON.parse(JSON.stringify(original.serialize())));

  assert.deepEqual(restored.serialize(), original.serialize());
  assert.deepEqual(outputAt(restored, 0.34, -0.72), outputAt(original, 0.34, -0.72));
});

test("bilateral symmetry produces matching predictions across the y axis", () => {
  const genome = NeuralArtGenome.random(31337);
  genome.style.rotation = 0;
  genome.style.offsetX = 0;
  genome.style.offsetY = 0;
  genome.style.symmetry = 1;

  for (const [x, y] of [[0.2, 0.1], [0.83, -0.66], [0.47, 0.92]]) {
    const left = outputAt(genome, -x, y);
    const right = outputAt(genome, x, y);
    left.forEach((value, index) => assert.ok(Math.abs(value - right[index]) < 1e-6));
  }
});

test("all activation mixtures stay finite and bounded across the canvas", () => {
  const genome = NeuralArtGenome.random(5150).mutate({ seed: 5151, strength: 2.5 });
  for (let row = 0; row <= 10; row += 1) {
    for (let column = 0; column <= 10; column += 1) {
      const output = genome.forward(column / 5 - 1, row / 5 - 1);
      for (const value of output) {
        assert.ok(Number.isFinite(value));
        assert.ok(value >= -8 && value <= 8);
      }
    }
  }
});
