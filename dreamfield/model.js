(function (root, factory) {
  const exports = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = exports;
  }
  root.DreamfieldMLP = exports.DreamfieldMLP;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  class SeededRandom {
    constructor(seed) {
      this.state = (seed >>> 0) || 0x6d2b79f5;
    }

    next() {
      let value = (this.state += 0x6d2b79f5);
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    }
  }

  class DreamfieldMLP {
    constructor(options = {}) {
      this.hiddenSize = options.hiddenSize || 20;
      this.bands = options.bands || 3;
      this.frequencyScale = options.frequencyScale || 1;
      this.learningRate = options.learningRate || 0.007;
      this.inputSize = 2 + this.bands * 4;
      this.seed = options.seed || 1;
      this.step = 0;

      this.beta1 = 0.9;
      this.beta2 = 0.999;
      this.epsilon = 1e-8;

      this.shapes = {
        w1: this.hiddenSize * this.inputSize,
        b1: this.hiddenSize,
        w2: this.hiddenSize * this.hiddenSize,
        b2: this.hiddenSize,
        w3: 3 * this.hiddenSize,
        b3: 3
      };

      this.params = {};
      this.moments = {};
      this.velocities = {};
      this.gradients = {};

      for (const [name, length] of Object.entries(this.shapes)) {
        this.params[name] = new Float32Array(length);
        this.moments[name] = new Float32Array(length);
        this.velocities[name] = new Float32Array(length);
        this.gradients[name] = new Float32Array(length);
      }

      this.featureScratch = new Float32Array(this.inputSize);
      this.h1 = new Float32Array(this.hiddenSize);
      this.h2 = new Float32Array(this.hiddenSize);
      this.output = new Float32Array(3);
      this.deltaH1 = new Float32Array(this.hiddenSize);
      this.deltaH2 = new Float32Array(this.hiddenSize);
      this.deltaOut = new Float32Array(3);

      this.reset(this.seed);
    }

    get parameterCount() {
      return Object.values(this.shapes).reduce((sum, length) => sum + length, 0);
    }

    reset(seed = this.seed) {
      this.seed = seed;
      this.step = 0;
      const random = new SeededRandom(seed);

      this._fillXavier(this.params.w1, this.inputSize, this.hiddenSize, random);
      this.params.b1.fill(0);
      this._fillXavier(this.params.w2, this.hiddenSize, this.hiddenSize, random);
      this.params.b2.fill(0);
      this._fillXavier(this.params.w3, this.hiddenSize, 3, random);
      this.params.b3.fill(0);

      for (const name of Object.keys(this.shapes)) {
        this.moments[name].fill(0);
        this.velocities[name].fill(0);
        this.gradients[name].fill(0);
      }
    }

    _fillXavier(target, fanIn, fanOut, random) {
      const limit = Math.sqrt(6 / (fanIn + fanOut));
      for (let index = 0; index < target.length; index += 1) {
        target[index] = (random.next() * 2 - 1) * limit;
      }
    }

    encode(x, y, target = new Float32Array(this.inputSize), offset = 0) {
      target[offset] = x;
      target[offset + 1] = y;
      let cursor = offset + 2;

      for (let band = 0; band < this.bands; band += 1) {
        const frequency = Math.pow(2, band) * this.frequencyScale * Math.PI;
        target[cursor] = Math.sin(x * frequency);
        target[cursor + 1] = Math.cos(x * frequency);
        target[cursor + 2] = Math.sin(y * frequency);
        target[cursor + 3] = Math.cos(y * frequency);
        cursor += 4;
      }

      return target;
    }

    forwardEncoded(features, offset = 0, target = this.output) {
      const { hiddenSize, inputSize, params, h1, h2 } = this;
      const { w1, b1, w2, b2, w3, b3 } = params;

      for (let row = 0; row < hiddenSize; row += 1) {
        let sum = b1[row];
        const weightOffset = row * inputSize;
        for (let column = 0; column < inputSize; column += 1) {
          sum += w1[weightOffset + column] * features[offset + column];
        }
        h1[row] = Math.tanh(sum);
      }

      for (let row = 0; row < hiddenSize; row += 1) {
        let sum = b2[row];
        const weightOffset = row * hiddenSize;
        for (let column = 0; column < hiddenSize; column += 1) {
          sum += w2[weightOffset + column] * h1[column];
        }
        h2[row] = Math.tanh(sum);
      }

      for (let channel = 0; channel < 3; channel += 1) {
        let sum = b3[channel];
        const weightOffset = channel * hiddenSize;
        for (let column = 0; column < hiddenSize; column += 1) {
          sum += w3[weightOffset + column] * h2[column];
        }
        target[channel] = this._sigmoid(sum);
      }

      return target;
    }

    predict(x, y, target = new Float32Array(3)) {
      this.encode(x, y, this.featureScratch);
      return this.forwardEncoded(this.featureScratch, 0, target);
    }

    _sigmoid(value) {
      if (value >= 0) {
        const z = Math.exp(-value);
        return 1 / (1 + z);
      }
      const z = Math.exp(value);
      return z / (1 + z);
    }

    lossForBatch(batch) {
      if (!batch.length) return 0;
      let loss = 0;
      for (const sample of batch) {
        const features = sample.features || this.encode(sample.x, sample.y, this.featureScratch);
        const prediction = this.forwardEncoded(features);
        for (let channel = 0; channel < 3; channel += 1) {
          const difference = prediction[channel] - sample.rgb[channel];
          loss += difference * difference / 3;
        }
      }
      return loss / batch.length;
    }

    computeGradients(batch) {
      for (const gradient of Object.values(this.gradients)) gradient.fill(0);
      if (!batch.length) return { loss: 0, gradients: this.gradients };

      const { hiddenSize, inputSize, params, gradients } = this;
      const { w2, w3 } = params;
      let loss = 0;

      for (const sample of batch) {
        const features = sample.features || this.encode(sample.x, sample.y, this.featureScratch);
        const prediction = this.forwardEncoded(features);

        for (let channel = 0; channel < 3; channel += 1) {
          const difference = prediction[channel] - sample.rgb[channel];
          loss += difference * difference / 3;
          this.deltaOut[channel] = (2 * difference / 3) * prediction[channel] * (1 - prediction[channel]);
        }

        this.deltaH2.fill(0);
        for (let channel = 0; channel < 3; channel += 1) {
          const delta = this.deltaOut[channel];
          const weightOffset = channel * hiddenSize;
          gradients.b3[channel] += delta;
          for (let unit = 0; unit < hiddenSize; unit += 1) {
            gradients.w3[weightOffset + unit] += delta * this.h2[unit];
            this.deltaH2[unit] += w3[weightOffset + unit] * delta;
          }
        }

        for (let unit = 0; unit < hiddenSize; unit += 1) {
          this.deltaH2[unit] *= 1 - this.h2[unit] * this.h2[unit];
        }

        this.deltaH1.fill(0);
        for (let row = 0; row < hiddenSize; row += 1) {
          const delta = this.deltaH2[row];
          const weightOffset = row * hiddenSize;
          gradients.b2[row] += delta;
          for (let column = 0; column < hiddenSize; column += 1) {
            gradients.w2[weightOffset + column] += delta * this.h1[column];
            this.deltaH1[column] += w2[weightOffset + column] * delta;
          }
        }

        for (let unit = 0; unit < hiddenSize; unit += 1) {
          this.deltaH1[unit] *= 1 - this.h1[unit] * this.h1[unit];
        }

        for (let row = 0; row < hiddenSize; row += 1) {
          const delta = this.deltaH1[row];
          const weightOffset = row * inputSize;
          gradients.b1[row] += delta;
          for (let column = 0; column < inputSize; column += 1) {
            gradients.w1[weightOffset + column] += delta * features[column];
          }
        }
      }

      const inverseBatch = 1 / batch.length;
      for (const gradient of Object.values(gradients)) {
        for (let index = 0; index < gradient.length; index += 1) {
          gradient[index] *= inverseBatch;
        }
      }

      return { loss: loss * inverseBatch, gradients };
    }

    applyGradients(learningRate = this.learningRate) {
      this.step += 1;
      const correction1 = 1 - Math.pow(this.beta1, this.step);
      const correction2 = 1 - Math.pow(this.beta2, this.step);

      for (const name of Object.keys(this.shapes)) {
        const values = this.params[name];
        const gradients = this.gradients[name];
        const moments = this.moments[name];
        const velocities = this.velocities[name];

        for (let index = 0; index < values.length; index += 1) {
          const gradient = Math.max(-2, Math.min(2, gradients[index]));
          moments[index] = this.beta1 * moments[index] + (1 - this.beta1) * gradient;
          velocities[index] = this.beta2 * velocities[index] + (1 - this.beta2) * gradient * gradient;
          const correctedMoment = moments[index] / correction1;
          const correctedVelocity = velocities[index] / correction2;
          values[index] -= learningRate * correctedMoment / (Math.sqrt(correctedVelocity) + this.epsilon);

          if (!Number.isFinite(values[index])) {
            throw new Error(`Non-finite parameter detected in ${name}[${index}]`);
          }
        }
      }
    }

    trainBatch(batch, learningRate = this.learningRate) {
      if (!batch.length) return 0;
      const result = this.computeGradients(batch);
      this.applyGradients(learningRate);
      return result.loss;
    }
  }

  return { DreamfieldMLP };
});
