(function (root, factory) {
  const exports = factory();
  if (typeof module === "object" && module.exports) module.exports = exports;
  root.DreamfieldStyleModel = exports.DreamfieldStyleModel;
  root.DreamfieldStyle = exports;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const TWO_PI = Math.PI * 2;
  const MAX_NORMAL_OFFSET = 0.085;
  const MAX_TANGENT_OFFSET = 0.045;

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  class SeededRandom {
    constructor(seed = 1) {
      this.state = (seed >>> 0) || 0x6d2b79f5;
    }

    next() {
      let value = (this.state += 0x6d2b79f5);
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    }
  }

  function hexToRgb(hex) {
    const clean = String(hex || "#171813").replace("#", "");
    const expanded = clean.length === 3
      ? clean.split("").map((character) => character + character).join("")
      : clean.padEnd(6, "0").slice(0, 6);
    return [0, 2, 4].map((offset) => parseInt(expanded.slice(offset, offset + 2), 16) / 255);
  }

  function pointDistance(a, b) {
    return Math.hypot(b.x - a.x, b.y - a.y);
  }

  function pathLength(points, closed = false) {
    let length = 0;
    for (let index = 1; index < points.length; index += 1) length += pointDistance(points[index - 1], points[index]);
    if (closed && points.length > 2) length += pointDistance(points[points.length - 1], points[0]);
    return length;
  }

  function resampleStroke(points, count = 48, closed = false) {
    if (!points || !points.length || count <= 0) return [];
    const source = points.map((point) => ({
      x: Number(point.x) || 0,
      y: Number(point.y) || 0,
      pressure: Number.isFinite(point.pressure) ? point.pressure : 0.5
    }));
    if (source.length === 1) return Array.from({ length: count }, () => ({ ...source[0] }));

    const segments = [];
    let total = 0;
    const segmentCount = closed ? source.length : source.length - 1;
    for (let index = 0; index < segmentCount; index += 1) {
      const start = source[index];
      const end = source[(index + 1) % source.length];
      const length = pointDistance(start, end);
      segments.push({ start, end, length, from: total });
      total += length;
    }
    if (total < 1e-8) return Array.from({ length: count }, () => ({ ...source[0] }));

    const result = [];
    let segmentIndex = 0;
    const divisor = closed ? count : Math.max(1, count - 1);
    for (let index = 0; index < count; index += 1) {
      const distance = total * index / divisor;
      while (segmentIndex < segments.length - 1 && distance > segments[segmentIndex].from + segments[segmentIndex].length) {
        segmentIndex += 1;
      }
      const segment = segments[segmentIndex];
      const progress = segment.length > 0 ? clamp((distance - segment.from) / segment.length, 0, 1) : 0;
      result.push({
        x: segment.start.x + (segment.end.x - segment.start.x) * progress,
        y: segment.start.y + (segment.end.y - segment.start.y) * progress,
        pressure: segment.start.pressure + (segment.end.pressure - segment.start.pressure) * progress
      });
    }
    return result;
  }

  function smoothStroke(points, radius = 3, passes = 2, closed = false) {
    let current = points.map((point) => ({ ...point }));
    if (current.length < 3) return current;
    for (let pass = 0; pass < passes; pass += 1) {
      current = current.map((point, index) => {
        if (!closed && (index === 0 || index === current.length - 1)) return { ...point };
        let sumX = 0;
        let sumY = 0;
        let sumPressure = 0;
        let weightSum = 0;
        for (let offset = -radius; offset <= radius; offset += 1) {
          let sourceIndex = index + offset;
          if (closed) sourceIndex = (sourceIndex + current.length) % current.length;
          else sourceIndex = clamp(sourceIndex, 0, current.length - 1);
          const weight = radius + 1 - Math.abs(offset);
          sumX += current[sourceIndex].x * weight;
          sumY += current[sourceIndex].y * weight;
          sumPressure += current[sourceIndex].pressure * weight;
          weightSum += weight;
        }
        return { x: sumX / weightSum, y: sumY / weightSum, pressure: sumPressure / weightSum };
      });
    }
    return current;
  }

  function pathGeometry(points, closed = false) {
    const totalLength = pathLength(points, closed);
    return points.map((point, index) => {
      const previousIndex = closed ? (index - 1 + points.length) % points.length : Math.max(0, index - 1);
      const nextIndex = closed ? (index + 1) % points.length : Math.min(points.length - 1, index + 1);
      const previous = points[previousIndex];
      const next = points[nextIndex];
      const dx = next.x - previous.x;
      const dy = next.y - previous.y;
      const magnitude = Math.hypot(dx, dy) || 1;
      const tangent = { x: dx / magnitude, y: dy / magnitude };

      const beforeX = point.x - previous.x;
      const beforeY = point.y - previous.y;
      const afterX = next.x - point.x;
      const afterY = next.y - point.y;
      const beforeMagnitude = Math.hypot(beforeX, beforeY) || 1;
      const afterMagnitude = Math.hypot(afterX, afterY) || 1;
      const cross = (beforeX * afterY - beforeY * afterX) / (beforeMagnitude * afterMagnitude);
      const dot = clamp((beforeX * afterX + beforeY * afterY) / (beforeMagnitude * afterMagnitude), -1, 1);
      const curvature = Math.atan2(cross, dot);
      return { tangent, normal: { x: -tangent.y, y: tangent.x }, curvature, totalLength };
    });
  }

  function encodeStyleFeatures(feature, target = new Float32Array(12)) {
    const t = clamp(Number(feature.t) || 0, 0, 1);
    const tangent = feature.tangent || { x: 1, y: 0 };
    target[0] = t * 2 - 1;
    target[1] = Math.sin(TWO_PI * t);
    target[2] = Math.cos(TWO_PI * t);
    target[3] = Math.sin(TWO_PI * 2 * t);
    target[4] = Math.cos(TWO_PI * 2 * t);
    target[5] = clamp(Number(tangent.x) || 0, -1, 1);
    target[6] = clamp(Number(tangent.y) || 0, -1, 1);
    target[7] = clamp((Number(feature.curvature) || 0) / 0.8, -1, 1);
    target[8] = clamp((Number(feature.length) || 0) / 3, 0, 1) * 2 - 1;
    target[9] = feature.closed ? 1 : -1;
    target[10] = clamp(Number(feature.grainNormal) || 0, -1, 1);
    target[11] = clamp(Number(feature.grainTangent) || 0, -1, 1);
    return target;
  }

  function brushStrength(stroke) {
    if (Number.isFinite(stroke.width)) return clamp(stroke.width, 0.08, 1);
    return ({ fine: 0.25, medium: 0.55, broad: 0.9 })[stroke.brush] || 0.55;
  }

  function extractStyleDataset(strokes, options = {}) {
    const samples = [];
    const paletteMap = new Map();
    let wobbleSum = 0;
    let turnSum = 0;
    let widthSum = 0;
    let pointCount = 0;

    (strokes || []).forEach((stroke, strokeIndex) => {
      if (!stroke.points || !stroke.points.length) return;
      const sourceLength = pathLength(stroke.points);
      const closed = stroke.closed === true || (stroke.points.length > 8 && pointDistance(stroke.points[0], stroke.points[stroke.points.length - 1]) < 0.075);
      const count = clamp(options.sampleCount || Math.round(sourceLength * 34), 20, 72);
      const observed = resampleStroke(stroke.points, count, closed);
      const intention = smoothStroke(observed, 4, 3, closed);
      const geometry = pathGeometry(intention, closed);
      const residuals = observed.map((point, index) => {
        const deltaX = point.x - intention[index].x;
        const deltaY = point.y - intention[index].y;
        const normal = geometry[index].normal;
        const tangent = geometry[index].tangent;
        return {
          normal: deltaX * normal.x + deltaY * normal.y,
          tangent: deltaX * tangent.x + deltaY * tangent.y
        };
      });
      const rmsNormal = Math.sqrt(residuals.reduce((sum, residual) => sum + residual.normal ** 2, 0) / residuals.length) + 0.0005;
      const rmsTangent = Math.sqrt(residuals.reduce((sum, residual) => sum + residual.tangent ** 2, 0) / residuals.length) + 0.0005;
      const color = hexToRgb(stroke.color);
      const width = brushStrength(stroke);
      const colorKey = String(stroke.color || "#171813").toUpperCase();
      paletteMap.set(colorKey, (paletteMap.get(colorKey) || 0) + count);

      observed.forEach((point, index) => {
        const t = closed ? index / observed.length : index / Math.max(1, observed.length - 1);
        const residual = residuals[index];
        const taper = closed ? 1 : 0.7 + 0.3 * Math.sin(Math.PI * t);
        const pressure = clamp(Number(point.pressure) || 0.5, 0, 1);
        const learnedWidth = clamp(width * taper * (0.82 + pressure * 0.36), 0.05, 1);
        const input = encodeStyleFeatures({
          t,
          tangent: geometry[index].tangent,
          curvature: geometry[index].curvature,
          length: geometry[index].totalLength,
          closed,
          grainNormal: residual.normal / (rmsNormal * 2.25),
          grainTangent: residual.tangent / (rmsTangent * 2.25)
        });
        const target = new Float32Array([
          clamp(residual.normal / MAX_NORMAL_OFFSET, -1, 1),
          clamp(residual.tangent / MAX_TANGENT_OFFSET, -1, 1),
          learnedWidth,
          color[0], color[1], color[2]
        ]);
        samples.push({ input, target, strokeIndex });
        wobbleSum += Math.abs(residual.normal);
        turnSum += Math.abs(geometry[index].curvature);
        widthSum += learnedWidth;
        pointCount += 1;
      });
    });

    const palette = Array.from(paletteMap.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([color, count]) => ({ color, count }));
    return {
      samples,
      palette,
      metrics: {
        wobble: pointCount ? wobbleSum / pointCount : 0,
        curvature: pointCount ? turnSum / pointCount : 0,
        width: pointCount ? widthSum / pointCount : 0,
        pointCount
      }
    };
  }

  function makeSmoothNoise(count, random) {
    let values = Array.from({ length: count }, () => random.next() * 2 - 1);
    for (let pass = 0; pass < 3; pass += 1) {
      values = values.map((value, index) => {
        const previous = values[Math.max(0, index - 1)];
        const next = values[Math.min(values.length - 1, index + 1)];
        return previous * 0.25 + value * 0.5 + next * 0.25;
      });
    }
    const maximum = Math.max(0.001, ...values.map(Math.abs));
    return values.map((value) => value / maximum);
  }

  function stylizePath(model, points, options = {}) {
    const closed = Boolean(options.closed);
    const count = clamp(options.sampleCount || Math.round(pathLength(points, closed) * 42), 24, 110);
    const backbone = resampleStroke(points, count, closed);
    const geometry = pathGeometry(backbone, closed);
    const random = new SeededRandom(options.seed || 1);
    const normalNoise = makeSmoothNoise(count, random);
    const tangentNoise = makeSmoothNoise(count, random);
    return backbone.map((point, index) => {
      const t = closed ? index / backbone.length : index / Math.max(1, backbone.length - 1);
      const input = encodeStyleFeatures({
        t,
        tangent: geometry[index].tangent,
        curvature: geometry[index].curvature,
        length: geometry[index].totalLength,
        closed,
        grainNormal: normalNoise[index],
        grainTangent: tangentNoise[index]
      });
      const prediction = model.predict(input);
      const normalOffset = prediction[0] * MAX_NORMAL_OFFSET;
      const tangentOffset = prediction[1] * MAX_TANGENT_OFFSET;
      const tangent = geometry[index].tangent;
      const normal = geometry[index].normal;
      return {
        x: point.x + normal.x * normalOffset + tangent.x * tangentOffset,
        y: point.y + normal.y * normalOffset + tangent.y * tangentOffset,
        width: clamp(prediction[2], 0.04, 1),
        color: [prediction[3], prediction[4], prediction[5]]
      };
    });
  }

  class DreamfieldStyleModel {
    constructor(options = {}) {
      this.inputSize = 12;
      this.hiddenSize = options.hiddenSize || 18;
      this.outputSize = 6;
      this.learningRate = options.learningRate || 0.009;
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
        w3: this.outputSize * this.hiddenSize,
        b3: this.outputSize
      };
      this.params = {};
      this.gradients = {};
      this.moments = {};
      this.velocities = {};
      for (const [name, length] of Object.entries(this.shapes)) {
        this.params[name] = new Float32Array(length);
        this.gradients[name] = new Float32Array(length);
        this.moments[name] = new Float32Array(length);
        this.velocities[name] = new Float32Array(length);
      }
      this.h1 = new Float32Array(this.hiddenSize);
      this.h2 = new Float32Array(this.hiddenSize);
      this.output = new Float32Array(this.outputSize);
      this.deltaH1 = new Float32Array(this.hiddenSize);
      this.deltaH2 = new Float32Array(this.hiddenSize);
      this.deltaOut = new Float32Array(this.outputSize);
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
      this._fillXavier(this.params.w2, this.hiddenSize, this.hiddenSize, random);
      this._fillXavier(this.params.w3, this.hiddenSize, this.outputSize, random);
      this.params.b1.fill(0);
      this.params.b2.fill(0);
      this.params.b3.fill(0);
      for (const name of Object.keys(this.shapes)) {
        this.gradients[name].fill(0);
        this.moments[name].fill(0);
        this.velocities[name].fill(0);
      }
    }

    _fillXavier(target, fanIn, fanOut, random) {
      const limit = Math.sqrt(6 / (fanIn + fanOut));
      for (let index = 0; index < target.length; index += 1) target[index] = (random.next() * 2 - 1) * limit;
    }

    _sigmoid(value) {
      if (value >= 0) {
        const inverse = Math.exp(-value);
        return 1 / (1 + inverse);
      }
      const exponential = Math.exp(value);
      return exponential / (1 + exponential);
    }

    forward(input, target = this.output) {
      const { hiddenSize, inputSize, outputSize, params } = this;
      for (let row = 0; row < hiddenSize; row += 1) {
        let sum = params.b1[row];
        const offset = row * inputSize;
        for (let column = 0; column < inputSize; column += 1) sum += params.w1[offset + column] * input[column];
        this.h1[row] = Math.tanh(sum);
      }
      for (let row = 0; row < hiddenSize; row += 1) {
        let sum = params.b2[row];
        const offset = row * hiddenSize;
        for (let column = 0; column < hiddenSize; column += 1) sum += params.w2[offset + column] * this.h1[column];
        this.h2[row] = Math.tanh(sum);
      }
      for (let channel = 0; channel < outputSize; channel += 1) {
        let sum = params.b3[channel];
        const offset = channel * hiddenSize;
        for (let column = 0; column < hiddenSize; column += 1) sum += params.w3[offset + column] * this.h2[column];
        target[channel] = channel < 2 ? Math.tanh(sum) : this._sigmoid(sum);
      }
      return target;
    }

    predict(input, target = new Float32Array(this.outputSize)) {
      return this.forward(input, target);
    }

    lossForBatch(batch) {
      if (!batch.length) return 0;
      let loss = 0;
      for (const sample of batch) {
        const prediction = this.forward(sample.input);
        for (let channel = 0; channel < this.outputSize; channel += 1) {
          const difference = prediction[channel] - sample.target[channel];
          loss += difference * difference / this.outputSize;
        }
      }
      return loss / batch.length;
    }

    computeGradients(batch) {
      for (const gradient of Object.values(this.gradients)) gradient.fill(0);
      if (!batch.length) return { loss: 0, gradients: this.gradients };
      let loss = 0;
      const { hiddenSize, inputSize, outputSize, params, gradients } = this;
      for (const sample of batch) {
        const prediction = this.forward(sample.input);
        for (let channel = 0; channel < outputSize; channel += 1) {
          const difference = prediction[channel] - sample.target[channel];
          loss += difference * difference / outputSize;
          const activationDerivative = channel < 2
            ? 1 - prediction[channel] * prediction[channel]
            : prediction[channel] * (1 - prediction[channel]);
          this.deltaOut[channel] = (2 * difference / outputSize) * activationDerivative;
        }

        this.deltaH2.fill(0);
        for (let channel = 0; channel < outputSize; channel += 1) {
          const delta = this.deltaOut[channel];
          const offset = channel * hiddenSize;
          gradients.b3[channel] += delta;
          for (let unit = 0; unit < hiddenSize; unit += 1) {
            gradients.w3[offset + unit] += delta * this.h2[unit];
            this.deltaH2[unit] += params.w3[offset + unit] * delta;
          }
        }
        for (let unit = 0; unit < hiddenSize; unit += 1) this.deltaH2[unit] *= 1 - this.h2[unit] * this.h2[unit];

        this.deltaH1.fill(0);
        for (let row = 0; row < hiddenSize; row += 1) {
          const delta = this.deltaH2[row];
          const offset = row * hiddenSize;
          gradients.b2[row] += delta;
          for (let column = 0; column < hiddenSize; column += 1) {
            gradients.w2[offset + column] += delta * this.h1[column];
            this.deltaH1[column] += params.w2[offset + column] * delta;
          }
        }
        for (let unit = 0; unit < hiddenSize; unit += 1) this.deltaH1[unit] *= 1 - this.h1[unit] * this.h1[unit];

        for (let row = 0; row < hiddenSize; row += 1) {
          const delta = this.deltaH1[row];
          const offset = row * inputSize;
          gradients.b1[row] += delta;
          for (let column = 0; column < inputSize; column += 1) gradients.w1[offset + column] += delta * sample.input[column];
        }
      }

      const inverseBatch = 1 / batch.length;
      for (const gradient of Object.values(gradients)) {
        for (let index = 0; index < gradient.length; index += 1) gradient[index] *= inverseBatch;
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
          const gradient = clamp(gradients[index], -3, 3);
          moments[index] = this.beta1 * moments[index] + (1 - this.beta1) * gradient;
          velocities[index] = this.beta2 * velocities[index] + (1 - this.beta2) * gradient * gradient;
          const correctedMoment = moments[index] / correction1;
          const correctedVelocity = velocities[index] / correction2;
          values[index] -= learningRate * correctedMoment / (Math.sqrt(correctedVelocity) + this.epsilon);
        }
      }
    }

    trainBatch(batch, learningRate = this.learningRate) {
      const result = this.computeGradients(batch);
      if (batch.length) this.applyGradients(learningRate);
      return result.loss;
    }
  }

  return {
    DreamfieldStyleModel,
    SeededRandom,
    encodeStyleFeatures,
    extractStyleDataset,
    hexToRgb,
    pathGeometry,
    pathLength,
    resampleStroke,
    smoothStroke,
    stylizePath,
    constants: { MAX_NORMAL_OFFSET, MAX_TANGENT_OFFSET }
  };
});
