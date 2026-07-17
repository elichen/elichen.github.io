(function (root, factory) {
  const exports = factory();
  if (typeof module === "object" && module.exports) module.exports = exports;
  root.NeuralArtGenome = exports.NeuralArtGenome;
  root.NeuralArtRandom = exports.SeededRandom;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const INPUT_SIZE = 3;
  const HIDDEN_SIZE = 12;
  const OUTPUT_SIZE = 4;
  const ACTIVATIONS = ["tanh", "sine", "gaussian", "absolute"];

  class SeededRandom {
    constructor(seed) {
      this.state = (seed >>> 0) || 0x6d2b79f5;
      this.spareNormal = null;
    }

    next() {
      let value = (this.state += 0x6d2b79f5);
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    }

    integer(maximum) {
      return Math.floor(this.next() * maximum);
    }

    normal() {
      if (this.spareNormal !== null) {
        const value = this.spareNormal;
        this.spareNormal = null;
        return value;
      }
      const u = Math.max(this.next(), 1e-9);
      const v = this.next();
      const magnitude = Math.sqrt(-2 * Math.log(u));
      this.spareNormal = magnitude * Math.sin(Math.PI * 2 * v);
      return magnitude * Math.cos(Math.PI * 2 * v);
    }
  }

  class NeuralArtGenome {
    constructor(data = {}) {
      this.seed = (data.seed >>> 0) || 1;
      this.lineageId = data.lineageId || `L${this.seed.toString(36).toUpperCase()}`;
      this.generationBorn = data.generationBorn || 1;
      this.params = {
        w1: toFloatArray(data.params && data.params.w1, INPUT_SIZE * HIDDEN_SIZE),
        b1: toFloatArray(data.params && data.params.b1, HIDDEN_SIZE),
        w2: toFloatArray(data.params && data.params.w2, HIDDEN_SIZE * HIDDEN_SIZE),
        b2: toFloatArray(data.params && data.params.b2, HIDDEN_SIZE),
        w3: toFloatArray(data.params && data.params.w3, OUTPUT_SIZE * HIDDEN_SIZE),
        b3: toFloatArray(data.params && data.params.b3, OUTPUT_SIZE)
      };
      this.activations1 = toActivationArray(data.activations1);
      this.activations2 = toActivationArray(data.activations2);
      this.style = normalizeStyle(data.style || {});
      this.h1 = new Float32Array(HIDDEN_SIZE);
      this.h2 = new Float32Array(HIDDEN_SIZE);
      this.output = new Float32Array(OUTPUT_SIZE);
    }

    static random(seed, options = {}) {
      const random = new SeededRandom(seed);
      const params = {
        w1: new Float32Array(INPUT_SIZE * HIDDEN_SIZE),
        b1: new Float32Array(HIDDEN_SIZE),
        w2: new Float32Array(HIDDEN_SIZE * HIDDEN_SIZE),
        b2: new Float32Array(HIDDEN_SIZE),
        w3: new Float32Array(OUTPUT_SIZE * HIDDEN_SIZE),
        b3: new Float32Array(OUTPUT_SIZE)
      };
      fillXavier(params.w1, INPUT_SIZE, HIDDEN_SIZE, random, 1.65);
      fillXavier(params.w2, HIDDEN_SIZE, HIDDEN_SIZE, random, 1.45);
      fillXavier(params.w3, HIDDEN_SIZE, OUTPUT_SIZE, random, 1.35);
      fillNormal(params.b1, random, 0.22);
      fillNormal(params.b2, random, 0.18);
      fillNormal(params.b3, random, 0.2);

      const activations1 = new Uint8Array(HIDDEN_SIZE);
      const activations2 = new Uint8Array(HIDDEN_SIZE);
      for (let index = 0; index < HIDDEN_SIZE; index += 1) {
        activations1[index] = random.integer(ACTIVATIONS.length);
        activations2[index] = random.integer(ACTIVATIONS.length);
      }
      activations1[random.integer(HIDDEN_SIZE)] = 1;
      activations2[random.integer(HIDDEN_SIZE)] = 2;

      const symmetryRoll = random.next();
      const symmetry = symmetryRoll < 0.18 ? 0 : symmetryRoll < 0.58 ? 1 : symmetryRoll < 0.83 ? 2 : 3;
      const style = {
        rotation: (random.next() * 2 - 1) * Math.PI,
        scale: 0.82 + random.next() * 1.2,
        offsetX: (random.next() * 2 - 1) * 0.2,
        offsetY: (random.next() * 2 - 1) * 0.2,
        threshold: -0.25 + random.next() * 0.42,
        softness: 0.1 + random.next() * 0.16,
        contrast: 0.9 + random.next() * 0.52,
        symmetry,
        paletteIndex: random.integer(10)
      };

      return new NeuralArtGenome({
        seed,
        lineageId: options.lineageId,
        generationBorn: options.generationBorn || 1,
        params,
        activations1,
        activations2,
        style
      });
    }

    get parameterCount() {
      return Object.values(this.params).reduce((sum, values) => sum + values.length, 0);
    }

    clone(overrides = {}) {
      const params = {};
      for (const [name, values] of Object.entries(this.params)) params[name] = new Float32Array(values);
      return new NeuralArtGenome({
        seed: overrides.seed === undefined ? this.seed : overrides.seed,
        lineageId: overrides.lineageId || this.lineageId,
        generationBorn: overrides.generationBorn || this.generationBorn,
        params,
        activations1: new Uint8Array(this.activations1),
        activations2: new Uint8Array(this.activations2),
        style: { ...this.style }
      });
    }

    mutate(options = {}) {
      const seed = (options.seed >>> 0) || mixSeed(this.seed, 0x9e3779b9);
      const strength = clamp(options.strength === undefined ? 1 : options.strength, 0, 3);
      const random = new SeededRandom(seed);
      const child = this.clone({
        seed,
        generationBorn: options.generationBorn || this.generationBorn + 1
      });
      if (strength === 0) return child;
      const mutationRate = 0.075 + strength * 0.055;
      const sigma = 0.045 + strength * 0.16;

      for (const values of Object.values(child.params)) {
        for (let index = 0; index < values.length; index += 1) {
          if (random.next() < mutationRate) values[index] += random.normal() * sigma;
          if (strength > 0 && random.next() < 0.0015 * strength) values[index] = random.normal() * 0.7;
          values[index] = clamp(values[index], -3.5, 3.5);
        }
      }

      mutateActivations(child.activations1, random, strength);
      mutateActivations(child.activations2, random, strength);
      child.style.rotation = wrapAngle(child.style.rotation + random.normal() * 0.12 * strength);
      child.style.scale = clamp(child.style.scale + random.normal() * 0.1 * strength, 0.52, 2.8);
      child.style.offsetX = clamp(child.style.offsetX + random.normal() * 0.045 * strength, -0.42, 0.42);
      child.style.offsetY = clamp(child.style.offsetY + random.normal() * 0.045 * strength, -0.42, 0.42);
      child.style.threshold = clamp(child.style.threshold + random.normal() * 0.055 * strength, -0.52, 0.48);
      child.style.softness = clamp(child.style.softness + random.normal() * 0.025 * strength, 0.055, 0.38);
      child.style.contrast = clamp(child.style.contrast + random.normal() * 0.08 * strength, 0.68, 1.82);
      if (random.next() < 0.055 * strength) child.style.symmetry = random.integer(4);
      if (random.next() < 0.075 * strength) child.style.paletteIndex = random.integer(10);
      return child;
    }

    static crossover(parentA, parentB, options = {}) {
      const seed = (options.seed >>> 0) || mixSeed(parentA.seed, parentB.seed);
      const alpha = clamp(options.alpha === undefined ? 0.5 : options.alpha, 0, 1);
      const random = new SeededRandom(seed);
      const child = parentA.clone({
        seed,
        lineageId: parentA.lineageId === parentB.lineageId
          ? parentA.lineageId
          : `${parentA.lineageId}×${parentB.lineageId}`,
        generationBorn: options.generationBorn || Math.max(parentA.generationBorn, parentB.generationBorn) + 1
      });

      for (const name of Object.keys(child.params)) {
        const target = child.params[name];
        const a = parentA.params[name];
        const b = parentB.params[name];
        for (let index = 0; index < target.length; index += 1) {
          const localAlpha = clamp(alpha + random.normal() * 0.08, 0.08, 0.92);
          target[index] = a[index] * localAlpha + b[index] * (1 - localAlpha);
        }
      }

      for (let index = 0; index < HIDDEN_SIZE; index += 1) {
        child.activations1[index] = random.next() < alpha ? parentA.activations1[index] : parentB.activations1[index];
        child.activations2[index] = random.next() < alpha ? parentA.activations2[index] : parentB.activations2[index];
      }
      for (const key of ["rotation", "scale", "offsetX", "offsetY", "threshold", "softness", "contrast"]) {
        child.style[key] = parentA.style[key] * alpha + parentB.style[key] * (1 - alpha);
      }
      child.style.symmetry = random.next() < alpha ? parentA.style.symmetry : parentB.style.symmetry;
      child.style.paletteIndex = random.next() < alpha ? parentA.style.paletteIndex : parentB.style.paletteIndex;
      child.style = normalizeStyle(child.style);
      return child;
    }

    forward(x, y, target = this.output) {
      const point = transformPoint(x, y, this.style);
      const inputs = [point.x, point.y, Math.min(1.5, Math.hypot(point.x, point.y))];
      const { w1, b1, w2, b2, w3, b3 } = this.params;

      for (let row = 0; row < HIDDEN_SIZE; row += 1) {
        let sum = b1[row];
        const offset = row * INPUT_SIZE;
        for (let column = 0; column < INPUT_SIZE; column += 1) sum += w1[offset + column] * inputs[column];
        this.h1[row] = activate(this.activations1[row], sum);
      }

      for (let row = 0; row < HIDDEN_SIZE; row += 1) {
        let sum = b2[row];
        const offset = row * HIDDEN_SIZE;
        for (let column = 0; column < HIDDEN_SIZE; column += 1) sum += w2[offset + column] * this.h1[column];
        this.h2[row] = activate(this.activations2[row], sum);
      }

      for (let output = 0; output < OUTPUT_SIZE; output += 1) {
        let sum = b3[output];
        const offset = output * HIDDEN_SIZE;
        for (let column = 0; column < HIDDEN_SIZE; column += 1) sum += w3[offset + column] * this.h2[column];
        target[output] = clamp(sum * this.style.contrast, -8, 8);
      }
      return target;
    }

    differenceFrom(other) {
      let difference = 0;
      let count = 0;
      for (const name of Object.keys(this.params)) {
        const a = this.params[name];
        const b = other.params[name];
        for (let index = 0; index < a.length; index += 1) {
          difference += Math.abs(a[index] - b[index]);
          count += 1;
        }
      }
      return difference / count;
    }

    serialize() {
      const params = {};
      for (const [name, values] of Object.entries(this.params)) params[name] = Array.from(values);
      return {
        version: 1,
        seed: this.seed,
        lineageId: this.lineageId,
        generationBorn: this.generationBorn,
        params,
        activations1: Array.from(this.activations1),
        activations2: Array.from(this.activations2),
        style: { ...this.style }
      };
    }

    static deserialize(data) {
      if (!data || data.version !== 1) throw new Error("Unsupported neural art genome");
      return new NeuralArtGenome(data);
    }
  }

  function transformPoint(x, y, style) {
    const cosine = Math.cos(style.rotation);
    const sine = Math.sin(style.rotation);
    let tx = (x * cosine - y * sine) * style.scale + style.offsetX;
    let ty = (x * sine + y * cosine) * style.scale + style.offsetY;
    if (style.symmetry === 1) {
      tx = Math.abs(tx);
    } else if (style.symmetry === 2) {
      tx = Math.abs(tx);
      ty = Math.abs(ty);
    } else if (style.symmetry === 3) {
      tx = Math.abs(tx);
      ty = Math.abs(ty);
      if (ty > tx) [tx, ty] = [ty, tx];
    }
    return { x: tx, y: ty };
  }

  function activate(type, value) {
    if (type === 1) return Math.sin(value * 2.05);
    if (type === 2) return 2 * Math.exp(-value * value) - 1;
    if (type === 3) return 2 * Math.tanh(Math.abs(value)) - 1;
    return Math.tanh(value);
  }

  function mutateActivations(activations, random, strength) {
    for (let index = 0; index < activations.length; index += 1) {
      if (random.next() < 0.018 * strength) activations[index] = random.integer(ACTIVATIONS.length);
    }
  }

  function normalizeStyle(style) {
    return {
      rotation: wrapAngle(Number.isFinite(style.rotation) ? style.rotation : 0),
      scale: clamp(Number.isFinite(style.scale) ? style.scale : 1, 0.52, 2.8),
      offsetX: clamp(Number.isFinite(style.offsetX) ? style.offsetX : 0, -0.42, 0.42),
      offsetY: clamp(Number.isFinite(style.offsetY) ? style.offsetY : 0, -0.42, 0.42),
      threshold: clamp(Number.isFinite(style.threshold) ? style.threshold : 0, -0.52, 0.48),
      softness: clamp(Number.isFinite(style.softness) ? style.softness : 0.16, 0.055, 0.38),
      contrast: clamp(Number.isFinite(style.contrast) ? style.contrast : 1, 0.68, 1.82),
      symmetry: clamp(Math.round(style.symmetry || 0), 0, 3),
      paletteIndex: clamp(Math.round(style.paletteIndex || 0), 0, 9)
    };
  }

  function toFloatArray(values, length) {
    const result = new Float32Array(length);
    if (!values) return result;
    const count = Math.min(length, values.length);
    for (let index = 0; index < count; index += 1) result[index] = values[index];
    return result;
  }

  function toActivationArray(values) {
    const result = new Uint8Array(HIDDEN_SIZE);
    if (!values) return result;
    const count = Math.min(HIDDEN_SIZE, values.length);
    for (let index = 0; index < count; index += 1) result[index] = clamp(Math.round(values[index]), 0, ACTIVATIONS.length - 1);
    return result;
  }

  function fillXavier(target, fanIn, fanOut, random, gain) {
    const limit = Math.sqrt(6 / (fanIn + fanOut)) * gain;
    for (let index = 0; index < target.length; index += 1) target[index] = (random.next() * 2 - 1) * limit;
  }

  function fillNormal(target, random, scale) {
    for (let index = 0; index < target.length; index += 1) target[index] = random.normal() * scale;
  }

  function wrapAngle(value) {
    let wrapped = value;
    while (wrapped > Math.PI) wrapped -= Math.PI * 2;
    while (wrapped < -Math.PI) wrapped += Math.PI * 2;
    return wrapped;
  }

  function mixSeed(a, b) {
    let value = (a ^ Math.imul(b, 0x45d9f3b)) >>> 0;
    value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
    value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
    return (value ^ (value >>> 16)) >>> 0;
  }

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  return { NeuralArtGenome, SeededRandom, ACTIVATIONS };
});
