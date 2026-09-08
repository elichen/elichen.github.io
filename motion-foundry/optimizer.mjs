/**
 * Browser-sized path synthesis: valid-geometry atlas → differential evolution
 * → Nelder–Mead refinement. It takes inspiration from retrieval/refinement in
 * LInK (https://arxiv.org/abs/2405.20592); this is a small analytic search, not
 * that paper's learned model or an arbitrary-topology linkage generator.
 *
 * Shapes are uniformly resampled by arc length. Ordered RMS checks cyclic
 * phase and both traversal directions; complex Procrustes removes placement,
 * scale, and orientation from the search variables. Reflection is implemented
 * by mirroring actual mechanism parameters, never by a nonrigid transform.
 */

import {
  validateDesign,
  sampleCoordinates,
  resampleCoordinates,
  resampleClosed,
} from "./kinematics.mjs";

export const FIT_SAMPLES = 64;
const CURVE_SAMPLES = 128;
const IDENTITY = { a: 1, b: 0, tx: 0, ty: 0 };
const pack = (points) => Float64Array.from(points.flatMap((p) => [p.x, p.y]));
const unpack = (values) =>
  Array.from({ length: values.length / 2 }, (_, i) => ({
    x: values[i * 2],
    y: values[i * 2 + 1],
  }));
const finitePoint = (p) => p && Number.isFinite(p.x) && Number.isFinite(p.y);

function prepareTarget(points, count = FIT_SAMPLES) {
  count = [32, 64, 128, 256].includes(count) ? count : FIT_SAMPLES;
  if (!Array.isArray(points) || points.length < 3 || !points.every(finitePoint))
    throw new TypeError(
      "Draw a finite closed path with at least three points.",
    );
  const samples = resampleClosed(points, count);
  if (!samples.length)
    throw new RangeError("The target path needs some length.");
  const values = pack(samples);
  let cx = 0,
    cy = 0,
    xmin = Infinity,
    xmax = -Infinity,
    ymin = Infinity,
    ymax = -Infinity;
  for (const p of points) {
    xmin = Math.min(xmin, p.x);
    xmax = Math.max(xmax, p.x);
    ymin = Math.min(ymin, p.y);
    ymax = Math.max(ymax, p.y);
  }
  const diagonal = Math.hypot(xmax - xmin, ymax - ymin);
  if (!Number.isFinite(diagonal) || diagonal < 1e-8)
    throw new RangeError("The target path is too small.");
  for (let i = 0; i < count; i++) {
    cx += values[i * 2];
    cy += values[i * 2 + 1];
  }
  cx /= count;
  cy /= count;
  const centered = values.slice();
  let energy = 0;
  for (let i = 0; i < count; i++) {
    centered[i * 2] -= cx;
    centered[i * 2 + 1] -= cy;
    energy += centered[i * 2] ** 2 + centered[i * 2 + 1] ** 2;
  }
  return { samples, values, centered, cx, cy, energy, diagonal, count };
}

/** Similarity/cyclic alignment of already arc-length-resampled packed points. */
function fitPacked(source, target, similarity = true) {
  if (!source) return null;
  const count = target.count,
    mask = count - 1;
  let cx = 0,
    cy = 0;
  for (let i = 0; i < count; i++) {
    cx += source[i * 2];
    cy += source[i * 2 + 1];
  }
  cx /= count;
  cy /= count;
  const centered = source.slice();
  let sourceEnergy = 0;
  for (let i = 0; i < count; i++) {
    centered[i * 2] -= cx;
    centered[i * 2 + 1] -= cy;
    sourceEnergy += centered[i * 2] ** 2 + centered[i * 2 + 1] ** 2;
  }
  if (sourceEnergy < 1e-14 || !Number.isFinite(sourceEnergy)) return null;
  const targetCentered = target.centered;
  const translationPenalty =
    count * ((cx - target.cx) ** 2 + (cy - target.cy) ** 2);
  let best = { sse: Infinity, phase: 0, direction: 1, reflected: false };
  for (const direction of [1, -1]) {
    for (let shift = 0; shift < count; shift++) {
      let xx = 0,
        yy = 0,
        xy = 0,
        yx = 0;
      for (let i = 0; i < count; i++) {
        const j = (shift + direction * i) & mask;
        const sx = centered[j * 2],
          sy = centered[j * 2 + 1];
        const tx = targetCentered[i * 2],
          ty = targetCentered[i * 2 + 1];
        xx += sx * tx;
        yy += sy * ty;
        xy += sx * ty;
        yx += sy * tx;
      }
      if (similarity) {
        for (const reflected of [false, true]) {
          const real = reflected ? xx - yy : xx + yy;
          const imaginary = reflected ? xy + yx : xy - yx;
          const sse =
            target.energy -
            (real * real + imaginary * imaginary) / sourceEnergy;
          if (sse < best.sse)
            best = { sse, phase: shift, direction, reflected };
        }
      } else {
        const sse =
          target.energy + sourceEnergy - 2 * (xx + yy) + translationPenalty;
        if (sse < best.sse)
          best = { sse, phase: shift, direction, reflected: false };
      }
    }
  }
  // A continuous sub-sample phase avoids a 1/64-cycle quantization floor.
  const atPhase = (phase) => {
    const whole = Math.floor(phase),
      fraction = phase - whole;
    let energy = 0,
      real = 0,
      imaginary = 0;
    for (let i = 0; i < count; i++) {
      const j = (whole + best.direction * i) & mask,
        next = (j + 1) & mask;
      const sx =
        centered[j * 2] * (1 - fraction) + centered[next * 2] * fraction;
      let sy =
        centered[j * 2 + 1] * (1 - fraction) +
        centered[next * 2 + 1] * fraction;
      if (best.reflected) sy = -sy;
      const tx = targetCentered[i * 2],
        ty = targetCentered[i * 2 + 1];
      real += sx * tx + sy * ty;
      imaginary += sx * ty - sy * tx;
      energy += sx * sx + sy * sy;
    }
    const sse = similarity
      ? target.energy - (real * real + imaginary * imaginary) / energy
      : target.energy + energy - 2 * real + translationPenalty;
    return { sse, phase, real, imaginary, energy };
  };
  let optimum = atPhase(best.phase);
  let low = best.phase - 0.65,
    high = best.phase + 0.65;
  const golden = (Math.sqrt(5) - 1) / 2;
  let left = atPhase(high - golden * (high - low));
  let right = atPhase(low + golden * (high - low));
  for (let i = 0; i < 10; i++) {
    if (left.sse < optimum.sse) optimum = left;
    if (right.sse < optimum.sse) optimum = right;
    if (left.sse < right.sse) {
      high = right.phase;
      right = left;
      left = atPhase(high - golden * (high - low));
    } else {
      low = left.phase;
      left = right;
      right = atPhase(low + golden * (high - low));
    }
  }
  const a = similarity ? optimum.real / optimum.energy : 1;
  const b = similarity ? optimum.imaginary / optimum.energy : 0;
  const sourceCY = best.reflected ? -cy : cy;
  const transform = similarity
    ? {
        a,
        b,
        tx: target.cx - a * cx + b * sourceCY,
        ty: target.cy - b * cx - a * sourceCY,
      }
    : { ...IDENTITY };
  return {
    source,
    error: Math.sqrt(Math.max(0, optimum.sse) / count) / target.diagonal,
    transform,
    phase: optimum.phase,
    direction: best.direction,
    reflected: best.reflected,
  };
}

function alignedCurve(fit) {
  const count = fit.source.length / 2,
    mask = count - 1;
  const whole = Math.floor(fit.phase),
    fraction = fit.phase - whole;
  const t = fit.transform;
  return Array.from({ length: count }, (_, i) => {
    const j = (whole + fit.direction * i) & mask,
      next = (j + 1) & mask;
    const x =
      fit.source[j * 2] * (1 - fraction) + fit.source[next * 2] * fraction;
    let y =
      fit.source[j * 2 + 1] * (1 - fraction) +
      fit.source[next * 2 + 1] * fraction;
    if (fit.reflected) y = -y;
    return { x: t.a * x - t.b * y + t.tx, y: t.b * x + t.a * y + t.ty };
  });
}

function mirroredDesign(design) {
  const params = { ...design.params, traceOffset: -design.params.traceOffset };
  if (design.family === "fourbar") params.branch *= -1;
  else params.railOffset *= -1;
  return { ...design, params };
}

function publicCandidate(candidate, target, keepTransform = false) {
  let design = candidate.design;
  const fit = candidate.fit;
  if (!keepTransform) {
    if (fit.reflected) design = mirroredDesign(design);
    design = {
      ...design,
      params: { ...design.params },
      transform: { ...fit.transform },
    };
  } else
    design = {
      ...design,
      params: { ...design.params },
      transform: { ...design.transform },
    };
  return {
    design,
    family: design.family,
    error: fit.error,
    curve: alignedCurve(fit),
    target: target.samples.map((p) => ({ ...p })),
    match: {
      phase: fit.phase,
      direction: fit.direction,
      reflected: fit.reflected,
      diagonal: target.diagonal,
      samples: target.count,
    },
    ...(candidate.sizeRatio === undefined
      ? {}
      : { sizeRatio: candidate.sizeRatio }),
  };
}

/** Refit a valid mechanism's placement, rotation, scale, assembly mirror, and phase. */
export function fitDesignToTarget(design, points) {
  design = validateDesign(design);
  if (!design) return null;
  const target = prepareTarget(points);
  const source = resampleCoordinates(
    sampleCoordinates(design, CURVE_SAMPLES, false),
    FIT_SAMPLES,
  );
  const fit = fitPacked(source, target, true);
  return fit ? publicCandidate({ design, fit }, target) : null;
}

/** Measure an edited mechanism in its current position; no placement refitting. */
export function measureDesign(design, points, { samples = FIT_SAMPLES } = {}) {
  design = validateDesign(design);
  if (!design) return null;
  const target = prepareTarget(points, samples);
  const sampled = sampleCoordinates(
    design,
    Math.max(CURVE_SAMPLES, target.count * 2),
    true,
  );
  const source = sampled && resampleCoordinates(sampled, target.count);
  const fit = fitPacked(source, target, false);
  return fit ? publicCandidate({ design, fit }, target, true) : null;
}

/** Compare closed curves directly, preserving their coordinates and traversal. */
export function compareCurves(sourcePoints, targetPoints) {
  const target = prepareTarget(targetPoints);
  const source = resampleClosed(sourcePoints, FIT_SAMPLES);
  if (!source.length) return null;
  const fit = fitPacked(pack(source), target, false);
  return fit
    ? { error: fit.error, curve: alignedCurve(fit), target: target.samples }
    : null;
}

function randomGenerator(seed) {
  let state = (Number(seed) || 0x41c6ce57) >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function halton(index, base) {
  let fraction = 1,
    value = 0;
  while (index > 0) {
    fraction /= base;
    value += fraction * (index % base);
    index = Math.floor(index / base);
  }
  return value;
}

function reflectUnit(value) {
  value = ((value % 2) + 2) % 2;
  return value > 1 ? 2 - value : value;
}

/** Every gene vector is full-rotation-valid by construction, including near bounds. */
function decode(family, genes) {
  let params;
  if (family === "fourbar") {
    const crank =
      genes[0] < 0.5 ? 0.1 + 1.68 * genes[0] : 1.06 + 1.88 * (genes[0] - 0.5);
    const sum = 1 + crank + 0.04 + 2.8 * genes[1];
    const difference = (Math.abs(1 - crank) - 0.02) * 0.94 * (2 * genes[2] - 1);
    params = {
      crank,
      coupler: (sum + difference) / 2,
      rocker: (sum - difference) / 2,
      traceAlong: -1 + 3 * genes[3],
      traceOffset: -1.5 + 3 * genes[4],
      branch: 1,
    };
  } else {
    const railOffset = -0.95 + 1.9 * genes[0];
    params = {
      crank: 1,
      coupler: 1 + Math.abs(railOffset) + 0.055 + 3.5 * genes[1],
      railOffset,
      traceAlong: -1 + 3 * genes[2],
      traceOffset: -1.5 + 3 * genes[3],
      branch: 1,
    };
  }
  return { family, params, transform: { ...IDENTITY } };
}

function geneDistance(a, b) {
  if (a.family !== b.family) return 1;
  let distance = 0;
  for (let i = 0; i < a.genes.length; i++)
    distance += (a.genes[i] - b.genes[i]) ** 2;
  return Math.sqrt(distance / a.genes.length);
}

/** Conservative camera-size bound; it does not change the reported RMS metric. */
function mechanismSizeRatio(design, curve, transform, diagonal) {
  const p = design.params;
  let xmin = -p.crank,
    xmax = Math.max(1, p.crank),
    ymin = -p.crank,
    ymax = p.crank;
  if (design.family === "fourbar") {
    xmin = Math.min(xmin, 1 - p.rocker);
    xmax = Math.max(xmax, 1 + p.rocker);
    ymin = Math.min(ymin, -p.rocker);
    ymax = Math.max(ymax, p.rocker);
  } else {
    xmin = Math.min(xmin, p.branch < 0 ? -p.crank - p.coupler : 0);
    xmax = Math.max(xmax, p.branch > 0 ? p.crank + p.coupler : 0);
    ymin = Math.min(ymin, p.railOffset);
    ymax = Math.max(ymax, p.railOffset);
  }
  for (let i = 0; i < curve.length; i += 2) {
    xmin = Math.min(xmin, curve[i]);
    xmax = Math.max(xmax, curve[i]);
    ymin = Math.min(ymin, curve[i + 1]);
    ymax = Math.max(ymax, curve[i + 1]);
  }
  const width = xmax - xmin,
    height = ymax - ymin;
  const worldWidth =
    Math.abs(transform.a) * width + Math.abs(transform.b) * height;
  const worldHeight =
    Math.abs(transform.b) * width + Math.abs(transform.a) * height;
  return Math.hypot(worldWidth, worldHeight) / diagonal;
}

export class MechanismSearch {
  constructor(
    points,
    { family = "all", seed = 2026, population = 64, maxGenerations = 150 } = {},
  ) {
    if (!["all", "fourbar", "slider"].includes(family))
      throw new RangeError(
        "Choose fourbar, slider, or all mechanism families.",
      );
    this.preparedTarget = prepareTarget(points);
    this.target = this.preparedTarget.samples.map((p) => ({ ...p }));
    this.random = randomGenerator(seed);
    this.generation = 0;
    this.evaluations = 0;
    this.maxGenerations = Math.max(
      5,
      Math.min(600, Math.round(maxGenerations) || 150),
    );
    this.evolutionGenerations = Math.max(
      3,
      Math.floor(this.maxGenerations * 0.72),
    );
    this.history = [];
    this.families = (family === "all" ? ["fourbar", "slider"] : [family]).map(
      (name) => ({
        family: name,
        size: Math.max(
          16,
          Math.min(
            96,
            Math.round(population / (family === "all" ? 2 : 1)) || 32,
          ),
        ),
        dimensions: name === "fourbar" ? 5 : 4,
        members: [],
        simplex: null,
      }),
    );
    for (const state of this.families) this.initialize(state);
    this.history.push({ generation: 0, error: this.bestCandidate().fit.error });
  }

  evaluate(family, genes) {
    this.evaluations++;
    const design = decode(family, genes);
    const raw = sampleCoordinates(design, CURVE_SAMPLES, false);
    const source = raw && resampleCoordinates(raw, FIT_SAMPLES);
    const fit = fitPacked(source, this.preparedTarget, true);
    const sizeRatio = fit
      ? mechanismSizeRatio(
          design,
          source,
          fit.transform,
          this.preparedTarget.diagonal,
        )
      : Infinity;
    // Reject machines over five target diagonals so a nearly stationary point
    // cannot win by magnifying a huge surrounding mechanism. This is a hard
    // admissibility condition; accepted candidates retain their actual RMS.
    if (fit && sizeRatio > 5) fit.error = Infinity;
    return {
      family,
      genes: Array.from(genes),
      design,
      sizeRatio,
      fit: fit || { error: Infinity },
    };
  }

  initialize(state) {
    const atlas = [];
    const shifts = Array.from({ length: state.dimensions }, () =>
      this.random(),
    );
    const primes = [2, 3, 5, 7, 11];
    const atlasSize = Math.max(128, state.size * 5);
    for (let i = 0; i < atlasSize; i++) {
      const genes = Array.from(
        { length: state.dimensions },
        (_, j) => (halton(i + 1, primes[j]) + shifts[j]) % 1,
      );
      atlas.push(this.evaluate(state.family, genes));
    }
    atlas.sort((a, b) => a.fit.error - b.fit.error);
    // Keep quality and geometric diversity, rather than seeding every member
    // with near-identical retrieved mechanisms.
    state.members.push(atlas[0]);
    for (const candidate of atlas.slice(1)) {
      if (state.members.length >= state.size * 0.8) break;
      if (state.members.every((other) => geneDistance(candidate, other) > 0.14))
        state.members.push(candidate);
    }
    for (const candidate of atlas) {
      if (state.members.length >= state.size) break;
      if (!state.members.includes(candidate)) state.members.push(candidate);
    }
    state.members.sort((a, b) => a.fit.error - b.fit.error);
  }

  evolve(state) {
    const members = state.members;
    const ranked = [...members].sort((a, b) => a.fit.error - b.fit.error);
    const indexExcept = (excluded) => {
      let index;
      do index = Math.floor(this.random() * members.length);
      while (excluded.includes(index));
      return index;
    };
    for (let i = 0; i < members.length; i++) {
      const x = members[i];
      const r1 = indexExcept([i]),
        r2 = indexExcept([i, r1]),
        r3 = indexExcept([i, r1, r2]);
      const pbest =
        ranked[
          Math.floor(
            this.random() * Math.max(2, Math.ceil(members.length * 0.2)),
          )
        ];
      const factor = 0.48 + this.random() * 0.42;
      const crossover = 0.7 + this.random() * 0.3;
      const forced = Math.floor(this.random() * state.dimensions);
      const global = this.random() < 0.3;
      const genes = x.genes.map((value, j) => {
        if (j !== forced && this.random() > crossover) return value;
        const mutant = global
          ? members[r1].genes[j] +
            factor * (members[r2].genes[j] - members[r3].genes[j])
          : value +
            factor * (pbest.genes[j] - value) +
            factor * (members[r1].genes[j] - members[r2].genes[j]);
        return reflectUnit(mutant);
      });
      const trial = this.evaluate(state.family, genes);
      if (trial.fit.error < x.fit.error) members[i] = trial;
    }
  }

  refine(state) {
    const dimensions = state.dimensions;
    if (!state.simplex) {
      const best = [...state.members].sort(
        (a, b) => a.fit.error - b.fit.error,
      )[0];
      state.simplex = [
        best,
        ...Array.from({ length: dimensions }, (_, j) => {
          const genes = [...best.genes];
          genes[j] = reflectUnit(genes[j] + (genes[j] < 0.94 ? 0.035 : -0.035));
          return this.evaluate(state.family, genes);
        }),
      ];
    }
    const simplex = state.simplex;
    simplex.sort((a, b) => a.fit.error - b.fit.error);
    const best = simplex[0],
      worst = simplex[dimensions];
    const centroid = Array.from(
      { length: dimensions },
      (_, j) =>
        simplex
          .slice(0, dimensions)
          .reduce((sum, item) => sum + item.genes[j], 0) / dimensions,
    );
    const make = (values) =>
      this.evaluate(state.family, values.map(reflectUnit));
    const reflected = make(
      centroid.map((value, j) => 2 * value - worst.genes[j]),
    );
    if (reflected.fit.error < best.fit.error) {
      const expanded = make(
        centroid.map((value, j) => value + 2 * (reflected.genes[j] - value)),
      );
      simplex[dimensions] =
        expanded.fit.error < reflected.fit.error ? expanded : reflected;
    } else if (reflected.fit.error < simplex[dimensions - 1].fit.error)
      simplex[dimensions] = reflected;
    else {
      const outside = reflected.fit.error < worst.fit.error;
      const contracted = make(
        centroid.map(
          (value, j) =>
            value + 0.5 * ((outside ? reflected : worst).genes[j] - value),
        ),
      );
      if (contracted.fit.error < (outside ? reflected : worst).fit.error)
        simplex[dimensions] = contracted;
      else
        for (let i = 1; i <= dimensions; i++)
          simplex[i] = make(
            simplex[i].genes.map((value, j) => (value + best.genes[j]) * 0.5),
          );
    }
    simplex.sort((a, b) => a.fit.error - b.fit.error);
    state.members.sort((a, b) => a.fit.error - b.fit.error);
    if (simplex[0].fit.error < state.members[0].fit.error)
      state.members[state.members.length - 1] = simplex[0];
  }

  bestCandidate() {
    let best = null;
    for (const state of this.families)
      for (const item of state.members) {
        if (!best || item.fit.error < best.fit.error) best = item;
      }
    return best;
  }

  snapshot() {
    const ranked = this.families
      .flatMap((state) => state.members)
      .filter((item) => Number.isFinite(item.fit.error))
      .sort((a, b) => a.fit.error - b.fit.error);
    const chosen = [ranked[0]];
    // Prefer a genuinely different mechanism family for the second alternative.
    const otherFamily = ranked.find((item) => item.family !== ranked[0].family);
    if (otherFamily) chosen.push(otherFamily);
    for (const item of ranked) {
      if (chosen.length >= 3) break;
      if (chosen.every((other) => geneDistance(item, other) > 0.1))
        chosen.push(item);
    }
    const candidates = chosen.map((candidate) =>
      publicCandidate(candidate, this.preparedTarget),
    );
    return {
      best: candidates[0],
      candidates,
      generation: this.generation,
      evaluations: this.evaluations,
      history: this.history.map((item) => ({ ...item })),
      done: this.generation >= this.maxGenerations,
      phase:
        this.generation < this.evolutionGenerations ? "evolving" : "refining",
    };
  }

  step(generations = 1) {
    generations = Math.max(0, Math.min(600, Math.floor(generations) || 0));
    for (
      let i = 0;
      i < generations && this.generation < this.maxGenerations;
      i++
    ) {
      for (const state of this.families) {
        if (this.generation < this.evolutionGenerations) this.evolve(state);
        else this.refine(state);
      }
      this.generation++;
      this.history.push({
        generation: this.generation,
        error: this.bestCandidate().fit.error,
      });
    }
    return this.snapshot();
  }
}
