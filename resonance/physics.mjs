/**
 * Fixed-edge, uniform-tension membrane modes on a Cartesian grid.
 *
 * The five-point finite-difference discretization of -Δu = λu uses zero
 * displacement outside the polygon (Dirichlet boundary conditions). Its sparse
 * symmetric positive-definite matrix is factored with banded Cholesky. Block
 * inverse iteration and a Rayleigh–Ritz projection recover the lowest modes,
 * including repeated eigenvalues. No network, dependencies, or baked-in modes.
 *
 * Frequencies at unit wave speed are sqrt(λ)/(2π). A physical membrane multiplies
 * these by sqrt(tension / arealDensity). This is a coarse membrane model, not a
 * bending-plate model; the polygon's boundary is approximated on the grid.
 */

const TAU = 2 * Math.PI;

/** Editable counterclockwise control points, all within [-1, 1]². */
export function makeOutline(preset = "circle", count = 16) {
  count = Math.max(8, Math.min(128, Math.round(count)));
  return Array.from({ length: count }, (_, i) => {
    const angle = (TAU * i) / count;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    let radius = 0.79;
    if (preset === "square" || preset === "rounded-square") {
      radius = 0.66 / Math.pow(c ** 4 + s ** 4, 0.25);
    } else if (preset === "petal") {
      radius = 0.69 + 0.15 * Math.cos(5 * angle);
    } else if (preset === "drop") {
      radius = 0.66 + 0.17 * s;
    }
    return { x: radius * c, y: radius * s };
  });
}

/** Strict polygon interior: the clamped boundary itself is excluded. */
export function insideOutline(outline, x, y) {
  let inside = false;
  for (let i = 0, j = outline.length - 1; i < outline.length; j = i++) {
    const a = outline[j];
    const b = outline[i];
    const cross = (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x);
    if (
      Math.abs(cross) < 1e-11 &&
      x >= Math.min(a.x, b.x) - 1e-11 &&
      x <= Math.max(a.x, b.x) + 1e-11 &&
      y >= Math.min(a.y, b.y) - 1e-11 &&
      y <= Math.max(a.y, b.y) + 1e-11
    )
      return false;
    if (
      a.y > y !== b.y > y &&
      x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x
    )
      inside = !inside;
  }
  return inside;
}

function dot(a, b) {
  let value = 0;
  for (let i = 0; i < a.length; i++) value += a[i] * b[i];
  return value;
}

function orthonormalize(vectors) {
  for (let i = 0; i < vectors.length; i++) {
    const v = vectors[i];
    // Two modified Gram–Schmidt passes retain orthogonality after inversion.
    for (let pass = 0; pass < 2; pass++) {
      for (let j = 0; j < i; j++) {
        const q = vectors[j];
        const projection = dot(v, q);
        for (let k = 0; k < v.length; k++) v[k] -= projection * q[k];
      }
    }
    const length = Math.sqrt(dot(v, v));
    if (length < 1e-14)
      throw new Error("Membrane eigensolver lost its independent basis.");
    for (let k = 0; k < v.length; k++) v[k] /= length;
  }
}

/** Jacobi diagonalization of a small symmetric Rayleigh–Ritz matrix. */
function symmetricEigen(matrix, size) {
  const a = matrix.slice();
  const rotation = new Float64Array(size * size);
  for (let i = 0; i < size; i++) rotation[i * size + i] = 1;
  for (let sweep = 0; sweep < 40; sweep++) {
    let largest = 0;
    for (let p = 0; p < size - 1; p++) {
      for (let q = p + 1; q < size; q++) {
        const off = a[p * size + q];
        largest = Math.max(largest, Math.abs(off));
        if (Math.abs(off) < 1e-14) continue;
        const tau = (a[q * size + q] - a[p * size + p]) / (2 * off);
        const t =
          Math.sign(tau || 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const cosine = 1 / Math.sqrt(1 + t * t);
        const sine = t * cosine;
        const app = a[p * size + p];
        const aqq = a[q * size + q];
        a[p * size + p] = app - t * off;
        a[q * size + q] = aqq + t * off;
        a[p * size + q] = a[q * size + p] = 0;
        for (let k = 0; k < size; k++) {
          if (k !== p && k !== q) {
            const akp = a[k * size + p];
            const akq = a[k * size + q];
            a[k * size + p] = a[p * size + k] = cosine * akp - sine * akq;
            a[k * size + q] = a[q * size + k] = sine * akp + cosine * akq;
          }
          const vkp = rotation[k * size + p];
          const vkq = rotation[k * size + q];
          rotation[k * size + p] = cosine * vkp - sine * vkq;
          rotation[k * size + q] = sine * vkp + cosine * vkq;
        }
      }
    }
    if (largest < 1e-12) break;
  }
  const order = Array.from({ length: size }, (_, i) => i).sort(
    (i, j) => a[i * size + i] - a[j * size + j],
  );
  return { values: order.map((i) => a[i * size + i]), rotation, order };
}

function makeOperator(indices, resolution, nodeCount) {
  const neighbors = new Int32Array(nodeCount * 4).fill(-1);
  let bandwidth = 1;
  for (let y = 1; y < resolution - 1; y++) {
    for (let x = 1; x < resolution - 1; x++) {
      const cell = y * resolution + x;
      const i = indices[cell];
      if (i < 0) continue;
      const adjacent = [
        cell - 1,
        cell + 1,
        cell - resolution,
        cell + resolution,
      ];
      for (let d = 0; d < 4; d++) {
        const j = indices[adjacent[d]];
        neighbors[i * 4 + d] = j;
        if (j >= 0) bandwidth = Math.max(bandwidth, Math.abs(i - j));
      }
    }
  }
  const stride = bandwidth + 1;
  // Lower triangular band: factor[i * stride + (i - j)] = L[i,j].
  const factor = new Float64Array(nodeCount * stride);
  for (let i = 0; i < nodeCount; i++) {
    const row = i * stride;
    const start = Math.max(0, i - bandwidth);
    let diagonal = 4;
    for (let j = start; j < i; j++) {
      let value = neighbors[i * 4] === j || neighbors[i * 4 + 2] === j ? -1 : 0;
      for (let k = start; k < j; k++) {
        value -= factor[row + i - k] * factor[j * stride + j - k];
      }
      value /= factor[j * stride];
      factor[row + i - j] = value;
      diagonal -= value * value;
    }
    factor[row] = Math.sqrt(diagonal);
  }
  return {
    apply(vector) {
      const out = new Float64Array(nodeCount);
      for (let i = 0; i < nodeCount; i++) {
        let value = 4 * vector[i];
        for (let d = 0; d < 4; d++) {
          const j = neighbors[i * 4 + d];
          if (j >= 0) value -= vector[j];
        }
        out[i] = value;
      }
      return out;
    },
    inverse(vector) {
      const out = vector.slice();
      for (let i = 0; i < nodeCount; i++) {
        let value = out[i];
        for (let j = Math.max(0, i - bandwidth); j < i; j++) {
          value -= factor[i * stride + i - j] * out[j];
        }
        out[i] = value / factor[i * stride];
      }
      for (let i = nodeCount - 1; i >= 0; i--) {
        let value = out[i];
        for (let j = i + 1; j <= Math.min(nodeCount - 1, i + bandwidth); j++) {
          value -= factor[j * stride + j - i] * out[j];
        }
        out[i] = value / factor[i * stride];
      }
      return out;
    },
  };
}

/**
 * Returns a structured-cloneable solution. Grid nodes span [-1,1]², with x
 * varying fastest in `indices`; a -1 index denotes fixed zero displacement.
 * `values` have maximum absolute magnitude 1. `mass` is their squared spatial
 * L2 norm (h² Σ values²), needed for point-force projection. `residualMax` is
 * max ||Av - λv||₂ / (λ ||v||₂) across the returned modes.
 */
export function solveMembrane(
  outline,
  { resolution = 31, modeCount = 12 } = {},
) {
  if (
    !Array.isArray(outline) ||
    outline.length < 3 ||
    outline.some(
      (p) =>
        !Number.isFinite(p.x) ||
        !Number.isFinite(p.y) ||
        Math.abs(p.x) > 1 ||
        Math.abs(p.y) > 1,
    )
  ) {
    throw new TypeError(
      "A membrane needs at least three finite outline points within [-1, 1].",
    );
  }
  if (!Number.isFinite(resolution) || !Number.isFinite(modeCount)) {
    throw new TypeError("Resolution and mode count must be finite numbers.");
  }
  resolution = Math.max(9, Math.min(65, Math.round(resolution)));
  modeCount = Math.max(1, Math.min(32, Math.round(modeCount)));
  const spacing = 2 / (resolution - 1);
  const indices = new Int32Array(resolution * resolution).fill(-1);
  const coordinates = [];
  for (let row = 1; row < resolution - 1; row++) {
    for (let col = 1; col < resolution - 1; col++) {
      const x = -1 + col * spacing;
      const y = -1 + row * spacing;
      if (insideOutline(outline, x, y)) {
        indices[row * resolution + col] = coordinates.length;
        coordinates.push({ x, y });
      }
    }
  }
  const nodeCount = coordinates.length;
  if (nodeCount < 4)
    throw new RangeError(
      "The membrane is too small for this grid. Enlarge the outline.",
    );
  modeCount = Math.min(modeCount, nodeCount);
  const blockSize = Math.min(
    nodeCount,
    Math.max(modeCount + 10, modeCount * 2),
  );
  const operator = makeOperator(indices, resolution, nodeCount);
  // Deterministic seeded vectors avoid biased symmetry classes and preserve
  // complete repeated-eigenvalue subspaces (a single Lanczos seed may not).
  let seed = 0x6d2b79f5;
  const random = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 4294967296 - 0.5;
  };
  let vectors = Array.from({ length: blockSize }, () =>
    Float64Array.from({ length: nodeCount }, random),
  );
  orthonormalize(vectors);
  let eigenvalues;
  let residualMax = Infinity;
  let iterations = 0;
  for (; iterations < 60; iterations++) {
    vectors = vectors.map((v) => operator.inverse(v));
    orthonormalize(vectors);
    // Project and diagonalize every other iteration; do not wait for the
    // individual iterates to separate degenerate eigenvalues.
    if (iterations < 3 || iterations % 2 === 0) continue;
    const applied = vectors.map((v) => operator.apply(v));
    const reduced = new Float64Array(blockSize * blockSize);
    for (let i = 0; i < blockSize; i++) {
      for (let j = 0; j <= i; j++) {
        reduced[i * blockSize + j] = reduced[j * blockSize + i] = dot(
          vectors[i],
          applied[j],
        );
      }
    }
    const eig = symmetricEigen(reduced, blockSize);
    eigenvalues = eig.values;
    vectors = eig.order.map((column) => {
      const out = new Float64Array(nodeCount);
      for (let j = 0; j < blockSize; j++) {
        const coefficient = eig.rotation[j * blockSize + column];
        const v = vectors[j];
        for (let k = 0; k < nodeCount; k++) out[k] += coefficient * v[k];
      }
      return out;
    });
    residualMax = 0;
    for (let i = 0; i < modeCount; i++) {
      const av = operator.apply(vectors[i]);
      let squared = 0;
      for (let j = 0; j < nodeCount; j++)
        squared += (av[j] - eigenvalues[i] * vectors[i][j]) ** 2;
      residualMax = Math.max(residualMax, Math.sqrt(squared) / eigenvalues[i]);
    }
    if (residualMax < 2e-7) break;
  }
  const modes = vectors.slice(0, modeCount).map((v, index) => {
    let peak = 0;
    let peakIndex = 0;
    for (let j = 0; j < nodeCount; j++) {
      if (Math.abs(v[j]) > peak) {
        peak = Math.abs(v[j]);
        peakIndex = j;
      }
    }
    // Stable sign makes a reload visually reproducible.
    const scale = (v[peakIndex] >= 0 ? 1 : -1) / peak;
    const values = Float32Array.from(v, (value) => value * scale);
    return {
      eigenvalue: eigenvalues[index] / (spacing * spacing),
      frequencyRatio: Math.sqrt(eigenvalues[index] / eigenvalues[0]),
      values,
      mass: dot(values, values) * spacing * spacing,
    };
  });
  return {
    resolution,
    spacing,
    coordinates,
    indices,
    modes,
    outline: outline.map(({ x, y }) => ({ x, y })),
    fundamental: Math.sqrt(modes[0].eigenvalue) / TAU,
    residualMax,
    iterations: iterations + 1,
  };
}

/** Bilinear mode displacement at a strike position; zero on/outside the rim. */
export function sampleMode(solution, modeIndex, x, y) {
  if (
    !solution.modes[modeIndex] ||
    !Number.isFinite(x) ||
    !Number.isFinite(y) ||
    x <= -1 ||
    x >= 1 ||
    y <= -1 ||
    y >= 1 ||
    !insideOutline(solution.outline, x, y)
  )
    return 0;
  const { resolution, indices } = solution;
  const gx = ((x + 1) * (resolution - 1)) / 2;
  const gy = ((y + 1) * (resolution - 1)) / 2;
  const col = Math.floor(gx);
  const row = Math.floor(gy);
  const tx = gx - col;
  const ty = gy - row;
  const values = solution.modes[modeIndex].values;
  const at = (cx, cy) => {
    const index = indices[cy * resolution + cx];
    return index >= 0 ? values[index] : 0;
  };
  return (
    (1 - ty) * ((1 - tx) * at(col, row) + tx * at(col + 1, row)) +
    ty * ((1 - tx) * at(col, row + 1) + tx * at(col + 1, row + 1))
  );
}
