// Binned-SAH bounding volume hierarchy, flattened for the GPU.
//
// Each GPU node is 16 floats (64 bytes) and stores the boxes of BOTH children,
// so one fetch is enough to test both and pick which side to visit first:
//   [ Lmin.xyz, La,  Lmax.xyz, Lb,  Rmin.xyz, Ra,  Rmax.xyz, Rb ]
// For a child, b > 0 means a leaf holding primitives [a, a + b);
// b = 0 means an inner node at index a. There are no empty children: a slab test
// treats an inverted box as unbounded, so a lone root leaf is written to both sides.

const BINS = 16;
const MAX_LEAF = 4;
const MAX_DEPTH = 40;          // the shader's traversal stack is this deep

// bmin/bmax/cent: Float32Array(3n) per-primitive boxes and centroids
export function buildBVH(bmin, bmax, cent, n) {
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;

  const cap = Math.max(2 * n, 2);
  const nMin = new Float32Array(cap * 3), nMax = new Float32Array(cap * 3);
  const nFirst = new Uint32Array(cap), nCount = new Uint32Array(cap);
  const nLeft = new Int32Array(cap).fill(-1);
  const nDepth = new Uint8Array(cap);
  let used = 1;

  // root box
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) {
    if (bmin[3 * i + k] < lo[k]) lo[k] = bmin[3 * i + k];
    if (bmax[3 * i + k] > hi[k]) hi[k] = bmax[3 * i + k];
  }
  nMin.set(lo, 0); nMax.set(hi, 0); nFirst[0] = 0; nCount[0] = n;

  const binCount = new Uint32Array(BINS * 3);
  const binBox = new Float32Array(BINS * 3 * 6);
  const rightArea = new Float32Array(BINS), rightCount = new Uint32Array(BINS);
  const rightBox = new Float32Array(BINS * 6);
  const bestL = new Float32Array(6), bestR = new Float32Array(6);
  const scale = [0, 0, 0];
  let maxDepth = 0;

  const area = (x0, y0, z0, x1, y1, z1) => {
    const dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
    return dx * dy + dy * dz + dz * dx;
  };

  const stack = [0];
  while (stack.length) {
    const node = stack.pop();
    const first = nFirst[node], count = nCount[node], depth = nDepth[node];
    if (depth > maxDepth) maxDepth = depth;
    if (count <= 1 || depth >= MAX_DEPTH) continue;

    // centroid bounds
    let c0 = Infinity, c1 = Infinity, c2 = Infinity, d0 = -Infinity, d1 = -Infinity, d2 = -Infinity;
    for (let i = first; i < first + count; i++) {
      const p = 3 * order[i];
      const x = cent[p], y = cent[p + 1], z = cent[p + 2];
      if (x < c0) c0 = x; if (x > d0) d0 = x;
      if (y < c1) c1 = y; if (y > d1) d1 = y;
      if (z < c2) c2 = z; if (z > d2) d2 = z;
    }
    const cmin = [c0, c1, c2], ext = [d0 - c0, d1 - c1, d2 - c2];

    // fill bins on all three axes in one pass
    binCount.fill(0);
    for (let b = 0; b < BINS * 3; b++) {
      const o = b * 6;
      binBox[o] = binBox[o + 1] = binBox[o + 2] = Infinity;
      binBox[o + 3] = binBox[o + 4] = binBox[o + 5] = -Infinity;
    }
    for (let a = 0; a < 3; a++) scale[a] = ext[a] > 1e-12 ? BINS / ext[a] : 0;
    for (let i = first; i < first + count; i++) {
      const pr = order[i], p = 3 * pr;
      for (let a = 0; a < 3; a++) {
        if (!scale[a]) continue;
        let k = ((cent[p + a] - cmin[a]) * scale[a]) | 0;
        if (k >= BINS) k = BINS - 1;
        const b = a * BINS + k, o = b * 6;
        binCount[b]++;
        if (bmin[p] < binBox[o]) binBox[o] = bmin[p];
        if (bmin[p + 1] < binBox[o + 1]) binBox[o + 1] = bmin[p + 1];
        if (bmin[p + 2] < binBox[o + 2]) binBox[o + 2] = bmin[p + 2];
        if (bmax[p] > binBox[o + 3]) binBox[o + 3] = bmax[p];
        if (bmax[p + 1] > binBox[o + 4]) binBox[o + 4] = bmax[p + 1];
        if (bmax[p + 2] > binBox[o + 5]) binBox[o + 5] = bmax[p + 2];
      }
    }

    // sweep for the cheapest split plane
    let bestCost = Infinity, bestAxis = -1, bestSplit = -1;
    for (let a = 0; a < 3; a++) {
      if (!scale[a]) continue;
      let rx0 = Infinity, ry0 = Infinity, rz0 = Infinity, rx1 = -Infinity, ry1 = -Infinity, rz1 = -Infinity, rc = 0;
      for (let k = BINS - 1; k > 0; k--) {
        const o = (a * BINS + k) * 6;
        rc += binCount[a * BINS + k];
        if (binBox[o] < rx0) rx0 = binBox[o]; if (binBox[o + 1] < ry0) ry0 = binBox[o + 1]; if (binBox[o + 2] < rz0) rz0 = binBox[o + 2];
        if (binBox[o + 3] > rx1) rx1 = binBox[o + 3]; if (binBox[o + 4] > ry1) ry1 = binBox[o + 4]; if (binBox[o + 5] > rz1) rz1 = binBox[o + 5];
        rightCount[k] = rc;
        rightArea[k] = rc ? area(rx0, ry0, rz0, rx1, ry1, rz1) : 0;
        const q = k * 6;
        rightBox[q] = rx0; rightBox[q + 1] = ry0; rightBox[q + 2] = rz0;
        rightBox[q + 3] = rx1; rightBox[q + 4] = ry1; rightBox[q + 5] = rz1;
      }
      let lx0 = Infinity, ly0 = Infinity, lz0 = Infinity, lx1 = -Infinity, ly1 = -Infinity, lz1 = -Infinity, lc = 0;
      for (let k = 0; k < BINS - 1; k++) {
        const o = (a * BINS + k) * 6;
        lc += binCount[a * BINS + k];
        if (binBox[o] < lx0) lx0 = binBox[o]; if (binBox[o + 1] < ly0) ly0 = binBox[o + 1]; if (binBox[o + 2] < lz0) lz0 = binBox[o + 2];
        if (binBox[o + 3] > lx1) lx1 = binBox[o + 3]; if (binBox[o + 4] > ly1) ly1 = binBox[o + 4]; if (binBox[o + 5] > lz1) lz1 = binBox[o + 5];
        const rcount = rightCount[k + 1];
        if (!lc || !rcount) continue;
        const cost = lc * area(lx0, ly0, lz0, lx1, ly1, lz1) + rcount * rightArea[k + 1];
        if (cost < bestCost) {
          bestCost = cost; bestAxis = a; bestSplit = k;
          bestL[0] = lx0; bestL[1] = ly0; bestL[2] = lz0; bestL[3] = lx1; bestL[4] = ly1; bestL[5] = lz1;
          for (let q = 0; q < 6; q++) bestR[q] = rightBox[(k + 1) * 6 + q];
        }
      }
    }

    const o = node * 3;
    const parentArea = area(nMin[o], nMin[o + 1], nMin[o + 2], nMax[o], nMax[o + 1], nMax[o + 2]);
    const splitCost = 1 + bestCost / Math.max(parentArea, 1e-30);   // traversal ~ one primitive test
    let mid;
    if (bestAxis >= 0 && (count > MAX_LEAF || splitCost < count)) {
      const a = bestAxis, s = scale[a], m = cmin[a];
      let i = first, j = first + count - 1;
      while (i <= j) {
        let k = ((cent[3 * order[i] + a] - m) * s) | 0;
        if (k >= BINS) k = BINS - 1;
        if (k <= bestSplit) i++;
        else { const t = order[i]; order[i] = order[j]; order[j] = t; j--; }
      }
      mid = i;
    } else if (count > MAX_LEAF) {
      mid = first + (count >> 1);                 // all centroids coincide: split by index
      bestL.set(boxOf(order, first, mid, bmin, bmax));
      bestR.set(boxOf(order, mid, first + count, bmin, bmax));
    } else continue;                              // cheaper as a leaf

    const l = used++, r = used++;
    nLeft[node] = l;
    nFirst[l] = first; nCount[l] = mid - first;
    nFirst[r] = mid; nCount[r] = first + count - mid;
    nMin.set(bestL.subarray(0, 3), l * 3); nMax.set(bestL.subarray(3, 6), l * 3);
    nMin.set(bestR.subarray(0, 3), r * 3); nMax.set(bestR.subarray(3, 6), r * 3);
    nDepth[l] = nDepth[r] = depth + 1;
    stack.push(r, l);
  }

  return { ...flatten(nMin, nMax, nFirst, nCount, nLeft, used), order, maxDepth };
}

function boxOf(order, from, to, bmin, bmax) {
  const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (let i = from; i < to; i++) for (let k = 0; k < 3; k++) {
    b[k] = Math.min(b[k], bmin[3 * order[i] + k]);
    b[k + 3] = Math.max(b[k + 3], bmax[3 * order[i] + k]);
  }
  return b;
}

// Depth-first layout: an inner node's left inner child follows it directly.
function flatten(nMin, nMax, nFirst, nCount, nLeft, used) {
  const outIndex = new Int32Array(used).fill(-1);
  let count = 0;
  const stack = [0];
  while (stack.length) {
    const b = stack.pop();
    outIndex[b] = count++;
    const l = nLeft[b];
    if (l < 0) continue;
    if (nLeft[l + 1] >= 0) stack.push(l + 1);
    if (nLeft[l] >= 0) stack.push(l);
  }
  const rootIsLeaf = nLeft[0] < 0;
  const nodeCount = rootIsLeaf ? 1 : count;
  const f = new Float32Array(nodeCount * 16), u = new Uint32Array(f.buffer);
  const writeChild = (o, c) => {
    f.set(nMin.subarray(c * 3, c * 3 + 3), o); f.set(nMax.subarray(c * 3, c * 3 + 3), o + 4);
    if (nLeft[c] < 0) { u[o + 3] = nFirst[c]; u[o + 7] = nCount[c]; }
    else { u[o + 3] = outIndex[c]; u[o + 7] = 0; }
  };
  if (rootIsLeaf) { writeChild(0, 0); writeChild(8, 0); }
  else for (let b = 0; b < used; b++) {
    if (nLeft[b] < 0 || outIndex[b] < 0) continue;
    const o = outIndex[b] * 16;
    writeChild(o, nLeft[b]);
    writeChild(o + 8, nLeft[b] + 1);
  }
  return { nodes: f, nodeCount, bounds: [...nMin.subarray(0, 3), ...nMax.subarray(0, 3)] };
}
