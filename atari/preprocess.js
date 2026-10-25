// The agent's view of the screen, exactly as Dopamine's AtariPreprocessing makes it:
// max of the last two grayscale frames, shrunk to 84x84 with OpenCV's INTER_AREA.

const f = Math.fround;

// OpenCV computeResizeAreaTab: which source pixels feed each output pixel, and with what weight (float32).
function areaTab(ssize, dsize) {
  const scale = ssize / dsize, tab = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale, fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1), sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) tab.push([dx, sx1 - 1, f((sx1 - fsx1) / cellWidth)]);
    for (let sx = sx1; sx < sx2; sx++) tab.push([dx, sx, f(1 / cellWidth)]);
    if (fsx2 - sx2 > 1e-3) tab.push([dx, sx2, f(Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth)]);
  }
  return tab;
}

// cvRound: round half to even
function cvRound(v) {
  const r = Math.round(v);
  return (Math.abs(v % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
}

export function makeResizer(sw = 160, sh = 210, dw = 84, dh = 84) {
  const xtab = areaTab(sw, dw), ytab = areaTab(sh, dh);
  const buf = new Float32Array(dw), sum = new Float32Array(dw);
  return function resize(src, dst = new Uint8Array(dw * dh)) {
    // OpenCV's resizeArea_Invoker: per source row, a horizontal pass into buf, then sum += beta * buf;
    // a destination row is written when the next table entry belongs to a new row.
    let prevDy = ytab[0][0];
    sum.fill(0);
    for (let j = 0; j < ytab.length; j++) {
      const [dy, sy, beta] = ytab[j];
      buf.fill(0);
      const row = sy * sw;
      for (const [dx, sx, alpha] of xtab) buf[dx] = f(buf[dx] + f(src[row + sx] * alpha));
      if (dy !== prevDy) {
        for (let dx = 0; dx < dw; dx++) {
          dst[prevDy * dw + dx] = Math.max(0, Math.min(255, cvRound(sum[dx])));
          sum[dx] = f(beta * buf[dx]);
        }
        prevDy = dy;
      } else {
        for (let dx = 0; dx < dw; dx++) sum[dx] = f(sum[dx] + f(beta * buf[dx]));
      }
    }
    for (let dx = 0; dx < dw; dx++) dst[prevDy * dw + dx] = Math.max(0, Math.min(255, cvRound(sum[dx])));
    return dst;
  };
}

// Max-pool two grayscale frames, then resize.
export function makeObserver() {
  const resize = makeResizer(), pooled = new Uint8Array(160 * 210);
  return (a, b, dst) => {
    for (let i = 0; i < pooled.length; i++) pooled[i] = a[i] > b[i] ? a[i] : b[i];
    return resize(pooled, dst);
  };
}
