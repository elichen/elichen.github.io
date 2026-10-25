// preprocess.js resize vs cv2.resize(INTER_AREA) on real frames (from resize_ref.py); must match to within 1 on a few pixels per million.
import { readFileSync } from 'node:fs';
import { makeResizer } from '../preprocess.js';
const src = readFileSync(new URL('./test/resize_src.bin', import.meta.url)), ref = readFileSync(new URL('./test/resize_ref.bin', import.meta.url));
const n = src.length / (160 * 210), resize = makeResizer();
let bad = 0, maxd = 0;
for (let i = 0; i < n; i++) {
  const out = resize(src.subarray(i * 33600, (i + 1) * 33600));
  for (let j = 0; j < 7056; j++) { const d = Math.abs(out[j] - ref[i * 7056 + j]); if (d) { bad++; maxd = Math.max(maxd, d); } }
}
// OpenCV's compiled loop rounds slightly differently: ~6 pixels per million end up 1 apart.
console.log(`${n} frames, ${bad} of ${n * 7056} pixels differ (max ${maxd})`, bad <= 1e-5 * n * 7056 && maxd <= 1 ? 'PASS' : 'FAIL');
