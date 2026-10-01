// Off the main thread: drum modes (finite elements) and the listener network.

import { drum, modes, crisscrossMesh, K_MODES } from './fem.js';
import { Listener } from './listener.js';

let listener = null;

function pack(d) {
  return {
    outline: d.outline, nodes: d.nodes, tri: d.tri, nRim: d.nRim,
    lam: d.lam, shapes: d.shapes,
  };
}

const transfers = (d) => [d.nodes.buffer, d.tri.buffer, d.lam.buffer, ...d.shapes.map((s) => s.buffer)];

globalThis.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      listener = await Listener.load(msg.model);
      postMessage({ type: 'ready' });
    } else if (msg.type === 'drum') {
      const t0 = performance.now();
      const d = pack(drum(Float64Array.from(msg.outline), K_MODES));
      postMessage({ type: 'drum', id: msg.id, drum: d, ms: performance.now() - t0 }, transfers(d));
    } else if (msg.type === 'hear') {
      const out = listener.hear(Float64Array.from(msg.lam), msg.K);
      postMessage({ type: 'heard', id: msg.id, K: msg.K, ...out });
    } else if (msg.type === 'gww') {
      // the isospectral pair on a mesh symmetric under the grid's reflections,
      // so the two spectra agree to rounding error, as the theory says
      const out = msg.outlines.map((p) => {
        const m = crisscrossMesh(Float64Array.from(p), msg.n);
        const r = modes(m, K_MODES);
        // report on the same unit-area scale as every other drum (area 3.5 here)
        const area = 3.5;
        return pack({ outline: Float64Array.from(p), ...m, lam: r.lam.map((v) => v * area), shapes: r.shapes });
      });
      postMessage({ type: 'gww', drums: out }, out.flatMap(transfers));
    }
  } catch (err) {
    postMessage({ type: 'error', id: msg.id, message: String((err && err.message) || err) });
  }
};

