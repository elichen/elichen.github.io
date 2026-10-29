// The brain half of BAAIWorm, started by worker.js. With the recorded sensory
// cycle as input the brain never needs the body, so it runs up to AHEAD control
// steps in front of it; each result is the same as in a strictly serial loop.
// It starts 3 s in: from v = -65 mV the first seconds swing the head round by 130°.

const AHEAD = 8;
const SETTLE = 30; // control steps of brain before the body moves
let w = null;
let credit = AHEAD;
let epoch = 0;
const mem = () => w.memory.buffer;
// views take the buffer after the export has returned its pointer (an export may grow memory)
const f64 = (p, n) => new Float64Array(mem(), p, n);

async function bytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Float64Array(await r.arrayBuffer());
}

function put(a) {
  const p = w.alloc(a.length);
  new Float64Array(mem(), p, a.length).set(a);
  return p;
}

async function load() {
  const [wasm, brain, body, wout, meta] = await Promise.all([
    WebAssembly.instantiateStreaming(fetch('engine.wasm')),
    bytes('model/brain.bin'),
    bytes('model/body.bin'),
    bytes('model/wout_offline.bin'),
    fetch('model/brain.json').then((r) => r.json()),
  ]);
  w = wasm.instance.exports;
  w.init(put(brain), brain.length, put(body), body.length, put(wout), put(new Float64Array([...meta.input, ...meta.output])));
  w.warm_up(SETTLE);
  postMessage({ type: 'ready' });
  loop();
}

function loop() {
  const t0 = performance.now();
  while (credit > 0 && performance.now() - t0 < 30) {
    w.think();
    credit--;
    const muscle = new Float64Array(f64(w.muscles(), 96));
    const motor = new Float32Array(f64(w.motor_v(), 80));
    const soma = new Float32Array(f64(w.soma_v(), 136));
    const p = w.node_v();
    const nodes = new Float32Array(mem(), p, 24496).slice();
    postMessage({ type: 'frame', epoch, muscle, motor, soma, nodes, stimulus: w.stimulus() }, [muscle.buffer, motor.buffer, soma.buffer, nodes.buffer]);
  }
  setTimeout(loop, credit > 0 ? 0 : 4);
}

onmessage = (e) => {
  const m = e.data;
  if (!w) return;
  if (m.type === 'credit') credit += m.n;
  else if (m.type === 'gain') {
    w.set_gain(1, m.syn);
    w.set_gain(2, m.syn);
    w.set_gain(0, m.gap);
  } else if (m.type === 'reset') {
    epoch = m.epoch;
    credit = AHEAD;
    w.reset();
    w.warm_up(SETTLE);
  }
};

load().catch((err) => postMessage({ type: 'error', text: String(err) }));
