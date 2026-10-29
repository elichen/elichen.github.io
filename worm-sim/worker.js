// Runs BAAIWorm (engine.wasm: brain + body) off the main thread, paced to real
// time: one control step = 100 ms of brain, then 1/30 s of body. After each batch
// of steps it posts the mesh, muscles and voltages to the page.

const CONTROL_DT = 1 / 30; // body seconds per control step
const WARM_UP = 20; // control steps of brain with the body held straight

let w = null;
let mem = () => w.memory.buffer;
let running = true;
let speed = 1;
let simTime = 0;
let wallStart = 0;
let simStart = 0;
let lastPost = 0;
let stepsSincePost = 0;
let busyMs = 0;
let iterSum = 0;

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
  const progress = (text) => postMessage({ type: 'progress', text });
  progress('Loading the circuit and body');
  const [wasm, brain, body, wout, meta] = await Promise.all([
    WebAssembly.instantiateStreaming(fetch('engine.wasm')),
    bytes('model/brain.bin'),
    bytes('model/body.bin'),
    bytes('model/wout_offline.bin'),
    fetch('model/brain.json').then((r) => r.json()),
  ]);
  w = wasm.instance.exports;
  w.init(put(brain), brain.length, put(body), body.length, put(wout), put(new Float64Array([...meta.input, ...meta.output])));
  progress('Warming up the brain');
  await new Promise((r) => setTimeout(r, 0));
  w.warm_up(WARM_UP);
  postMessage({ type: 'ready', vertices: w.n_vertices() });
  restartClock();
  loop();
}

function restartClock() {
  wallStart = performance.now();
  simStart = simTime;
}

function post() {
  const nv = w.n_vertices();
  const x = new Float32Array(new Float64Array(mem(), w.positions(), nv * 3));
  const muscle = new Float32Array(new Float64Array(mem(), w.muscles(), 96));
  const motor = new Float32Array(new Float64Array(mem(), w.motor_v(), 80));
  const soma = new Float32Array(new Float64Array(mem(), w.soma_v(), 136));
  const nodes = new Float32Array(mem(), w.node_v(), 24496).slice();
  const now = performance.now();
  const wall = (now - lastPost) / 1000;
  postMessage(
    {
      type: 'state', x, muscle, motor, soma, nodes,
      stimulus: w.stimulus(), time: simTime, steps: stepsSincePost,
      // how much body time one wall second buys at full speed
      capacity: stepsSincePost ? (stepsSincePost * CONTROL_DT) / (busyMs / 1000) : 0,
      rate: wall > 0 ? (stepsSincePost * CONTROL_DT) / wall : 0,
      iterations: stepsSincePost ? iterSum / stepsSincePost : 0,
    },
    [x.buffer, muscle.buffer, motor.buffer, soma.buffer, nodes.buffer],
  );
  lastPost = now;
  stepsSincePost = 0;
  busyMs = 0;
  iterSum = 0;
}

function loop() {
  if (running) {
    const t0 = performance.now();
    const due = simStart + ((t0 - wallStart) / 1000) * speed;
    // never fall further behind than a few steps: a slow machine runs slower than real time
    if (due - simTime > 4 * CONTROL_DT) simTime = due - 4 * CONTROL_DT, simStart = simTime, wallStart = t0;
    while (simTime + CONTROL_DT <= due + 1e-9 && performance.now() - t0 < 40) {
      w.think();
      w.moving();
      iterSum += w.iterations();
      simTime += CONTROL_DT;
      stepsSincePost++;
    }
    busyMs += performance.now() - t0;
    if (stepsSincePost) post();
  }
  setTimeout(loop, 4);
}

onmessage = (e) => {
  const m = e.data;
  if (!w) return;
  if (m.type === 'gain') {
    w.set_gain(1, m.syn); // excitatory
    w.set_gain(2, m.syn); // inhibitory
    w.set_gain(0, m.gap);
  } else if (m.type === 'reset') {
    w.reset();
    w.warm_up(WARM_UP);
    simTime = 0;
    restartClock();
    post();
  } else if (m.type === 'run') {
    running = m.on;
    restartClock();
  } else if (m.type === 'speed') {
    speed = m.speed;
    restartClock();
  }
};

load().catch((err) => postMessage({ type: 'error', text: String(err) }));
