// Runs BAAIWorm's body (engine.wasm) off the main thread, paced to real time, and
// starts brain-worker.js for the brain. One control step = 100 ms of brain (done
// ahead in the brain worker), then 100 ms of body crawling on agar here, with the
// food-seeking layer (behavior.rs) between the two. After each batch of steps it
// posts the mesh, muscles, voltages and what the worm smells to the page.

const CONTROL_DT = 0.1; // body seconds per control step
const FADE = 10; // control steps over which the muscles fade in after a start
const FOOD = [3.0, -4.0]; // where the food starts: 8 mm from the worm, ahead and to the dorsal side

let w = null;
const mem = () => w.memory.buffer;
const brain = new Worker('brain-worker.js');
const frames = [];
let epoch = 0;
let brainReady = false;
let running = true;
let speed = 1;
let simTime = 0;
let wallStart = 0;
let simStart = 0;
let lastPost = 0;
let stepsSincePost = 0;
let busyMs = 0;
let iterSum = 0;
let shown = null; // brain data of the last step taken
let muscleBuf = 0;
let stepsTaken = 0;

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

brain.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'frame') {
    if (m.epoch === epoch) frames.push(m);
  } else if (m.type === 'ready') {
    brainReady = true;
    maybeStart();
  } else postMessage(m);
};

async function load() {
  postMessage({ type: 'progress', text: 'Loading the circuit and body' });
  const [wasm, brainData, body, wout, meta] = await Promise.all([
    WebAssembly.instantiateStreaming(fetch('engine.wasm')),
    bytes('model/brain.bin'),
    bytes('model/body_sym.bin'),
    bytes('model/wout_offline.bin'),
    fetch('model/brain.json').then((r) => r.json()),
  ]);
  w = wasm.instance.exports;
  w.init(put(brainData), brainData.length, put(body), body.length, put(wout), put(new Float64Array([...meta.input, ...meta.output])));
  muscleBuf = w.alloc(96);
  w.agar_on(FOOD[0], FOOD[1], (Math.random() * 1e9) | 0);
  postMessage({ type: 'progress', text: 'Warming up the brain' });
  maybeStart();
}

let started = false;
function maybeStart() {
  if (started || !w || !brainReady) return;
  started = true;
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
  const p = w.positions();
  const x = new Float32Array(new Float64Array(mem(), p, nv * 3));
  const muscle = new Float32Array(shown.muscle);
  const { motor, soma, nodes } = shown;
  const seek = new Float64Array(mem(), w.seek_state(), 8).slice();
  const now = performance.now();
  const wall = (now - lastPost) / 1000;
  postMessage(
    {
      type: 'state', epoch, x, muscle, motor, soma, nodes, seek,
      stimulus: shown.stimulus, time: simTime, steps: stepsSincePost,
      // how much body time one wall second buys at full speed (body side)
      capacity: stepsSincePost ? (stepsSincePost * CONTROL_DT) / (busyMs / 1000) : 0,
      rate: wall > 0 ? (stepsSincePost * CONTROL_DT) / wall : 0,
      iterations: stepsSincePost ? iterSum / stepsSincePost : 0,
      queue: frames.length, // brain steps ready ahead of the body
    },
    [x.buffer, muscle.buffer],
  );
  shown = null;
  lastPost = now;
  stepsSincePost = 0;
  busyMs = 0;
  iterSum = 0;
}

function loop() {
  if (running) {
    const t0 = performance.now();
    // while the brain has nothing ready (starting up after a restart) the clock waits
    if (!frames.length) (wallStart = t0), (simStart = simTime);
    const due = simStart + ((t0 - wallStart) / 1000) * speed;
    let used = 0;
    while (frames.length && simTime + CONTROL_DT <= due + 1e-9 && performance.now() - t0 < 40) {
      const f = frames.shift();
      const fade = Math.min(1, stepsTaken / FADE);
      new Float64Array(mem(), muscleBuf, 96).set(fade < 1 ? f.muscle.map((a) => a * fade) : f.muscle);
      stepsTaken++;
      w.set_muscles(muscleBuf);
      w.moving();
      iterSum += w.iterations();
      simTime += CONTROL_DT;
      stepsSincePost++;
      shown = f;
      used++;
    }
    // never fall further behind than a few steps: a slow machine runs slower than real time
    // (the clock starts again from where the body is; it never skips body time)
    if (due - simTime > 4 * CONTROL_DT) (simStart = simTime), (wallStart = performance.now() - (4 * CONTROL_DT * 1000) / speed);
    if (used) brain.postMessage({ type: 'credit', n: used });
    busyMs += performance.now() - t0;
    if (stepsSincePost && shown) post();
  }
  setTimeout(loop, 4);
}

onmessage = (e) => {
  const m = e.data;
  if (m.type === 'gain') brain.postMessage(m);
  if (!w) return;
  if (m.type === 'food') {
    w.seek_food(m.x, m.z);
  } else if (m.type === 'reset') {
    epoch = m.epoch;
    frames.length = 0;
    stepsTaken = 0;
    w.reset();
    brain.postMessage({ type: 'reset', epoch });
    simTime = 0;
    restartClock();
  } else if (m.type === 'run') {
    running = m.on;
    restartClock();
  } else if (m.type === 'speed') {
    speed = m.speed;
    restartClock();
  }
};

load().catch((err) => postMessage({ type: 'error', text: String(err) }));
