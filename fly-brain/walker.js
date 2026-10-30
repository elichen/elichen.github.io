// flybody's trained walking policy (Vaxenburg et al. 2025) in the browser: a 741-input network that
// reads the fly's joints and sensors and a preview of a path, and sets 59 actuators. This is not the
// connectome: it moves the legs. The page's command is a speed and a turn rate, which make the path.
// Shared by the page and tools/walk_parity.mjs.
//
// The observation matches flybody's walk_imitation (tools/walk_ref.py records it): sensors are the mean over
// the 10 physics steps of a control step; path preview = 65 poses, in the fly's frame.

export const CONTROL_STEPS = 10;   // physics steps per control step (0.2 ms, 2 ms)
export const CONTROL_DT = 0.002;
const PREVIEW = 65, HEIGHT = 0.1278;

export class WalkPolicy {
    // weights: flat Float32Array as written by tools/walk_ref.py
    constructor(weights) {
        let at = 0;
        const take = n => { const x = weights.subarray(at, at + n); at += n; return x; };
        this.w0 = take(741 * 512); this.b0 = take(512); this.off = take(512); this.scale = take(512);
        this.layers = [0, 1, 2].map(() => ({ w: take(512 * 512), b: take(512) }));
        this.wm = take(512 * 59); this.bm = take(59);
        this.h = new Float32Array(512); this.n = new Float32Array(512); this.out = new Float32Array(59);
    }

    // out = b + x W, W rows are inputs
    static mul(x, w, b, nIn, nOut, out) {
        out.set(b);
        for (let i = 0; i < nIn; i++) {
            const xi = x[i], row = i * nOut;
            if (xi === 0) continue;
            for (let j = 0; j < nOut; j++) out[j] += xi * w[row + j];
        }
    }

    run(obs) {
        let h = this.h, n = this.n;
        WalkPolicy.mul(obs, this.w0, this.b0, 741, 512, h);
        let mean = 0, v = 0;
        for (let j = 0; j < 512; j++) mean += h[j];
        mean /= 512;
        for (let j = 0; j < 512; j++) v += (h[j] - mean) ** 2;
        const s = Math.sqrt(v / 512 + 1e-5);
        for (let j = 0; j < 512; j++) h[j] = Math.tanh((h[j] - mean) / s * this.scale[j] + this.off[j]);
        for (const { w, b } of this.layers) {
            WalkPolicy.mul(h, w, b, 512, 512, n);
            for (let j = 0; j < 512; j++) n[j] = n[j] > 0 ? n[j] : Math.expm1(n[j]);
            [h, n] = [n, h];
        }
        WalkPolicy.mul(h, this.wm, this.bm, 512, 59, this.out);
        for (let j = 0; j < 59; j++) this.out[j] = Math.max(-1, Math.min(1, this.out[j]));
        return this.out;
    }
}

const quatMul = (a, b) => [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
];

// the same network from engine.wasm (SIMD), ~4x faster; engine: an instance's exports
export class WasmPolicy {
    constructor(engine, weights, [inputs, hidden, layers, outputs] = [741, 512, 3, 59]) {
        this.e = engine;
        this.inputs = inputs; this.outputs = outputs;
        const p = engine.policy_alloc(weights.length * 4);
        new Float32Array(engine.memory.buffer, p, weights.length).set(weights);
        engine.policy_init(inputs, hidden, layers, outputs);
        this.obsP = engine.policy_obs();
        this.outP = engine.policy_out();
    }

    run(obs) {
        new Float32Array(this.e.memory.buffer, this.obsP, this.inputs).set(obs);
        this.e.policy_run();
        return new Float32Array(this.e.memory.buffer, this.outP, this.outputs);
    }
}

export class Walker {
    // model, data: MuJoCo; meta: tools/walk_ref.py's names; policy: WalkPolicy or WasmPolicy
    constructor(mujoco, model, data, meta, policy) {
        Object.assign(this, { mj: mujoco, model, data, meta, policy });
        const id = (type, n) => {
            const i = mujoco.mj_name2id(model, mujoco.mjtObj[type].value, n);
            if (i < 0) throw new Error(`no ${type} ${n}`);
            return i;
        };
        this.jq = meta.joints.map(n => model.jnt_qposadr[id('mjOBJ_JOINT', n)]);
        this.jv = meta.joints.map(n => model.jnt_dofadr[id('mjOBJ_JOINT', n)]);
        this.sites = meta.sites.map(n => id('mjOBJ_SITE', n));
        this.root = id('mjOBJ_BODY', meta.root);
        const adr = names => names.map(n => {
            const s = id('mjOBJ_SENSOR', n);
            return [model.sensor_adr[s], model.sensor_dim[s]];
        });
        this.sens = {};
        for (const k of ['accelerometer', 'force', 'gyro', 'touch', 'velocimeter']) this.sens[k] = adr(meta[k]);
        this.ctrl = meta.action.map(n => id('mjOBJ_ACTUATOR', n.startsWith('walker/') ? n : 'walker/' + n));
        this.freeq = model.jnt_qposadr[id('mjOBJ_JOINT', 'walker/')];
        // the policy sees the activations of its own 59 actuators, in the model's order
        this.actObs = this.ctrl.map(i => model.actuator_actadr[i]).sort((x, y) => x - y);
        // Python's state -> this model's, by name (this model may have mouth joints and no ghost)
        const has = (t, n) => mujoco.mj_name2id(model, mujoco.mjtObj[t].value, n) >= 0;
        this.mapQ = []; this.mapV = []; this.mapA = [];
        let pq = 0, pv = 0;
        meta.names_q.forEach(n => {
            const free = n.endsWith('/');
            if (has('mjOBJ_JOINT', n)) {
                const j = id('mjOBJ_JOINT', n);
                for (let i = 0; i < (free ? 7 : 1); i++) this.mapQ.push([pq + i, model.jnt_qposadr[j] + i]);
                for (let i = 0; i < (free ? 6 : 1); i++) this.mapV.push([pv + i, model.jnt_dofadr[j] + i]);
            }
            pq += free ? 7 : 1; pv += free ? 6 : 1;
        });
        meta.names_a.forEach((n, i) => { if (has('mjOBJ_ACTUATOR', n)) this.mapA.push([i, model.actuator_actadr[id('mjOBJ_ACTUATOR', n)]]); });
        // sensor running mean over the control step
        this.buf = {};
        for (const k in this.sens) this.buf[k] = new Float64Array(this.sens[k].reduce((n, [, d]) => n + d, 0));
        this.nbuf = 0;
        this.obs = new Float32Array(741);
        this.speed = 2; this.yaw = 0;
        this.ref = { x: 0, y: 0, heading: 0 };
    }

    // state as recorded in Python (or meta.qpos0)
    setState(q, v, a) {
        const d = this.data;
        for (const [p, o] of this.mapQ) d.qpos[o] = q[p];
        if (v) for (const [p, o] of this.mapV) d.qvel[o] = v[p];
        if (a) for (const [p, o] of this.mapA) d.act[o] = a[p];
    }

    // the path starts where the fly stands
    setRef(x, y, heading) { this.ref = { x, y, heading }; }

    readSensors() {
        const s = this.data.sensordata;
        for (const k in this.sens) {
            let n = 0;
            const b = this.buf[k];
            for (const [a, d] of this.sens[k]) for (let i = 0; i < d; i++) b[n++] += s[a + i];
        }
        this.nbuf++;
    }

    writeObs() {
        const { data: d, obs, root } = this, xm = d.xmat, xp = d.xpos;
        const R = [...xm.subarray(9 * root, 9 * root + 9)], P = [...xp.subarray(3 * root, 3 * root + 3)];
        let n = 0;
        const sensor = k => { for (const v of this.buf[k]) obs[n++] = this.nbuf ? v / this.nbuf : 0; };
        const local = (x, y, z) => {   // world vector -> thorax frame
            obs[n++] = x * R[0] + y * R[3] + z * R[6];
            obs[n++] = x * R[1] + y * R[4] + z * R[7];
            obs[n++] = x * R[2] + y * R[5] + z * R[8];
        };
        sensor('accelerometer');
        for (const a of this.actObs) obs[n++] = d.act[a];
        for (const s of this.sites) local(d.site_xpos[3 * s] - P[0], d.site_xpos[3 * s + 1] - P[1], d.site_xpos[3 * s + 2] - P[2]);
        sensor('force'); sensor('gyro');
        for (const q of this.jq) obs[n++] = d.qpos[q];
        for (const v of this.jv) obs[n++] = d.qvel[v];
        // the path: 65 poses from the reference, integrated with the current command
        const q0 = [...d.qpos.subarray(this.freeq + 3, this.freeq + 7)];
        const conj = [q0[0], -q0[1], -q0[2], -q0[3]];
        const dt = CONTROL_DT;
        let { x, y, heading: th } = this.ref;
        const quats = [];
        for (let k = 0; k < PREVIEW; k++) {
            local(x - d.qpos[this.freeq], y - d.qpos[this.freeq + 1], HEIGHT - d.qpos[this.freeq + 2]);
            quats.push(quatMul(conj, [Math.cos(th / 2), 0, 0, Math.sin(th / 2)]));
            x += this.speed * dt * Math.cos(th); y += this.speed * dt * Math.sin(th); th += this.yaw * dt;
        }
        for (const q of quats) for (const v of q) obs[n++] = v;
        sensor('touch'); sensor('velocimeter');
        obs[n++] = R[6]; obs[n++] = R[7]; obs[n++] = R[8];
        return n;
    }

    setAction(a) {
        const { lo, hi } = this.meta;
        for (let i = 0; i < 59; i++) this.data.ctrl[this.ctrl[i]] = lo[i] + (a[i] + 1) / 2 * (hi[i] - lo[i]);
    }

    // one control step: read, act, 10 physics steps, advance the path
    step() {
        this.writeObs();
        this.setAction(this.policy.run(this.obs));
        for (const k in this.buf) this.buf[k].fill(0);
        this.nbuf = 0;
        for (let i = 0; i < CONTROL_STEPS; i++) {
            this.mj.mj_step(this.model, this.data);
            this.readSensors();
        }
        const r = this.ref;
        r.x += this.speed * CONTROL_DT * Math.cos(r.heading);
        r.y += this.speed * CONTROL_DT * Math.sin(r.heading);
        r.heading += this.yaw * CONTROL_DT;
    }
}
