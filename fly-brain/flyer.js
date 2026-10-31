// flybody's trained flight policy (Vaxenburg et al. 2025) in the browser. This is not the connectome: it moves the wings.
// The wings follow a recorded wingbeat (218 Hz) at a frequency the policy adjusts, plus the policy's corrections; head and
// abdomen follow the policy. The command is a speed and a turn rate, which make the path it tracks.
// The observation matches flybody's flight_imitation (tools/pack_flyer.py records it): sensors are the mean over the 4
// physics steps of a control step; path preview = 6 poses, in the fly's frame.
export const CONTROL_STEPS = 4;    // physics steps per control step (50 us, 200 us)
export const CONTROL_DT = 2e-4;
const PREVIEW = 6, HEIGHT = 1;
const PITCH = -47.5 * Math.PI / 180;   // thorax axis, as flybody trained it; photos of free flight look this upright
const COM = [-0.03697732, 0.00029205, -0.0142447];   // flybody: centre of mass from the root, in the thorax frame
// real saccades in flybody's flight recordings: 27 deg of roll at a peak of 43 rad/s
export const ROLL_PER_YAW = (27 * Math.PI / 180) / 43;

const mulq = (a, b) => [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
];
const axis = (th, k) => { const q = [Math.cos(th / 2), 0, 0, 0]; q[k + 1] = Math.sin(th / 2); return q; };
const rotate = (q, v) => {
    const u = [q[1], q[2], q[3]], s = q[0];
    const c = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const d = [u[1] * c[2] - u[2] * c[1], u[2] * c[0] - u[0] * c[2], u[0] * c[1] - u[1] * c[0]];
    return [0, 1, 2].map(i => v[i] + 2 * (s * c[i] + d[i]));
};

export class Flyer {
    constructor(mujoco, model, data, meta, policy) {
        Object.assign(this, { mj: mujoco, model, data, meta, policy });
        const id = (t, n) => {
            const i = mujoco.mj_name2id(model, mujoco.mjtObj[t].value, n);
            if (i < 0) throw new Error(`no ${t} ${n}`);
            return i;
        };
        this.jq = meta.joints.map(n => model.jnt_qposadr[id('mjOBJ_JOINT', n)]);
        this.jv = meta.joints.map(n => model.jnt_dofadr[id('mjOBJ_JOINT', n)]);
        this.wq = meta.wing_joints.map(n => model.jnt_qposadr[id('mjOBJ_JOINT', n)]);
        this.wv = meta.wing_joints.map(n => model.jnt_dofadr[id('mjOBJ_JOINT', n)]);
        this.root = id('mjOBJ_BODY', meta.joints.length ? 'walker/thorax' : '');
        this.freeq = model.jnt_qposadr[id('mjOBJ_JOINT', 'walker/')];
        this.freev = model.jnt_dofadr[id('mjOBJ_JOINT', 'walker/')];
        this.sens = ['accelerometer', 'gyro', 'velocimeter'].map(k => {
            const s = id('mjOBJ_SENSOR', meta[k]);
            return model.sensor_adr[s];
        });
        this.ctrl = meta.action.filter(n => !n.startsWith('user')).map(n => id('mjOBJ_ACTUATOR', 'walker/' + n));
        this.buf = new Float64Array(9);
        this.nbuf = 0;
        this.obs = new Float32Array(104);
        this.real = new Float64Array(12);
        this.pattern = meta.pattern;
        this.phase = 0;
        this.freq = meta.base_freq;
        this.speed = 30; this.yaw = 0;
        this.pitch = PITCH;     // thorax axis asked of the policy, rad
        this.height = HEIGHT;   // centre of mass height asked of the policy, cm
        this.ref = { x: 0, y: 0, heading: 0 };
        // Python's state -> this model's, by name (this model has no ghost)
        this.mapQ = []; this.mapV = [];
        let pq = 0, pv = 0;
        meta.names_q.forEach(n => {
            const free = n.endsWith('/');
            if (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n) >= 0) {
                const j = id('mjOBJ_JOINT', n);
                for (let i = 0; i < (free ? 7 : 1); i++) this.mapQ.push([pq + i, model.jnt_qposadr[j] + i]);
                for (let i = 0; i < (free ? 6 : 1); i++) this.mapV.push([pv + i, model.jnt_dofadr[j] + i]);
            }
            pq += free ? 7 : 1; pv += free ? 6 : 1;
        });
    }

    setState(q, v) {
        const d = this.data;
        for (const [p, o] of this.mapQ) d.qpos[o] = q[p];
        if (v) for (const [p, o] of this.mapV) d.qvel[o] = v[p];
    }

    pose(x, y, th, yawRate) {
        const roll = -ROLL_PER_YAW * yawRate;                  // a left turn banks left
        const q = mulq(axis(th, 2), mulq(axis(roll, 0), axis(this.pitch, 1)));
        const o = rotate(q, COM);
        return { p: [x - o[0], y - o[1], this.height - o[2]], q };
    }

    // start in the air on the path, moving along it, wings at a phase
    start(x = 0, y = 0, heading = 0, phase = 0) {
        const { mj, model, data: d, meta } = this;
        mj.mj_resetData(model, d);
        this.setState(meta.qpos0);
        for (let i = 0; i < d.qvel.length; i++) d.qvel[i] = 0;
        this.ref = { x, y, heading };
        const { p, q } = this.pose(x, y, heading, this.yaw);
        for (let i = 0; i < 3; i++) d.qpos[this.freeq + i] = p[i];
        for (let i = 0; i < 4; i++) d.qpos[this.freeq + 3 + i] = q[i];
        d.qvel[this.freev] = this.speed * Math.cos(heading);
        d.qvel[this.freev + 1] = this.speed * Math.sin(heading);
        this.phase = phase;
        this.freq = meta.base_freq;
        const a = this.wing(phase), b = this.wing(phase + meta.base_freq * CONTROL_DT);
        for (let i = 0; i < 6; i++) {
            d.qpos[this.wq[i]] = a[i];
            d.qvel[this.wv[i]] = (b[i] - a[i]) / CONTROL_DT;
        }
        mj.mj_forward(model, d);
        this.buf.fill(0); this.nbuf = 0;
    }

    // target wing angles at a phase in beats: yaw, roll, pitch of the pattern, the same for both wings
    wing(phase) {
        const P = this.pattern, n = P.length;
        const f = (phase - Math.floor(phase)) * n, i = Math.floor(f) % n, j = (i + 1) % n, w = f - Math.floor(f);
        const v = [0, 1, 2].map(k => P[i][k] * (1 - w) + P[j][k] * w);
        return [...v, ...v];
    }

    readSensors() {
        const s = this.data.sensordata;
        this.sens.forEach((a, k) => { for (let i = 0; i < 3; i++) this.buf[3 * k + i] += s[a + i]; });
        this.nbuf++;
    }

    writeObs() {
        const { data: d, obs, root } = this, xm = d.xmat;
        const R = [...xm.subarray(9 * root, 9 * root + 9)];
        const f = this.freeq, P = [d.qpos[f], d.qpos[f + 1], d.qpos[f + 2]];
        const q0 = [d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
        const conj = [q0[0], -q0[1], -q0[2], -q0[3]];
        let n = 0;
        const sensor = k => { for (let i = 0; i < 3; i++) obs[n++] = this.nbuf ? this.buf[3 * k + i] / this.nbuf : 0; };
        sensor(0); sensor(1);          // accelerometer, gyro
        for (const q of this.jq) obs[n++] = d.qpos[q];
        for (const v of this.jv) obs[n++] = d.qvel[v];
        let { x, y, heading: th } = this.ref;
        const quats = [];
        for (let k = 0; k < PREVIEW; k++) {
            const { p, q } = this.pose(x, y, th, this.yaw);
            const dx = p[0] - P[0], dy = p[1] - P[1], dz = p[2] - P[2];
            obs[n++] = dx * R[0] + dy * R[3] + dz * R[6];
            obs[n++] = dx * R[1] + dy * R[4] + dz * R[7];
            obs[n++] = dx * R[2] + dy * R[5] + dz * R[8];
            quats.push(mulq(conj, q));
            x += this.speed * CONTROL_DT * Math.cos(th); y += this.speed * CONTROL_DT * Math.sin(th); th += this.yaw * CONTROL_DT;
        }
        for (const q of quats) for (const v of q) obs[n++] = v;
        sensor(2);                     // velocimeter
        obs[n++] = R[6]; obs[n++] = R[7]; obs[n++] = R[8];
        return n;
    }

    step() {
        const { data: d, meta } = this;
        this.writeObs();
        const a = this.policy.run(this.obs), real = this.real;
        for (let i = 0; i < 12; i++) real[i] = meta.lo[i] + (a[i] + 1) / 2 * (meta.hi[i] - meta.lo[i]);
        const want = meta.base_freq * (1 + meta.rel_range * real[meta.user_action]);
        const rate = Math.exp(-CONTROL_DT / meta.ctrl_filter);
        this.freq = this.freq * rate + want * (1 - rate);
        this.phase += this.freq * CONTROL_DT;
        const target = this.wing(this.phase);
        meta.wing_action.forEach((i, k) => { real[i] += target[k] - d.qpos[this.wq[k]]; });
        for (let i = 0; i < this.ctrl.length; i++) d.ctrl[this.ctrl[i]] = real[i];
        this.buf.fill(0); this.nbuf = 0;
        for (let i = 0; i < CONTROL_STEPS; i++) {
            this.mj.mj_step(this.model, d);
            this.readSensors();
        }
        const r = this.ref;
        r.x += this.speed * CONTROL_DT * Math.cos(r.heading);
        r.y += this.speed * CONTROL_DT * Math.sin(r.heading);
        r.heading += this.yaw * CONTROL_DT;
    }

    position() { const f = this.freeq, q = this.data.qpos; return [q[f], q[f + 1], q[f + 2]]; }
}
