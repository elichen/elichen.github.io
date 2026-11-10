// Getting up: one open-loop move of the legs and wings that turns a fallen fly over, found by random search (tools/right.mjs). Not a policy, and not from recordings.
// `near` is the side of the body nearer the ground. Each phase sets the leg joints of each side to a share of their range, pushes the wings, and may hold the feet down.
export const PHASES = 2;
// The move the page uses. Leg joints in the model's order: coxa abduct, coxa twist, coxa, femur twist, femur, tibia, tarsus, tarsus 2; wings: yaw, roll, pitch (torque, -1 to 1).
export const GET_UP = { phases: [
    { T: 0.0900, near: [0.346, 0.156, 0.214, 0.446, 0.808, 0.75, 0.826, 0.325], far: [0.108, 0.924, 0.194, 0.282, 0.152, 0.597, 0.883, 0.189], wingNear: [0.265, -0.712, -0.232], wingFar: [-0.459, -0.184, -0.418], stickNear: true, stickFar: true },
    { T: 0.0894, near: [0.009, 0.495, 0.136, 0.779, 0.163, 0.209, 0.277, 0.696], far: [0.941, 0.611, 0.912, 0.586, 0.312, 0.968, 0.516, 0.924], wingNear: [-0.894, -0.796, 0.703], wingFar: [-0.844, -0.827, -0.151], stickNear: true, stickFar: true },
] };
const smooth = x => { const u = Math.min(1, Math.max(0, x)); return u * u * (3 - 2 * u); };

// a fallen fly, for the search: turned by roll and pitch, dropped from 3 mm, legs slack at the angles they had, settled for 0.25 s
export function fallen(mujoco, m, d, W, F, L, { roll, pitch, yaw }) {
    const f = F.freeq, q = (a, k) => { const r = [Math.cos(a / 2), 0, 0, 0]; r[k] = Math.sin(a / 2); return r; };
    const mul = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
    mujoco.mj_resetData(m, d); W.setState(W.meta.qpos0);
    for (const w of F.wq) d.qpos[w] = m.qpos_spring[w];
    const o = mul(q(yaw, 3), mul(q(roll, 1), q(pitch, 2)));
    for (let i = 0; i < 4; i++) d.qpos[f + 3 + i] = o[i];
    d.qpos[f + 2] = 0.3;
    for (const l of L.legs) d.ctrl[l.a] = d.qpos[l.q];
    m.opt.timestep = 5e-5; mujoco.mj_forward(m, d);
    for (let i = 0; i < 5000; i++) mujoco.mj_step(m, d);
    L.mode = 'down';
}

// starts the move from the pose the fly is in; returns a function that sets the commands of the next control step (0.2 ms) and returns false after the last
export function rightingMove(L, M) {
    const { model: m, data: d, F } = L, f = F.freeq, q = d.qpos;
    // the body's left axis in the world: if it points down, the left side is the near side
    const nearLeft = 2 * (q[f + 5] * q[f + 6] + q[f + 3] * q[f + 4]) < 0;
    const stick = L.adhere.map(a => /left/.test(L.mj.mj_id2name(m, L.mj.mjtObj.mjOBJ_ACTUATOR.value, a)) === nearLeft);
    const wing = i => F.ctrl[F.meta.wing_action[i]];
    let from = L.legs.map(l => q[l.q]), at = 0, k = 0;
    return () => {
        const p = M.phases[at];
        if (!p) { for (let i = 0; i < 6; i++) d.ctrl[wing(i)] = 0; return false; }
        const n = Math.round(p.T / 2e-4), u = smooth((k + 1) / n / 0.6);
        L.legs.forEach((l, i) => { const share = ((l.leg % 2 === 0) === nearLeft ? p.near : p.far)[L.byLeg[l.leg].indexOf(l)]; d.ctrl[l.a] = from[i] * (1 - u) + (l.lo + share * (l.hi - l.lo)) * u; });
        for (let i = 0; i < 6; i++) d.ctrl[wing(i)] = ((i < 3) === nearLeft ? p.wingNear : p.wingFar)[i % 3];
        L.adhere.forEach((a, i) => { d.ctrl[a] = (stick[i] ? p.stickNear : p.stickFar) ? 1 : 0; });
        if (++k >= n) { k = 0; at++; from = L.legs.map(l => d.ctrl[l.a]); }
        return true;
    };
}
