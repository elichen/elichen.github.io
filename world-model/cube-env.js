// OGBench single-cube manipulation (UR5e + Robotiq 2F-85) on MuJoCo's WebAssembly build.
// Shared by the page and tools/parity_env.mjs, so both run identical physics & DiffIK code.
//
// Action space (5D in [-1, 1]): [dx, dy, dz, dyaw, dgripper]
//   unnormalized to [-0.05..0.05] m, [-0.3..0.3] rad, [-1..1] gripper opening.
// Control rate: 20 Hz (25 physics substeps of 2 ms = 50 ms per control step).

export const SUBSTEPS = 25;
export const DQ_MAX = 0.35;
export const JNT_LO = [-6.2, -3.1, -3.0, -6.2, -6.2, -6.2];
export const JNT_HI = [ 6.2,  0.1,  3.0,  6.2,  6.2,  6.2];
export const ACTION_HIGH = [0.05, 0.05, 0.05, 0.3, 1.0];
export const WORKSPACE_LO = [0.25, -0.35, 0.02];
export const WORKSPACE_HI = [0.60, 0.35, 0.35];
export const HOME_QPOS = [-Math.PI / 2, -Math.PI / 2, Math.PI / 2, -Math.PI / 2, -Math.PI / 2, 0];

const DOWN_QUAT = [0, 1, 0, 0];
const T_PA_POS = [0, 0, -0.1558];
const T_PA_QUAT = [Math.SQRT1_2, 0, 0, Math.SQRT1_2];
const MAX_ANGLE_CHANGE = Math.PI / 4;

export function makeRandom(seed = (Math.random() * 2 ** 32) >>> 0) {
  let s = seed >>> 0;
  const uniform = () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(Math.max(1e-7, 1 - uniform()))) * Math.cos(2 * Math.PI * uniform());
  return { uniform, normal, range: (lo, hi) => lo + (hi - lo) * uniform() };
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function quatMul(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

function quatConj(q) {
  return [q[0], -q[1], -q[2], -q[3]];
}

function quatFromZ(theta) {
  return [Math.cos(0.5 * theta), 0, 0, Math.sin(0.5 * theta)];
}

function quatRotateVec(q, v) {
  const [w, x, y, z] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}

// Matches MuJoCo's mju_mat2Quat on a 3x3 row-major rotation matrix
function mat2Quat(m, off = 0) {
  const m00 = m[off], m01 = m[off + 1], m02 = m[off + 2];
  const m10 = m[off + 3], m11 = m[off + 4], m12 = m[off + 5];
  const m20 = m[off + 6], m21 = m[off + 7], m22 = m[off + 8];
  const tr = m00 + m11 + m22;
  let q;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s];
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    q = [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s];
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    q = [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s];
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s];
  }
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

// Matches MuJoCo's mju_quat2Vel(res, quat, dt=1.0)
function quat2Vel(q) {
  const ax = q[1], ay = q[2], az = q[3];
  const sinHalf = Math.hypot(ax, ay, az);
  if (sinHalf < 1e-14) return [0, 0, 0];
  let theta = 2 * Math.atan2(sinHalf, q[0]);
  if (theta > Math.PI) theta -= 2 * Math.PI;
  const s = theta / sinHalf;
  return [ax * s, ay * s, az * s];
}

function computeYaw(q) {
  const [q0, q1, q2, q3] = q;
  return Math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3));
}

// Solve 6x6 linear system A x = b via Gaussian elimination with partial pivoting
function solve6(A, b) {
  const M = new Float64Array(36);
  const rhs = Float64Array.from(b);
  M.set(A);
  for (let col = 0; col < 6; col++) {
    let piv = col, best = Math.abs(M[col * 6 + col]);
    for (let r = col + 1; r < 6; r++) {
      const v = Math.abs(M[r * 6 + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (piv !== col) {
      for (let c = col; c < 6; c++) {
        const tmp = M[col * 6 + c]; M[col * 6 + c] = M[piv * 6 + c]; M[piv * 6 + c] = tmp;
      }
      const tr = rhs[col]; rhs[col] = rhs[piv]; rhs[piv] = tr;
    }
    const diag = M[col * 6 + col];
    if (Math.abs(diag) < 1e-18) continue;
    for (let r = col + 1; r < 6; r++) {
      const f = M[r * 6 + col] / diag;
      for (let c = col + 1; c < 6; c++) M[r * 6 + c] -= f * M[col * 6 + c];
      rhs[r] -= f * rhs[col];
    }
  }
  const x = new Float64Array(6);
  for (let r = 5; r >= 0; r--) {
    let s = rhs[r];
    for (let c = r + 1; c < 6; c++) s -= M[r * 6 + c] * x[c];
    x[r] = Math.abs(M[r * 6 + r]) > 1e-18 ? s / M[r * 6 + r] : 0;
  }
  return x;
}

export class DiffIKController {
  constructor(mujoco, ikXml, damping = 1e-12) {
    this.mj = mujoco;
    this.model = mujoco.MjModel.from_xml_string(ikXml);
    this.data = new mujoco.MjData(this.model);
    this.damping = damping;
    this.siteId = mujoco.mj_name2id(this.model, mujoco.mjtObj.mjOBJ_SITE.value, 'attachment_site');
    this.jacpBuf = new mujoco.DoubleBuffer(3 * this.model.nv);
    this.jacrBuf = new mujoco.DoubleBuffer(3 * this.model.nv);
    this._J = new Float64Array(36);
    this._H = new Float64Array(36);
    this._err = new Float64Array(6);
  }

  solve(pos, quat, currQpos, maxIters = 20, posThresh = 1e-4, oriThresh = 1e-4) {
    const { mj, model: m, data: d, siteId, _J: J, _H: H, _err: err } = this;
    for (let i = 0; i < 6; i++) d.qpos[i] = currQpos[i];

    for (let iter = 0; iter < maxIters; iter++) {
      mj.mj_kinematics(m, d);
      mj.mj_comPos(m, d);

      const sPos = 3 * siteId;
      err[0] = pos[0] - d.site_xpos[sPos];
      err[1] = pos[1] - d.site_xpos[sPos + 1];
      err[2] = pos[2] - d.site_xpos[sPos + 2];

      const siteQuat = mat2Quat(d.site_xmat, 9 * siteId);
      const errQuat = quatMul(quat, quatConj(siteQuat));
      const vel = quat2Vel(errQuat);
      err[3] = vel[0]; err[4] = vel[1]; err[5] = vel[2];

      if (Math.hypot(err[0], err[1], err[2]) <= posThresh &&
          Math.hypot(err[3], err[4], err[5]) <= oriThresh) {
        break;
      }

      mj.mj_jacSite(m, d, this.jacpBuf, this.jacrBuf, siteId);
      const jp = this.jacpBuf.GetView();
      const jr = this.jacrBuf.GetView();
      for (let i = 0; i < 18; i++) { J[i] = jp[i]; J[18 + i] = jr[i]; }

      // H = J * J^T + damping * I
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) {
          let s = r === c ? this.damping : 0;
          for (let k = 0; k < 6; k++) s += J[r * 6 + k] * J[c * 6 + k];
          H[r * 6 + c] = s;
        }
      }
      const y = solve6(H, err);
      const upd = new Float64Array(6);
      let maxAbs = 0;
      for (let j = 0; j < 6; j++) {
        let s = 0;
        for (let r = 0; r < 6; r++) s += J[r * 6 + j] * y[r];
        upd[j] = s;
        const a = Math.abs(s);
        if (a > maxAbs) maxAbs = a;
      }
      if (maxAbs > MAX_ANGLE_CHANGE) {
        const sc = MAX_ANGLE_CHANGE / maxAbs;
        for (let j = 0; j < 6; j++) upd[j] *= sc;
      }
      for (let j = 0; j < 6; j++) d.qpos[j] += upd[j];
    }
    return Float64Array.from(d.qpos.subarray(0, 6));
  }
}

export class CubeEnv {
  // files: { 'scene.xml': text, 'ur5e_ik.xml': text, 'meshes/*.stl': Uint8Array }
  constructor(mujoco, files, random = makeRandom()) {
    this.mj = mujoco;
    this.random = random;
    this.vfs = new mujoco.MjVFS();
    const enc = new TextEncoder();
    for (const [name, data] of Object.entries(files)) {
      if (name === 'ur5e_ik.xml') continue;
      this.vfs.addBuffer(name, typeof data === 'string' ? enc.encode(data) : data);
    }
    const m = this.model = mujoco.MjModel.from_xml_string(files['scene.xml'], this.vfs);
    this.data = new mujoco.MjData(m);
    this.ik = new DiffIKController(mujoco, files['ur5e_ik.xml']);

    const id = (type, name) => mujoco.mj_name2id(m, mujoco.mjtObj[type].value, name);
    this.cubeBody = id('mjOBJ_BODY', 'object_0');
    this.targetBody = id('mjOBJ_BODY', 'object_target_0');
    this.targetMocap = m.body_mocapid[this.targetBody];
    this.pinchSite = id('mjOBJ_SITE', 'ur5e/robotiq/pinch');
    this.gripperJoint = id('mjOBJ_JOINT', 'ur5e/robotiq/right_driver_joint');
    this.gripperQposAdr = m.jnt_qposadr[this.gripperJoint];
    const cubeJoint = m.body_jntadr[this.cubeBody];
    this.cubeQposAdr = m.jnt_qposadr[cubeJoint];
    this.cubeDofAdr = m.jnt_dofadr[cubeJoint];
    this.dt = m.opt.timestep * SUBSTEPS; // 0.05 s
    this.goal = [0.45, 0, 0.02];
    this.gripCmd = 0;
    this.pull = null;
  }

  setState(qpos, qvel, goal = null) {
    const d = this.data;
    for (let i = 0; i < this.model.nq; i++) d.qpos[i] = qpos[i];
    for (let i = 0; i < this.model.nv; i++) d.qvel[i] = qvel[i];
    for (let i = 0; i < this.model.nu; i++) d.ctrl[i] = i < 6 ? qpos[i] : 0;
    if (goal) this.setGoal(goal);
    this.mj.mj_forward(this.model, d);
  }

  // Random reachable start matching OGBench Cube place distribution
  reset() {
    const { mj, model: m, data: d, random: r } = this;
    mj.mj_resetData(m, d);

    // Random initial effector above table, solved via IK from HOME_QPOS
    const effPos = [r.range(0.32, 0.52), r.range(-0.22, 0.22), r.range(0.20, 0.30)];
    const effYaw = r.range(-Math.PI, Math.PI);
    const effOri = quatMul(quatFromZ(effYaw), DOWN_QUAT);
    const posWa = effPos.map((v, i) => v + quatRotateVec(effOri, T_PA_POS)[i]);
    const quatWa = quatMul(effOri, T_PA_QUAT);
    const qInit = this.ik.solve(posWa, quatWa, HOME_QPOS);
    for (let i = 0; i < 6; i++) {
      d.qpos[i] = qInit[i];
      d.ctrl[i] = qInit[i];
    }
    d.ctrl[6] = 0;
    this.gripCmd = 0;

    // Random cube pose on table
    const cx = r.range(0.30, 0.54), cy = r.range(-0.24, 0.24);
    const cyaw = r.range(-Math.PI, Math.PI);
    this.placeCube([cx, cy, 0.02], quatFromZ(cyaw));

    // Random goal 12-25 cm away from the cube inside the workspace
    this.setGoal(this.sampleGoal([cx, cy]));
    mj.mj_forward(m, d);
  }

  sampleGoal(avoidXY = null) {
    const r = this.random;
    const ref = avoidXY ?? this.cubePos();
    for (let k = 0; k < 50; k++) {
      const gx = r.range(0.30, 0.55), gy = r.range(-0.25, 0.25);
      const dist = Math.hypot(gx - ref[0], gy - ref[1]);
      if (dist >= 0.12 && dist <= 0.25) return [gx, gy, 0.02];
    }
    return [r.range(0.30, 0.55), r.range(-0.25, 0.25), 0.02];
  }

  placeCube(pos, quat = [1, 0, 0, 0]) {
    const d = this.data, a = this.cubeQposAdr;
    d.qpos[a] = pos[0]; d.qpos[a + 1] = pos[1]; d.qpos[a + 2] = pos[2];
    d.qpos[a + 3] = quat[0]; d.qpos[a + 4] = quat[1]; d.qpos[a + 5] = quat[2]; d.qpos[a + 6] = quat[3];
    for (let i = 0; i < 6; i++) d.qvel[this.cubeDofAdr + i] = 0;
    this.mj.mj_forward(this.model, d);
  }

  setGoal(pos) {
    this.goal = [clamp(pos[0], 0.27, 0.58), clamp(pos[1], -0.32, 0.32), pos[2] ?? 0.02];
    const k = 3 * this.targetMocap;
    this.data.mocap_pos[k] = this.goal[0];
    this.data.mocap_pos[k + 1] = this.goal[1];
    this.data.mocap_pos[k + 2] = this.goal[2];
    const qk = 4 * this.targetMocap;
    this.data.mocap_quat[qk] = 1;
    this.data.mocap_quat[qk + 1] = 0;
    this.data.mocap_quat[qk + 2] = 0;
    this.data.mocap_quat[qk + 3] = 0;
    this.mj.mj_forward(this.model, this.data);
  }

  cubePos() {
    const p = this.data.xpos, b = 3 * this.cubeBody;
    return [p[b], p[b + 1], p[b + 2]];
  }

  effectorPos() {
    const p = this.data.site_xpos, s = 3 * this.pinchSite;
    return [p[s], p[s + 1], p[s + 2]];
  }

  goalError() {
    const c = this.cubePos();
    return Math.hypot(c[0] - this.goal[0], c[1] - this.goal[1], c[2] - this.goal[2]);
  }

  gripClosed() {
    return clamp(this.data.qpos[this.gripperQposAdr] / 0.8, 0, 1);
  }

  respawnCube() {
    const r = this.random;
    const cx = r.range(0.32, 0.52), cy = r.range(-0.22, 0.22);
    this.placeCube([cx, cy, 0.02], quatFromZ(r.range(-Math.PI, Math.PI)));
  }

  // Exact 33D observation matching UR5eArena.obs() + add_fovea() in train_dagger_v4.py:
  //   proprio (23D: sin q, cos q, qvel/5, eff_pos, grip_closed, grip_cmd)
  //   + visual cube readout (7D: cube_est, cube_est - eff_pos, in_hand_est) -- ZERO privileged cube state!
  //   + foveated cube zoom (3D: tanh((cube_est - eff_pos) / 0.04))
  arenaObs(cubeEst, contactEst = 0) {
    const d = this.data;
    const out = new Float32Array(33);
    for (let i = 0; i < 6; i++) {
      out[i] = Math.sin(d.qpos[i]);
      out[6 + i] = Math.cos(d.qpos[i]);
      out[12 + i] = d.qvel[i] / 5.0;
    }
    const eff = this.effectorPos();
    out[18] = eff[0]; out[19] = eff[1]; out[20] = eff[2];
    const gc = this.gripClosed();
    out[21] = gc;
    out[22] = this.gripCmd;
    out[23] = cubeEst[0]; out[24] = cubeEst[1]; out[25] = cubeEst[2];
    out[26] = cubeEst[0] - eff[0];
    out[27] = cubeEst[1] - eff[1];
    out[28] = cubeEst[2] - eff[2];
    const dist = Math.hypot(out[26], out[27], out[28]);
    out[29] = (dist < 0.065 && gc > 0.25) || (contactEst > 0.5 && dist < 0.085) ? 1.0 : 0.0;
    out[30] = Math.tanh(out[26] / 0.04);
    out[31] = Math.tanh(out[27] / 0.04);
    out[32] = Math.tanh(out[28] / 0.04);
    return out;
  }

  // Native 7D joint-space delta + gripper action (matches UR5eArena.step() 1:1!)
  setJointControl(action) {
    const d = this.data;
    for (let i = 0; i < 6; i++) {
      const dq = clamp(action[i], -1, 1) * DQ_MAX;
      d.ctrl[i] = clamp(d.qpos[i] + dq, JNT_LO[i], JNT_HI[i]);
    }
    const g = 0.5 * (clamp(action[6], -1, 1) + 1.0);
    this.gripCmd = g;
    d.ctrl[6] = 255.0 * g;
  }

  stepJoints(action) {
    this.setJointControl(action);
    for (let k = 0; k < SUBSTEPS; k++) this.substep();
  }

  setControl(action) {
    const d = this.data;
    const aPos = [action[0] * ACTION_HIGH[0], action[1] * ACTION_HIGH[1], action[2] * ACTION_HIGH[2]];
    const aOri = action[3] * ACTION_HIGH[3];
    const aGrip = action[4] * ACTION_HIGH[4];

    const effPos = this.effectorPos();
    const effQuat = mat2Quat(d.site_xmat, 9 * this.pinchSite);
    const effYaw = computeYaw(effQuat);
    const gripOpen = clamp(d.qpos[this.gripperQposAdr] / 0.8, 0, 1);

    const tgtPos = [
      clamp(effPos[0] + aPos[0], WORKSPACE_LO[0], WORKSPACE_HI[0]),
      clamp(effPos[1] + aPos[1], WORKSPACE_LO[1], WORKSPACE_HI[1]),
      clamp(effPos[2] + aPos[2], WORKSPACE_LO[2], WORKSPACE_HI[2]),
    ];
    // Matches OGBench: Rz(a_ori) @ Rz(eff_yaw) @ down_rot^-1 -> compute yaw -> Rz(yaw) @ down_rot
    const tmpOri = quatMul(quatMul(quatFromZ(aOri), quatFromZ(effYaw)), quatConj(DOWN_QUAT));
    const yaw = clamp(computeYaw(tmpOri), -Math.PI, Math.PI);
    const tgtOri = quatMul(quatFromZ(yaw), DOWN_QUAT);
    const tgtGrip = clamp(gripOpen + aGrip, 0, 1);

    // Pinch pose in world frame -> attachment_site pose in world frame
    const rotTpa = quatRotateVec(tgtOri, T_PA_POS);
    const posWa = [tgtPos[0] + rotTpa[0], tgtPos[1] + rotTpa[1], tgtPos[2] + rotTpa[2]];
    const quatWa = quatMul(tgtOri, T_PA_QUAT);

    const qTarget = this.ik.solve(posWa, quatWa, d.qpos.subarray(0, 6));
    for (let i = 0; i < 6; i++) d.ctrl[i] = qTarget[i];
    d.ctrl[6] = 255.0 * tgtGrip;
  }

  substep() {
    this.pull?.();
    this.mj.mj_step(this.model, this.data);
  }

  step(action) {
    this.setControl(action);
    for (let k = 0; k < SUBSTEPS; k++) this.substep();
  }
}
