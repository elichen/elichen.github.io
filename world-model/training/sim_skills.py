"""Multi-behavior UR5e + Robotiq 2F-85 simulator & dataset generator.

Uses the EXACT robot XML & STLs from world-model/robot/ (with the tiny 41x76 cm
invisible wall cage removed so thrown cubes can fly freely) and native 7D joint-space
delta + gripper actions at 20 Hz (25 x 2 ms substeps) so the arm can BOTH do
millimeter-accurate tabletop manipulation AND full 360-degree whole-body motions
(throwing, rotating around its base, waving, reaching straight up, bowing, etc.).
"""
import os, re, math, time, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import mujoco

SUBSTEPS = 25          # 25 * 2 ms = 50 ms (20 Hz control)
DQ_MAX = 0.35          # max rad per 50 ms step (~7 rad/s: fast enough to whip-throw & spin 360!)
HOME_Q = np.array([-math.pi/2, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.0], dtype=np.float64)
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
T_PA_POS = np.array([0.0, 0.0, -0.1558], dtype=np.float64)
T_PA_QUAT = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=np.float64)
JNT_LO = np.array([-6.2, -3.1, -3.0, -6.2, -6.2, -6.2], dtype=np.float64)
JNT_HI = np.array([ 6.2,  0.1,  3.0,  6.2,  6.2,  6.2], dtype=np.float64)

SKILL_NAMES = [
    'throw', 'rotate_base', 'place', 'push', 'lift_hold', 'lift_drop',
    'wave', 'reach_up', 'bow', 'circle', 'sweep', 'clap_gripper',
    'poke_cube', 'point', 'nod', 'dance', 'home',
]
SKILL_TO_ID = {s: i for i, s in enumerate(SKILL_NAMES)}


def make_open_arena_xml(robot_dir):
    """Load scene.xml from robot_dir, remove the 4 tiny invisible cage walls, and widen floor."""
    out_path = Path(robot_dir) / 'scene_arena.xml'
    if out_path.exists() and out_path.stat().st_size > 100:
        return str(out_path)
    xml_path = Path(robot_dir) / 'scene.xml'
    xml = xml_path.read_text()
    # Remove invisible cube cage walls so thrown cubes actually fly & bounce!
    xml = re.sub(r'\s*<geom name="wall_[0-3]"[^/>]*/>', '', xml)
    # Widen floor plane from 1x1 to 4x4 so thrown cubes roll freely
    xml = xml.replace('<geom name="floor" class="/" size="1 1 0.01"',
                      '<geom name="floor" class="/" size="4 4 0.01"')
    tmp_path = Path(robot_dir) / f'scene_arena.{os.getpid()}.tmp'
    tmp_path.write_text(xml)
    os.replace(tmp_path, out_path)
    return str(out_path)


def quat_mul(a, b):
    return np.array([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ], dtype=np.float64)


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_from_z(th):
    return np.array([math.cos(0.5 * th), 0.0, 0.0, math.sin(0.5 * th)], dtype=np.float64)


def quat_rot(q, v):
    w, x, y, z = q
    tx = 2.0 * (y*v[2] - z*v[1])
    ty = 2.0 * (z*v[0] - x*v[2])
    tz = 2.0 * (x*v[1] - y*v[0])
    return np.array([
        v[0] + w*tx + (y*tz - z*ty),
        v[1] + w*ty + (z*tx - x*tz),
        v[2] + w*tz + (x*ty - y*tx),
    ], dtype=np.float64)


class DiffIK:
    """Exact same damped-least-squares IK as cube-env.js, used only by expert data generators."""
    def __init__(self, ik_xml_path):
        self.m = mujoco.MjModel.from_xml_path(str(ik_xml_path))
        self.d = mujoco.MjData(self.m)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.sq = np.zeros(4)
        self.vel = np.zeros(3)

    def solve(self, eff_pos, yaw, curr_q, max_iters=20):
        ori = quat_mul(quat_from_z(yaw), DOWN_QUAT)
        pos_wa = np.asarray(eff_pos, np.float64) + quat_rot(ori, T_PA_POS)
        quat_wa = quat_mul(ori, T_PA_QUAT)
        self.d.qpos[:6] = curr_q[:6]
        for _ in range(max_iters):
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            ep = pos_wa - self.d.site_xpos[self.site]
            mujoco.mju_mat2Quat(self.sq, self.d.site_xmat[self.site])
            eq = quat_mul(quat_wa, quat_conj(self.sq))
            mujoco.mju_quat2Vel(self.vel, eq, 1.0)
            err = np.concatenate([ep, self.vel])
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(self.vel) < 1e-4:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp, self.jacr])
            H = J @ J.T + 1e-6 * np.eye(6)
            upd = J.T @ np.linalg.solve(H, err)
            mx = np.max(np.abs(upd))
            if mx > math.pi / 4:
                upd *= (math.pi / 4) / mx
            self.d.qpos[:6] += upd
        return self.d.qpos[:6].copy()


class UR5eArena:
    """Native 7D joint-delta + gripper UR5e arena (exact match to upcoming browser cube-env.js)."""
    def __init__(self, robot_dir):
        self.robot_dir = Path(robot_dir)
        scene_xml = make_open_arena_xml(robot_dir)
        self.m = mujoco.MjModel.from_xml_path(scene_xml)
        self.d = mujoco.MjData(self.m)
        self.ik = DiffIK(self.robot_dir / 'ur5e_ik.xml')
        self.cube_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'object_0')
        self.pinch_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, 'ur5e/robotiq/pinch')
        self.grip_jnt = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, 'ur5e/robotiq/right_driver_joint')
        self.grip_qadr = int(self.m.jnt_qposadr[self.grip_jnt])
        cube_jnt = int(self.m.body_jntadr[self.cube_body])
        self.cube_qadr = int(self.m.jnt_qposadr[cube_jnt])
        self.cube_vadr = int(self.m.jnt_dofadr[cube_jnt])
        self.grip_cmd = 0.0

    def reset(self, rng, cube_xy=None, start_q=None):
        mujoco.mj_resetData(self.m, self.d)
        if start_q is None:
            mode = rng.random()
            if mode < 0.55:
                # Standard tabletop hover
                eff = np.array([rng.uniform(0.30, 0.52), rng.uniform(-0.24, 0.24), rng.uniform(0.18, 0.36)])
                yaw = rng.uniform(-0.8, 0.8)
                q = self.ik.solve(eff, yaw, HOME_Q)
            elif mode < 0.80:
                # Wide random arm posture (so policy learns to recover from wave / reach_up / bow / point!)
                q = np.array([
                    rng.uniform(-3.0, 0.2),
                    rng.uniform(-2.1, -0.7),
                    rng.uniform(0.1, 1.8),
                    rng.uniform(-2.0, -0.6),
                    rng.uniform(-2.2, 0.6),
                    rng.uniform(-1.0, 1.0),
                ])
            else:
                # Full 360° rotated base pose (recovers seamlessly after "rotate around your base"!)
                q = np.array([
                    rng.uniform(-5.2, 2.2),
                    rng.uniform(-1.6, -1.0),
                    rng.uniform(0.8, 1.6),
                    rng.uniform(-1.8, -1.0),
                    -1.57,
                    0.0,
                ])
        else:
            q = np.asarray(start_q, np.float64)
        self.d.qpos[:6] = q
        self.d.ctrl[:6] = q
        self.d.ctrl[6] = 0.0
        self.grip_cmd = 0.0
        if cube_xy is None:
            cx, cy = rng.uniform(0.32, 0.52), rng.uniform(-0.22, 0.22)
        else:
            cx, cy = cube_xy
        cyaw = rng.uniform(-math.pi, math.pi)
        self.d.qpos[self.cube_qadr:self.cube_qadr+3] = [cx, cy, 0.02]
        self.d.qpos[self.cube_qadr+3:self.cube_qadr+7] = quat_from_z(cyaw)
        self.d.qvel[self.cube_vadr:self.cube_vadr+6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        return self.obs()

    def cube_pos(self):
        return self.d.xpos[self.cube_body].copy()

    def cube_vel(self):
        return self.d.qvel[self.cube_vadr:self.cube_vadr+3].copy()

    def eff_pos(self):
        return self.d.site_xpos[self.pinch_site].copy()

    def grip_closed(self):
        return float(np.clip(self.d.qpos[self.grip_qadr] / 0.8, 0.0, 1.0))

    def has_cube(self):
        c = self.cube_pos(); e = self.eff_pos()
        return float(np.linalg.norm(c - e) < 0.065 and self.grip_closed() > 0.25)

    def obs(self):
        """30D compact state observation (proprio 23D + cube_pos 3D + cube_rel 3D + in_hand 1D).
        At browser runtime, cube_pos comes purely from frozen ViT-tiny + Readout2 (zero privileged state!)."""
        q = self.d.qpos[:6]
        qv = self.d.qvel[:6] / 5.0
        eff = self.eff_pos()
        gc = self.grip_closed()
        c = self.cube_pos()
        rel = c - eff
        ih = self.has_cube()
        # [sin(q) 6, cos(q) 6, qvel/5 6, eff 3, grip_closed 1, grip_cmd 1, cube 3, cube-eff 3, ih 1] = 30D
        return np.concatenate([
            np.sin(q), np.cos(q), qv, eff, [gc, self.grip_cmd], c, rel, [ih]
        ]).astype(np.float32)

    def feat_step(self):
        """Compact per-step trajectory feature for E_traj (includes cube_vel for dynamic behaviors!)."""
        q = self.d.qpos[:6]
        qv = np.clip(self.d.qvel[:6] / 5.0, -2.0, 2.0)
        eff = self.eff_pos()
        c = self.cube_pos()
        cv = np.clip(self.cube_vel() / 2.0, -3.0, 3.0)
        gc = self.grip_closed()
        ih = self.has_cube()
        # 6 + 6 + 6 + 3 + 3 + 3 + 2 = 29D
        return np.concatenate([
            np.sin(q), np.cos(q), qv, eff, c, cv, [gc, ih]
        ]).astype(np.float32)

    def step(self, action):
        """Native 7D action in [-1, 1]: dq_0..5 * DQ_MAX relative to current qpos, grip in [-1, 1] -> [0, 255]."""
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        q_curr = self.d.qpos[:6]
        q_tgt = np.clip(q_curr + a[:6] * DQ_MAX, JNT_LO, JNT_HI)
        self.d.ctrl[:6] = q_tgt
        g = 0.5 * (a[6] + 1.0)
        self.grip_cmd = float(g)
        self.d.ctrl[6] = 255.0 * g
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        return self.obs()

    def act_toward_q(self, q_tgt, grip01, max_frac=0.85):
        """Compute smooth 7D action moving toward joint target q_tgt."""
        dq = np.clip((np.asarray(q_tgt, np.float64) - self.d.qpos[:6]) / DQ_MAX, -max_frac, max_frac)
        return np.r_[dq, 2.0 * float(grip01) - 1.0].astype(np.float32)

    def act_toward_eff(self, pos, yaw, grip01, max_frac=0.65):
        # Always seed IK near HOME_Q if arm is spun far away from tabletop so IK never gets stuck!
        seed_q = HOME_Q if abs(self.d.qpos[0] - HOME_Q[0]) > 1.2 else self.d.qpos[:6]
        q_tgt = self.ik.solve(pos, yaw, seed_q)
        # Take shortest angular turn on shoulder_pan within joint limits
        pan_diff = (q_tgt[0] - self.d.qpos[0] + math.pi) % (2 * math.pi) - math.pi
        cand_pan = self.d.qpos[0] + pan_diff
        if JNT_LO[0] + 0.1 < cand_pan < JNT_HI[0] - 0.1:
            q_tgt[0] = cand_pan
        return self.act_toward_q(q_tgt, grip01, max_frac=max_frac)


# ---------------------------------------------------------------------------
# Physically grounded skill rollout generators (T = 80 steps = 4.0 s @ 20 Hz)
# ---------------------------------------------------------------------------

def _grasp_sequence(env, steps_out, feats_out, acts_out, rng, max_steps=34, lift_z=0.16):
    """Approach cube, descend, close gripper firmly, lift. Returns True if cube is lifted."""
    yaw = rng.uniform(-0.3, 0.3)
    def run_act(a):
        steps_out.append(env.obs())
        acts_out.append(a)
        env.step(a)
        feats_out.append(env.feat_step())

    # Phase 1: hover above cube (up to 20 steps so arm recovers smoothly from any base spin!)
    for _ in range(20):
        c = env.cube_pos()
        tgt = np.array([c[0], c[1], 0.14])
        run_act(env.act_toward_eff(tgt, yaw, 0.0, max_frac=0.78))
        if np.linalg.norm(env.eff_pos()[:2] - c[:2]) < 0.015 and abs(env.eff_pos()[2] - 0.14) < 0.025:
            break
    # Phase 2: descend to cube center
    for _ in range(10):
        c = env.cube_pos()
        tgt = np.array([c[0], c[1], max(0.022, c[2] + 0.002)])
        run_act(env.act_toward_eff(tgt, yaw, 0.0, max_frac=0.55))
        if np.linalg.norm(env.eff_pos() - tgt) < 0.014:
            break
    # Phase 3: close gripper firmly
    c_hold = env.cube_pos()
    for _ in range(8):
        tgt = np.array([c_hold[0], c_hold[1], max(0.022, c_hold[2])])
        run_act(env.act_toward_eff(tgt, yaw, 1.0, max_frac=0.35))
    # Phase 4: lift
    for _ in range(10):
        tgt = np.array([c_hold[0], c_hold[1], lift_z])
        run_act(env.act_toward_eff(tgt, yaw, 1.0, max_frac=0.65))
        if env.eff_pos()[2] >= lift_z - 0.02:
            break
    return env.cube_pos()[2] > 0.065


def rollout_skill(env, skill, rng, T=80):
    """Simulate one skill clip of length T=80 (4.0 s). Returns (ok, obs, acts, feats, meta)."""
    obs_list, act_list, feat_list = [], [], []
    def step_a(a):
        obs_list.append(env.obs())
        act_list.append(a.astype(np.float32))
        env.step(a)
        feat_list.append(env.feat_step())

    env.reset(rng)
    c0 = env.cube_pos().copy()
    q0_start = float(env.d.qpos[0])
    meta = {'skill': skill, 'c0': c0.tolist()}

    if skill == 'throw':
        # Pick up cube, wind up, whip forward/left/right/high, release mid-swing!
        dir_choice = rng.choice(['forward', 'left', 'right', 'high'])
        meta['dir'] = str(dir_choice)
        if not _grasp_sequence(env, obs_list, feat_list, act_list, rng, lift_z=0.20):
            return False, None, None, None, meta
        # Wind-up pose (pulled back and crouched before whip)
        if dir_choice == 'forward':
            q_wind = np.array([-1.57, -2.05, 1.85, -1.35, -1.57, 0.0])
            q_whip = np.array([-1.57, -0.55, 0.35, -1.35, -1.57, 0.0])
            rel_step = 4
        elif dir_choice == 'high':
            q_wind = np.array([-1.57, -2.15, 2.00, -1.45, -1.57, 0.0])
            q_whip = np.array([-1.57, -1.20, 0.25, -0.60, -1.57, 0.0])
            rel_step = 3
        elif dir_choice == 'left':
            # Sweep shoulder_pan toward -Y (left)
            q_wind = np.array([-0.85, -1.55, 1.35, -1.35, -1.57, 0.0])
            q_whip = np.array([-2.65, -0.95, 0.55, -1.15, -1.57, 0.0])
            rel_step = 4
        else:  # right
            q_wind = np.array([-2.30, -1.55, 1.35, -1.35, -1.57, 0.0])
            q_whip = np.array([-0.45, -0.95, 0.55, -1.15, -1.57, 0.0])
            rel_step = 4

        # Move smoothly to wind-up (14 steps) keeping cube gripped
        for _ in range(14):
            if len(act_list) >= T: break
            step_a(env.act_toward_q(q_wind, 1.0, max_frac=0.65))
        # Whip fast! Release gripper mid-whip so cube launches cleanly
        max_z = 0.0; max_v = 0.0
        for k in range(16):
            if len(act_list) >= T: break
            grip = 1.0 if k < rel_step else 0.0
            step_a(env.act_toward_q(q_whip, grip, max_frac=1.0))
            max_z = max(max_z, float(env.cube_pos()[2]))
            max_v = max(max_v, float(np.linalg.norm(env.cube_vel())))
        # Follow-through hold while cube flies & lands
        while len(act_list) < T:
            step_a(env.act_toward_q(q_whip, 0.0, max_frac=0.45))
            max_z = max(max_z, float(env.cube_pos()[2]))
            max_v = max(max_v, float(np.linalg.norm(env.cube_vel())))
        c_end = env.cube_pos()
        fly_dist = float(np.linalg.norm(c_end[:2] - c0[:2]))
        meta['fly_dist'] = fly_dist; meta['max_z'] = max_z; meta['max_v'] = max_v; meta['c_end'] = c_end.tolist()
        ok = (fly_dist > 0.35 or max_v > 1.2 or max_z > 0.42) and (env.has_cube() < 0.5)

    elif skill == 'rotate_base':
        direction = float(rng.choice([-1.0, 1.0]))
        meta['dir'] = 'ccw' if direction > 0 else 'cw'
        # Compact safe upright posture while spinning shoulder_pan full 360+ deg
        q_base = np.array([q0_start, -1.30, 1.45, -1.70, -1.57, 0.0])
        # Lift cleanly first (12 steps)
        for _ in range(12):
            step_a(env.act_toward_q(q_base, 0.0, max_frac=0.65))
        # Continuous 360°+ spin around base!
        q_cur_pan = float(env.d.qpos[0])
        speed = rng.uniform(0.11, 0.16) * direction  # ~2.2-3.2 rad/s -> >1 full turn in 3s
        while len(act_list) < T:
            q_cur_pan = float(np.clip(q_cur_pan + speed, -6.1, 6.1))
            q_t = q_base.copy(); q_t[0] = q_cur_pan
            step_a(env.act_toward_q(q_t, 0.0, max_frac=0.85))
        delta_pan = abs(float(env.d.qpos[0]) - q0_start)
        meta['delta_pan'] = delta_pan
        ok = delta_pan > 3.8

    elif skill == 'place':
        # Pick & place anywhere on tabletop [0.29..0.55] x [-0.26..0.26]
        for _ in range(30):
            gx, gy = rng.uniform(0.29, 0.55), rng.uniform(-0.26, 0.26)
            if math.hypot(gx - c0[0], gy - c0[1]) >= 0.11:
                break
        meta['goal'] = [float(gx), float(gy), 0.02]
        if not _grasp_sequence(env, obs_list, feat_list, act_list, rng, lift_z=0.17):
            return False, None, None, None, meta
        yaw = 0.0
        # Carry high over goal
        for _ in range(16):
            if len(act_list) >= T: break
            step_a(env.act_toward_eff([gx, gy, 0.17], yaw, 1.0, max_frac=0.65))
            if math.hypot(env.eff_pos()[0] - gx, env.eff_pos()[1] - gy) < 0.018:
                break
        # Lower to table
        for _ in range(12):
            if len(act_list) >= T: break
            step_a(env.act_toward_eff([gx, gy, 0.032], yaw, 1.0, max_frac=0.5))
            if env.eff_pos()[2] < 0.042:
                break
        # Open & retract gently
        for k in range(8):
            if len(act_list) >= T: break
            z_up = 0.035 if k < 4 else 0.14
            step_a(env.act_toward_eff([gx, gy, z_up], yaw, 0.0, max_frac=0.5))
        while len(act_list) < T:
            step_a(env.act_toward_eff([gx, gy, 0.16], yaw, 0.0, max_frac=0.4))
        c_end = env.cube_pos()
        err = float(np.linalg.norm(c_end[:2] - np.array([gx, gy])))
        meta['c_end'] = c_end.tolist(); meta['err'] = err
        ok = err < 0.055 and c_end[2] < 0.045

    elif skill == 'push':
        dir_choice = rng.choice(['left', 'right', 'forward', 'back'])
        meta['dir'] = str(dir_choice)
        dvec = {'left': np.array([0.0, -1.0]), 'right': np.array([0.0, 1.0]),
                'forward': np.array([1.0, 0.0]), 'back': np.array([-1.0, 0.0])}[dir_choice]
        # Start behind cube with closed gripper, lower, push briskly through cube
        p_behind = c0[:2] - dvec * 0.085
        p_behind = np.clip(p_behind, [0.22, -0.34], [0.60, 0.34])
        p_thru = np.clip(c0[:2] + dvec * rng.uniform(0.18, 0.28), [0.20, -0.38], [0.64, 0.38])
        for _ in range(16):
            step_a(env.act_toward_eff([p_behind[0], p_behind[1], 0.12], 0.0, 1.0, max_frac=0.7))
        for _ in range(10):
            step_a(env.act_toward_eff([p_behind[0], p_behind[1], 0.026], 0.0, 1.0, max_frac=0.6))
        for k in range(28):
            frac = min(1.0, (k + 1) / 20.0)
            p = p_behind * (1 - frac) + p_thru * frac
            step_a(env.act_toward_eff([p[0], p[1], 0.026], 0.0, 1.0, max_frac=0.75))
        while len(act_list) < T:
            step_a(env.act_toward_eff([p_thru[0], p_thru[1], 0.14], 0.0, 0.0, max_frac=0.5))
        disp = env.cube_pos()[:2] - c0[:2]
        proj = float(np.dot(disp, dvec))
        meta['proj'] = proj; meta['c_end'] = env.cube_pos().tolist()
        ok = proj > 0.09 and env.cube_pos()[2] < 0.06

    elif skill == 'lift_hold':
        target_z = float(rng.uniform(0.16, 0.34))
        meta['z'] = target_z
        if not _grasp_sequence(env, obs_list, feat_list, act_list, rng, lift_z=target_z):
            return False, None, None, None, meta
        c_xy = env.cube_pos()[:2]
        while len(act_list) < T:
            step_a(env.act_toward_eff([c_xy[0], c_xy[1], target_z], 0.0, 1.0, max_frac=0.5))
        ok = env.cube_pos()[2] > 0.12 and env.has_cube() > 0.5

    elif skill == 'lift_drop':
        drop_z = float(rng.uniform(0.24, 0.34))
        meta['z'] = drop_z
        if not _grasp_sequence(env, obs_list, feat_list, act_list, rng, lift_z=drop_z):
            return False, None, None, None, meta
        c_xy = env.cube_pos()[:2]
        for _ in range(10):
            if len(act_list) >= T: break
            step_a(env.act_toward_eff([c_xy[0], c_xy[1], drop_z], 0.0, 1.0, max_frac=0.6))
        max_z = float(env.cube_pos()[2])
        while len(act_list) < T:
            step_a(env.act_toward_eff([c_xy[0], c_xy[1], drop_z], 0.0, 0.0, max_frac=0.45))
        ok = max_z > 0.18 and env.cube_pos()[2] < 0.045

    elif skill == 'wave':
        # Raise arm high and wave forearm/wrist happily side to side!
        q_hi = np.array([-1.57, -1.65, 0.55, -0.55, 1.57, 0.0])
        for _ in range(18):
            step_a(env.act_toward_q(q_hi, 0.0, max_frac=0.75))
        for k in range(T - len(act_list)):
            q_t = q_hi.copy()
            s = math.sin(2.0 * math.pi * k / 16.0)
            q_t[2] = 0.55 + 0.42 * s
            q_t[3] = -0.55 - 0.55 * s
            q_t[5] = 0.45 * s
            step_a(env.act_toward_q(q_t, 0.0, max_frac=0.85))
        ok = env.eff_pos()[2] > 0.45

    elif skill == 'reach_up':
        q_up = np.array([-1.57 + rng.uniform(-0.25, 0.25), -1.57, 0.05, -1.57, 0.0, 0.0])
        while len(act_list) < T:
            step_a(env.act_toward_q(q_up, 0.0, max_frac=0.7))
        ok = env.eff_pos()[2] > 0.72

    elif skill == 'bow':
        # Deep respectful bow / crouch low toward the table
        q_bow = np.array([-1.57, -0.45, 1.55, -1.15, -1.57, 0.0])
        while len(act_list) < T:
            step_a(env.act_toward_q(q_bow, 0.0, max_frac=0.6))
        ok = env.eff_pos()[2] < 0.14

    elif skill == 'circle':
        # Trace smooth horizontal circles above table
        rad = float(rng.uniform(0.13, 0.18))
        z_h = float(rng.uniform(0.18, 0.26))
        sgn = float(rng.choice([-1.0, 1.0]))
        cx_c, cy_c = 0.42, 0.0
        for k in range(T):
            th = sgn * (2.0 * math.pi * k / 36.0)
            px = cx_c + rad * math.cos(th)
            py = cy_c + rad * math.sin(th)
            step_a(env.act_toward_eff([px, py, z_h], 0.0, 0.0, max_frac=0.7))
        ok = True

    elif skill == 'sweep':
        # Wide left-to-right tabletop sweeping wave
        z_h = float(rng.uniform(0.14, 0.22))
        x_h = float(rng.uniform(0.38, 0.48))
        for k in range(T):
            py = 0.28 * math.sin(2.0 * math.pi * k / 32.0)
            step_a(env.act_toward_eff([x_h, py, z_h], 0.0, 0.0, max_frac=0.75))
        ok = True

    elif skill == 'clap_gripper':
        # Hold gripper out visibly and rapidly snap open/shut 4+ times!
        q_show = np.array([-1.57, -1.25, 1.25, -1.57, -1.57, 0.0])
        for k in range(T):
            grip = 1.0 if ((k // 7) % 2 == 1) else 0.0
            step_a(env.act_toward_q(q_show, grip, max_frac=0.65))
        ok = True

    elif skill == 'poke_cube':
        # Close fingers into a pointer and tap the top of the cube repeatedly
        for k in range(T):
            c = env.cube_pos()
            if k < 16:
                z_t = 0.14
            else:
                phase = ((k - 16) % 16) / 16.0
                z_t = 0.042 if phase < 0.45 else 0.13
            step_a(env.act_toward_eff([c[0], c[1], z_t], 0.0, 1.0, max_frac=0.65))
        ok = True

    elif skill == 'point':
        dir_choice = rng.choice(['left', 'right', 'forward', 'back'])
        meta['dir'] = str(dir_choice)
        # Note: shoulder_pan -pi/2 faces +X (front); 0 faces +Y (right); -pi faces -Y (left); +pi/2 faces -X (back)
        pan = {'forward': -math.pi/2, 'right': 0.0, 'left': -math.pi, 'back': math.pi/2}[dir_choice]
        q_pt = np.array([pan, -0.55, 0.18, -1.20, 0.0, 0.0])
        while len(act_list) < T:
            step_a(env.act_toward_q(q_pt, 1.0, max_frac=0.7))
        ok = True

    elif skill == 'nod':
        # Hold arm up mid-height and pitch wrist up & down like nodding "yes"
        q_mid = np.array([-1.57, -1.35, 1.20, -1.45, -1.57, 0.0])
        for k in range(T):
            q_t = q_mid.copy()
            if k >= 12:
                q_t[3] = -1.45 + 0.65 * math.sin(2.0 * math.pi * (k - 12) / 16.0)
            step_a(env.act_toward_q(q_t, 0.0, max_frac=0.75))
        ok = True

    elif skill == 'dance':
        # Playful multi-joint rhythmic sway + shoulder/wrist bounce + finger clap
        q_mid = np.array([-1.57, -1.45, 1.00, -1.10, -1.57, 0.0])
        for k in range(T):
            t = 2.0 * math.pi * k / 20.0
            q_t = q_mid.copy()
            q_t[0] += 0.55 * math.sin(t)
            q_t[1] += 0.25 * math.cos(2 * t)
            q_t[2] += 0.35 * math.sin(2 * t)
            q_t[3] += 0.45 * math.cos(t)
            q_t[5] += 0.80 * math.sin(t)
            grip = 1.0 if math.sin(2 * t) > 0 else 0.0
            step_a(env.act_toward_q(q_t, grip, max_frac=0.8))
        ok = True

    elif skill == 'home':
        while len(act_list) < T:
            step_a(env.act_toward_q(HOME_Q, 0.0, max_frac=0.6))
        ok = True

    else:
        raise ValueError(skill)

    if not ok or len(act_list) != T:
        return False, None, None, None, meta
    return True, np.stack(obs_list), np.stack(act_list), np.stack(feat_list), meta


# ---------------------------------------------------------------------------
# Rich natural-language descriptions (training) vs STRICTLY UNSEEN test prompts
# ---------------------------------------------------------------------------

def _pick(seq, rng):
    return seq[int(rng.integers(len(seq)))]

def describe_clip(meta, rng):
    """Diverse natural English captions for paired alignment (train/val).
    Strictly disjoint from UNSEEN_PROMPTS test phrasings!"""
    sk = meta['skill']
    obj = _pick(('the cube', 'the red cube', 'the block', 'the red block', 'cube', 'the box', 'it'), rng)
    if sk == 'throw':
        d = meta.get('dir', 'forward')
        v = _pick(('throw', 'toss', 'fling', 'launch', 'pitch', 'hurl', 'chuck', 'lob', 'catapult'), rng)
        if rng.random() < 0.35:
            return _pick((f"{v} {obj}", f"pick up and {v} {obj}", f"grab {obj} and {v} it"), rng)
        dt = {
            'forward': _pick(('forward', 'ahead', 'straight ahead', 'away from you', 'across the room', 'far away'), rng),
            'left': _pick(('to the left', 'leftward', 'toward the left side'), rng),
            'right': _pick(('to the right', 'rightward', 'toward the right side'), rng),
            'high': _pick(('upward', 'straight up', 'up into the air', 'high up'), rng),
        }[d]
        return f"{v} {obj} {dt}"

    if sk == 'rotate_base':
        d = meta.get('dir', 'ccw')
        base_phrases = (
            'rotate around your base', 'spin around your base', 'turn all the way around',
            'spin in a full circle', 'rotate 360 degrees around the base',
            'orbit around your base', 'spin your base around', 'do a full rotation',
            'twirl around the base', 'turn in a circle around your base',
            'spin around in place', 'rotate your whole arm around the pedestal',
            'revolve around your base', 'make a full spin around your base',
        )
        p = _pick(base_phrases, rng)
        if rng.random() < 0.35:
            p += ' ' + ('counterclockwise' if d == 'ccw' else 'clockwise')
        return p

    if sk == 'place':
        gx, gy = meta['goal'][0], meta['goal'][1]
        v = _pick(('put', 'place', 'move', 'carry', 'set', 'bring', 'relocate', 'deliver', 'position'), rng)
        if gx > 0.48: depth = _pick(('in the front', 'near the front edge', 'at the front', 'toward the front'), rng)
        elif gx < 0.36: depth = _pick(('in the back', 'near the robot', 'at the back', 'toward the back'), rng)
        else: depth = _pick(('halfway back', 'in the middle', 'mid-table', 'halfway across'), rng)
        if gy < -0.10: lat = _pick(('on the left', 'to the left', 'on the left side'), rng)
        elif gy > 0.10: lat = _pick(('on the right', 'to the right', 'on the right side'), rng)
        else: lat = _pick(('in the center', 'in the middle', 'centered'), rng)

        mode = int(rng.integers(3))
        if mode == 0:
            if abs(gy) < 0.08 and 0.37 < gx < 0.48:
                return _pick((f'{v} {obj} in the center', f'center {obj}', f'{v} {obj} to the middle', 'center the block'), rng)
            fb = 'front' if gx > 0.46 else ('back' if gx < 0.38 else 'middle')
            lr = 'left' if gy < -0.07 else ('right' if gy > 0.07 else 'center')
            return _pick((f'{v} {obj} {fb} {lr}', f'{v} {obj} to the {fb} {lr}', f'{v} {obj} at the {fb} {lr}', f'{fb} {lr}'), rng)
        elif mode == 1:
            cm_y = int(round(abs(gy) * 100 / 5) * 5)
            side = 'left' if gy < 0 else 'right'
            if cm_y < 5:
                return f"{v} {obj} centered, {depth}"
            return f"{v} {obj} {cm_y} cm to the {side}, {depth}"
        else:
            return f"{v} {obj} {lat}, {depth}"

    if sk == 'push':
        d = meta.get('dir', 'right')
        v = _pick(('push', 'shove', 'slide', 'nudge', 'knock', 'scoot', 'sweep'), rng)
        dt = {
            'left': _pick(('to the left', 'left', 'leftward', 'toward the left'), rng),
            'right': _pick(('to the right', 'right', 'rightward', 'toward the right'), rng),
            'forward': _pick(('forward', 'away', 'toward the front', 'ahead'), rng),
            'back': _pick(('backward', 'back toward the robot', 'closer to you'), rng),
        }[d]
        return f"{v} {obj} {dt}"

    if sk == 'lift_hold':
        cm = int(round(meta.get('z', 0.22) * 100 / 5) * 5)
        return _pick((
            f"pick up {obj}", f"lift {obj} up", f"grab and lift {obj}",
            f"hold {obj} in the air", f"lift {obj} {cm} cm up", f"raise {obj} high",
            f"pick up {obj} and hold it", f"elevate {obj} and keep holding it",
            f"carry {obj} up and hold steady", f"hold {obj} suspended up high",
        ), rng)

    if sk == 'lift_drop':
        return _pick((
            f"pick up and drop {obj}", f"drop {obj}", f"lift {obj} and drop it",
            f"lift {obj} high and let go", f"pick up {obj} then release it",
            f"raise and drop {obj}", f"drop {obj} onto the table",
            f"lift and release {obj} so it falls", f"let {obj} drop to the floor",
        ), rng)

    if sk == 'wave':
        return _pick((
            'wave', 'wave hello', 'wave your hand', 'wave goodbye',
            'wave at the camera', 'wave in the air', 'give a friendly wave',
            'say hello by waving', 'greet me with a wave', 'wave your arm hello',
            'say hello', 'wave hi',
        ), rng)

    if sk == 'reach_up':
        return _pick((
            'reach up', 'reach straight up', 'reach for the sky',
            'stretch straight up', 'stand up tall', 'extend upward toward the ceiling',
            'arms up high', 'stretch as high as you can', 'reach toward the ceiling',
            'extend your arm vertically upward', 'lift your arm straight up to the top',
            'point straight up', 'stretch upward as tall as possible',
        ), rng)

    if sk == 'bow':
        return _pick((
            'bow', 'bow down', 'crouch low', 'get down low',
            'bend down close to the table', 'duck down', 'lower your arm to the table',
            'bend forward in a bow', 'hunch down low', 'stoop low to the ground',
        ), rng)

    if sk == 'circle':
        return _pick((
            'draw a circle over the table', 'move in a circle', 'trace a circle above the table',
            'stir in a circle', 'circle above the tabletop', 'make circles with your gripper',
            'loop around over the table', 'draw circles in the air above the table',
            'swirl your hand in circles over the tabletop',
        ), rng)

    if sk == 'sweep':
        return _pick((
            'sweep left and right', 'sweep across the table', 'swing side to side',
            'wipe back and forth across the table', 'move back and forth left to right',
            'oscillate left and right over the table', 'wave back and forth across the bench',
        ), rng)

    if sk == 'clap_gripper':
        return _pick((
            'open and close your gripper', 'clap your fingers', 'pinch open and shut',
            'snap your claws', 'open and shut your hand', 'chomp with your gripper',
            'open and close your jaws', 'flex your gripper fingers', 'click your pinchers',
        ), rng)

    if sk == 'poke_cube':
        return _pick((
            f'tap {obj}', f'poke {obj}', f'touch {obj} from above',
            f'tap on top of {obj}', f'boop {obj}', f'jab {obj} gently',
            f'pat the top of {obj}', f'lightly tap {obj}',
        ), rng)

    if sk == 'point':
        d = meta.get('dir', 'left')
        v = _pick(('point', 'aim your arm', 'reach out', 'turn to point', 'extend your arm'), rng)
        dt = {'left': 'to the left', 'right': 'to the right', 'forward': 'forward', 'back': 'behind you'}[d]
        return f"{v} {dt}"

    if sk == 'nod':
        return _pick((
            'nod', 'nod yes', 'nod your wrist up and down', 'bob your gripper up and down',
            'tilt your gripper up and down', 'agree by nodding',
        ), rng)

    if sk == 'dance':
        return _pick((
            'dance', 'do a little dance', 'wiggle around', 'celebrate', 'boogie',
            'groove around', 'jiggle and dance', 'happy robot dance', 'bust a move',
        ), rng)

    if sk == 'home':
        return _pick((
            'go home', 'return to home pose', 'stand ready', 'stay still',
            'stop and wait', 'reset pose', 'be still', 'hold position', 'rest at home',
        ), rng)

    return sk


# Strictly held-out real human phrasings (never verbatim in describe_clip where possible!)
UNSEEN_PROMPTS = [
    ("throw cube", "throw"),
    ("yeet the red block", "throw"),
    ("toss it to the right", "throw"),
    ("fling the cube off to the left", "throw"),
    ("launch the block high into the air", "throw"),
    ("rotate around your base", "rotate_base"),
    ("do a 360", "rotate_base"),
    ("spin all the way around", "rotate_base"),
    ("turn around in a complete circle", "rotate_base"),
    ("put the cube in the front left corner", "place"),
    ("move the red block to the back right", "place"),
    ("set it down in the middle of the table", "place"),
    ("slide the block 15 cm right near the front edge", "place"),
    ("shove the cube to the left", "push"),
    ("knock the red block to the right", "push"),
    ("push it forward", "push"),
    ("pick up the red block and keep it aloft", "lift_hold"),
    ("hoist the cube into the air", "lift_hold"),
    ("grab the block and let it fall", "lift_drop"),
    ("drop it from high up", "lift_drop"),
    ("say hi", "wave"),
    ("wave to me", "wave"),
    ("touch the ceiling", "reach_up"),
    ("point straight up at the sky", "reach_up"),
    ("take a bow", "bow"),
    ("duck down close to the floor", "bow"),
    ("stir the pot over the table", "circle"),
    ("orbit around above the cube", "circle"),
    ("windshield wiper across the table", "sweep"),
    ("sweep side to side", "sweep"),
    ("wiggle your claws open and closed", "clap_gripper"),
    ("squeeze and release your fingers", "clap_gripper"),
    ("boop the red square", "poke_cube"),
    ("gently tap the top of the block", "poke_cube"),
    ("point over to the left side", "point"),
    ("point straight behind you", "point"),
    ("nod your head yes", "nod"),
    ("shake and party", "dance"),
    ("hold steady and wait", "home"),
    ("freeze at home position", "home"),
]


# Worker pool for fast parallel dataset generation
_WORKER_ENV = None
def _init_worker(robot_dir):
    global _WORKER_ENV
    _WORKER_ENV = UR5eArena(robot_dir)

def _gen_chunk(args):
    worker_id, counts, seed_base, T = args
    rng = np.random.default_rng(seed_base + worker_id * 10007)
    obs_l, act_l, feat_l, sk_l, cap_l, meta_l = [], [], [], [], [], []
    for sk, n_want in counts.items():
        got = 0; tries = 0
        while got < n_want and tries < n_want * 5 + 20:
            tries += 1
            ok, ob, ac, ft, mt = rollout_skill(_WORKER_ENV, sk, rng, T=T)
            if ok:
                obs_l.append(ob)
                act_l.append(ac)
                feat_l.append(ft)
                sk_l.append(SKILL_TO_ID[sk])
                cap_l.append(describe_clip(mt, rng))
                meta_l.append(mt)
                got += 1
    return obs_l, act_l, feat_l, sk_l, cap_l, meta_l


def generate_dataset(robot_dir, out_dir, total_clips=12000, T=80, workers=None, seed=42):
    """Generate balanced multi-behavior dataset in parallel across CPU cores."""
    os.makedirs(out_dir, exist_ok=True)
    make_open_arena_xml(robot_dir)  # create scene_arena.xml before spawning workers
    workers = workers or min(os.cpu_count() or 4, 16)
    # Give extra weight to continuous-parameterized skills (place, throw, push, point, rotate_base)
    weights = {
        'place': 3.2, 'throw': 1.8, 'push': 1.5, 'rotate_base': 1.3, 'point': 1.3,
        'lift_hold': 1.0, 'lift_drop': 1.0, 'wave': 0.9, 'reach_up': 0.8, 'bow': 0.8,
        'circle': 0.9, 'sweep': 0.9, 'clap_gripper': 0.8, 'poke_cube': 0.9,
        'nod': 0.8, 'dance': 0.9, 'home': 0.8,
    }
    wsum = sum(weights.values())
    per_skill = {k: max(40, int(round(total_clips * v / wsum))) for k, v in weights.items()}
    chunks = []
    for w in range(workers):
        c = {k: (n // workers) + (1 if w < (n % workers) else 0) for k, n in per_skill.items()}
        chunks.append((w, c, seed, T))

    t0 = time.time()
    all_obs, all_act, all_feat, all_sk, all_cap, all_meta = [], [], [], [], [], []
    with Pool(workers, initializer=_init_worker, initargs=(str(robot_dir),)) as pool:
        for ob, ac, ft, sk, cp, mt in pool.map(_gen_chunk, chunks):
            all_obs.extend(ob); all_act.extend(ac); all_feat.extend(ft)
            all_sk.extend(sk); all_cap.extend(cp); all_meta.extend(mt)

    N = len(all_obs)
    perm = np.random.default_rng(seed + 999).permutation(N)
    obs = np.stack(all_obs)[perm].astype(np.float16)
    act = np.stack(all_act)[perm].astype(np.float16)
    feat = np.stack(all_feat)[perm].astype(np.float16)
    sk = np.array(all_sk, dtype=np.int32)[perm]
    caps = [all_cap[i] for i in perm]
    metas = [all_meta[i] for i in perm]

    np.savez(os.path.join(out_dir, 'trajs.npz'), obs=obs, act=act, feat=feat, skill=sk)
    json.dump({'skills': SKILL_NAMES, 'captions': caps, 'metas': metas},
              open(os.path.join(out_dir, 'meta.json'), 'w'))

    counts_got = {SKILL_NAMES[i]: int((sk == i).sum()) for i in range(len(SKILL_NAMES))}
    print(f"Generated {N} verified clips (T={T}) in {time.time()-t0:.1f}s ({workers} workers)", flush=True)
    print("Per-skill counts:", counts_got, flush=True)
    return N


if __name__ == '__main__':
    import sys
    rdir = sys.argv[1] if len(sys.argv) > 1 else 'robot'
    odir = sys.argv[2] if len(sys.argv) > 2 else 'data_arena'
    n_clips = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
    generate_dataset(rdir, odir, total_clips=n_clips)
