"""GPU-vectorized Shadow Hand + 26-cubelet 3x3x3 Rubik's Cube RL environment on MuJoCo-Warp.

Exact physics & observation parity with rubiks-hand/rubik-env.js:
  - 50 Hz control (5 physics substeps of 4 ms)
  - EMA-filtered absolute joint targets (alpha = 0.4) over all 20 Shadow Hand actuators
  - Task 'reorient' (140D obs): bring any target face normal to +Z while keeping all 6 faces locked to 90 deg
  - Task 'twist'    (143D obs): clamp bottom 2 layers & twist top (+Z) face by +90 deg or -90 deg
"""
import math
import numpy as np
import torch
import mujoco
import warp as wp
import mujoco_warp as mjw

from cube_state import (
    FACE_AXES, CUBELET_HOME, QPOS_CUBE_POS, QPOS_CUBE_QUAT, QPOS_CENTERS, QPOS_CUBELETS,
    apply_move_to_qpos, quat_mul, quat_from_axis_angle,
)

SUBSTEPS = 5
ACTION_EMA = 0.4
OBS_REORIENT_DIM = 140
OBS_TWIST_DIM = 143

CRADLE_CTRL = np.array([
    0.0, 0.0,
    0.3, 0.9, 0.0, 0.2, 0.3,
    0.0, 0.5, 0.8,
    0.0, 0.5, 0.8,
    0.0, 0.5, 0.8,
    0.2, 0.0, 0.5, 0.8,
], dtype=np.float32)

TIP_NAMES = ["tip_ff", "tip_mf", "tip_rf", "tip_lf", "tip_th"]
ALL_MOVES = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2",
             "D", "D'", "D2", "L", "L'", "L2", "B", "B'", "B2"]


def _build_24_cube_quats():
    """All 24 proper rotational symmetries of a cube as unit quaternions [w, x, y, z]."""
    # 6 face-up base rotations (putting face 0..5 on +Z), each crossed with 4 yaw multiples of 90 deg
    face_up_base = [
        quat_from_axis_angle([0, 1, 0], -math.pi / 2),  # 0: +X -> +Z
        quat_from_axis_angle([0, 1, 0],  math.pi / 2),  # 1: -X -> +Z
        quat_from_axis_angle([1, 0, 0],  math.pi / 2),  # 2: +Y -> +Z
        quat_from_axis_angle([1, 0, 0], -math.pi / 2),  # 3: -Y -> +Z
        np.array([1.0, 0.0, 0.0, 0.0]),                 # 4: +Z -> +Z
        quat_from_axis_angle([1, 0, 0],  math.pi),      # 5: -Z -> +Z
    ]
    quats = []
    for f_idx, q_base in enumerate(face_up_base):
        for y in range(4):
            q_yaw = quat_from_axis_angle([0, 0, 1], y * math.pi / 2)
            q = quat_mul(q_yaw, q_base)
            if q[0] < 0:
                q = -q
            quats.append(q)
    return np.asarray(quats, dtype=np.float32)  # (24, 4), grouped 4 per top-face!


CUBE_24_QUATS = _build_24_cube_quats()


def wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi


def wrap90(a: torch.Tensor) -> torch.Tensor:
    h = math.pi * 0.5
    return torch.remainder(a + math.pi * 0.25, h) - math.pi * 0.25


def round90(a: torch.Tensor) -> torch.Tensor:
    h = math.pi * 0.5
    return torch.round(a / h) * h


def quat_mul_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def quat_rotate_vec_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def yaw_harmonic4_t(xmat: torch.Tensor):
    """xmat: (N, 3, 3) -> (sin(4*yaw), cos(4*yaw)), each (N,)."""
    rx = xmat[:, 0, :]  # (N, 3)
    ry = xmat[:, 1, :]  # (N, 3)
    rx2 = rx * rx
    ry2 = ry * ry
    c4 = 0.5 * (rx2 * rx2 + ry2 * ry2 - 6.0 * rx2 * ry2).sum(dim=-1)
    s4 = 2.0 * (rx2 * rx * ry - rx * ry2 * ry).sum(dim=-1)
    return s4, c4


def build_cradle_qpos(mj_model: mujoco.MjModel) -> np.ndarray:
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    q = mj_data.qpos.copy().astype(np.float32)
    q[:7] = CRADLE_CTRL[:7]
    q[19] = CRADLE_CTRL[16]
    for q_off, a_off in [(7, 7), (11, 10), (15, 13), (20, 17)]:
        q[q_off] = CRADLE_CTRL[a_off]
        q[q_off + 1] = CRADLE_CTRL[a_off + 1]
        q[q_off + 2] = 0.5 * CRADLE_CTRL[a_off + 2]
        q[q_off + 3] = 0.5 * CRADLE_CTRL[a_off + 2]
    return q


def build_scramble_pool(mj_model: mujoco.MjModel, pool_size: int = 2048, seed: int = 0) -> np.ndarray:
    """Pre-generate pool_size physically scrambled Rubik's cube qpos[31:117] states."""
    rng = np.random.default_rng(seed)
    base_qpos = build_cradle_qpos(mj_model).astype(np.float64)
    pool = np.zeros((pool_size, mj_model.nq - QPOS_CENTERS), dtype=np.float32)
    for i in range(pool_size):
        q = base_qpos.copy()
        # 10% solved, 90% scrambled 1..20 moves
        n_moves = 0 if (i % 10 == 0) else int(rng.integers(1, 21))
        for _ in range(n_moves):
            apply_move_to_qpos(q, ALL_MOVES[int(rng.integers(0, 18))])
        # 40% of states get a realistic +-22 deg single-face skew so Reorient learns to straighten crooked faces cleanly
        if i % 5 < 2:
            fid = int(rng.integers(0, 6))
            axis = FACE_AXES[fid]
            ang = float(rng.uniform(-0.38, 0.38))
            q_rot = quat_from_axis_angle(axis, ang)
            q[QPOS_CENTERS + fid] += ang
            for k in range(20):
                off = QPOS_CUBELETS + 4 * k
                qk = q[off:off + 4]
                # Rotate cubelet home pos by qk
                w, x, y, z = qk
                h = CUBELET_HOME[k]
                t = 2.0 * np.cross(np.array([x, y, z]), h)
                pos = h + w * t + np.cross(np.array([x, y, z]), t)
                if np.dot(pos, axis) > 0.5:
                    qn = quat_mul(q_rot, qk)
                    q[off:off + 4] = qn / (np.linalg.norm(qn) + 1e-12)
        pool[i] = q[QPOS_CENTERS:].astype(np.float32)
    return pool


class RubikWarpEnv:
    def __init__(
        self,
        xml_path: str,
        task: str = "reorient",
        num_envs: int = 1024,
        seed: int = 1,
        handover_path: str | None = None,
        handover_prob: float = 0.5,
        twist_start_deg: float = 22.0,
        brake_settle: bool = False,
        nconmax: int = 200,
        njmax: int = 450,
    ):
        assert task in ("reorient", "twist")
        self.task = task
        self.num_envs = num_envs
        self.obs_dim = OBS_REORIENT_DIM if task == "reorient" else OBS_TWIST_DIM
        self.act_dim = 20
        self.handover_prob = handover_prob
        self.brake_settle = brake_settle

        wp.init()
        self.device = torch.device("cuda:0")
        torch.manual_seed(seed)

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.cube_body = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube_core")
        self.grasp_site = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.tip_sites = torch.tensor(
            [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in TIP_NAMES],
            device=self.device, dtype=torch.long,
        )

        self.mw = mjw.put_model(self.mj_model)
        if hasattr(self.mw.opt, "warn_overflow"):
            self.mw.opt.warn_overflow = 0
        self.dw = mjw.put_data(self.mj_model, self.mj_data, nworld=num_envs, nconmax=nconmax, njmax=njmax)

        # Zero-copy PyTorch views into Warp GPU buffers
        self.qpos = wp.to_torch(self.dw.qpos)          # (N, 117)
        self.qvel = wp.to_torch(self.dw.qvel)          # (N, 96)
        self.ctrl = wp.to_torch(self.dw.ctrl)          # (N, 20)
        self.qacc = wp.to_torch(self.dw.qacc)          # (N, 96)
        self.qacc_ws = wp.to_torch(self.dw.qacc_warmstart) if hasattr(self.dw, "qacc_warmstart") else None
        self.site_xpos = wp.to_torch(self.dw.site_xpos)  # (N, nsite, 3)
        self.xmat = wp.to_torch(self.dw.xmat)            # (N, nbody, 3, 3) or (N, nbody, 9)

        # Actuator ranges
        cr = self.mj_model.actuator_ctrlrange.astype(np.float32)
        self.act_lo = torch.tensor(cr[:, 0], device=self.device)
        self.act_hi = torch.tensor(cr[:, 1], device=self.device)
        self.act_mid = 0.5 * (self.act_lo + self.act_hi)
        self.act_half = 0.5 * (self.act_hi - self.act_lo)

        # Hand joint limits for reset jitter
        jr = self.mj_model.jnt_range[:24].astype(np.float32)
        self.hand_lo = torch.tensor(jr[:, 0], device=self.device)
        self.hand_hi = torch.tensor(jr[:, 1], device=self.device)

        self.cradle_ctrl = torch.tensor(CRADLE_CTRL, device=self.device)
        self.cradle_qpos = torch.tensor(build_cradle_qpos(self.mj_model), device=self.device)
        self.face_axes = torch.tensor(FACE_AXES, device=self.device, dtype=torch.float32)  # (6, 3)
        self.cubelet_home = torch.tensor(CUBELET_HOME, device=self.device, dtype=torch.float32)  # (20, 3)
        self.cube_24_quats = torch.tensor(CUBE_24_QUATS, device=self.device, dtype=torch.float32)  # (24, 4)

        # Twist angle curriculum: starts at twist_start_deg remaining and advances automatically to full 90 deg (pi/2)
        self.twist_curr_rad = math.radians(twist_start_deg) if task == "twist" else math.pi * 0.5

        # Pre-generated pool of scrambled internal cube states on GPU
        self.scramble_pool = torch.tensor(
            build_scramble_pool(self.mj_model, pool_size=2048, seed=seed),
            device=self.device,
        )

        # Optional handover state pool (real terminal states from the partner policy!)
        self.ho_qpos = self.ho_qvel = self.ho_ctrl = None
        if handover_path is not None:
            ho = np.load(handover_path)
            self.ho_qpos = torch.tensor(ho["qpos"], device=self.device, dtype=torch.float32)
            self.ho_qvel = torch.tensor(ho["qvel"], device=self.device, dtype=torch.float32)
            self.ho_ctrl = torch.tensor(ho["ctrl"], device=self.device, dtype=torch.float32)
            print(f"[{task}] Loaded {len(self.ho_qpos)} handover states from {handover_path} (p={handover_prob})", flush=True)

        # Capture CUDA graph for 5 substeps (1 control step at 50 Hz) + fast kinematics update for resets
        mjw.forward(self.mw, self.dw)
        mjw.kinematics(self.mw, self.dw)
        mjw.step(self.mw, self.dw)
        wp.synchronize()
        with wp.ScopedCapture() as cap_step:
            for _ in range(SUBSTEPS):
                mjw.step(self.mw, self.dw)
        self.step_graph = cap_step.graph
        with wp.ScopedCapture() as cap_fwd:
            mjw.kinematics(self.mw, self.dw)
        self.fwd_graph = cap_fwd.graph
        wp.synchronize()

        # Per-env task buffers
        N = num_envs
        self.ep_steps = torch.zeros(N, device=self.device, dtype=torch.long)
        self.goal_steps = torch.zeros(N, device=self.device, dtype=torch.long)
        self.consec_success = torch.zeros(N, device=self.device, dtype=torch.long)
        self.prev_action = torch.zeros((N, 20), device=self.device)
        self.target_face = torch.full((N,), 4, device=self.device, dtype=torch.long)
        self.twist_dir = torch.ones(N, device=self.device)
        self.goal_angle = torch.zeros(N, device=self.device)
        self.prev_err = torch.zeros(N, device=self.device)

        # Episode stats (running window)
        self.max_ep_steps = 800 if task == "reorient" else 600
        self.max_goal_steps = 250 if task == "reorient" else 200
        self.reset_all()

    def _cube_xmat_3x3(self) -> torch.Tensor:
        xm = self.xmat[:, self.cube_body]
        return xm.reshape(self.num_envs, 3, 3)

    def _cube_quat_canon(self) -> torch.Tensor:
        q = self.qpos[:, QPOS_CUBE_QUAT:QPOS_CUBE_QUAT + 4]
        sign = torch.where(q[:, :1] < 0.0, -1.0, 1.0)
        return q * sign

    def _top_face_ids(self) -> torch.Tensor:
        """Which face (0..5) currently points most upward (+Z) in each env."""
        xm = self._cube_xmat_3x3()  # (N, 3, 3): row 2 is world Z dot cube axes
        # dot(R @ FACE_AXES[i], +Z) = (R[2, :] @ FACE_AXES.T)
        z_dots = xm[:, 2, :] @ self.face_axes.T  # (N, 6)
        return torch.argmax(z_dots, dim=-1)

    def _target_normal_world(self) -> torch.Tensor:
        xm = self._cube_xmat_3x3()  # (N, 3, 3)
        axes = self.face_axes[self.target_face]  # (N, 3)
        return torch.bmm(xm, axes.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    def _pretwist_top_layer(self, ids: torch.Tensor, top_faces: torch.Tensor, pre_angles: torch.Tensor):
        """Rigidly rotate the top face slice (center + its 8 cubelets) by pre_angles on GPU."""
        m = len(ids)
        if m == 0:
            return
        axes = self.face_axes[top_faces]  # (m, 3)
        # Update center hinge angles
        self.qpos[ids, QPOS_CENTERS + top_faces] += pre_angles
        # Quaternion for rotation around axis by pre_angles: [cos(a/2), axis*sin(a/2)]
        half = 0.5 * pre_angles.unsqueeze(-1)  # (m, 1)
        q_rot = torch.cat([torch.cos(half), axes * torch.sin(half)], dim=-1)  # (m, 4)
        # Rotate all 20 cubelets belonging to this face slice
        q_all = self.qpos[ids, QPOS_CUBELETS:QPOS_CUBELETS + 80].reshape(m, 20, 4)
        home = self.cubelet_home.unsqueeze(0).expand(m, 20, 3)
        pos = quat_rotate_vec_t(q_all, home)  # (m, 20, 3)
        in_slice = (pos * axes.unsqueeze(1)).sum(dim=-1) > 0.5  # (m, 20)
        q_rot_exp = q_rot.unsqueeze(1).expand(m, 20, 4)
        q_new = quat_mul_t(q_rot_exp, q_all)
        q_new = q_new / (q_new.norm(dim=-1, keepdim=True) + 1e-8)
        q_out = torch.where(in_slice.unsqueeze(-1), q_new, q_all)
        self.qpos[ids, QPOS_CUBELETS:QPOS_CUBELETS + 80] = q_out.reshape(m, 80)

    def _sample_new_goal(self, ids: torch.Tensor, initial: bool = False):
        m = len(ids)
        if m == 0:
            return
        if self.task == "reorient":
            # Pick a new target face != current top face
            cur_top = self._top_face_ids()[ids]
            delta = torch.randint(1, 6, (m,), device=self.device)
            new_face = (cur_top + delta) % 6
            self.target_face[ids] = new_face
            # Compute initial angle error
            xm = self._cube_xmat_3x3()[ids]
            nw = torch.bmm(xm, self.face_axes[new_face].unsqueeze(-1)).squeeze(-1)
            self.prev_err[ids] = torch.acos(torch.clamp(nw[:, 2], -1.0, 1.0))
        else:
            # Twist: active face is whichever face is currently pointing UP (+Z)
            top = self._top_face_ids()[ids]
            self.target_face[ids] = top
            # 50/50 CW (-1) or CCW (+1)
            dirs = torch.where(torch.rand(m, device=self.device) < 0.5, -1.0, 1.0)
            self.twist_dir[ids] = dirs
            cur_ang = self.qpos[ids, QPOS_CENTERS + top]
            ga = round90(cur_ang) + dirs * (math.pi * 0.5)
            # During angle curriculum (< pi/2), pre-twist initial resets partway toward ga so early successes happen immediately!
            if initial and self.twist_curr_rad < (math.pi * 0.5 - 1e-3):
                # Sample remaining angle in [0.7*curr, curr]
                rem_target = self.twist_curr_rad * (0.7 + 0.3 * torch.rand(m, device=self.device))
                pre_ang = dirs * ((math.pi * 0.5) - rem_target)
                self._pretwist_top_layer(ids, top, pre_ang)
                wp.capture_launch(self.fwd_graph)
                wp.synchronize()
                cur_ang = self.qpos[ids, QPOS_CENTERS + top]
            self.goal_angle[ids] = ga
            self.prev_err[ids] = torch.abs(wrap_pi(ga - cur_ang))
        self.goal_steps[ids] = 0

    def reset_ids(self, ids: torch.Tensor):
        m = len(ids)
        if m == 0:
            return
        # Decide how many come from real handover pool vs synthetic
        use_ho = torch.zeros(m, device=self.device, dtype=torch.bool)
        if self.ho_qpos is not None and len(self.ho_qpos) > 0:
            use_ho = torch.rand(m, device=self.device) < self.handover_prob

        syn_ids = ids[~use_ho]
        ho_ids = ids[use_ho]

        if len(syn_ids) > 0:
            ns = len(syn_ids)
            q = self.cradle_qpos.unsqueeze(0).expand(ns, -1).clone()
            # Mild hand joint jitter
            q[:, :24] = torch.clamp(
                q[:, :24] + (torch.rand((ns, 24), device=self.device) * 2.0 - 1.0) * 0.08,
                self.hand_lo, self.hand_hi,
            )
            # Cube position jitter right above palm
            q[:, QPOS_CUBE_POS:QPOS_CUBE_POS + 3] += (torch.rand((ns, 3), device=self.device) * 2.0 - 1.0) * torch.tensor([0.008, 0.008, 0.004], device=self.device)
            # Random cube orientation from the 24 axis-aligned orientations + small tilt/yaw jitter
            q24 = self.cube_24_quats[torch.randint(0, 24, (ns,), device=self.device)]
            if self.task == "reorient":
                # Small random 3D tilt (up to ~18 deg)
                axis = torch.randn((ns, 3), device=self.device)
                axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
                ang = (torch.rand((ns, 1), device=self.device) * 2.0 - 1.0) * 0.30
                q_jit = torch.cat([torch.cos(0.5 * ang), axis * torch.sin(0.5 * ang)], dim=-1)
                q24 = quat_mul_t(q_jit, q24)
            else:
                # For twist: yaw jitter up to +-20 deg + slight tilt up to +-8 deg
                yaw = (torch.rand((ns, 1), device=self.device) * 2.0 - 1.0) * 0.35
                q_yaw = torch.cat([torch.cos(0.5 * yaw), torch.zeros((ns, 2), device=self.device), torch.sin(0.5 * yaw)], dim=-1)
                axis = torch.randn((ns, 3), device=self.device)
                axis[:, 2] = 0.0
                axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
                tilt = (torch.rand((ns, 1), device=self.device) * 2.0 - 1.0) * 0.14
                q_tilt = torch.cat([torch.cos(0.5 * tilt), axis * torch.sin(0.5 * tilt)], dim=-1)
                q24 = quat_mul_t(q_tilt, quat_mul_t(q_yaw, q24))
            q[:, QPOS_CUBE_QUAT:QPOS_CUBE_QUAT + 4] = q24 / (q24.norm(dim=-1, keepdim=True) + 1e-8)

            # Sample scrambled full cubelet configuration from pre-built pool
            p_idx = torch.randint(0, len(self.scramble_pool), (ns,), device=self.device)
            q[:, QPOS_CENTERS:] = self.scramble_pool[p_idx]

            self.qpos[syn_ids] = q
            self.qvel[syn_ids] = 0.0
            self.ctrl[syn_ids] = self.cradle_ctrl.unsqueeze(0).expand(ns, -1)

        if len(ho_ids) > 0:
            nh = len(ho_ids)
            h_idx = torch.randint(0, len(self.ho_qpos), (nh,), device=self.device)
            qh = self.ho_qpos[h_idx].clone()
            vh = self.ho_qvel[h_idx].clone()
            if self.task == "reorient":
                # On 25% of post-twist handovers, add bailout-like tilt (up to 35 deg), offset, and angular velocity
                # so Reorient robustly catches slipping cubes right after Twist bailouts!
                pert = torch.rand(nh, device=self.device) < 0.25
                np_cnt = int(pert.sum().item())
                if np_cnt > 0:
                    p_idx = torch.nonzero(pert, as_tuple=False).squeeze(-1)
                    axis = torch.randn((np_cnt, 3), device=self.device)
                    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
                    ang = (torch.rand((np_cnt, 1), device=self.device) * 2.0 - 1.0) * 0.60
                    q_jit = torch.cat([torch.cos(0.5 * ang), axis * torch.sin(0.5 * ang)], dim=-1)
                    q_cur = qh[p_idx, QPOS_CUBE_QUAT:QPOS_CUBE_QUAT + 4]
                    q_new = quat_mul_t(q_jit, q_cur)
                    qh[p_idx, QPOS_CUBE_QUAT:QPOS_CUBE_QUAT + 4] = q_new / (q_new.norm(dim=-1, keepdim=True) + 1e-8)
                    qh[p_idx, QPOS_CUBE_POS:QPOS_CUBE_POS + 3] += (torch.rand((np_cnt, 3), device=self.device) * 2.0 - 1.0) * torch.tensor([0.012, 0.012, 0.008], device=self.device)
                    vh[p_idx, 24:27] += (torch.rand((np_cnt, 3), device=self.device) * 2.0 - 1.0) * 0.12
                    vh[p_idx, 27:30] += (torch.rand((np_cnt, 3), device=self.device) * 2.0 - 1.0) * 2.0
            self.qpos[ho_ids] = qh
            self.qvel[ho_ids] = vh
            self.ctrl[ho_ids] = self.ho_ctrl[h_idx]

        self.qacc[ids] = 0.0
        if self.qacc_ws is not None:
            self.qacc_ws[ids] = 0.0
        self.ep_steps[ids] = 0
        self.consec_success[ids] = 0
        self.prev_action[ids] = (self.ctrl[ids] - self.act_mid) / self.act_half

        wp.capture_launch(self.fwd_graph)
        wp.synchronize()
        self._sample_new_goal(ids, initial=True)

    def reset_all(self):
        self.reset_ids(torch.arange(self.num_envs, device=self.device, dtype=torch.long))
        return self.get_obs()

    def get_obs(self) -> torch.Tensor:
        N = self.num_envs
        hand_qpos = self.qpos[:, :24]
        hand_qvel = torch.clamp(self.qvel[:, :24] * 0.1, -5.0, 5.0)
        ctrl = self.ctrl[:, :20]

        cube_pos = self.qpos[:, QPOS_CUBE_POS:QPOS_CUBE_POS + 3]  # (N, 3)
        tips = self.site_xpos[:, self.tip_sites, :]               # (N, 5, 3)
        tip_rel = ((tips - cube_pos.unsqueeze(1)) * 10.0).reshape(N, 15)

        grasp_pos = self.site_xpos[:, self.grasp_site, :]         # (N, 3)
        cube_rel_grasp = (cube_pos - grasp_pos) * 10.0            # (N, 3)

        cube_linvel = torch.clamp(self.qvel[:, 24:27], -5.0, 5.0)
        cube_quat = self._cube_quat_canon()
        cube_angvel = torch.clamp(self.qvel[:, 27:30] * 0.2, -5.0, 5.0)

        xm3 = self._cube_xmat_3x3()                               # (N, 3, 3)
        cube_xmat = xm3.reshape(N, 9)

        face_ang = self.qpos[:, QPOS_CENTERS:QPOS_CENTERS + 6]    # (N, 6)
        face_mis = wrap90(face_ang)
        face_sin4 = torch.sin(4.0 * face_ang)
        face_cos4 = torch.cos(4.0 * face_ang)
        face_vel = torch.clamp(self.qvel[:, 30:36] * 0.2, -5.0, 5.0)

        nw = self._target_normal_world()                          # (N, 3)

        if self.task == "reorient":
            onehot = torch.zeros((N, 6), device=self.device)
            onehot.scatter_(1, self.target_face.unsqueeze(1), 1.0)
            ys4, yc4 = yaw_harmonic4_t(xm3)
            obs = torch.cat([
                hand_qpos, hand_qvel, ctrl, tip_rel, cube_rel_grasp,
                cube_linvel, cube_quat, cube_angvel, cube_xmat,
                face_mis, face_sin4, face_cos4, face_vel,
                nw, onehot, ys4.unsqueeze(1), yc4.unsqueeze(1),
            ], dim=-1)
            return obs

        # Twist task (143D)
        row_idx = torch.arange(N, device=self.device)
        cur_top = face_ang[row_idx, self.target_face]
        rem = wrap_pi(self.goal_angle - cur_top)                  # (N,)
        top_vel = face_vel[row_idx, self.target_face]
        rel_faces = -face_mis.clone()
        rel_faces[row_idx, self.target_face] = rem

        obs = torch.cat([
            hand_qpos, hand_qvel, ctrl, tip_rel, cube_rel_grasp,
            cube_linvel, cube_quat, cube_angvel, cube_xmat,
            face_mis, face_sin4, face_cos4, face_vel,
            nw, rem.unsqueeze(1), torch.sin(rem).unsqueeze(1), torch.cos(rem).unsqueeze(1),
            self.twist_dir.unsqueeze(1), top_vel.unsqueeze(1), rel_faces,
        ], dim=-1)
        return obs

    def step(self, action: torch.Tensor):
        N = self.num_envs
        a = torch.clamp(action, -1.0, 1.0)
        target = self.act_mid + self.act_half * a
        new_ctrl = torch.clamp((1.0 - ACTION_EMA) * self.ctrl + ACTION_EMA * target, self.act_lo, self.act_hi)
        self.ctrl.copy_(new_ctrl)

        wp.capture_launch(self.step_graph)
        wp.synchronize()

        self.ep_steps += 1
        self.goal_steps += 1

        # Compute state quantities
        cube_pos = self.qpos[:, QPOS_CUBE_POS:QPOS_CUBE_POS + 3]
        grasp_pos = self.site_xpos[:, self.grasp_site, :]
        pos_dist = torch.norm(cube_pos - grasp_pos, dim=-1)
        dropped = (pos_dist > 0.14) | (cube_pos[:, 2] < -0.05) | (~torch.isfinite(pos_dist))

        face_ang = self.qpos[:, QPOS_CENTERS:QPOS_CENTERS + 6]
        face_mis_abs = torch.abs(wrap90(face_ang))
        xm3 = self._cube_xmat_3x3()
        nw = self._target_normal_world()
        dot_z = torch.clamp(nw[:, 2], -1.0, 1.0)
        tilt = torch.acos(dot_z)

        act_diff = ((a - self.prev_action) ** 2).sum(dim=-1)
        vel_pen = (self.qvel[:, :24] ** 2).sum(dim=-1)
        self.prev_action.copy_(a)

        if self.task == "reorient":
            # Potential progress toward target face pointing +Z
            progress = 8.0 * (self.prev_err - tilt)
            self.prev_err.copy_(tilt)
            orient_bonus = 1.5 / (1.0 + 4.0 * tilt)
            _, yc4 = yaw_harmonic4_t(xm3)
            yaw_bonus = 0.25 * torch.clamp((dot_z - 0.85) / 0.15, 0.0, 1.0) * (0.5 * (yc4 + 1.0))
            face_lock_pen = -3.0 * face_mis_abs.sum(dim=-1)
            center_pen = (-6.0 if self.brake_settle else -4.0) * pos_dist
            smooth_pen = -0.02 * act_diff - 0.001 * vel_pen

            max_mis = face_mis_abs.max(dim=-1).values
            if self.brake_settle:
                cube_angv = torch.norm(self.qvel[:, 27:30], dim=-1)
                near_r = torch.clamp(1.0 - tilt / 0.40, min=0.0)
                brake_pen = -0.35 * near_r * cube_angv
                succeeded = (~dropped) & (tilt < 0.22) & (max_mis < 0.13) & (cube_angv < 2.2) & (pos_dist < 0.045)
            else:
                brake_pen = 0.0
                succeeded = (~dropped) & (tilt < 0.24) & (max_mis < 0.13)
            reward = progress + orient_bonus + yaw_bonus + face_lock_pen + center_pen + smooth_pen + brake_pen
            reward = torch.where(succeeded, reward + 25.0, reward)
            reward = torch.where(dropped, reward - 15.0, reward)

            timeout = (self.ep_steps >= self.max_ep_steps) | (self.goal_steps >= self.max_goal_steps)
            done = dropped | timeout
        else:
            row_idx = torch.arange(N, device=self.device)
            cur_top = face_ang[row_idx, self.target_face]
            rem = torch.abs(wrap_pi(self.goal_angle - cur_top))
            progress = 30.0 * (self.prev_err - rem)
            self.prev_err.copy_(rem)

            # Active fingertip engagement reward (kills "splayed serving tray" local optimum!):
            # Pulls thumb, index, middle fingertips up onto the top layer (+19mm) & ring/pinky onto lower sides!
            tips = self.site_xpos[:, self.tip_sites, :]  # (N, 5, 3): [ff, mf, rf, lf, th]
            top_center = cube_pos + 0.019 * nw           # (N, 3)
            bot_center = cube_pos - 0.010 * nw           # (N, 3)
            d_top_ff = torch.norm(tips[:, 0, :] - top_center, dim=-1)
            d_top_mf = torch.norm(tips[:, 1, :] - top_center, dim=-1)
            d_top_th = torch.norm(tips[:, 4, :] - top_center, dim=-1)
            d_bot_rf = torch.norm(tips[:, 2, :] - bot_center, dim=-1)
            d_bot_lf = torch.norm(tips[:, 3, :] - bot_center, dim=-1)
            r_reach = (
                0.25 * torch.exp(-20.0 * torch.clamp(d_top_ff - 0.032, min=0.0))
                + 0.25 * torch.exp(-20.0 * torch.clamp(d_top_mf - 0.032, min=0.0))
                + 0.25 * torch.exp(-20.0 * torch.clamp(d_top_th - 0.032, min=0.0))
                + 0.15 * torch.exp(-20.0 * torch.clamp(d_bot_rf - 0.034, min=0.0))
                + 0.15 * torch.exp(-20.0 * torch.clamp(d_bot_lf - 0.034, min=0.0))
            )

            # Directional angular velocity reward while turning, plus smooth braking near target when brake_settle=True
            top_vel_raw = self.qvel[row_idx, 30 + self.target_face]
            top_vel_signed = top_vel_raw * self.twist_dir
            cube_angv = torch.norm(self.qvel[:, 27:30], dim=-1)
            r_vel = torch.where(rem > 0.22, 0.35 * torch.clamp(top_vel_signed, -2.0, 4.0), torch.zeros_like(rem))
            if self.brake_settle:
                near = torch.clamp(1.0 - rem / 0.28, min=0.0)
                r_vel = r_vel - 0.45 * near * torch.abs(top_vel_raw) - 0.25 * near * cube_angv

            # Closeness shaping (steep peak at rem=0, modest baseline so active twisting >> idling)
            twist_close = 1.5 / (1.0 + 3.0 * rem)

            # Mask out active top face to penalize unintended rotation of the other 5 faces
            other_mask = torch.ones((N, 6), device=self.device)
            other_mask[row_idx, self.target_face] = 0.0
            other_err_sum = (face_mis_abs * other_mask).sum(dim=-1)
            other_err_max = (face_mis_abs * other_mask).max(dim=-1).values

            if self.brake_settle:
                upright_pen = -2.2 * tilt - 4.5 * pos_dist
            else:
                upright_pen = -1.0 * tilt - 2.5 * pos_dist
            other_pen = -2.5 * other_err_sum
            smooth_pen = -0.01 * act_diff - 0.0005 * vel_pen

            flipped = dot_z < 0.35
            failed = dropped | flipped
            if self.brake_settle:
                succeeded = (~failed) & (rem < 0.13) & (torch.abs(top_vel_raw) < 1.8) & (cube_angv < 2.5) & (pos_dist < 0.045) & (other_err_max < 0.14) & (tilt < 0.30)
            else:
                succeeded = (~failed) & (rem < 0.16) & (other_err_max < 0.18) & (tilt < 0.40)

            reward = progress + r_reach + r_vel + twist_close + upright_pen + other_pen + smooth_pen
            reward = torch.where(succeeded, reward + 60.0, reward)
            reward = torch.where(failed, reward - 15.0, reward)

            goal_lim = 90 if self.twist_curr_rad < (math.pi * 0.5 - 1e-3) else self.max_goal_steps
            timeout = (self.ep_steps >= self.max_ep_steps) | (self.goal_steps >= goal_lim)
            done = failed | timeout

        # Record metrics BEFORE chaining/reset
        n_succ = int(succeeded.sum().item())
        n_drop = int((dropped if self.task == "reorient" else failed).sum().item())
        mean_err_deg = float((tilt if self.task == "reorient" else rem).mean().item() * 180.0 / math.pi)

        # Capture successful terminal states if requested
        succ_ids = torch.nonzero(succeeded & (~done), as_tuple=False).squeeze(-1)
        if len(succ_ids) > 0:
            self.consec_success[succ_ids] += 1
            self._sample_new_goal(succ_ids, initial=False)

        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self.reset_ids(done_ids)

        obs = self.get_obs()
        # Guard against rare physics NaNs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        reward = torch.nan_to_num(reward, nan=-15.0, posinf=50.0, neginf=-50.0)
        info = {
            "n_succ": n_succ,
            "n_drop": n_drop,
            "n_done": len(done_ids),
            "succ_ids": succ_ids,
            "mean_err_deg": mean_err_deg,
        }
        return obs, reward, done, info
