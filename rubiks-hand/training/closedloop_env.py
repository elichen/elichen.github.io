"""GPU-vectorized Closed-Loop Alternating Rubik's Cube Env (Reorient <-> Twist) on MuJoCo-Warp.

v4 ("Iron-Grip Sim-to-WASM"):
- Exact observation parity with rubiks-hand/rubik-env.js (Reorient 140D, Twist 143D)
- Safety-first reward structure: -150 drop penalty, -45 Twist bailout penalty, nonlinear palm-rim
  barrier & velocity discipline so policies cannot trade grip safety for speed
- Live physics domain randomization on training slice ONLY (per-env EMA [0.32, 0.48], action scale
  [0.93, 1.07], random lateral slip micro-impulses, 30% skewed-face scrambles) to eliminate
  MuJoCo-Warp -> MuJoCo-WASM sim-to-sim contact overfitting!
- Live GPU 40-step ring buffer mining BOTH pre-drop AND pre-bailout / post-bailout-catch states
  into a 4096-state Hard-Negative Pool
- Dedicated clean deterministic evaluation slice (envs 0..eval_envs-1) with zero jitter/DR
"""
import math
import os
import sys
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube_state import (
    FACE_AXES, CUBELET_HOME, apply_move_to_qpos, quat_mul, quat_from_axis_angle,
    QPOS_CUBE_POS, QPOS_CUBE_QUAT, QPOS_CENTERS, QPOS_CUBELETS,
)
from rubik_warp_env import (
    SUBSTEPS, ACTION_EMA, CRADLE_CTRL, TIP_NAMES,
    OBS_REORIENT_DIM, OBS_TWIST_DIM, CUBE_24_QUATS,
    wrap_pi, wrap90, round90, yaw_harmonic4_t,
)


class ClosedLoopRubikEnv:
    def __init__(
        self,
        xml_path: str,
        num_envs: int = 1280,
        eval_envs: int = 256,
        target_moves: int = 25,
        hard_neg_prob: float = 0.30,
        dr_enabled: bool = True,
        seed: int = 42,
        device: str = "cuda:0",
    ):
        self.num_envs = num_envs
        self.eval_envs = min(eval_envs, num_envs)
        self.target_moves = target_moves
        self.hard_neg_prob = hard_neg_prob
        self.dr_enabled = dr_enabled
        self.device = torch.device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.nu = self.mj_model.nu

        self.cube_body = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube_core")
        self.grasp_site = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.tip_sites = torch.tensor(
            [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in TIP_NAMES],
            device=self.device, dtype=torch.long,
        )

        lo = self.mj_model.actuator_ctrlrange[:, 0].astype(np.float32)
        hi = self.mj_model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.act_lo = torch.tensor(lo, device=self.device)
        self.act_hi = torch.tensor(hi, device=self.device)
        self.act_mid = 0.5 * (self.act_lo + self.act_hi)
        self.act_half = 0.5 * (self.act_hi - self.act_lo)

        # Cradle base pose
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.ctrl[:] = CRADLE_CTRL
        self.mj_data.qpos[:7] = CRADLE_CTRL[:7]
        self.mj_data.qpos[19] = CRADLE_CTRL[16]
        for q_off, a_off in [(7, 7), (11, 10), (15, 13), (20, 17)]:
            self.mj_data.qpos[q_off] = CRADLE_CTRL[a_off]
            self.mj_data.qpos[q_off + 1] = CRADLE_CTRL[a_off + 1]
            self.mj_data.qpos[q_off + 2] = 0.5 * CRADLE_CTRL[a_off + 2]
            self.mj_data.qpos[q_off + 3] = 0.5 * CRADLE_CTRL[a_off + 2]
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.cradle_qpos = torch.tensor(self.mj_data.qpos, dtype=torch.float32, device=self.device)
        self.cradle_ctrl = torch.tensor(CRADLE_CTRL, dtype=torch.float32, device=self.device)
        self.hand_lo = torch.tensor(self.mj_model.jnt_range[:24, 0], dtype=torch.float32, device=self.device)
        self.hand_hi = torch.tensor(self.mj_model.jnt_range[:24, 1], dtype=torch.float32, device=self.device)

        self.face_axes = torch.tensor(FACE_AXES, dtype=torch.float32, device=self.device)
        self.cube_24_quats = torch.tensor(CUBE_24_QUATS, dtype=torch.float32, device=self.device)

        # Pre-generate 1024 scrambled cube states:
        #   clean_pool (512): exact 90-deg aligned scrambles for eval slice + 70% train resets
        #   skew_pool  (512): +-20 deg single-face skewed states for 30% train resets (unjam/straightening skill!)
        rng = np.random.default_rng(seed + 101)
        faces = ["U", "R", "F", "D", "L", "B"]
        mods = ["", "'", "2"]
        clean_pool, skew_pool = [], []
        for idx in range(512):
            qp = self.mj_data.qpos.copy()
            last = -1
            for _ in range(int(rng.integers(5, 22))):
                f = int(rng.integers(0, 6))
                while f == last:
                    f = int(rng.integers(0, 6))
                last = f
                apply_move_to_qpos(qp, faces[f] + mods[int(rng.integers(0, 3))])
            clean_pool.append(qp[QPOS_CENTERS:].astype(np.float32))

            qp_s = qp.copy()
            fid = int(rng.integers(0, 6))
            axis = FACE_AXES[fid]
            ang = float(rng.uniform(-0.35, 0.35))
            q_rot = quat_from_axis_angle(axis, ang)
            qp_s[QPOS_CENTERS + fid] += ang
            for k in range(20):
                off = QPOS_CUBELETS + 4 * k
                qk = qp_s[off:off + 4]
                w, x, y, z = qk
                h = CUBELET_HOME[k]
                t = 2.0 * np.cross(np.array([x, y, z]), h)
                pos = h + w * t + np.cross(np.array([x, y, z]), t)
                if np.dot(pos, axis) > 0.5:
                    qn = quat_mul(q_rot, qk)
                    qp_s[off:off + 4] = qn / (np.linalg.norm(qn) + 1e-12)
            skew_pool.append(qp_s[QPOS_CENTERS:].astype(np.float32))
        self.scramble_pool = torch.tensor(np.stack(clean_pool), device=self.device)
        self.skew_pool = torch.tensor(np.stack(skew_pool), device=self.device)

        # MuJoCo-Warp GPU buffers (per-world nconmax=200, njmax=450 for exact 26-cubelet contacts)
        wp.init()
        self.mw = mjw.put_model(self.mj_model)
        if hasattr(self.mw.opt, "warn_overflow"):
            self.mw.opt.warn_overflow = 0
        self.dw = mjw.put_data(self.mj_model, self.mj_data, nworld=num_envs, nconmax=200, njmax=450)

        self.qpos = wp.to_torch(self.dw.qpos)
        self.qvel = wp.to_torch(self.dw.qvel)
        self.ctrl = wp.to_torch(self.dw.ctrl)
        self.qacc = wp.to_torch(self.dw.qacc)
        self.qacc_ws = wp.to_torch(self.dw.qacc_warmstart) if hasattr(self.dw, "qacc_warmstart") else None
        self.site_xpos = wp.to_torch(self.dw.site_xpos)
        self.xmat = wp.to_torch(self.dw.xmat)

        # Warmup & capture CUDA graphs
        self.qpos[:] = self.cradle_qpos.unsqueeze(0)
        self.qvel[:] = 0.0
        self.ctrl[:] = self.cradle_ctrl.unsqueeze(0)
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

        N = num_envs
        # Per-env DR parameters (locked to exact nominal on eval slice 0..eval_envs-1!)
        self.is_train_env = torch.arange(N, device=self.device) >= self.eval_envs
        self.env_ema = torch.full((N, 1), ACTION_EMA, device=self.device)
        self.env_act_scale = torch.ones((N, 1), device=self.device)

        # Per-env closed-loop solver state
        self.phase = torch.zeros(N, device=self.device, dtype=torch.long)       # 0=reorient, 1=twist
        self.target_face = torch.full((N,), 4, device=self.device, dtype=torch.long)
        self.twist_dir = torch.ones(N, device=self.device)
        self.goal_angle = torch.zeros(N, device=self.device)
        self.prev_err = torch.zeros(N, device=self.device)
        self.prev_action = torch.zeros((N, 20), device=self.device)
        self.ep_steps = torch.zeros(N, device=self.device, dtype=torch.long)
        self.phase_steps = torch.zeros(N, device=self.device, dtype=torch.long)
        self.moves_done = torch.zeros(N, device=self.device, dtype=torch.long)
        self.unjam_mode = torch.zeros(N, device=self.device, dtype=torch.bool)
        self.max_ep_steps = 4000

        # Live GPU 40-step ring buffer per env for hard-negative failure & bailout mining
        self.ring_len = 40
        self.ring_ptr = 0
        self.ring_qpos = torch.zeros((self.ring_len, N, self.mj_model.nq), device=self.device)
        self.ring_qvel = torch.zeros((self.ring_len, N, self.mj_model.nv), device=self.device)
        self.ring_ctrl = torch.zeros((self.ring_len, N, 20), device=self.device)
        self.ring_phase = torch.zeros((self.ring_len, N), device=self.device, dtype=torch.long)
        self.ring_face = torch.zeros((self.ring_len, N), device=self.device, dtype=torch.long)
        self.ring_dir = torch.ones((self.ring_len, N), device=self.device)
        self.ring_goal = torch.zeros((self.ring_len, N), device=self.device)
        self.ring_ep_steps = torch.zeros((self.ring_len, N), device=self.device, dtype=torch.long)

        # Hard-negative replay buffer (mined live from drops + bailouts!)
        self.hn_cap = 4096
        self.hn_count = 0
        self.hn_ptr = 0
        self.hn_qpos = torch.zeros((self.hn_cap, self.mj_model.nq), device=self.device)
        self.hn_qvel = torch.zeros((self.hn_cap, self.mj_model.nv), device=self.device)
        self.hn_ctrl = torch.zeros((self.hn_cap, 20), device=self.device)
        self.hn_phase = torch.zeros(self.hn_cap, device=self.device, dtype=torch.long)
        self.hn_face = torch.zeros(self.hn_cap, device=self.device, dtype=torch.long)
        self.hn_dir = torch.ones(self.hn_cap, device=self.device)
        self.hn_goal = torch.zeros(self.hn_cap, device=self.device)

        self.reset_all()

    def _cube_xmat_3x3(self):
        return self.xmat[:, self.cube_body].reshape(self.num_envs, 3, 3)

    def _cube_quat_canon(self):
        q = self.qpos[:, QPOS_CUBE_QUAT:QPOS_CUBE_QUAT + 4]
        return q * torch.where(q[:, :1] < 0.0, -1.0, 1.0)

    def _top_face_ids(self):
        xm = self._cube_xmat_3x3()
        return torch.argmax(xm[:, 2, :] @ self.face_axes.T, dim=-1)

    def _target_normal_world(self):
        xm = self._cube_xmat_3x3()
        return torch.bmm(xm, self.face_axes[self.target_face].unsqueeze(-1)).squeeze(-1)

    def _record_ring(self):
        p = self.ring_ptr
        self.ring_qpos[p].copy_(self.qpos)
        self.ring_qvel[p].copy_(self.qvel)
        self.ring_ctrl[p].copy_(self.ctrl)
        self.ring_phase[p].copy_(self.phase)
        self.ring_face[p].copy_(self.target_face)
        self.ring_dir[p].copy_(self.twist_dir)
        self.ring_goal[p].copy_(self.goal_angle)
        self.ring_ep_steps[p].copy_(self.ep_steps)
        self.ring_ptr = (p + 1) % self.ring_len

    def _push_hn(self, qpos, qvel, ctrl, phase, face, tdir, goal):
        m = len(phase)
        if m == 0:
            return
        dst = (torch.arange(m, device=self.device) + self.hn_ptr) % self.hn_cap
        self.hn_qpos[dst] = qpos
        self.hn_qvel[dst] = qvel
        self.hn_ctrl[dst] = ctrl
        self.hn_phase[dst] = phase
        self.hn_face[dst] = face
        self.hn_dir[dst] = tdir
        self.hn_goal[dst] = goal
        self.hn_ptr = (self.hn_ptr + m) % self.hn_cap
        self.hn_count = min(self.hn_cap, self.hn_count + m)

    def _mine_from_ring(self, event_ids: torch.Tensor, lo_back: int = 20, hi_back: int = 31):
        if len(event_ids) == 0:
            return
        valid = self.ep_steps[event_ids] >= (hi_back + 2)
        d_ids = event_ids[valid]
        if len(d_ids) == 0:
            return
        lookback = torch.randint(lo_back, hi_back, (len(d_ids),), device=self.device)
        slots = (self.ring_ptr - 1 - lookback) % self.ring_len
        snap_ep = self.ring_ep_steps[slots, d_ids]
        same_ep = (snap_ep > 0) & (snap_ep < self.ep_steps[d_ids])
        d_ids = d_ids[same_ep]
        slots = slots[same_ep]
        if len(d_ids) == 0:
            return
        self._push_hn(
            self.ring_qpos[slots, d_ids],
            self.ring_qvel[slots, d_ids],
            self.ring_ctrl[slots, d_ids],
            self.ring_phase[slots, d_ids],
            self.ring_face[slots, d_ids],
            self.ring_dir[slots, d_ids],
            self.ring_goal[slots, d_ids],
        )

    def _mine_bailout_catch(self, bail_ids: torch.Tensor):
        """Also push current post-bailout states (in phase=0 Reorient) so Reorient drills emergency catches!"""
        if len(bail_ids) == 0:
            return
        # Cap per-step catch pushes to avoid flooding buffer
        if len(bail_ids) > 16:
            bail_ids = bail_ids[:16]
        self._push_hn(
            self.qpos[bail_ids],
            self.qvel[bail_ids],
            self.ctrl[bail_ids],
            torch.zeros(len(bail_ids), device=self.device, dtype=torch.long),
            self.target_face[bail_ids],
            self.twist_dir[bail_ids],
            self.goal_angle[bail_ids],
        )

    def _init_reorient_err(self, ids: torch.Tensor):
        if len(ids) == 0:
            return
        xm = self._cube_xmat_3x3()[ids]
        nw = torch.bmm(xm, self.face_axes[self.target_face[ids]].unsqueeze(-1)).squeeze(-1)
        self.prev_err[ids] = torch.acos(torch.clamp(nw[:, 2], -1.0, 1.0))

    def _init_twist_err(self, ids: torch.Tensor):
        if len(ids) == 0:
            return
        cur = self.qpos[ids, QPOS_CENTERS + self.target_face[ids]]
        self.prev_err[ids] = torch.abs(wrap_pi(self.goal_angle[ids] - cur))

    def reset_ids(self, ids: torch.Tensor):
        m = len(ids)
        if m == 0:
            return
        # Resample per-env DR for train slice on reset (eval slice stays exact nominal!)
        tr_ids = ids[ids >= self.eval_envs]
        if self.dr_enabled and len(tr_ids) > 0:
            self.env_ema[tr_ids, 0] = 0.32 + 0.16 * torch.rand(len(tr_ids), device=self.device)
            self.env_act_scale[tr_ids, 0] = 0.93 + 0.14 * torch.rand(len(tr_ids), device=self.device)

        use_hn = torch.zeros(m, device=self.device, dtype=torch.bool)
        if self.hn_count >= 256 and self.hard_neg_prob > 0.0:
            is_train = ids >= self.eval_envs
            use_hn = is_train & (torch.rand(m, device=self.device) < self.hard_neg_prob)

        syn_ids = ids[~use_hn]
        hn_ids = ids[use_hn]

        if len(syn_ids) > 0:
            ns = len(syn_ids)
            q = self.cradle_qpos.unsqueeze(0).expand(ns, -1).clone()
            tr_mask = (syn_ids >= self.eval_envs).unsqueeze(-1).float()
            q[:, :24] = torch.clamp(
                q[:, :24] + tr_mask * (torch.rand((ns, 24), device=self.device) * 2.0 - 1.0) * 0.04,
                self.hand_lo, self.hand_hi,
            )
            pos_jit = (torch.rand((ns, 3), device=self.device) * 2.0 - 1.0) * torch.tensor([0.006, 0.006, 0.003], device=self.device)
            pos_jit[:, 2] = torch.abs(pos_jit[:, 2])  # never push cube downward into palm
            q[:, QPOS_CUBE_POS:QPOS_CUBE_POS + 3] += tr_mask * pos_jit
            q24 = self.cube_24_quats[torch.randint(0, 24, (ns,), device=self.device)]
            q_id = self.cradle_qpos[QPOS_CUBE_QUAT:QPOS_CUBE_QUAT + 4].unsqueeze(0).expand(ns, -1)
            q[:, QPOS_CUBE_QUAT:QPOS_CUBE_QUAT + 4] = torch.where(tr_mask > 0.5, q24, q_id)

            p_idx = torch.randint(0, len(self.scramble_pool), (ns,), device=self.device)
            clean_c = self.scramble_pool[p_idx]
            skew_c = self.skew_pool[p_idx]
            use_skew = (syn_ids >= self.eval_envs) & (torch.rand(ns, device=self.device) < 0.30)
            q[:, QPOS_CENTERS:] = torch.where(use_skew.unsqueeze(-1), skew_c, clean_c)

            self.qpos[syn_ids] = q
            self.qvel[syn_ids] = 0.0
            self.ctrl[syn_ids] = self.cradle_ctrl.unsqueeze(0).expand(ns, -1)
            self.phase[syn_ids] = 0
            self.target_face[syn_ids] = torch.randint(0, 6, (ns,), device=self.device)
            self.twist_dir[syn_ids] = torch.where(torch.rand(ns, device=self.device) < 0.5, -1.0, 1.0)
            self.goal_angle[syn_ids] = 0.0

        if len(hn_ids) > 0:
            nh = len(hn_ids)
            h_idx = torch.randint(0, self.hn_count, (nh,), device=self.device)
            self.qpos[hn_ids] = self.hn_qpos[h_idx]
            self.qvel[hn_ids] = self.hn_qvel[h_idx]
            self.ctrl[hn_ids] = self.hn_ctrl[h_idx]
            self.phase[hn_ids] = self.hn_phase[h_idx]
            self.target_face[hn_ids] = self.hn_face[h_idx]
            self.twist_dir[hn_ids] = self.hn_dir[h_idx]
            self.goal_angle[hn_ids] = self.hn_goal[h_idx]

        self.qacc[ids] = 0.0
        if self.qacc_ws is not None:
            self.qacc_ws[ids] = 0.0
        self.ep_steps[ids] = 0
        self.phase_steps[ids] = 0
        self.moves_done[ids] = 0
        self.unjam_mode[ids] = False
        self.prev_action[ids] = (self.ctrl[ids] - self.act_mid) / self.act_half

        wp.capture_launch(self.fwd_graph)
        wp.synchronize()

        re_ids = ids[self.phase[ids] == 0]
        tw_ids = ids[self.phase[ids] == 1]
        self._init_reorient_err(re_ids)
        self._init_twist_err(tw_ids)

    def reset_all(self):
        self.reset_ids(torch.arange(self.num_envs, device=self.device, dtype=torch.long))
        return self.get_both_obs()

    def get_both_obs(self):
        """Return (obs_reorient [N, 140], obs_twist [N, 143]) matching rubik-env.js bitwise."""
        N = self.num_envs
        hand_qpos = self.qpos[:, :24]
        hand_qvel = torch.clamp(self.qvel[:, :24] * 0.1, -5.0, 5.0)
        ctrl = self.ctrl[:, :20]

        cube_pos = self.qpos[:, QPOS_CUBE_POS:QPOS_CUBE_POS + 3]
        tips = self.site_xpos[:, self.tip_sites, :]
        tip_rel = ((tips - cube_pos.unsqueeze(1)) * 10.0).reshape(N, 15)

        grasp_pos = self.site_xpos[:, self.grasp_site, :]
        cube_rel_grasp = (cube_pos - grasp_pos) * 10.0

        cube_linvel = torch.clamp(self.qvel[:, 24:27], -5.0, 5.0)
        cube_quat = self._cube_quat_canon()
        cube_angvel = torch.clamp(self.qvel[:, 27:30] * 0.2, -5.0, 5.0)

        xm3 = self._cube_xmat_3x3()
        cube_xmat = xm3.reshape(N, 9)

        face_ang = self.qpos[:, QPOS_CENTERS:QPOS_CENTERS + 6]
        face_mis = wrap90(face_ang)
        face_sin4 = torch.sin(4.0 * face_ang)
        face_cos4 = torch.cos(4.0 * face_ang)
        face_vel = torch.clamp(self.qvel[:, 30:36] * 0.2, -5.0, 5.0)

        nw = self._target_normal_world()

        base = torch.cat([
            hand_qpos, hand_qvel, ctrl, tip_rel, cube_rel_grasp,
            cube_linvel, cube_quat, cube_angvel, cube_xmat,
            face_mis, face_sin4, face_cos4, face_vel,
        ], dim=-1)  # (N, 129)

        # Reorient 140D
        onehot = torch.zeros((N, 6), device=self.device)
        onehot.scatter_(1, self.target_face.unsqueeze(1), 1.0)
        ys4, yc4 = yaw_harmonic4_t(xm3)
        obs_re = torch.cat([base, nw, onehot, ys4.unsqueeze(1), yc4.unsqueeze(1)], dim=-1)

        # Twist 143D
        row_idx = torch.arange(N, device=self.device)
        cur_top = face_ang[row_idx, self.target_face]
        rem = wrap_pi(self.goal_angle - cur_top)
        top_vel = face_vel[row_idx, self.target_face]
        rel_faces = -face_mis.clone()
        rel_faces[row_idx, self.target_face] = rem
        obs_tw = torch.cat([
            base, nw, rem.unsqueeze(1), torch.sin(rem).unsqueeze(1), torch.cos(rem).unsqueeze(1),
            self.twist_dir.unsqueeze(1), top_vel.unsqueeze(1), rel_faces,
        ], dim=-1)

        obs_re = torch.nan_to_num(obs_re, nan=0.0, posinf=10.0, neginf=-10.0)
        obs_tw = torch.nan_to_num(obs_tw, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs_re, obs_tw

    def step(self, action: torch.Tensor):
        N = self.num_envs
        row_idx = torch.arange(N, device=self.device)
        phase_before = self.phase.clone()

        self._record_ring()

        a = torch.clamp(action, -1.0, 1.0)
        # Apply DR action scale & per-env EMA ONLY on training slice (eval slice stays 100% nominal!)
        if self.dr_enabled:
            a_eff = torch.clamp(a * self.env_act_scale, -1.0, 1.0)
            target = self.act_mid + self.act_half * a_eff
            new_ctrl = torch.clamp((1.0 - self.env_ema) * self.ctrl + self.env_ema * target, self.act_lo, self.act_hi)
            # Random slip micro-perturbation impulses on 12% of train envs to kill fragile fingertip tricks!
            kick_mask = self.is_train_env & (torch.rand(N, device=self.device) < 0.12)
            if kick_mask.any():
                k_m = kick_mask.unsqueeze(-1).float()
                dv = (torch.rand((N, 3), device=self.device) * 2.0 - 1.0) * 0.04
                dw = (torch.rand((N, 3), device=self.device) * 2.0 - 1.0) * 0.85
                self.qvel[:, 24:27].add_(k_m * dv)
                self.qvel[:, 27:30].add_(k_m * dw)
        else:
            target = self.act_mid + self.act_half * a
            new_ctrl = torch.clamp((1.0 - ACTION_EMA) * self.ctrl + ACTION_EMA * target, self.act_lo, self.act_hi)

        self.ctrl.copy_(new_ctrl)

        wp.capture_launch(self.step_graph)
        wp.synchronize()

        self.ep_steps += 1
        self.phase_steps += 1

        cube_pos = self.qpos[:, QPOS_CUBE_POS:QPOS_CUBE_POS + 3]
        grasp_pos = self.site_xpos[:, self.grasp_site, :]
        pos_dist = torch.norm(cube_pos - grasp_pos, dim=-1)
        dropped = (pos_dist > 0.14) | (cube_pos[:, 2] < -0.05) | (~torch.isfinite(pos_dist))

        face_ang = self.qpos[:, QPOS_CENTERS:QPOS_CENTERS + 6]
        face_mis_abs = torch.abs(wrap90(face_ang))
        max_mis = face_mis_abs.max(dim=-1).values
        xm3 = self._cube_xmat_3x3()
        nw = self._target_normal_world()
        dot_z = torch.clamp(nw[:, 2], -1.0, 1.0)
        tilt = torch.acos(dot_z)
        cube_linv = torch.norm(self.qvel[:, 24:27], dim=-1)
        cube_angv = torch.norm(self.qvel[:, 27:30], dim=-1)

        act_diff = ((a - self.prev_action) ** 2).sum(dim=-1)
        vel_pen = (self.qvel[:, :24] ** 2).sum(dim=-1)
        self.prev_action.copy_(a)

        is_re = phase_before == 0
        is_tw = phase_before == 1

        # Nonlinear palm-rim safety barrier & anti-hover time cost (IMPOSSIBLE to reward-farm by stalling!)
        rim_excess = torch.clamp(pos_dist - 0.036, min=0.0)
        step_cost = -0.08

        # --- IRON-GRIP REORIENT REWARD (zero positive hover reward -> cannot stall!) ---
        re_progress = 7.0 * (self.prev_err - tilt)
        _, yc4 = yaw_harmonic4_t(xm3)
        yaw_bonus = 0.08 * torch.clamp((dot_z - 0.90) / 0.10, 0.0, 1.0) * (0.5 * (yc4 + 1.0))
        face_lock_pen = -3.5 * face_mis_abs.sum(dim=-1)
        re_center_pen = -8.0 * pos_dist - 50.0 * rim_excess
        re_throw_pen = -2.5 * cube_linv * torch.clamp(pos_dist - 0.030, min=0.0) - 0.12 * torch.clamp(cube_angv - 5.5, min=0.0)
        near_r = torch.clamp(1.0 - tilt / 0.40, min=0.0)
        re_brake_pen = -0.45 * near_r * cube_angv
        re_smooth_pen = -0.05 * act_diff - 0.003 * vel_pen
        rew_re = re_progress + step_cost + yaw_bonus + face_lock_pen + re_center_pen + re_throw_pen + re_brake_pen + re_smooth_pen

        re_dot_req = torch.where(self.phase_steps > 120, 0.94, 0.975)
        re_mis_req = torch.where(self.phase_steps > 120, 0.18, 0.13)
        re_ang_req = torch.where(self.phase_steps > 120, 4.0, 2.2)
        re_pos_req = torch.where(self.phase_steps > 120, 0.065, 0.045)
        re_succ = is_re & (~dropped) & (~self.unjam_mode) & (dot_z > re_dot_req) & (max_mis < re_mis_req) & (cube_angv < re_ang_req) & (pos_dist < re_pos_req)
        re_unjam_done = is_re & (~dropped) & self.unjam_mode & ((max_mis < 0.14) | (self.phase_steps > 60))
        re_stall = is_re & (~dropped) & (~re_succ) & (~self.unjam_mode) & (self.phase_steps > 200)
        re_stall_tw = re_stall & (dot_z > 0.88) & (max_mis < 0.24)
        re_stall_unjam = re_stall & (~re_stall_tw)
        re_enter_tw = re_succ | re_stall_tw

        rew_re = torch.where(re_succ, rew_re + 25.0, rew_re)
        rew_re = torch.where(re_stall_unjam, rew_re - 30.0, rew_re)
        rew_re = torch.where(dropped, rew_re - 180.0, rew_re)

        # --- IRON-GRIP TWIST REWARD (zero positive hover reward -> cannot stall!) ---
        cur_top = face_ang[row_idx, self.target_face]
        rem = torch.abs(wrap_pi(self.goal_angle - cur_top))
        tw_progress = 24.0 * (self.prev_err - rem)

        tips = self.site_xpos[:, self.tip_sites, :]
        top_center = cube_pos + 0.019 * nw
        bot_center = cube_pos - 0.010 * nw
        r_reach = (
            0.12 * torch.exp(-20.0 * torch.clamp(torch.norm(tips[:, 0, :] - top_center, dim=-1) - 0.032, min=0.0))
            + 0.12 * torch.exp(-20.0 * torch.clamp(torch.norm(tips[:, 1, :] - top_center, dim=-1) - 0.032, min=0.0))
            + 0.12 * torch.exp(-20.0 * torch.clamp(torch.norm(tips[:, 4, :] - top_center, dim=-1) - 0.032, min=0.0))
            + 0.08 * torch.exp(-20.0 * torch.clamp(torch.norm(tips[:, 2, :] - bot_center, dim=-1) - 0.034, min=0.0))
            + 0.08 * torch.exp(-20.0 * torch.clamp(torch.norm(tips[:, 3, :] - bot_center, dim=-1) - 0.034, min=0.0))
        )
        top_vel_raw = self.qvel[row_idx, 30 + self.target_face]
        top_vel_signed = top_vel_raw * self.twist_dir
        r_vel = torch.where(rem > 0.22, 0.20 * torch.clamp(top_vel_signed, -2.0, 3.5), torch.zeros_like(rem))
        near_t = torch.clamp(1.0 - rem / 0.28, min=0.0)
        r_vel = r_vel - 0.50 * near_t * torch.abs(top_vel_raw) - 0.35 * near_t * cube_angv

        other_mask = torch.ones((N, 6), device=self.device)
        other_mask[row_idx, self.target_face] = 0.0
        other_err_sum = (face_mis_abs * other_mask).sum(dim=-1)
        other_err_max = (face_mis_abs * other_mask).max(dim=-1).values
        upright_pen = -3.0 * tilt - 10.0 * torch.clamp(tilt - 0.24, min=0.0) - 6.0 * pos_dist - 45.0 * rim_excess
        other_pen = -3.0 * other_err_sum
        tw_smooth_pen = -0.025 * act_diff - 0.0015 * vel_pen
        rew_tw = tw_progress + step_cost + r_reach + r_vel + upright_pen + other_pen + tw_smooth_pen

        tw_rem_req = torch.where(self.phase_steps > 140, 0.16, 0.13)
        tw_tv_req = torch.where(self.phase_steps > 140, 2.8, 1.8)
        tw_ang_req = torch.where(self.phase_steps > 140, 4.0, 2.5)
        tw_pos_req = torch.where(self.phase_steps > 140, 0.065, 0.045)
        tw_dot_req = torch.where(self.phase_steps > 140, 0.90, 0.955)
        tw_succ = is_tw & (~dropped) & (rem < tw_rem_req) & (torch.abs(top_vel_raw) < tw_tv_req) & (cube_angv < tw_ang_req) & (pos_dist < tw_pos_req) & (other_err_max < 0.15) & (dot_z > tw_dot_req)
        tw_bail = is_tw & (~dropped) & (~tw_succ) & ((dot_z < 0.62) | (pos_dist > 0.075) | (self.phase_steps > 220))

        rew_tw = torch.where(tw_succ, rew_tw + 45.0, rew_tw)
        rew_tw = torch.where(tw_bail, rew_tw - 50.0, rew_tw)
        rew_tw = torch.where(dropped, rew_tw - 180.0, rew_tw)

        reward = torch.where(is_re, rew_re, rew_tw)
        reward = torch.nan_to_num(reward, nan=-180.0, posinf=150.0, neginf=-220.0)

        self.prev_err = torch.where(is_re, tilt, rem)

        # --- STATE TRANSITIONS ---
        re_to_tw_ids = torch.nonzero(re_enter_tw, as_tuple=False).squeeze(-1)
        if len(re_to_tw_ids) > 0:
            self.phase[re_to_tw_ids] = 1
            self.phase_steps[re_to_tw_ids] = 0
            cur_a = self.qpos[re_to_tw_ids, QPOS_CENTERS + self.target_face[re_to_tw_ids]]
            self.goal_angle[re_to_tw_ids] = round90(cur_a) + self.twist_dir[re_to_tw_ids] * (math.pi * 0.5)
            self._init_twist_err(re_to_tw_ids)

        uj_start = torch.nonzero(re_stall_unjam, as_tuple=False).squeeze(-1)
        if len(uj_start) > 0:
            self.target_face[uj_start] = (self.target_face[uj_start] + 2) % 6
            self.unjam_mode[uj_start] = True
            self.phase_steps[uj_start] = 0
            self._init_reorient_err(uj_start)

        uj_done = torch.nonzero(re_unjam_done, as_tuple=False).squeeze(-1)
        if len(uj_done) > 0:
            self.unjam_mode[uj_done] = False
            cur_top_u = self._top_face_ids()[uj_done]
            self.target_face[uj_done] = (cur_top_u + torch.randint(1, 6, (len(uj_done),), device=self.device)) % 6
            self.phase_steps[uj_done] = 0
            self._init_reorient_err(uj_done)

        self.moves_done += tw_succ.long()
        solved_ep = tw_succ & (self.moves_done >= self.target_moves)
        reward = torch.where(solved_ep, reward + 80.0, reward)

        tw_cont_ids = torch.nonzero(tw_succ & (~solved_ep), as_tuple=False).squeeze(-1)
        if len(tw_cont_ids) > 0:
            same_face = torch.rand(len(tw_cont_ids), device=self.device) < 0.20
            x2_ids = tw_cont_ids[same_face]
            next_re_ids = tw_cont_ids[~same_face]
            if len(x2_ids) > 0:
                self.phase[x2_ids] = 1
                self.phase_steps[x2_ids] = 0
                cur_a = self.qpos[x2_ids, QPOS_CENTERS + self.target_face[x2_ids]]
                self.goal_angle[x2_ids] = round90(cur_a) + self.twist_dir[x2_ids] * (math.pi * 0.5)
                self._init_twist_err(x2_ids)
            if len(next_re_ids) > 0:
                self.phase[next_re_ids] = 0
                self.phase_steps[next_re_ids] = 0
                cur_top_n = self._top_face_ids()[next_re_ids]
                self.target_face[next_re_ids] = (cur_top_n + torch.randint(1, 6, (len(next_re_ids),), device=self.device)) % 6
                self.twist_dir[next_re_ids] = torch.where(torch.rand(len(next_re_ids), device=self.device) < 0.5, -1.0, 1.0)
                self._init_reorient_err(next_re_ids)

        bail_ids = torch.nonzero(tw_bail, as_tuple=False).squeeze(-1)
        if len(bail_ids) > 0:
            # Mine pre-bailout states (15..26 steps before bail) AND post-bailout emergency catch states!
            self._mine_from_ring(bail_ids, lo_back=15, hi_back=26)
            self.phase[bail_ids] = 0
            self.phase_steps[bail_ids] = 0
            self.unjam_mode[bail_ids] = False
            self._mine_bailout_catch(bail_ids)
            self._init_reorient_err(bail_ids)

        timeout = (self.ep_steps >= self.max_ep_steps) & (~dropped) & (~solved_ep)
        done = dropped | solved_ep | timeout

        drop_ids = torch.nonzero(dropped, as_tuple=False).squeeze(-1)
        self._mine_from_ring(drop_ids, lo_back=20, hi_back=31)

        ev_slice = slice(0, self.eval_envs)
        eval_solved = int(solved_ep[ev_slice].sum().item())
        eval_dropped = int(dropped[ev_slice].sum().item())
        eval_timeout = int(timeout[ev_slice].sum().item())
        eval_done_mask = done[ev_slice]
        eval_moves_sum = float(self.moves_done[ev_slice][eval_done_mask].sum().item()) if eval_done_mask.any() else 0.0

        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self.reset_ids(done_ids)

        obs_re, obs_tw = self.get_both_obs()
        info = {
            "phase_before": phase_before,
            "dropped": dropped,
            "solved_ep": solved_ep,
            "timeout": timeout,
            "done": done,
            "n_re_succ": int(re_succ.sum().item()),
            "n_tw_succ": int(tw_succ.sum().item()),
            "n_tw_bail": int(tw_bail.sum().item()),
            "n_drop": len(drop_ids),
            "n_re_drop": int((dropped & is_re).sum().item()),
            "n_tw_drop": int((dropped & is_tw).sum().item()),
            "eval_solved": eval_solved,
            "eval_dropped": eval_dropped,
            "eval_timeout": eval_timeout,
            "eval_done": eval_solved + eval_dropped + eval_timeout,
            "eval_moves_sum": eval_moves_sum,
            "hn_count": self.hn_count,
        }
        return obs_re, obs_tw, reward, done, info
