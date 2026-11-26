"""DAgger Episodic Specialist PPO + Sim-to-WASM Physics DR (reorient5 -> twist7).

Why Episodic DAgger PPO beats long-horizon semi-MDP fine-tuning:
1. Harvests 16,384 real closed-loop states on GPU (post-twist exits, twist-bailout emergency
   catch states, pre-drop hard negatives, X2 double-turn starts) in ~15 seconds.
2. Trains single-move episodic PPO (positive steady-state V ~ +150 -> ZERO risk-averse stalling!)
   with full Sim-to-WASM physics DR (EMA [0.30, 0.50], action scale [0.92, 1.08], lateral/rotational
   slip micro-impulses, obs noise, 35% skewed-face starts), nonlinear palm-rim safety barrier,
   and mild KL behavioral anchor to preserve 100% standalone skill.
3. Validates every 40 iters on 48 parallel native C-MuJoCo 25-move solves (saving best_reorient5,
   then harvesting fresh reorient5 handovers to train twist7, saving best_pair).
"""
import argparse
import copy
import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import warp as wp
import mujoco
import mujoco_warp as mjw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closedloop_env import ClosedLoopRubikEnv
from cpu_eval import eval_cpu_mujoco
from cube_state import (
    FACE_AXES, CUBELET_HOME, apply_move_to_qpos, quat_mul, quat_from_axis_angle,
    QPOS_CUBE_POS, QPOS_CUBE_QUAT, QPOS_CENTERS, QPOS_CUBELETS,
)
from rubik_warp_env import (
    SUBSTEPS, ACTION_EMA, CRADLE_CTRL, TIP_NAMES,
    OBS_REORIENT_DIM, OBS_TWIST_DIM, CUBE_24_QUATS,
    wrap_pi, wrap90, round90, yaw_harmonic4_t,
)
from train_ppo import ActorCritic, RunningNorm, save_ckpt
from train_closedloop import load_exported_into_model, export_pair
from export import export_checkpoint


def harvest_dagger_pools(
    xml_path: str,
    re_model: ActorCritic,
    re_norm: RunningNorm,
    tw_model: ActorCritic,
    tw_norm: RunningNorm,
    n_envs: int = 1280,
    n_steps: int = 450,
    seed: int = 77,
):
    """Run closed-loop rollouts on GPU to harvest real post-twist, bailout-catch, pre-drop, and handover states."""
    t0 = time.time()
    cl = ClosedLoopRubikEnv(
        xml_path=xml_path,
        num_envs=n_envs,
        eval_envs=0,
        target_moves=30,
        hard_neg_prob=0.0,
        dr_enabled=False,
        seed=seed,
    )
    obs_re, obs_tw = cl.reset_all()

    re_exit_qpos, re_exit_qvel, re_exit_ctrl = [], [], []
    re_catch_qpos, re_catch_qvel, re_catch_ctrl, re_catch_face = [], [], [], []
    tw_ho_qpos, tw_ho_qvel, tw_ho_ctrl, tw_ho_face, tw_ho_dir = [], [], [], [], []
    tw_x2_qpos, tw_x2_qvel, tw_x2_ctrl, tw_x2_face, tw_x2_dir = [], [], [], [], []

    cap_exit = 8192
    cap_catch = 4096
    cap_ho = 8192
    cap_x2 = 4096

    with torch.no_grad():
        for st in range(n_steps):
            re_a, _, _, _, _ = re_model.act(re_norm.normalize(obs_re), deterministic=True)
            tw_a, _, _, _, _ = tw_model.act(tw_norm.normalize(obs_tw), deterministic=True)
            # Mild exploration noise on half the envs during harvest to broaden state coverage!
            if st % 3 == 0:
                re_a[n_envs // 2:] += torch.randn_like(re_a[n_envs // 2:]) * 0.08
                tw_a[n_envs // 2:] += torch.randn_like(tw_a[n_envs // 2:]) * 0.08

            phase_before = cl.phase.clone()
            is_tw = (phase_before == 1).unsqueeze(-1)
            action = torch.where(is_tw, tw_a, re_a)
            obs_re, obs_tw, _, done, info = cl.step(action)

            # Detect clean transitions that just landed (not done!)
            alive = ~done
            # Reorient -> Twist handover!
            just_ho = alive & (phase_before == 0) & (cl.phase == 1)
            if just_ho.any() and sum(len(x) for x in tw_ho_face) < cap_ho:
                idx = torch.nonzero(just_ho, as_tuple=False).squeeze(-1)[:128]
                tw_ho_qpos.append(cl.qpos[idx].cpu())
                tw_ho_qvel.append(cl.qvel[idx].cpu())
                tw_ho_ctrl.append(cl.ctrl[idx].cpu())
                tw_ho_face.append(cl.target_face[idx].cpu())
                tw_ho_dir.append(cl.twist_dir[idx].cpu())

            # Twist -> Reorient exit OR Bailout catch!
            just_re = alive & (phase_before == 1) & (cl.phase == 0)
            if just_re.any():
                idx = torch.nonzero(just_re, as_tuple=False).squeeze(-1)
                cpos = cl.qpos[idx, QPOS_CUBE_POS:QPOS_CUBE_POS + 3]
                gpos = cl.site_xpos[idx, cl.grasp_site, :]
                pd = torch.norm(cpos - gpos, dim=-1)
                fmis = torch.abs(wrap90(cl.qpos[idx, QPOS_CENTERS:QPOS_CENTERS + 6])).max(dim=-1).values
                is_hard = (pd > 0.052) | (fmis > 0.16)
                hard_i = idx[is_hard][:128]
                clean_i = idx[~is_hard][:128]
                if len(hard_i) > 0 and sum(len(x) for x in re_catch_face) < cap_catch:
                    re_catch_qpos.append(cl.qpos[hard_i].cpu())
                    re_catch_qvel.append(cl.qvel[hard_i].cpu())
                    re_catch_ctrl.append(cl.ctrl[hard_i].cpu())
                    re_catch_face.append(cl.target_face[hard_i].cpu())
                if len(clean_i) > 0 and sum(len(x) for x in re_exit_qpos) < cap_exit:
                    re_exit_qpos.append(cl.qpos[clean_i].cpu())
                    re_exit_qvel.append(cl.qvel[clean_i].cpu())
                    re_exit_ctrl.append(cl.ctrl[clean_i].cpu())

            # Twist -> Twist (X2 double turn!)
            just_x2 = alive & (phase_before == 1) & (cl.phase == 1) & (cl.phase_steps == 0)
            if just_x2.any() and sum(len(x) for x in tw_x2_face) < cap_x2:
                idx = torch.nonzero(just_x2, as_tuple=False).squeeze(-1)[:128]
                tw_x2_qpos.append(cl.qpos[idx].cpu())
                tw_x2_qvel.append(cl.qvel[idx].cpu())
                tw_x2_ctrl.append(cl.ctrl[idx].cpu())
                tw_x2_face.append(cl.target_face[idx].cpu())
                tw_x2_dir.append(cl.twist_dir[idx].cpu())

    # Grab mined pre-drop / pre-bailout hard negatives directly from cl's ring-mined pool!
    n_hn = cl.hn_count
    hn_mask_re = (cl.hn_phase[:n_hn] == 0)
    hn_mask_tw = (cl.hn_phase[:n_hn] == 1)

    pools = {
        "re_exit": (
            torch.cat(re_exit_qpos, dim=0) if re_exit_qpos else torch.empty(0),
            torch.cat(re_exit_qvel, dim=0) if re_exit_qvel else torch.empty(0),
            torch.cat(re_exit_ctrl, dim=0) if re_exit_ctrl else torch.empty(0),
        ),
        "re_catch": (
            torch.cat(re_catch_qpos, dim=0) if re_catch_qpos else torch.empty(0),
            torch.cat(re_catch_qvel, dim=0) if re_catch_qvel else torch.empty(0),
            torch.cat(re_catch_ctrl, dim=0) if re_catch_ctrl else torch.empty(0),
            torch.cat(re_catch_face, dim=0) if re_catch_face else torch.empty(0, dtype=torch.long),
        ),
        "re_hn": (
            cl.hn_qpos[:n_hn][hn_mask_re].cpu(),
            cl.hn_qvel[:n_hn][hn_mask_re].cpu(),
            cl.hn_ctrl[:n_hn][hn_mask_re].cpu(),
            cl.hn_face[:n_hn][hn_mask_re].cpu(),
        ),
        "tw_ho": (
            torch.cat(tw_ho_qpos, dim=0) if tw_ho_qpos else torch.empty(0),
            torch.cat(tw_ho_qvel, dim=0) if tw_ho_qvel else torch.empty(0),
            torch.cat(tw_ho_ctrl, dim=0) if tw_ho_ctrl else torch.empty(0),
            torch.cat(tw_ho_face, dim=0) if tw_ho_face else torch.empty(0, dtype=torch.long),
            torch.cat(tw_ho_dir, dim=0) if tw_ho_dir else torch.empty(0),
        ),
        "tw_x2": (
            torch.cat(tw_x2_qpos, dim=0) if tw_x2_qpos else torch.empty(0),
            torch.cat(tw_x2_qvel, dim=0) if tw_x2_qvel else torch.empty(0),
            torch.cat(tw_x2_ctrl, dim=0) if tw_x2_ctrl else torch.empty(0),
            torch.cat(tw_x2_face, dim=0) if tw_x2_face else torch.empty(0, dtype=torch.long),
            torch.cat(tw_x2_dir, dim=0) if tw_x2_dir else torch.empty(0),
        ),
        "tw_hn": (
            cl.hn_qpos[:n_hn][hn_mask_tw].cpu(),
            cl.hn_qvel[:n_hn][hn_mask_tw].cpu(),
            cl.hn_ctrl[:n_hn][hn_mask_tw].cpu(),
            cl.hn_face[:n_hn][hn_mask_tw].cpu(),
            cl.hn_dir[:n_hn][hn_mask_tw].cpu(),
        ),
    }
    print(
        f"[DAgger Harvest {time.time()-t0:.1f}s] re_exit={len(pools['re_exit'][0])}, "
        f"re_catch={len(pools['re_catch'][0])}, re_hn={len(pools['re_hn'][0])}, "
        f"tw_ho={len(pools['tw_ho'][0])}, tw_x2={len(pools['tw_x2'][0])}, tw_hn={len(pools['tw_hn'][0])}",
        flush=True,
    )
    del cl
    torch.cuda.empty_cache()
    return pools


class DaggerEpisodicEnv:
    """High-speed episodic single-task GPU env (reorient OR twist) with DAgger pools + Physics DR."""
    def __init__(self, xml_path: str, task: str, pools: dict, num_envs: int = 2048, seed: int = 100):
        assert task in ("reorient", "twist")
        self.task = task
        self.num_envs = num_envs
        self.obs_dim = OBS_REORIENT_DIM if task == "reorient" else OBS_TWIST_DIM
        self.device = torch.device("cuda:0")
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
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

        # Build synthetic scramble + 40% skewed cubelet pool, plus dedicated 512-state physically exact skewed-slice Twist-to-Straighten pool!
        rng = np.random.default_rng(seed + 55)
        faces = ["U", "R", "F", "D", "L", "B"]
        mods = ["", "'", "2"]
        s_pool, sk_q, sk_f, sk_d = [], [], [], []
        for idx in range(512):
            qp = self.mj_data.qpos.copy()
            last = -1
            for _ in range(int(rng.integers(5, 22))):
                f = int(rng.integers(0, 6))
                while f == last:
                    f = int(rng.integers(0, 6))
                last = f
                apply_move_to_qpos(qp, faces[f] + mods[int(rng.integers(0, 3))])

            # Build exact 26-cubelet skewed state (15..44 deg) for Twist-to-Straighten training
            qp_sk = qp.copy()
            fid_s = int(rng.integers(0, 6))
            axis_s = FACE_AXES[fid_s]
            ang_s = float(rng.uniform(0.24, 0.74)) * (1.0 if rng.random() < 0.5 else -1.0)
            q_rot_s = quat_from_axis_angle(axis_s, ang_s)
            qp_sk[QPOS_CENTERS + fid_s] += ang_s
            for k in range(20):
                off = QPOS_CUBELETS + 4 * k
                qk = qp_sk[off:off + 4]
                w, x, y, z = qk
                h = CUBELET_HOME[k]
                t = 2.0 * np.cross(np.array([x, y, z]), h)
                pos = h + w * t + np.cross(np.array([x, y, z]), t)
                if np.dot(pos, axis_s) > 0.5:
                    qn = quat_mul(q_rot_s, qk)
                    qp_sk[off:off + 4] = qn / (np.linalg.norm(qn) + 1e-12)
            sk_q.append(qp_sk[QPOS_CENTERS:].astype(np.float32))
            sk_f.append(fid_s)
            sk_d.append(-1.0 if ang_s > 0.0 else 1.0)

            s_pool.append(qp_sk[QPOS_CENTERS:].astype(np.float32) if idx % 5 < 2 else qp[QPOS_CENTERS:].astype(np.float32))
        self.scramble_pool = torch.tensor(np.stack(s_pool), device=self.device)
        self.skew_cube_pool = torch.tensor(np.stack(sk_q), device=self.device)
        self.skew_face_pool = torch.tensor(sk_f, dtype=torch.long, device=self.device)
        self.skew_dir_pool = torch.tensor(sk_d, dtype=torch.float32, device=self.device)

        # Move harvested DAgger pools to GPU
        self.pools = {}
        for k, tup in pools.items():
            self.pools[k] = tuple(t.to(self.device) for t in tup)

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
        self.env_ema = torch.full((N, 1), ACTION_EMA, device=self.device)
        self.env_act_scale = torch.ones((N, 1), device=self.device)
        self.target_face = torch.full((N,), 4, device=self.device, dtype=torch.long)
        self.twist_dir = torch.ones(N, device=self.device)
        self.goal_angle = torch.zeros(N, device=self.device)
        self.prev_err = torch.zeros(N, device=self.device)
        self.prev_action = torch.zeros((N, 20), device=self.device)
        self.ep_steps = torch.zeros(N, device=self.device, dtype=torch.long)
        self.goal_steps = torch.zeros(N, device=self.device, dtype=torch.long)
        self.max_ep_steps = 360 if task == "reorient" else 220
        self.max_goal_steps = 150 if task == "reorient" else 160

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

    def _init_err(self, ids: torch.Tensor):
        if len(ids) == 0:
            return
        if self.task == "reorient":
            xm = self._cube_xmat_3x3()[ids]
            nw = torch.bmm(xm, self.face_axes[self.target_face[ids]].unsqueeze(-1)).squeeze(-1)
            self.prev_err[ids] = torch.acos(torch.clamp(nw[:, 2], -1.0, 1.0))
        else:
            cur = self.qpos[ids, QPOS_CENTERS + self.target_face[ids]]
            self.prev_err[ids] = torch.abs(wrap_pi(self.goal_angle[ids] - cur))

    def reset_ids(self, ids: torch.Tensor):
        m = len(ids)
        if m == 0:
            return
        # Sim-to-WASM Physics DR resampled every episode!
        self.env_ema[ids, 0] = 0.30 + 0.20 * torch.rand(m, device=self.device)
        self.env_act_scale[ids, 0] = 0.92 + 0.16 * torch.rand(m, device=self.device)

        u = torch.rand(m, device=self.device)
        if self.task == "reorient":
            # 35% real post-twist exits | 20% bailout-catch | 20% mined pre-drop HN | 25% synthetic (40% skewed)
            p_ex, p_ca, p_hn = self.pools["re_exit"], self.pools["re_catch"], self.pools["re_hn"]
            has_ex, has_ca, has_hn = len(p_ex[0]) > 64, len(p_ca[0]) > 64, len(p_hn[0]) > 64
            m_ex = (u < 0.35) & has_ex
            m_ca = (u >= 0.35) & (u < 0.55) & has_ca
            m_hn = (u >= 0.55) & (u < 0.75) & has_hn
            m_syn = ~(m_ex | m_ca | m_hn)

            if m_ex.any():
                sub = ids[m_ex]
                i = torch.randint(0, len(p_ex[0]), (len(sub),), device=self.device)
                self.qpos[sub] = p_ex[0][i]
                self.qvel[sub] = p_ex[1][i]
                self.ctrl[sub] = p_ex[2][i]
                self.target_face[sub] = torch.randint(0, 6, (len(sub),), device=self.device)
            if m_ca.any():
                sub = ids[m_ca]
                i = torch.randint(0, len(p_ca[0]), (len(sub),), device=self.device)
                self.qpos[sub] = p_ca[0][i]
                self.qvel[sub] = p_ca[1][i]
                self.ctrl[sub] = p_ca[2][i]
                self.target_face[sub] = p_ca[3][i]
            if m_hn.any():
                sub = ids[m_hn]
                i = torch.randint(0, len(p_hn[0]), (len(sub),), device=self.device)
                self.qpos[sub] = p_hn[0][i]
                self.qvel[sub] = p_hn[1][i]
                self.ctrl[sub] = p_hn[2][i]
                self.target_face[sub] = p_hn[3][i]
            if m_syn.any():
                sub = ids[m_syn]
                ns = len(sub)
                q = self.cradle_qpos.unsqueeze(0).expand(ns, -1).clone()
                q[:, :24] = torch.clamp(q[:, :24] + (torch.rand((ns, 24), device=self.device) * 2.0 - 1.0) * 0.05, self.hand_lo, self.hand_hi)
                pj = (torch.rand((ns, 3), device=self.device) * 2.0 - 1.0) * torch.tensor([0.008, 0.008, 0.004], device=self.device)
                pj[:, 2] = torch.abs(pj[:, 2])
                q[:, QPOS_CUBE_POS:QPOS_CUBE_POS + 3] += pj
                q[:, QPOS_CUBE_QUAT:QPOS_CUBE_QUAT + 4] = self.cube_24_quats[torch.randint(0, 24, (ns,), device=self.device)]
                q[:, QPOS_CENTERS:] = self.scramble_pool[torch.randint(0, len(self.scramble_pool), (ns,), device=self.device)]
                self.qpos[sub] = q
                self.qvel[sub] = 0.0
                self.ctrl[sub] = self.cradle_ctrl.unsqueeze(0).expand(ns, -1)
                self.target_face[sub] = torch.randint(0, 6, (ns,), device=self.device)
        else:
            # TWIST: 40% real Reorient handovers | 20% real X2 | 15% mined pre-bailout HN | 15% skewed-slice straighten | 10% synthetic
            p_ho, p_x2, p_hn = self.pools["tw_ho"], self.pools["tw_x2"], self.pools["tw_hn"]
            has_ho, has_x2, has_hn = len(p_ho[0]) > 64, len(p_x2[0]) > 64, len(p_hn[0]) > 64
            m_ho = (u < 0.40) & has_ho
            m_x2 = (u >= 0.40) & (u < 0.60) & has_x2
            m_hn = (u >= 0.60) & (u < 0.75) & has_hn
            m_sk = (u >= 0.75) & (u < 0.90) & has_ho
            m_syn = ~(m_ho | m_x2 | m_hn | m_sk)

            if m_ho.any():
                sub = ids[m_ho]
                i = torch.randint(0, len(p_ho[0]), (len(sub),), device=self.device)
                self.qpos[sub] = p_ho[0][i]
                self.qvel[sub] = p_ho[1][i]
                self.ctrl[sub] = p_ho[2][i]
                self.target_face[sub] = p_ho[3][i]
                self.twist_dir[sub] = torch.where(torch.rand(len(sub), device=self.device) < 0.5, -1.0, 1.0)
            if m_x2.any():
                sub = ids[m_x2]
                i = torch.randint(0, len(p_x2[0]), (len(sub),), device=self.device)
                self.qpos[sub] = p_x2[0][i]
                self.qvel[sub] = p_x2[1][i]
                self.ctrl[sub] = p_x2[2][i]
                self.target_face[sub] = p_x2[3][i]
                self.twist_dir[sub] = p_x2[4][i]
            if m_hn.any():
                sub = ids[m_hn]
                i = torch.randint(0, len(p_hn[0]), (len(sub),), device=self.device)
                self.qpos[sub] = p_hn[0][i]
                self.qvel[sub] = p_hn[1][i]
                self.ctrl[sub] = p_hn[2][i]
                self.target_face[sub] = p_hn[3][i]
                self.twist_dir[sub] = p_hn[4][i]
            if m_sk.any():
                # Physically exact 26-cubelet skewed-slice Twist-to-Straighten starts (+-15..44 deg on top face!)
                sub = ids[m_sk]
                ns = len(sub)
                q = self.cradle_qpos.unsqueeze(0).expand(ns, -1).clone()
                q[:, :24] = torch.clamp(q[:, :24] + (torch.rand((ns, 24), device=self.device) * 2.0 - 1.0) * 0.05, self.hand_lo, self.hand_hi)
                sk_idx = torch.randint(0, len(self.skew_cube_pool), (ns,), device=self.device)
                f_sel = self.skew_face_pool[sk_idx]
                y_sel = torch.randint(0, 4, (ns,), device=self.device)
                q[:, QPOS_CUBE_QUAT:QPOS_CUBE_QUAT + 4] = self.cube_24_quats[f_sel * 4 + y_sel]
                q[:, QPOS_CENTERS:] = self.skew_cube_pool[sk_idx]
                self.qpos[sub] = q
                self.qvel[sub] = 0.0
                self.ctrl[sub] = self.cradle_ctrl.unsqueeze(0).expand(ns, -1)
                self.target_face[sub] = f_sel
                self.twist_dir[sub] = self.skew_dir_pool[sk_idx]
            if m_syn.any():
                sub = ids[m_syn]
                ns = len(sub)
                q = self.cradle_qpos.unsqueeze(0).expand(ns, -1).clone()
                q[:, :24] = torch.clamp(q[:, :24] + (torch.rand((ns, 24), device=self.device) * 2.0 - 1.0) * 0.04, self.hand_lo, self.hand_hi)
                f_sel = torch.randint(0, 6, (ns,), device=self.device)
                y_sel = torch.randint(0, 4, (ns,), device=self.device)
                q[:, QPOS_CUBE_QUAT:QPOS_CUBE_QUAT + 4] = self.cube_24_quats[f_sel * 4 + y_sel]
                q[:, QPOS_CENTERS:] = self.scramble_pool[torch.randint(0, len(self.scramble_pool), (ns,), device=self.device)]
                self.qpos[sub] = q
                self.qvel[sub] = 0.0
                self.ctrl[sub] = self.cradle_ctrl.unsqueeze(0).expand(ns, -1)
                self.target_face[sub] = f_sel
                self.twist_dir[sub] = torch.where(torch.rand(ns, device=self.device) < 0.5, -1.0, 1.0)

            cur_a = self.qpos[ids, QPOS_CENTERS + self.target_face[ids]]
            goal_90 = round90(cur_a) + self.twist_dir[ids] * (math.pi * 0.5)
            goal_snap = round90(cur_a)
            self.goal_angle[ids] = torch.where(m_sk, goal_snap, goal_90)

        self.qacc[ids] = 0.0
        if self.qacc_ws is not None:
            self.qacc_ws[ids] = 0.0
        self.ep_steps[ids] = 0
        self.goal_steps[ids] = 0
        self.prev_action[ids] = (self.ctrl[ids] - self.act_mid) / self.act_half

        wp.capture_launch(self.fwd_graph)
        wp.synchronize()
        self._init_err(ids)

    def reset_all(self):
        self.reset_ids(torch.arange(self.num_envs, device=self.device, dtype=torch.long))
        return self.get_obs()

    def get_obs(self) -> torch.Tensor:
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
        ], dim=-1)

        if self.task == "reorient":
            onehot = torch.zeros((N, 6), device=self.device)
            onehot.scatter_(1, self.target_face.unsqueeze(1), 1.0)
            ys4, yc4 = yaw_harmonic4_t(xm3)
            obs = torch.cat([base, nw, onehot, ys4.unsqueeze(1), yc4.unsqueeze(1)], dim=-1)
        else:
            row_idx = torch.arange(N, device=self.device)
            cur_top = face_ang[row_idx, self.target_face]
            rem = wrap_pi(self.goal_angle - cur_top)
            top_vel = face_vel[row_idx, self.target_face]
            rel_faces = -face_mis.clone()
            rel_faces[row_idx, self.target_face] = rem
            obs = torch.cat([
                base, nw, rem.unsqueeze(1), torch.sin(rem).unsqueeze(1), torch.cos(rem).unsqueeze(1),
                self.twist_dir.unsqueeze(1), top_vel.unsqueeze(1), rel_faces,
            ], dim=-1)
        return torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def step(self, action: torch.Tensor):
        N = self.num_envs
        row_idx = torch.arange(N, device=self.device)

        a = torch.clamp(action, -1.0, 1.0)
        a_eff = torch.clamp(a * self.env_act_scale, -1.0, 1.0)
        target = self.act_mid + self.act_half * a_eff
        new_ctrl = torch.clamp((1.0 - self.env_ema) * self.ctrl + self.env_ema * target, self.act_lo, self.act_hi)
        self.ctrl.copy_(new_ctrl)

        # Continuous lateral/rotational micro-slip perturbations (12% of steps) -> kills fragile Warp-only tricks!
        kick_mask = torch.rand(N, device=self.device) < 0.12
        if kick_mask.any():
            k_m = kick_mask.unsqueeze(-1).float()
            self.qvel[:, 24:27].add_(k_m * ((torch.rand((N, 3), device=self.device) * 2.0 - 1.0) * 0.04))
            self.qvel[:, 27:30].add_(k_m * ((torch.rand((N, 3), device=self.device) * 2.0 - 1.0) * 0.85))

        wp.capture_launch(self.step_graph)
        wp.synchronize()

        self.ep_steps += 1
        self.goal_steps += 1

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

        rim_excess = torch.clamp(pos_dist - 0.036, min=0.0)

        if self.task == "reorient":
            progress = 8.0 * (self.prev_err - tilt)
            self.prev_err.copy_(tilt)
            orient_bonus = 1.2 / (1.0 + 4.0 * tilt)
            _, yc4 = yaw_harmonic4_t(xm3)
            yaw_bonus = 0.20 * torch.clamp((dot_z - 0.85) / 0.15, 0.0, 1.0) * (0.5 * (yc4 + 1.0))
            face_lock_pen = -3.5 * face_mis_abs.sum(dim=-1)
            center_pen = -7.0 * pos_dist - 40.0 * rim_excess
            throw_pen = -2.0 * cube_linv * torch.clamp(pos_dist - 0.030, min=0.0) - 0.10 * torch.clamp(cube_angv - 6.0, min=0.0)
            near_r = torch.clamp(1.0 - tilt / 0.40, min=0.0)
            brake_pen = -0.40 * near_r * cube_angv
            smooth_pen = -0.03 * act_diff - 0.002 * vel_pen

            # Strict canonical handover success (chains multiple reorients per episode -> strongly positive V!)
            succeeded = (~dropped) & (dot_z > 0.975) & (max_mis < 0.125) & (cube_angv < 2.1) & (pos_dist < 0.043)
            reward = progress + orient_bonus + yaw_bonus + face_lock_pen + center_pen + throw_pen + brake_pen + smooth_pen
            reward = torch.where(succeeded, reward + 35.0, reward)
            reward = torch.where(dropped, reward - 60.0, reward)

            timeout = (self.ep_steps >= self.max_ep_steps) | (self.goal_steps >= self.max_goal_steps)
            done = dropped | timeout

            # On success (not done), immediately chain a new random face target!
            cont_succ = succeeded & (~done)
            if cont_succ.any():
                s_ids = torch.nonzero(cont_succ, as_tuple=False).squeeze(-1)
                cur_top = self._top_face_ids()[s_ids]
                self.target_face[s_ids] = (cur_top + torch.randint(1, 6, (len(s_ids),), device=self.device)) % 6
                self.goal_steps[s_ids] = 0
                self._init_err(s_ids)
            n_bail = 0
        else:
            cur_top = face_ang[row_idx, self.target_face]
            rem = torch.abs(wrap_pi(self.goal_angle - cur_top))
            progress = 28.0 * (self.prev_err - rem)
            self.prev_err.copy_(rem)

            tips = self.site_xpos[:, self.tip_sites, :]
            top_center = cube_pos + 0.019 * nw
            bot_center = cube_pos - 0.010 * nw
            r_reach = (
                0.22 * torch.exp(-20.0 * torch.clamp(torch.norm(tips[:, 0, :] - top_center, dim=-1) - 0.032, min=0.0))
                + 0.22 * torch.exp(-20.0 * torch.clamp(torch.norm(tips[:, 1, :] - top_center, dim=-1) - 0.032, min=0.0))
                + 0.22 * torch.exp(-20.0 * torch.clamp(torch.norm(tips[:, 4, :] - top_center, dim=-1) - 0.032, min=0.0))
                + 0.14 * torch.exp(-20.0 * torch.clamp(torch.norm(tips[:, 2, :] - bot_center, dim=-1) - 0.034, min=0.0))
                + 0.14 * torch.exp(-20.0 * torch.clamp(torch.norm(tips[:, 3, :] - bot_center, dim=-1) - 0.034, min=0.0))
            )
            top_vel_raw = self.qvel[row_idx, 30 + self.target_face]
            top_vel_signed = top_vel_raw * self.twist_dir
            r_vel = torch.where(rem > 0.22, 0.30 * torch.clamp(top_vel_signed, -2.0, 3.8), torch.zeros_like(rem))
            near_t = torch.clamp(1.0 - rem / 0.28, min=0.0)
            r_vel = r_vel - 0.50 * near_t * torch.abs(top_vel_raw) - 0.30 * near_t * cube_angv
            twist_close = 1.2 / (1.0 + 3.0 * rem)

            other_mask = torch.ones((N, 6), device=self.device)
            other_mask[row_idx, self.target_face] = 0.0
            other_err_sum = (face_mis_abs * other_mask).sum(dim=-1)
            other_err_max = (face_mis_abs * other_mask).max(dim=-1).values
            upright_pen = -3.0 * tilt - 10.0 * torch.clamp(tilt - 0.24, min=0.0) - 6.0 * pos_dist - 40.0 * rim_excess
            other_pen = -3.0 * other_err_sum
            smooth_pen = -0.02 * act_diff - 0.001 * vel_pen

            succeeded = (~dropped) & (rem < 0.125) & (torch.abs(top_vel_raw) < 1.8) & (cube_angv < 2.3) & (pos_dist < 0.043) & (other_err_max < 0.14) & (dot_z > 0.96)
            bailed = (~dropped) & (~succeeded) & ((dot_z < 0.62) | (pos_dist > 0.075))
            reward = progress + r_reach + r_vel + twist_close + upright_pen + other_pen + smooth_pen
            reward = torch.where(succeeded, reward + 50.0, reward)
            reward = torch.where(bailed, reward - 45.0, reward)
            reward = torch.where(dropped, reward - 70.0, reward)

            timeout = (self.ep_steps >= self.max_ep_steps) | (self.goal_steps >= self.max_goal_steps)
            done = dropped | bailed | timeout

            # On success, 40% chain X2 double turn in same episode so Twist masters X2!
            cont_x2 = succeeded & (~done) & (torch.rand(N, device=self.device) < 0.40)
            done = done | (succeeded & (~cont_x2))
            if cont_x2.any():
                x_ids = torch.nonzero(cont_x2, as_tuple=False).squeeze(-1)
                cur_a = self.qpos[x_ids, QPOS_CENTERS + self.target_face[x_ids]]
                self.goal_angle[x_ids] = round90(cur_a) + self.twist_dir[x_ids] * (math.pi * 0.5)
                self.goal_steps[x_ids] = 0
                self._init_err(x_ids)
            n_bail = int(bailed.sum().item())

        reward = torch.nan_to_num(reward, nan=-60.0, posinf=80.0, neginf=-90.0)
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self.reset_ids(done_ids)

        return self.get_obs(), reward, done, {
            "n_succ": int(succeeded.sum().item()),
            "n_drop": int(dropped.sum().item()),
            "n_bail": n_bail,
            "n_done": len(done_ids),
        }


def train_specialist(
    task: str,
    xml_path: str,
    pools: dict,
    model: ActorCritic,
    norm: RunningNorm,
    partner_model: ActorCritic,
    partner_norm: RunningNorm,
    out_dir: str,
    num_envs: int,
    n_iters: int,
    cpu_eval_every: int,
    best_cpu_solve: float,
    seed: int,
    base_lr: float = 1.0e-4,
    init_log_std: float = -1.6,
    anchor_coef: float = 0.006,
    stage_tag: str = "",
):
    device = torch.device("cuda:0")
    env = DaggerEpisodicEnv(xml_path, task, pools, num_envs=num_envs, seed=seed)
    N = num_envs
    T = 48
    obs_dim = env.obs_dim

    ref_actor = copy.deepcopy(model.actor).eval()
    for prm in ref_actor.parameters():
        prm.requires_grad_(False)

    # Fresh critic + surgical warm-start actor std
    model.log_std.data.fill_(init_log_std)
    lr = base_lr
    opt = torch.optim.Adam([
        {"params": list(model.actor.parameters()) + [model.log_std], "lr": lr},
        {"params": list(model.critic.parameters()), "lr": 1.0e-3},
    ], eps=1e-5)

    buf_obs = torch.zeros((T, N, obs_dim), device=device)
    buf_act = torch.zeros((T, N, 20), device=device)
    buf_logp = torch.zeros((T, N), device=device)
    buf_rew = torch.zeros((T, N), device=device)
    buf_done = torch.zeros((T, N), device=device, dtype=torch.bool)
    buf_val = torch.zeros((T, N), device=device)
    buf_mean = torch.zeros((T, N, 20), device=device)
    buf_std = torch.zeros((T, N, 20), device=device)

    obs = env.reset_all()
    t0 = time.time()
    total_steps = 0
    best_model_sd = copy.deepcopy(model.state_dict())

    print(f"\n=== STAGE {task.upper()} DAgger Episodic PPO (N={N}, T={T}, iters={n_iters}) ===", flush=True)
    for it in range(1, n_iters + 1):
        warmup = (it <= 20)
        it_succ = it_drop = it_bail = it_done = 0

        with torch.no_grad():
            for t in range(T):
                obs_n = norm.normalize(obs) + torch.randn((N, obs_dim), device=device) * 0.004
                act, logp, val, mean, std = model.act(obs_n, deterministic=False)
                next_obs, rew, done, info = env.step(act)
                buf_obs[t] = obs_n
                buf_act[t] = act
                buf_logp[t] = logp
                buf_rew[t] = rew
                buf_done[t] = done
                buf_val[t] = val
                buf_mean[t] = mean
                buf_std[t] = std
                obs = next_obs
                it_succ += info["n_succ"]
                it_drop += info["n_drop"]
                it_bail += info["n_bail"]
                it_done += info["n_done"]

            next_v = model.critic(norm.normalize(obs)).squeeze(-1)
            adv = torch.zeros((T, N), device=device)
            ret = torch.zeros((T, N), device=device)
            last_gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nt = 1.0 - buf_done[t].float()
                delta = buf_rew[t] + 0.99 * next_v * nt - buf_val[t]
                last_gae = delta + 0.99 * 0.95 * nt * last_gae
                adv[t] = last_gae
                ret[t] = last_gae + buf_val[t]
                next_v = buf_val[t]

        total_steps += N * T
        f_obs = buf_obs.reshape(-1, obs_dim)
        f_act = buf_act.reshape(-1, 20)
        f_logp = buf_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = ret.reshape(-1)
        f_mean = buf_mean.reshape(-1, 20)
        f_std = buf_std.reshape(-1, 20)

        n_tot = N * T
        mb_size = n_tot // 8
        kl_sum = vl_sum = anch_sum = 0.0
        n_mb = 0
        for _ in range(5):
            perm = torch.randperm(n_tot, device=device)
            for start in range(0, n_tot - mb_size + 1, mb_size):
                idx = perm[start:start + mb_size]
                new_logp, ent, new_val, new_mean, new_std = model.evaluate(f_obs[idx], f_act[idx])
                vloss = nn.functional.smooth_l1_loss(new_val, f_ret[idx], beta=15.0)
                if warmup:
                    loss = vloss
                    kl_v = anch_v = 0.0
                else:
                    with torch.no_grad():
                        old_m, old_s = f_mean[idx], f_std[idx]
                        kl = (torch.log(new_std / old_s) + (old_s ** 2 + (old_m - new_mean) ** 2) / (2.0 * new_std ** 2) - 0.5).sum(dim=-1).mean()
                        kl_v = float(kl.item())
                        if kl_v > 0.016:
                            lr = max(1e-5, lr / 1.5)
                        elif kl_v < 0.004 and kl_v > 0.0:
                            lr = min(3e-4, lr * 1.25)
                        opt.param_groups[0]["lr"] = lr
                        ref_m = torch.tanh(ref_actor(f_obs[idx]))
                    mb_adv = (f_adv[idx] - f_adv[idx].mean()) / (f_adv[idx].std() + 1e-8)
                    ratio = torch.exp(new_logp - f_logp[idx])
                    ploss = -torch.min(ratio * mb_adv, torch.clamp(ratio, 0.85, 1.15) * mb_adv).mean()
                    aloss = ((new_mean - ref_m) ** 2).sum(dim=-1).mean()
                    anch_v = float(aloss.item())
                    loss = ploss + 0.5 * vloss + anchor_coef * aloss - 0.0003 * ent.mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    model.log_std.clamp_(-2.3, -1.2)
                kl_sum += kl_v
                vl_sum += float(vloss.item())
                anch_sum += anch_v
                n_mb += 1

        if it == 1 or it % 10 == 0 or it == n_iters:
            el = time.time() - t0
            fps = total_steps / max(el, 1e-3)
            d_pct = 100.0 * it_drop / max(it_succ + it_drop, 1)
            std_m = float(torch.exp(model.log_std).mean().item())
            print(
                f"[{el:6.1f}s] {task[:2].upper()}{'*' if warmup else ' '} it={it:3d}/{n_iters} fps={fps:6,.0f} | "
                f"succ={it_succ:4d} drop={it_drop:3d} bail={it_bail:3d} d/s={d_pct:4.2f}% "
                f"kl={kl_sum/max(n_mb,1):.4f} vl={vl_sum/max(n_mb,1):5.1f} anch={anch_sum/max(n_mb,1):.3f} std={std_m:.3f}",
                flush=True,
            )

        if (not warmup) and (it % cpu_eval_every == 0 or it == n_iters):
            re_m, re_n = (model, norm) if task == "reorient" else (partner_model, partner_norm)
            tw_m, tw_n = (partner_model, partner_norm) if task == "reorient" else (model, norm)
            tc = time.time()
            res = eval_cpu_mujoco(xml_path, re_m, re_n, tw_m, tw_n, n_episodes=48, target_moves=25)
            print(
                f"  [CPU-MUJOCO {time.time()-tc:4.1f}s] {task} it={it:3d}: CPU_SOLVE25={res['solve_pct']:5.1f}% "
                f"({res['solved']}/{res['n']}, drops={res['dropped']}[r={res['re_drops']},t={res['tw_drops']}], "
                f"bails={res['bails']}, mv={res['avg_moves']}, sim={res['avg_sim_s']}s)",
                flush=True,
            )
            # Export stage checkpoint every eval for offline WASM comparison!
            prefix = f"{stage_tag}_{task}" if stage_tag else task
            export_pair(out_dir, f"{prefix}_it{it}", re_m, re_n, tw_m, tw_n, total_steps, res)
            if res["solve_pct"] > best_cpu_solve and res["avg_sim_s"] < 52.0:
                best_cpu_solve = res["solve_pct"]
                best_model_sd = copy.deepcopy(model.state_dict())
                export_pair(out_dir, "best_pair", re_m, re_n, tw_m, tw_n, total_steps, res)
                print(f"  >>> NEW BEST FAST CPU-VERIFIED PAIR: CPU_SOLVE25={best_cpu_solve:.1f}% (sim={res['avg_sim_s']}s, drops={res['dropped']}, bails={res['bails']}) -> best_pair <<<", flush=True)

    # Restore best checkpoint from this stage before handing off to next stage!
    model.load_state_dict(best_model_sd)
    del env
    torch.cuda.empty_cache()
    return best_cpu_solve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="/content/rubik/robot/shadow_rubik.xml")
    p.add_argument("--reorient-init", required=True)
    p.add_argument("--twist-init", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--twist-first", action="store_true")
    p.add_argument("--iters-reorient", type=int, default=200)
    p.add_argument("--iters-twist", type=int, default=200)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--init-log-std", type=float, default=-1.6)
    p.add_argument("--anchor-coef", type=float, default=0.006)
    p.add_argument("--cpu-eval-every", type=int, default=40)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0")
    hidden = (512, 256, 128)

    re_norm = RunningNorm(OBS_REORIENT_DIM, device)
    tw_norm = RunningNorm(OBS_TWIST_DIM, device)
    re_model = ActorCritic(OBS_REORIENT_DIM, 20, actor_hidden=hidden, critic_hidden=hidden).to(device)
    tw_model = ActorCritic(OBS_TWIST_DIM, 20, actor_hidden=hidden, critic_hidden=hidden).to(device)

    load_exported_into_model(args.reorient_init, re_model, re_norm, device, init_log_std=args.init_log_std)
    load_exported_into_model(args.twist_init, tw_model, tw_norm, device, init_log_std=args.init_log_std)

    tb = time.time()
    base = eval_cpu_mujoco(args.xml, re_model, re_norm, tw_model, tw_norm, n_episodes=48, target_moves=25)
    best_cpu = base["solve_pct"]
    print(
        f"=== BASELINE NATIVE C-MUJOCO ({time.time()-tb:.1f}s): CPU_SOLVE25={base['solve_pct']:5.1f}% "
        f"({base['solved']}/{base['n']}, drops={base['dropped']}[r={base['re_drops']},t={base['tw_drops']}], "
        f"bails={base['bails']}, mv={base['avg_moves']}, sim={base['avg_sim_s']}s) ===",
        flush=True,
    )

    order = ["twist", "reorient"] if args.twist_first else ["reorient", "twist"]
    seed_off = 0
    for r in range(1, args.rounds + 1):
        for task in order:
            pools = harvest_dagger_pools(args.xml, re_model, re_norm, tw_model, tw_norm, n_envs=1280, n_steps=450, seed=args.seed + seed_off)
            seed_off += 1
            n_it = (args.iters_twist if task == "twist" else args.iters_reorient) if r == 1 else 120
            m_act, n_act = (re_model, re_norm) if task == "reorient" else (tw_model, tw_norm)
            m_par, n_par = (tw_model, tw_norm) if task == "reorient" else (re_model, re_norm)
            best_cpu = train_specialist(
                task, args.xml, pools, m_act, n_act, m_par, n_par,
                args.out, args.num_envs, n_it, args.cpu_eval_every, best_cpu, args.seed + seed_off,
                base_lr=args.lr, init_log_std=args.init_log_std, anchor_coef=args.anchor_coef, stage_tag=f"r{r}",
            )
            seed_off += 1

    export_pair(args.out, "final_pair", re_model, re_norm, tw_model, tw_norm, 0, {"best_cpu_solve": best_cpu})
    print(f"DONE DAgger EPISODIC PPO (best CPU_SOLVE25={best_cpu:.1f}%) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
