"""Parallel native C-MuJoCo 25-move closed-loop evaluator (exact physics parity with MuJoCo WASM).

Uses pure NumPy actor inference inside spawned worker processes so it never conflicts with PyTorch/CUDA
and measures true zero-gap CPU/WASM transfer live during GPU training!
"""
import concurrent.futures
import math
import multiprocessing as mp
import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube_state import (
    FACE_AXES, apply_move_to_qpos, QPOS_CUBE_POS, QPOS_CUBE_QUAT, QPOS_CENTERS,
)

SUBSTEPS = 5
ACTION_EMA = 0.4
TIP_NAMES = ["tip_ff", "tip_mf", "tip_rf", "tip_lf", "tip_th"]
CRADLE_CTRL = np.array([
    0.0, 0.0,
    0.3, 0.9, 0.0, 0.2, 0.3,
    0.0, 0.5, 0.8,
    0.0, 0.5, 0.8,
    0.0, 0.5, 0.8,
    0.2, 0.0, 0.5, 0.8,
], dtype=np.float64)
FACES = ["U", "R", "F", "D", "L", "B"]
MODS = ["", "'", "2"]


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _wrap90(a: float) -> float:
    h = math.pi * 0.5
    return (a + math.pi * 0.25) % h - math.pi * 0.25


def _round90(a: float) -> float:
    h = math.pi * 0.5
    return round(a / h) * h


def _mlp_act(obs: np.ndarray, params: dict) -> np.ndarray:
    x = np.clip((obs - params["mean"]) / params["std"], -10.0, 10.0)
    ws = params["W"]
    bs = params["b"]
    for i in range(len(ws) - 1):
        x = x @ ws[i].T + bs[i]
        # ELU
        x = np.where(x >= 0.0, x, np.expm1(np.minimum(x, 0.0)))
    x = x @ ws[-1].T + bs[-1]
    return np.tanh(x)


def extract_numpy_policy(model, norm) -> dict:
    """Snapshot ActorCritic actor weights + RunningNorm into plain NumPy arrays (picklable)."""
    mean = norm.mean_np()
    std = norm.std_np()
    ws, bs = [], []
    sd = model.actor.state_dict()
    for k in sd.keys():
        if k.endswith(".weight"):
            bk = k[:-7] + ".bias"
            ws.append(sd[k].detach().cpu().numpy().astype(np.float32))
            bs.append(sd[bk].detach().cpu().numpy().astype(np.float32))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "W": ws, "b": bs}


def _eval_worker(args):
    xml_path, re_params, tw_params, seeds, target_moves = args
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    cube_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube_core")
    grasp_site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    tip_sites = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in TIP_NAMES]
    act_lo = mj_model.actuator_ctrlrange[:, 0].copy()
    act_hi = mj_model.actuator_ctrlrange[:, 1].copy()
    act_mid = 0.5 * (act_lo + act_hi)
    act_half = 0.5 * (act_hi - act_lo)

    # Build clean cradle qpos
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.ctrl[:] = CRADLE_CTRL
    mj_data.qpos[:7] = CRADLE_CTRL[:7]
    mj_data.qpos[19] = CRADLE_CTRL[16]
    for q_off, a_off in [(7, 7), (11, 10), (15, 13), (20, 17)]:
        mj_data.qpos[q_off] = CRADLE_CTRL[a_off]
        mj_data.qpos[q_off + 1] = CRADLE_CTRL[a_off + 1]
        mj_data.qpos[q_off + 2] = 0.5 * CRADLE_CTRL[a_off + 2]
        mj_data.qpos[q_off + 3] = 0.5 * CRADLE_CTRL[a_off + 2]
    cradle_qpos = mj_data.qpos.copy()

    obs_re = np.zeros(140, dtype=np.float32)
    obs_tw = np.zeros(143, dtype=np.float32)
    face_axes = np.asarray(FACE_AXES, dtype=np.float64)

    solved = 0
    dropped = 0
    re_drops = 0
    tw_drops = 0
    bails = 0
    moves_sum = 0
    steps_solved = 0

    for seed in seeds:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:] = cradle_qpos
        mj_data.qvel[:] = 0.0
        mj_data.ctrl[:] = CRADLE_CTRL
        last = -1
        for _ in range(18):
            f = int(rng.integers(0, 6))
            while f == last:
                f = int(rng.integers(0, 6))
            last = f
            apply_move_to_qpos(mj_data.qpos, FACES[f] + MODS[int(rng.integers(0, 3))])
        mujoco.mj_forward(mj_model, mj_data)

        phase = 0
        target_face = int(rng.integers(0, 6))
        twist_dir = -1.0 if rng.random() < 0.5 else 1.0
        goal_angle = 0.0
        phase_steps = 0
        moves_done = 0
        unjam = False

        for ep_step in range(1, 4001):
            qpos = mj_data.qpos
            qvel = mj_data.qvel
            ctrl = mj_data.ctrl
            xm = mj_data.xmat[cube_body].reshape(3, 3)
            cpos = qpos[QPOS_CUBE_POS:QPOS_CUBE_POS + 3]
            gpos = mj_data.site_xpos[grasp_site]
            cq = qpos[QPOS_CUBE_QUAT:QPOS_CUBE_QUAT + 4]
            if cq[0] < 0.0:
                cq = -cq
            fang = qpos[QPOS_CENTERS:QPOS_CENTERS + 6]
            fmis = np.array([_wrap90(float(a)) for a in fang], dtype=np.float32)
            fvel = np.clip(qvel[30:36] * 0.2, -5.0, 5.0).astype(np.float32)
            nw = xm @ face_axes[target_face]

            # Fill shared 129D base
            base = obs_re if phase == 0 else obs_tw
            base[0:24] = qpos[:24]
            base[24:48] = np.clip(qvel[:24] * 0.1, -5.0, 5.0)
            base[48:68] = ctrl[:20]
            k = 68
            for ts in tip_sites:
                rel = (mj_data.site_xpos[ts] - cpos) * 10.0
                base[k:k + 3] = rel
                k += 3
            base[83:86] = (cpos - gpos) * 10.0
            base[86:89] = np.clip(qvel[24:27], -5.0, 5.0)
            base[89:93] = cq
            base[93:96] = np.clip(qvel[27:30] * 0.2, -5.0, 5.0)
            base[96:105] = xm.reshape(9)
            base[105:111] = fmis
            base[111:117] = np.sin(4.0 * fang)
            base[117:123] = np.cos(4.0 * fang)
            base[123:129] = fvel
            base[129:132] = nw

            if phase == 0:
                base[132:138] = 0.0
                base[132 + target_face] = 1.0
                rx = xm[0]
                ry = xm[1]
                rx2 = rx * rx
                ry2 = ry * ry
                c4 = 0.5 * float(np.sum(rx2 * rx2 + ry2 * ry2 - 6.0 * rx2 * ry2))
                s4 = 2.0 * float(np.sum(rx2 * rx * ry - rx * ry2 * ry))
                base[138] = s4
                base[139] = c4
                act = _mlp_act(obs_re, re_params)
            else:
                rem = _wrap_pi(goal_angle - float(fang[target_face]))
                base[132] = rem
                base[133] = math.sin(rem)
                base[134] = math.cos(rem)
                base[135] = twist_dir
                base[136] = fvel[target_face]
                base[137:143] = -fmis
                base[137 + target_face] = rem
                act = _mlp_act(obs_tw, tw_params)

            target_ctrl = act_mid + act_half * act
            mj_data.ctrl[:] = np.clip((1.0 - ACTION_EMA) * mj_data.ctrl + ACTION_EMA * target_ctrl, act_lo, act_hi)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(mj_model, mj_data)
            phase_steps += 1

            cpos = mj_data.qpos[QPOS_CUBE_POS:QPOS_CUBE_POS + 3]
            gpos = mj_data.site_xpos[grasp_site]
            pos_dist = float(np.linalg.norm(cpos - gpos))
            if pos_dist > 0.14 or cpos[2] < -0.05 or not math.isfinite(pos_dist):
                dropped += 1
                if phase == 0:
                    re_drops += 1
                else:
                    tw_drops += 1
                break

            xm = mj_data.xmat[cube_body].reshape(3, 3)
            nw = xm @ face_axes[target_face]
            dot_z = float(np.clip(nw[2], -1.0, 1.0))
            angv = float(np.linalg.norm(mj_data.qvel[27:30]))
            fang = mj_data.qpos[QPOS_CENTERS:QPOS_CENTERS + 6]
            fmis_abs = [abs(_wrap90(float(a))) for a in fang]
            max_mis = max(fmis_abs)

            if phase == 0:
                if unjam:
                    if max_mis < 0.14 or phase_steps > 60:
                        unjam = False
                        cur_top = int(np.argmax(xm[2] @ face_axes.T))
                        target_face = (cur_top + int(rng.integers(1, 6))) % 6
                        phase_steps = 0
                    continue
                dot_req = 0.94 if phase_steps > 120 else 0.975
                mis_req = 0.18 if phase_steps > 120 else 0.13
                ang_req = 4.0 if phase_steps > 120 else 2.2
                pos_req = 0.065 if phase_steps > 120 else 0.045
                if dot_z > dot_req and max_mis < mis_req and angv < ang_req and pos_dist < pos_req:
                    phase = 1
                    phase_steps = 0
                    goal_angle = _round90(float(fang[target_face])) + twist_dir * (math.pi * 0.5)
                elif phase_steps > 200:
                    if dot_z > 0.88 and max_mis < 0.24:
                        phase = 1
                        phase_steps = 0
                        goal_angle = _round90(float(fang[target_face])) + twist_dir * (math.pi * 0.5)
                    else:
                        target_face = (target_face + 2) % 6
                        unjam = True
                        phase_steps = 0
            else:
                rem = abs(_wrap_pi(goal_angle - float(fang[target_face])))
                tv = abs(float(mj_data.qvel[30 + target_face]))
                other_max = max(fmis_abs[i] for i in range(6) if i != target_face)
                rem_req = 0.16 if phase_steps > 140 else 0.13
                tv_req = 2.8 if phase_steps > 140 else 1.8
                ang_req = 4.0 if phase_steps > 140 else 2.5
                pos_req = 0.065 if phase_steps > 140 else 0.045
                dot_req = 0.90 if phase_steps > 140 else 0.955
                if rem < rem_req and tv < tv_req and angv < ang_req and pos_dist < pos_req and other_max < 0.15 and dot_z > dot_req:
                    moves_done += 1
                    if moves_done >= target_moves:
                        solved += 1
                        steps_solved += ep_step
                        break
                    if rng.random() < 0.20:
                        phase = 1
                        phase_steps = 0
                        goal_angle = _round90(float(fang[target_face])) + twist_dir * (math.pi * 0.5)
                    else:
                        phase = 0
                        phase_steps = 0
                        cur_top = int(np.argmax(xm[2] @ face_axes.T))
                        target_face = (cur_top + int(rng.integers(1, 6))) % 6
                        twist_dir = -1.0 if rng.random() < 0.5 else 1.0
                elif dot_z < 0.62 or pos_dist > 0.075 or phase_steps > 220:
                    bails += 1
                    phase = 0
                    phase_steps = 0
                    unjam = False

        moves_sum += moves_done

    return solved, dropped, re_drops, tw_drops, bails, moves_sum, steps_solved


def eval_cpu_mujoco(
    xml_path: str,
    re_model,
    re_norm,
    tw_model,
    tw_norm,
    n_episodes: int = 48,
    target_moves: int = 25,
    base_seed: int = 1000,
    max_workers: int | None = None,
) -> dict:
    """Evaluate policy pair on native C MuJoCo (exact WASM physics parity) across parallel CPU workers."""
    re_params = extract_numpy_policy(re_model, re_norm)
    tw_params = extract_numpy_policy(tw_model, tw_norm)
    n_workers = max_workers or min(8, os.cpu_count() or 4)
    all_seeds = [base_seed + i for i in range(n_episodes)]
    chunks = [all_seeds[i::n_workers] for i in range(n_workers) if all_seeds[i::n_workers]]

    ctx = mp.get_context("spawn")
    solved = dropped = re_drops = tw_drops = bails = moves_sum = steps_solved = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as ex:
        futs = [ex.submit(_eval_worker, (xml_path, re_params, tw_params, ch, target_moves)) for ch in chunks]
        for fut in futs:
            s, d, rd, td, b, ms, ss = fut.result()
            solved += s
            dropped += d
            re_drops += rd
            tw_drops += td
            bails += b
            moves_sum += ms
            steps_solved += ss

    return {
        "n": n_episodes,
        "solved": solved,
        "solve_pct": round(100.0 * solved / max(n_episodes, 1), 1),
        "dropped": dropped,
        "re_drops": re_drops,
        "tw_drops": tw_drops,
        "bails": bails,
        "avg_moves": round(moves_sum / max(n_episodes, 1), 1),
        "avg_sim_s": round(0.025 * steps_solved / max(solved, 1), 1) if solved > 0 else 0.0,
    }
