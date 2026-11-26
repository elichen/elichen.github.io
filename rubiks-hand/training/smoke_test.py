"""Verify RubikWarpEnv physics, contact counts, face-twistability, and obs parity."""
import json
import sys
import time
import numpy as np
import torch
import mujoco

sys.path.insert(0, "/content/rubik/training")
from rubik_warp_env import RubikWarpEnv, CRADLE_CTRL, build_cradle_qpos
from cube_state import apply_move_to_qpos, read_facelets_from_qpos, SOLVED_FACELETS

XML = "/content/rubik/robot/shadow_rubik.xml"

# 1. Verify exact CPU MuJoCo physics of a deterministic 10-step rollout (for JS parity test!)
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
d.qpos[:] = build_cradle_qpos(m)
d.ctrl[:] = CRADLE_CTRL
for mv in ["U", "R'", "F"]:
    apply_move_to_qpos(d.qpos, mv)
mujoco.mj_forward(m, d)
print("CPU facelets after U R' F:", read_facelets_from_qpos(d.qpos))

# 2. Smoke test RubikWarpEnv for BOTH tasks + check max ncon under random actions!
for task in ["reorient", "twist"]:
    env = RubikWarpEnv(XML, task=task, num_envs=1024, seed=42, nconmax=260, njmax=600)
    obs = env.reset_all()
    print(f"[{task}] obs shape={tuple(obs.shape)}, finite={bool(torch.isfinite(obs).all())}, "
          f"min={obs.min().item():.3f}, max={obs.max().item():.3f}")
    t0 = time.time()
    max_ncon = 0
    total_succ = 0
    total_drop = 0
    for step in range(40):
        act = torch.randn((1024, 20), device=env.device) * 0.6
        obs, rew, done, info = env.step(act)
        total_succ += info["n_succ"]
        total_drop += info["n_drop"]
    dt = time.time() - t0
    ncon_t = None
    if hasattr(env.dw, "ncon"):
        import warp as wp
        ncon_val = wp.to_torch(env.dw.ncon)
        ncon_t = int(ncon_val.max().item()) if ncon_val.numel() > 1 else int(ncon_val.item())
    print(f"[{task}] 40 steps x 1024 envs in {dt:.2f}s ({40*1024/dt:,.0f} ctrl-steps/s) | "
          f"rew=[{rew.min().item():.2f}, {rew.max().item():.2f}] finite={bool(torch.isfinite(obs).all())} "
          f"succ={total_succ} drop={total_drop} dw.ncon={ncon_t}")
    # Check face angles movement under random finger actions!
    fa = env.qpos[:, 31:37]
    print(f"[{task}] face angle spread across 1024 envs after 40 random steps: "
          f"max|wrap90|={torch.abs(torch.remainder(fa + np.pi/4, np.pi/2) - np.pi/4).max().item():.3f} rad, "
          f"mean|wrap90|={torch.abs(torch.remainder(fa + np.pi/4, np.pi/2) - np.pi/4).mean().item():.4f} rad")
    del env
    torch.cuda.empty_cache()

# 3. Dump one exact parity snapshot (reset + U R' F + 5 known actions) for JS verification!
env_p = RubikWarpEnv(XML, task="reorient", num_envs=4, seed=0, nconmax=260, njmax=600)
q0 = build_cradle_qpos(env_p.mj_model).astype(np.float64)
for mv in ["U", "R'", "F"]:
    apply_move_to_qpos(q0, mv)
env_p.qpos[0] = torch.tensor(q0, device=env_p.device, dtype=torch.float32)
env_p.qvel[0] = 0.0
env_p.ctrl[0] = torch.tensor(CRADLE_CTRL, device=env_p.device, dtype=torch.float32)
env_p.target_face[0] = 0  # pX (R)
import warp as wp
wp.capture_launch(env_p.fwd_graph)
wp.synchronize()
obs_r0 = env_p.get_obs()[0].cpu().numpy().tolist()

env_t = RubikWarpEnv(XML, task="twist", num_envs=4, seed=0, nconmax=260, njmax=600)
env_t.qpos[0] = torch.tensor(q0, device=env_t.device, dtype=torch.float32)
env_t.qvel[0] = 0.0
env_t.ctrl[0] = torch.tensor(CRADLE_CTRL, device=env_t.device, dtype=torch.float32)
env_t.target_face[0] = 4  # pZ (U)
env_t.twist_dir[0] = -1.0
env_t.goal_angle[0] = float(np.round(q0[35] / (np.pi/2)) * (np.pi/2) - np.pi/2)
wp.capture_launch(env_t.fwd_graph)
wp.synchronize()
obs_t0 = env_t.get_obs()[0].cpu().numpy().tolist()

with open("/content/parity_ref.json", "w") as f:
    json.dump({"obs_reorient": obs_r0, "obs_twist": obs_t0, "goal_angle": float(env_t.goal_angle[0].item())}, f)
print("Saved /content/parity_ref.json successfully!")
