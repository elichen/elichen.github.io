"""Collect mixed handovers from twist5 best+latest, then fine-tune reorient3 (25M steps) for seamless catching."""
import os
import subprocess
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/content/rubik/training")
from rubik_warp_env import RubikWarpEnv
from train_ppo import ActorCritic, RunningNorm, collect_handovers

os.makedirs("/content/runs/reorient3", exist_ok=True)
mixed_ho = "/content/runs/twist5/handovers_mixed.npz"

# Collect 2048 states from twist5/latest.pt and combine with 2048 from twist5/handovers.npz (best.pt)
env = RubikWarpEnv(
    xml_path="/content/rubik/robot/shadow_rubik.xml",
    task="twist",
    num_envs=1024,
    seed=99,
    handover_path="/content/runs/reorient2/handovers.npz",
    handover_prob=0.65,
    twist_start_deg=90.0,
    brake_settle=True,
)
ck = torch.load("/content/runs/twist5/latest.pt", map_location=env.device, weights_only=False)
norm = RunningNorm(env.obs_dim, env.device)
norm.load_np(ck["obs_mean"], ck["obs_std"])
model = ActorCritic(env.obs_dim, env.act_dim, actor_hidden=(512, 256, 128), critic_hidden=(512, 256, 128)).to(env.device)
model.actor.load_state_dict(ck["actor_state_dict"])
lat_ho = "/content/runs/twist5/handovers_latest.npz"
collect_handovers(env, model, norm, n_target=2048, out_path=lat_ho)

# Combine 2048 from best + 2048 from latest
b = np.load("/content/runs/twist5/handovers.npz")
l = np.load(lat_ho)
np.savez_compressed(
    mixed_ho,
    qpos=np.concatenate([b["qpos"][:2048], l["qpos"][:2048]], axis=0),
    qvel=np.concatenate([b["qvel"][:2048], l["qvel"][:2048]], axis=0),
    ctrl=np.concatenate([b["ctrl"][:2048], l["ctrl"][:2048]], axis=0),
)
print(f"Saved mixed twist5 handovers (4096) -> {mixed_ho}", flush=True)
del env, model, norm
torch.cuda.empty_cache()

log_path = "/content/runs/reorient3/stdout.log"
subprocess.run(["pkill", "-f", "train_ppo.py"], check=False)
time.sleep(1)

cmd = [
    sys.executable, "-u", "/content/rubik/training/train_ppo.py",
    "--task", "reorient",
    "--xml", "/content/rubik/robot/shadow_rubik.xml",
    "--out", "/content/runs/reorient3",
    "--restore", "/content/runs/reorient2/best.pt",
    "--handover", mixed_ho,
    "--handover-prob", "0.70",
    "--num-envs", "1024",
    "--horizon", "32",
    "--steps", "25000000",
    "--lr", "1.5e-4",
    "--desired-kl", "0.015",
    "--entropy", "0.001",
    "--hidden", "512,256,128",
    "--seed", "7",
    "--save-every", "50",
    "--collect-handovers", "0",
]

with open(log_path, "w") as lf:
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

print(f"Started Reorient3 fine-tune PID={proc.pid}, log={log_path}", flush=True)
for _ in range(10):
    time.sleep(5)
    txt = open(log_path).read() if os.path.exists(log_path) else ""
    if "it=  10" in txt or proc.poll() is not None:
        break
print("=== Initial log ===")
lines = open(log_path).read().splitlines()
print("\n".join(lines[-20:]))
print("Process alive:", proc.poll() is None)
