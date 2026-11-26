"""Launch detached Stage 2 Twist PPO training on Colab L4 using real Reorient handover states."""
import os
import subprocess
import sys
import time

os.makedirs("/content/runs/twist4", exist_ok=True)
log_path = "/content/runs/twist4/stdout.log"

subprocess.run(["pkill", "-f", "train_ppo.py"], check=False)
time.sleep(1)

cmd = [
    sys.executable, "-u", "/content/rubik/training/train_ppo.py",
    "--task", "twist",
    "--xml", "/content/rubik/robot/shadow_rubik.xml",
    "--out", "/content/runs/twist4",
    "--restore", "/content/runs/twist3/latest.pt",
    "--handover", "/content/runs/reorient1/handovers.npz",
    "--handover-prob", "0.5",
    "--num-envs", "1024",
    "--horizon", "32",
    "--steps", "120000000",
    "--lr", "3e-4",
    "--desired-kl", "0.016",
    "--entropy", "0.001",
    "--hidden", "512,256,128",
    "--seed", "4",
    "--save-every", "50",
    "--collect-handovers", "4096",
]

with open(log_path, "w") as lf:
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

print(f"Started Twist PPO PID={proc.pid}, log={log_path}", flush=True)
for _ in range(10):
    time.sleep(5)
    txt = open(log_path).read() if os.path.exists(log_path) else ""
    if "it=  10" in txt or proc.poll() is not None:
        break
print("=== Initial log ===")
lines = open(log_path).read().splitlines()
print("\n".join(lines[-30:]))
print("Process alive:", proc.poll() is None)
