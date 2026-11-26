"""Launch detached Reorient PPO training on Colab L4 and verify initial speed + metrics."""
import os
import subprocess
import sys
import time

os.makedirs("/content/runs/reorient1", exist_ok=True)
log_path = "/content/runs/reorient1/stdout.log"

# Kill any stale training process if present
subprocess.run(["pkill", "-f", "train_ppo.py"], check=False)
time.sleep(1)

cmd = [
    sys.executable, "-u", "/content/rubik/training/train_ppo.py",
    "--task", "reorient",
    "--xml", "/content/rubik/robot/shadow_rubik.xml",
    "--out", "/content/runs/reorient1",
    "--num-envs", "1024",
    "--horizon", "32",
    "--steps", "60000000",
    "--lr", "3e-4",
    "--desired-kl", "0.016",
    "--entropy", "0.002",
    "--hidden", "512,256,128",
    "--seed", "1",
    "--save-every", "50",
    "--collect-handovers", "4096",
]

with open(log_path, "w") as lf:
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

print(f"Started Reorient PPO PID={proc.pid}, log={log_path}", flush=True)
# Wait up to 45s for first few iterations to appear
for _ in range(9):
    time.sleep(5)
    txt = open(log_path).read() if os.path.exists(log_path) else ""
    if "it=  10" in txt or proc.poll() is not None:
        break
print("=== Initial log ===")
lines = open(log_path).read().splitlines()
print("\n".join(lines[-40:]))
print("Process alive:", proc.poll() is None)
