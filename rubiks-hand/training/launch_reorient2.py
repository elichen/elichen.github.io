"""Fine-tune Reorient (reorient2) on twist4's 4,096 real post-twist handover states for seamless closed-loop solves."""
import os
import subprocess
import sys
import time

os.makedirs("/content/runs/reorient2", exist_ok=True)
log_path = "/content/runs/reorient2/stdout.log"

subprocess.run(["pkill", "-f", "train_ppo.py"], check=False)
time.sleep(1)

cmd = [
    sys.executable, "-u", "/content/rubik/training/train_ppo.py",
    "--task", "reorient",
    "--xml", "/content/rubik/robot/shadow_rubik.xml",
    "--out", "/content/runs/reorient2",
    "--restore", "/content/runs/reorient1/best.pt",
    "--handover", "/content/runs/twist4/handovers.npz",
    "--handover-prob", "0.65",
    "--num-envs", "1024",
    "--horizon", "32",
    "--steps", "35000000",
    "--lr", "1.5e-4",
    "--desired-kl", "0.015",
    "--entropy", "0.001",
    "--hidden", "512,256,128",
    "--seed", "5",
    "--save-every", "50",
    "--collect-handovers", "4096",
]

with open(log_path, "w") as lf:
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

print(f"Started Reorient2 handover fine-tune PID={proc.pid}, log={log_path}", flush=True)
for _ in range(10):
    time.sleep(5)
    txt = open(log_path).read() if os.path.exists(log_path) else ""
    if "it=  10" in txt or proc.poll() is not None:
        break
print("=== Initial log ===")
lines = open(log_path).read().splitlines()
print("\n".join(lines[-25:]))
print("Process alive:", proc.poll() is None)
