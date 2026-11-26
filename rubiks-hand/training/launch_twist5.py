"""Fine-tune Twist (twist5) with explicit dead-stop braking (<1.8 rad/s) on reorient2's real handover states."""
import os
import subprocess
import sys
import time

os.makedirs("/content/runs/twist5", exist_ok=True)
log_path = "/content/runs/twist5/stdout.log"

subprocess.run(["pkill", "-f", "train_ppo.py"], check=False)
time.sleep(1)

cmd = [
    sys.executable, "-u", "/content/rubik/training/train_ppo.py",
    "--task", "twist",
    "--xml", "/content/rubik/robot/shadow_rubik.xml",
    "--out", "/content/runs/twist5",
    "--restore", "/content/runs/twist4/best.pt",
    "--handover", "/content/runs/reorient2/handovers.npz",
    "--handover-prob", "0.65",
    "--start-deg", "90",
    "--brake-settle",
    "--num-envs", "1024",
    "--horizon", "32",
    "--steps", "35000000",
    "--lr", "1.5e-4",
    "--desired-kl", "0.015",
    "--entropy", "0.001",
    "--hidden", "512,256,128",
    "--seed", "6",
    "--save-every", "50",
    "--collect-handovers", "4096",
]

with open(log_path, "w") as lf:
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

print(f"Started Twist5 brake-settle fine-tune PID={proc.pid}, log={log_path}", flush=True)
for _ in range(10):
    time.sleep(5)
    txt = open(log_path).read() if os.path.exists(log_path) else ""
    if "it=  10" in txt or proc.poll() is not None:
        break
print("=== Initial log ===")
lines = open(log_path).read().splitlines()
print("\n".join(lines[-25:]))
print("Process alive:", proc.poll() is None)
