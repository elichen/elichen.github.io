"""Train canonical-manifold pair sequentially on Colab L4:
  1. reorient4 (18M steps, ~21 min): restores reorient3/best.pt on twist5 handovers with --brake-settle
     (dead-still angv < 2.2 rad/s, centered pos_dist < 4.5 cm, tilt < 12.6 deg), collects 4096 handovers.
  2. twist6 (22M steps, ~29 min): restores twist5/best.pt on 80% reorient4 canonical handovers with --brake-settle
     (anti-tip upright_pen, dead-still face_vel < 1.8 & angv < 2.5, centered pos_dist < 4.5 cm).
"""
import os
import subprocess
import sys
import time

os.makedirs("/content/runs/canonical", exist_ok=True)
runner = "/content/runs/canonical/run_pipeline.sh"
pipe_log = "/content/runs/canonical/pipeline.log"

script = f"""#!/bin/bash
set -euo pipefail
echo "=== STAGE A: reorient4 (canonical dead-stop centered handover) ==="
{sys.executable} -u /content/rubik/training/train_ppo.py \\
  --task reorient \\
  --xml /content/rubik/robot/shadow_rubik.xml \\
  --out /content/runs/reorient4 \\
  --restore /content/runs/reorient3/best.pt \\
  --handover /content/runs/twist5/handovers_mixed.npz \\
  --handover-prob 0.70 \\
  --brake-settle \\
  --num-envs 1024 --horizon 32 --steps 18000000 \\
  --lr 1.5e-4 --desired-kl 0.015 --entropy 0.001 \\
  --hidden 512,256,128 --seed 8 --save-every 50 --collect-handovers 4096 \\
  > /content/runs/reorient4_stdout.log 2>&1

echo "=== STAGE B: twist6 (80% reorient4 canonical handovers + anti-tip + brake-settle) ==="
{sys.executable} -u /content/rubik/training/train_ppo.py \\
  --task twist \\
  --xml /content/rubik/robot/shadow_rubik.xml \\
  --out /content/runs/twist6 \\
  --restore /content/runs/twist5/best.pt \\
  --handover /content/runs/reorient4/handovers.npz \\
  --handover-prob 0.80 \\
  --start-deg 90 --brake-settle \\
  --num-envs 1024 --horizon 32 --steps 22000000 \\
  --lr 1.5e-4 --desired-kl 0.015 --entropy 0.001 \\
  --hidden 512,256,128 --seed 9 --save-every 50 --collect-handovers 4096 \\
  > /content/runs/twist6_stdout.log 2>&1

echo "=== ALL DONE CANONICAL PAIR ==="
"""

os.makedirs("/content/runs/reorient4", exist_ok=True)
os.makedirs("/content/runs/twist6", exist_ok=True)
with open(runner, "w") as f:
    f.write(script)
os.chmod(runner, 0o755)

subprocess.run(["pkill", "-f", "train_ppo.py"], check=False)
subprocess.run(["pkill", "-f", "run_pipeline.sh"], check=False)
time.sleep(1)

with open(pipe_log, "w") as lf:
    proc = subprocess.Popen(["bash", runner], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

print(f"Started canonical pair pipeline PID={proc.pid}, log={pipe_log}", flush=True)
rlog = "/content/runs/reorient4_stdout.log"
for _ in range(10):
    time.sleep(5)
    txt = open(rlog).read() if os.path.exists(rlog) else ""
    if "it=  10" in txt or proc.poll() is not None:
        break
print("=== reorient4 initial log ===")
if os.path.exists(rlog):
    print("\n".join(open(rlog).read().splitlines()[-20:]))
print("Pipeline alive:", proc.poll() is None)
