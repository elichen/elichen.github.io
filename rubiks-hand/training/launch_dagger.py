"""Spawn detached dg2 Round 3 surgical polish (Twist-to-Straighten training + Reorient polish) from dg1 best_pair (79.2% CPU!)."""
import os
import subprocess
import sys

os.makedirs("/content/rubik", exist_ok=True)
os.makedirs("/content/runs/dg2", exist_ok=True)

runner = "/content/runs/dg2/run.sh"
log_path = "/content/runs/dg2/stdout.log"

script = f"""#!/bin/bash
set -euo pipefail
echo "=== Unpacking dg2 bundle ==="
tar -xzf /content/rubik_dg2_bundle.tar.gz -C /content/rubik

echo "=== Launching train_dagger.py (dg2: Round 3 Surgical Polish + Skewed-Slice Straightening from dg1 79.2% CPU!) ==="
{sys.executable} -u /content/rubik/training/train_dagger.py \\
  --xml /content/rubik/robot/shadow_rubik.xml \\
  --reorient-init /content/runs/dg1/best_pair/reorient \\
  --twist-init /content/runs/dg1/best_pair/twist \\
  --out /content/runs/dg2 \\
  --num-envs 2048 \\
  --rounds 1 \\
  --twist-first \\
  --iters-twist 120 \\
  --iters-reorient 120 \\
  --lr 6.0e-5 \\
  --init-log-std -1.8 \\
  --anchor-coef 0.015 \\
  --cpu-eval-every 30 \\
  --seed 200
"""

with open(runner, "w") as f:
    f.write(script)
os.chmod(runner, 0o755)

subprocess.run(["pkill", "-f", "train_dagger.py"], check=False)
with open(log_path, "w") as lf:
    proc = subprocess.Popen(["bash", runner], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

print(f"DETACHED dg2 PID={proc.pid} log={log_path}", flush=True)
