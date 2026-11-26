"""Spawn detached Iron-Grip v5 (anti-stall + warm-started from cl4_best@64.6%CPU) immediately."""
import os
import subprocess
import sys

os.makedirs("/content/rubik", exist_ok=True)
os.makedirs("/content/runs/cl5", exist_ok=True)

runner = "/content/runs/cl5/run.sh"
log_path = "/content/runs/cl5/stdout.log"

script = f"""#!/bin/bash
set -euo pipefail
echo "=== Unpacking cl5 bundle ==="
tar -xzf /content/rubik5_bundle.tar.gz -C /content/rubik

if ! {sys.executable} -c "import mujoco, mujoco_warp" 2>/dev/null; then
  echo "=== Installing mujoco + mujoco-warp ==="
  {sys.executable} -m pip install -q mujoco mujoco-warp
fi

echo "=== Launching train_closedloop.py (cl5: Anti-Stall Iron-Grip warm-started from cl4_best 64.6% CPU!) ==="
{sys.executable} -u -c '
import torch, os
print("GPU:", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB | CPUs:", os.cpu_count())
'

{sys.executable} -u /content/rubik/training/train_closedloop.py \\
  --xml /content/rubik/robot/shadow_rubik.xml \\
  --reorient-init /content/runs/cl4/best_pair/reorient \\
  --twist-init /content/runs/cl4/best_pair/twist \\
  --out /content/runs/cl5 \\
  --num-envs 1280 \\
  --eval-envs 256 \\
  --horizon 64 \\
  --rounds 2 \\
  --reorient-first \\
  --iters-reorient 125 \\
  --iters-twist 100 \\
  --critic-warmup 25 \\
  --target-moves 25 \\
  --hard-neg-prob 0.30 \\
  --lr 3.0e-5 \\
  --critic-lr 1.0e-3 \\
  --desired-kl 0.004 \\
  --entropy 0.0001 \\
  --anchor-coef 0.025 \\
  --blame -120.0 \\
  --cpu-eval-every 25 \\
  --cpu-eval-episodes 48 \\
  --seed 36
"""

with open(runner, "w") as f:
    f.write(script)
os.chmod(runner, 0o755)

subprocess.run(["pkill", "-f", "train_closedloop.py"], check=False)
with open(log_path, "w") as lf:
    proc = subprocess.Popen(["bash", runner], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

print(f"DETACHED cl5 PID={proc.pid} log={log_path}", flush=True)
