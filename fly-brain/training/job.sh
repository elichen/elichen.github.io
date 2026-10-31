#!/bin/bash
# Run train.py with the settings in ~/play/flyfly/jobs/current.env. From Windows: C:\w\gns\launch.ps1 -Job ../flyfly/job.sh -TaskName FlyFly
cd ~/play/flyfly
set -a; source jobs/current.env; set +a
export PYTHONUNBUFFERED=1 CUDA_MODULE_LOADING=EAGER XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$OUT/logs"
/home/aif_eng/play/.venv-jax011-cu13-braxmain-mj311/bin/python train.py > "$OUT/logs/$RUN_ID.log" 2>&1
echo "exit $?" >> "$OUT/logs/$RUN_ID.log"
