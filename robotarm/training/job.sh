#!/bin/bash
# Run train.py with the settings in ~/play/robotarm/jobs/$JOB.env, e.g. from Windows:
#   C:\w\gns\launch.ps1 -Job ../robotarm/job.sh  (after copying the env file to jobs/current.env)
cd ~/play/robotarm
set -a; source jobs/current.env; set +a
export PYTHONUNBUFFERED=1 CUDA_MODULE_LOADING=EAGER XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$OUT/logs"
/home/aif_eng/play/.venv-jax011-cu13-braxmain-mj311/bin/python train.py > "$OUT/logs/$RUN_ID.log" 2>&1
echo "exit $?" >> "$OUT/logs/$RUN_ID.log"
