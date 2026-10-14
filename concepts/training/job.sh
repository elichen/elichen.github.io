#!/bin/bash
# Launched from Windows: C:\w\gns\launch.ps1 -Job ../sae/job.sh -TaskName SAE. Runs the steps listed in jobs/current.
cd ~/play/sae
export PYTHONUNBUFFERED=1
PY=/home/aif_eng/play/torch-venv/bin/python
mkdir -p logs
while read -r step; do
  [ -z "$step" ] && continue
  echo "=== $step $(date)" >> logs/job.log
  $PY $step >> logs/job.log 2>&1 || { echo "FAILED $step" >> logs/job.log; break; }
done < jobs/current
echo "=== JOB DONE $(date)" >> logs/job.log
