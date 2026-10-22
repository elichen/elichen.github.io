#!/bin/bash
# Nitro: ES on lap time from the DAgger network (C:\w\gns\launch.ps1 -Job ../grand-prix/es_job.sh -TaskName GrandPrixES)
cd ~/play/grand-prix
export PYTHONPATH=~/play:~/play/grand-prix XLA_PYTHON_CLIENT_PREALLOCATE=false
~/play/.venv-jax011-cu13-braxmain-mj311/bin/python es.py /mnt/c/w/grand_prix/dagger4/dagger_round_15 /mnt/c/w/grand_prix/es1 --generations 300 > /mnt/c/w/grand_prix/es1.log 2>&1
