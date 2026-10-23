#!/bin/bash
# Nitro: ES on lap time (C:\w\gns\launch.ps1 -Job ../grand-prix/es_job.sh -TaskName GrandPrixES)
# es1: 300 generations from dagger4/dagger_round_15; es2: 600 more from es1/es_best, sigma 0.002 -> 0.0008
cd ~/play/grand-prix
export PYTHONPATH=~/play:~/play/grand-prix XLA_PYTHON_CLIENT_PREALLOCATE=false
~/play/.venv-jax011-cu13-braxmain-mj311/bin/python es.py /mnt/c/w/grand_prix/es1/es_best /mnt/c/w/grand_prix/es2 --generations 600 --sigma 0.002 --sigma-end 0.0008 > /mnt/c/w/grand_prix/es2.log 2>&1
