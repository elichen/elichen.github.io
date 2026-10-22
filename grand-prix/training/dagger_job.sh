#!/bin/bash
# Nitro: DAgger from the path-following teacher (C:\w\gns\launch.ps1 -Job ../grand-prix/dagger_job.sh -TaskName GrandPrixDagger)
# base.params = mjx-rl-experiments' 22-track policy (fastest_racer.params, sha256 c5d3de12...)
cd ~/play/grand-prix
export PYTHONPATH=~/play:~/play/grand-prix XLA_PYTHON_CLIENT_PREALLOCATE=false
~/play/.venv-jax011-cu13-braxmain-mj311/bin/python dagger.py base.params profiles_0.9.npz /mnt/c/w/grand_prix/dagger4 --rounds 16 > /mnt/c/w/grand_prix/dagger4.log 2>&1
