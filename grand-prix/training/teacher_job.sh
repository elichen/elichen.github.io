#!/bin/bash
# Nitro: grid-search the teacher's steering gains (C:\w\gns\launch.ps1 -Job ../grand-prix/teacher_job.sh -TaskName GrandPrixTeacher)
cd ~/play/grand-prix
export PYTHONPATH=~/play:~/play/grand-prix XLA_PYTHON_CLIENT_PREALLOCATE=false
~/play/.venv-jax011-cu13-braxmain-mj311/bin/python teacher.py --profile profiles_0.85.npz profiles_0.9.npz --search > /mnt/c/w/grand_prix/teacher_search.log 2>&1
