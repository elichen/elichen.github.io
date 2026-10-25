#!/bin/bash
# On Nitro, on the CPU (training keeps the GPU): score each snapshot of a run over 20 games with eval_ref.py.
#   evals.sh <run> <Game>   writes /mnt/c/w/bbf/<run>/evals.txt, one line per snapshot
cd ~/play/bbf/export && source ../.venv/bin/activate
export JAX_PLATFORMS=cpu BBF_WORK=~/play/bbf/work PYTHONPATH=~/play/bbf/work/shims:~/play/bbf/work
for p in /mnt/c/w/bbf/$1/params_*.pkl; do
  grep -q "$(basename $p)" /mnt/c/w/bbf/$1/evals.txt 2>/dev/null || python eval_ref.py $p $2 20 2>/dev/null | tail -1 >> /mnt/c/w/bbf/$1/evals.txt
done
