#!/bin/bash
# On Nitro, overnight: score every snapshot of each run (evals.sh) one run at a time, waiting for each run's
# training and BBF's final evaluation to finish first. Launched as scheduled task BBFEvals (launch.ps1).
cd ~/play/bbf/export
done_training() { grep -q "Average undiscounted return per evaluation" /mnt/c/w/bbf/$1/log.txt 2>/dev/null && grep -q "^end " /mnt/c/w/bbf/$1/log.txt; }
while pgrep -f "evals.sh qbert" > /dev/null; do sleep 60; done
for r in qbert:Qbert mspacman:MsPacman kungfu:KungFuMaster spaceinvaders:SpaceInvaders mspacman2:MsPacman; do
  run=${r%%:*} game=${r#*:}
  until done_training $run; do sleep 300; done
  ./evals.sh $run $game
  echo "$(date) scored $run" >> /mnt/c/w/bbf/evalq.log
done
