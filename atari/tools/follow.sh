#!/bin/bash
# Fetch each new snapshot of the queued runs (fetch.sh) until every run has its 100k snapshot.
#   nohup tools/follow.sh > /tmp/follow.log 2>&1 &
cd "$(dirname "$0")/.."
runs="mspacman:ms_pacman qbert:qbert kungfu:kung_fu_master spaceinvaders:space_invaders"
for i in $(seq 1 400); do
  left=0
  for r in $runs; do
    run=${r%%:*} rom=${r#*:}
    list=$(printf 'ls /mnt/c/w/bbf/%s/ 2>/dev/null | grep -o "params_[0-9]*" | grep -o "[0-9]*$"' "$run" |
      ssh -o User=aif-engineering -o ConnectTimeout=20 nitro 'wsl.exe -d Ubuntu -u aif_eng -- bash' 2>/dev/null | grep -E '^[0-9]{6}$')
    for s in $list; do
      [ -f model/$rom/$s/policy.bin ] || { tools/fetch.sh $run $rom $((10#$s)) | tail -1; chmod 644 model/$rom/$s/*; }
    done
    [ -f model/$rom/100000/policy.bin ] || left=1
  done
  [ $left = 0 ] && { echo "all done $(date)"; break; }
  sleep 300
done
