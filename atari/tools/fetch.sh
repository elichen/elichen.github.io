#!/bin/bash
# Export BBF snapshots on Nitro and copy them here.
#   tools/fetch.sh <run> <rom name> <step> [<step>...]   e.g. tools/fetch.sh roadrunner road_runner 10000 100000
# On Nitro: ~/play/bbf/export holds export.py, bbf_net.py and test/states_<rom>.npy; training writes
# /mnt/c/w/bbf/<run>/params_<step>.pkl. Exports go to C:\w\bbf\<run>\export\<step> and then to model/<rom>/<step>.
set -e
run=$1 rom=$2; shift 2
here=$(cd "$(dirname "$0")/.." && pwd)
for step in "$@"; do
  s=$(printf '%06d' $step)
  script="cd ~/play/bbf/export && JAX_PLATFORMS=cpu BBF_WORK=~/play/bbf/work ../.venv/bin/python export.py /mnt/c/w/bbf/$run/params_$s.pkl /mnt/c/w/bbf/$run/export/$s --states test/states_$rom.npy 2>&1 | tail -1"
  for i in 1 2 3 4 5; do
    out=$(printf '%s' "$script" | ssh -o User=aif-engineering -o ConnectTimeout=20 nitro 'wsl.exe -d Ubuntu -u aif_eng -- bash' 2>&1) && echo "$out" | grep -q "MB" && break
    echo "retry ($out)"; sleep 4
  done
  echo "$rom $step: $out"
  mkdir -p "$here/model/$rom/$s"
  for f in policy.json policy.bin test.json; do
    for i in 1 2 3 4 5; do scp -q -o User=aif-engineering nitro:"C:/w/bbf/$run/export/$s/$f" "$here/model/$rom/$s/$f" && break; sleep 4; done
  done
done
