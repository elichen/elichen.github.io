#!/bin/bash
# Pull GPU-trained candidate weights from Nitro: ./fetch.sh <run> [file ...]  -> runs/nitro/<run>/
run=$1; shift; files=${@:-best.npz}
mkdir -p runs/nitro/$run
echo "cd /mnt/c/w/qwop/$run && tar cz $files history.jsonl | base64 -w0" | \
  ssh -o User=aif-engineering nitro 'wsl.exe -d Ubuntu -u aif_eng -- bash' | base64 -d | tar xz -C runs/nitro/$run
ls runs/nitro/$run
