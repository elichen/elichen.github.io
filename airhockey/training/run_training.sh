#!/bin/bash
TIMESTEPS=${1:-2000000}
source ../venv/bin/activate
python train_selfplay_v3.py --timesteps $TIMESTEPS --device auto