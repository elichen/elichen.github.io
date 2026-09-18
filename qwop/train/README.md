# Training the QWOP runner

Everything here runs on a laptop CPU (tested on an M4 Pro: ~1M env-steps/s).

```bash
python3 -m venv .venv && .venv/bin/pip install numpy numba torch
.venv/bin/python parity.py          # physics.js must match sim.py
.venv/bin/python teacher.py 1       # CEM search for the scripted teacher -> teacher.json
.venv/bin/python train.py imitate   # behaviour cloning + DAgger -> bc.pt, dagger.pt
.venv/bin/python train.py ppo --out runs/ppo_init   # ~35 min for 300M steps
.venv/bin/python train.py finaleval --rl rl.pt      # 2,048 episodes per stage
.venv/bin/python train.py export                    # -> ../models.json
```

- `sim.py` — numba rigid-body simulator (sequential impulses, 9 bodies, 8 joints,
  11 ground-contact circles). `../physics.js` is a line-for-line mirror.
- `teacher.py` — four-phase finite-state machine keyed on the hip split angle.
- `train.py` — BC, DAgger, PPO, evaluation, export.

## Results (2,048 episodes, randomised starts, 100 s cap)

| stage | finish 100 m | mean time | speed | falls |
|---|---|---|---|---|
| teacher (scripted) | 0% | – | 0.56 m/s | 27% |
| behaviour cloning | 0% | – | 0.56 m/s | 27% |
| DAgger (10 rounds) | 0% | – | 0.56 m/s | 27% |
| PPO from DAgger, 300M steps | 99.8% | 21.4 ± 0.03 s | 4.67 m/s | 0.2% |
| PPO from scratch, 130M steps (512 ep.) | 75% | 38.0 s | 2.57 m/s | 25% |
| PPO from DAgger at 126M steps (512 ep.) | 99% | 22.1 s | – | 1% |

Notes:
- BC matches the teacher on 99.98% of actions even from two demonstration runs,
  because the teacher is a simple function of state plus the previous key (which
  the network observes). There was no distribution shift for DAgger to repair.
- The pretraining pays off in RL: at equal steps the pretrained run is ~1.7x faster
  and far more reliable than PPO from random weights (one seed each, and the
  shorter scratch run had a faster learning-rate decay, so treat as indicative).
- PPO's gait is a lunging gallop that touches a knee down ~14% of the time, the
  same family as the "knee scoot" human QWOP players find. Penalising knee
  contact (`--knee-penalty 1.5`) gives a clean upright walk but only 1.55 m/s.
