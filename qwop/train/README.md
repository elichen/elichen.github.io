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
# optional GPU continuation (needs jax[cuda] + optax): see "GPU hillclimb" below
python jppo.py --init init_rl.npz --out /path/to/run --lr 5e-4 --ent 0.05 --steps 1200000000
```

- `sim.py` — numba rigid-body simulator (sequential impulses, 9 bodies, 8 joints,
  11 ground-contact circles). `../physics.js` is a line-for-line mirror.
- `teacher.py` — four-phase finite-state machine keyed on the hip split angle.
- `train.py` — BC, DAgger, PPO, evaluation, export.
- `gait.py` — form diagnostics: knee contact, lead-leg balance, stride coverage.

## Results (2,048 episodes, randomised starts, 100 s cap)

| stage | finish 100 m | mean time | speed | falls |
|---|---|---|---|---|
| teacher (scripted) | 0% | – | 0.56 m/s | 27% |
| behaviour cloning | 0% | – | 0.56 m/s | 27% |
| DAgger (10 rounds) | 0% | – | 0.56 m/s | 27% |
| PPO from DAgger, 300M steps (Mac CPU) | 99.8% | 21.4 ± 0.03 s | 4.67 m/s | 0.2% |
| + 4.4B more steps on GPU, high entropy (shipped) | 99.9% | 18.59 ± 0.01 s | 5.38 m/s | 0.1% |
| alternating-stride sprinter, +4.2B GPU steps from the CPU PPO (shipped as "Sprinter") | 99.9% | 19.95 ± 0.01 s | 5.01 m/s | 0.1% |
| PPO from scratch, 130M steps (512 ep.) | 75% | 38.0 s | 2.57 m/s | 25% |
| PPO from DAgger at 126M steps (512 ep.) | 99% | 22.1 s | – | 1% |

## GPU hillclimb

`jsim.py` is a JAX port of `sim.py` (`jparity.py`: agrees to 1e-14 in float64) and
`jppo.py` runs PPO end to end on the GPU at ~1M steps/s on an RTX 5050 laptop.
It trains in float32, so its checkpoints are only candidates: every number in
the table comes from `train.py finaleval` in the float64 numba simulator, which
the browser matches. `fetch.sh` pulls checkpoints back.

What moved the 100 m time (each row continues from the previous winner):

| run | change | reference-sim time |
|---|---|---|
| start | CPU PPO result | 21.42 s |
| s1_a/b | just keep training (lr 1e-4, ent 0.003; also gamma 0.995), 600M | ~21.0 s (GPU eval) |
| s1_c | lr 3e-4, entropy 0.02, 600M | 20.41 s |
| s2_b | lr 5e-4, entropy 0.05, 1.2B | 18.74 s |
| s3_a | same again, 1.3B | 18.59 s (plateau) |

Fresh starts did worse at equal budget: 1B steps from the DAgger init reached
~19.8 s, and from random weights 23.4 s (with a more upright, 4%-knee gait).

Notes:
- BC matches the teacher on 99.98% of actions even from two demonstration runs,
  because the teacher is a simple function of state plus the previous key (which
  the network observes). There was no distribution shift for DAgger to repair.
- The pretraining pays off in RL: at equal steps the pretrained run is ~1.7x faster
  and far more reliable than PPO from random weights (one seed each, and the
  shorter scratch run had a faster learning-rate decay, so treat as indicative).
## Alternating-stride sprinter

`gait.py` measures form in the reference simulator. The shipped PPO gait keeps the
right leg in front 78% of the time and none of its distance is covered by proper
strides; the knee-penalty walker never touches a knee but leads with the right leg
100% of the time (a shuffle). So penalising knees alone does not give a run.

`jppo.py --stride 0.7 --knee-fatal 1` does. 70% of the progress reward is paid only
per valid stride: a foot strike that lands at least 0.2 m ahead of the other foot,
by the opposite foot from the last valid stride, with no knee contact in between
(the stride is credited with the torso distance since the previous one, capped at
3 m). A gallop or shuffle keeps only the 30% base rate. A knee on the ground ends
the episode like a fall. The network and its observations are unchanged.

| run | change (each continues from the row above) | steps | 100 m | knee down | stride coverage | right leg leads |
|---|---|---|---|---|---|---|
| shipped PPO gallop | – | – | 18.59 s | 8.0% | 0% | 78% |
| g1_a | from the 21.4 s gallop: stride reward, knee penalty 0.3/step | 300M | 24.27 s | 0.36% | 97.7% | 56% |
| g2_a | knee contact ends the episode | 1.2B | 20.80 s | 0.02% | 99.3% | 48% |
| g3_a | same again | 1.5B | 20.17 s | 0.00% | 99.2% | 48% |
| g4_a | entropy 0.1 | 1.2B | 20.16 s (GPU eval) | 0.00% | 99% | – |
| g4_b (`sprint.npz`, shipped) | from g3_a: gamma 0.995 | 1.2B | 19.95 ± 0.01 s | 0.00% | 99.0% | 48% |

Form columns are `gait.py` (256 episodes); the shipped time is `train.py finaleval`
(2,048 episodes, 99.9% finish). The clean sprint costs about 1.4 s against the gallop.

Everything uses lr 5e-4 and entropy 0.05 unless noted. Starting from the gallop, the
alternating gait appears within the first ~100M steps; after ~3B it plateaus near 20 s.

```bash
python jppo.py --init init_rl.npz --critic-warmup 20 --stride 0.7 --knee-penalty 0.3 --out g1 --steps 300000000
python jppo.py --init g1/latest_full.pkl --stride 0.7 --knee-fatal 1 --out g2 --steps 1200000000
.venv/bin/python train.py finaleval --sprint sprint.npz && .venv/bin/python train.py export
.venv/bin/python gait.py sprint.npz
```

- PPO's gait is a lunging gallop that touches a knee down ~8% of the time (14% before the GPU hillclimb), the
  same family as the "knee scoot" human QWOP players find. Penalising knee
  contact (`--knee-penalty 1.5`) gives a clean upright walk but only 1.55 m/s.
