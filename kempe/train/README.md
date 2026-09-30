# Kempe's Machine: how the data was made

The web app (`../index.html`) searches a library of machines, then tunes the most promising ones to
your drawing. Everything it ships lives in `../data/`: the library (`index.bin.gz`, `index.json`) and
the tuning predictor (`ranker.json`). This folder rebuilds them.

| Step | Script | Where it ran |
| --- | --- | --- |
| 1. Random machines | `gen_torch.py` | GPU (RTX 5050 laptop), ~45 min for 20M machines |
| 2. Doodles | `qd_fetch.py` | anywhere; Google Quick, Draw! (CC BY 4.0) |
| 3. Choose the library | `select_index.py` | GPU, ~15 min |
| 4. Pack the library | `export_index.py` | CPU |
| 5. Tune machines to doodles, log outcomes | `../tools/tune_doodles.mjs` | Node, 13 processes, ~1 h |
| 6. Add the tuned machines | `merge_tuned.py`, then `export_index.py` again | CPU |
| 7. Train the tuning predictor | `../tools/ranker_data.mjs`, `train_ranker.py` | CPU, minutes |
| 8. Evaluate end to end | `../tools/eval_app.mjs` | Node |

`linkage.py` and `refine.py` are the reference kinematics and Levenberg–Marquardt tuner;
`../mech.js` is the browser's port, checked against them by `golden.py` and
`../tools/golden_test.mjs`. `eval_match.py` was the first brute-force experiment.

A machine is a list of joints in construction order: fixed pivots, a motor crank, up to two more
cranks belted to the motor at whole-number speed ratios, then dyads (a joint tied by two bars to two
earlier joints). Each has exactly one degree of freedom, so every pose is a sequence of
circle-circle intersections. Machines are stored pruned to the joints their pen depends on, in the
frame of the pen's path (centroid at the origin, RMS radius 1).

`encoder.py` and `train_encoder.py` train a curve encoder by distilling the exact alignment error.
It didn't beat the Fourier-magnitude fingerprint the app uses for retrieval, so the app doesn't ship
it; the learned part that ships is the tuning predictor.

Doodles: `export_heldout.py` writes the evaluation set (the first 4,000 of a seeded permutation, never
used for selection, tuning or training); `export_doodles.py` writes the training doodles and
distorted test shapes (`targets.py`) for the tuner.
