# Hear the Shape of a Drum: how the listener was made

The page (`../index.html`) computes a drum's notes in the browser (`../fem.js`) and hands their ratios
to a small network (`../listener.js`, weights in `../model/`) that draws the outline it hears. This
folder rebuilds that network and the quiz drums.

| Step | Script | Notes |
| --- | --- | --- |
| 1. Doodles | `../../kempe/train/qd_fetch.py` | Google Quick, Draw! (CC BY 4.0), up to 9,000 per category |
| 2. Clean them up | `prep_doodles.py` | the same smoothing the page applies to a drawing (`shapes.doodle`) |
| 3. Their notes | `gen_spectra.py` | P1 finite elements (`fem.py`), 40 lowest modes per drum; ~580k drums took about an hour on a 16-thread laptop |
| 4. Synthetic drums | `gen_spectra.py` | blobs, star-shaped curves, polygons, ellipses (`shapes.py`) |
| 5. One training set | `merge_spectra.py` | doodle seeds index one Quick, Draw! file, so the split stays consistent |
| 6. Train | `train_hear.py` | 512 wide, 4 residual blocks, 60k steps of 1,024 drums, ~6 min on an RTX 5050 laptop GPU |
| 7. Evaluate | `eval_hear.py`, `../tools/quiz_eval.mjs` | overlap by notes heard and shape family; quiz win rate |
| 8. Export | `export_listener.py`, `export_quiz.py` | float16 weights; quiz drums are held-out doodles |

Every drum is scaled to unit area before meshing, so the mesh spacing means the same thing for every
shape, and the network only sees ratios of eigenvalues, which don't depend on size or tension. Drums
with `seed % 10 == 0` are held out from training; a stretched copy of a doodle (seed 10,000,000 +
index) shares its original's split.

The loss lines each of the network's four candidate outlines up with the true one over rotation,
mirror image and starting point (the notes can't carry any of those), then scores what's left; the
best candidate takes the loss (winner takes all) and a confidence head learns which one that will be.

`golden.py` and `../tools/golden_test.mjs` check that the browser's mesher and eigensolver match
`fem.py` (to about 1e-14); `export_listener.py --golden` and `../tools/listener_test.mjs` do the
same for the network. The isospectral pair is meshed with `crisscross_mesh`, which is symmetric under
the grid's reflections, so their computed spectra agree to rounding error as the theory demands.
