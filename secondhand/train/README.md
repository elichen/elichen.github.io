# Training the handwriting network

The page (`../index.html`, working title "Second Hand") primes a Graves-style
handwriting synthesis network with the visitor's own line of "The quick brown fox",
then writes whatever they type in that hand. The shipped weights are `../model/hand.bin`
(fp16, 7.6 MB) and `../model/hand.json`.

## Data
BRUSH (Kotani, Tellex and Tompkin, ECCV 2020): 170 writers, 27,649 short lines written on a
tablet. Licence: non-commercial research only (Eli OK'd using it on the site with credit).
The page ships only weights, plus five lines the network invented (`../model/hands.json`),
and no BRUSH handwriting. Download (567 MB, no account needed):

    uvx gdown 1NIIXDfmpUhI6i80Dg2363PIdllY7FRVQ -O data/brush.zip
    cd data && unzip -q brush.zip -x '*_inter' '*_resample25' '*_resample20'

## Pipeline (run from this folder)
1. `python3 prep.py` → `data/brush_norm.pkl`. Baseline and x-height come from BRUSH's
   per-point character labels. Coordinates are in x-heights, and each stroke is resampled
   every 0.2 x-heights. 27,173 samples, ~17 points per character.
2. Train. The shipped run is h1: 20k steps, 2 h on Nitro's RTX 5050.

       python3 train.py --out runs/h1 --batch 64 --maxlen 900 --nparts 0.15,0.5,0.35 \
         --steps 20000 --eval_every 1000 --graph 1

   - 3×400 LSTM, 10-component attention window, 20-component MDN, 3.79M params.
   - Each training sequence puts 1–3 lines by one writer on a shared baseline, so carrying
     on in someone's hand is exactly what it practises. That's what priming relies on.
   - The same shear/scale augmentation is applied to the whole sequence.
   - 12 writers are held out: 17 20 23 27 43 53 61 90 151 155 156 159.
   - Final val NLL is −2.91 (xy) and 0.120 (pen).
3. `python3 sample.py runs/h1/latest.pt --bias 1.0` → primed samples for held-out writers.
4. `python3 eval_style.py runs/h1/latest.pt --n 6 --seeds 3 --bias 1.0`. Final results:
   - Pairwise "closer to the right writer than another": 80% primed, 55% unprimed
     (chance 50%).
   - Top-1 of 12: 33% primed vs 17% unprimed. Top-3: 58% vs 25%.
   - Correlation of generated vs real across writers: slant 0.80, width/letter 0.82,
     strokes/letter 0.47, ascender 0.28, descender 0.13.
5. `python3 export.py runs/h1/latest.pt export --golden`, then `node web/parity.js export`:
   max |Δ| about 4e-4 against torch, ~2.4 ms/step in Node and ~3 ms in a browser worker.
6. Borrowed hands: `python3 gen_hands.py runs/h1/latest.pt` writes a contact sheet.
   Then `--pick 2,10,13,24,7` writes `../model/hands.json`, trimming stray trailing dots.
7. `calib.py` → `calib.json` (copied to `../model/calib.json`). It estimates baseline,
   x-height and tilt from a visitor's prompt line using ink-height quantiles. Writer-CV
   error: x-height 12.6%, baseline 0.08 x-height.

## Notes
- `--graph 1` captures the whole training step (forward, backward, Adam) as one CUDA
  graph, which needs a fixed T (`--fixT 1`, the default) and fixed text padding
  (`--upad 72`). It runs at 0.37 s/step vs 1.11 s eager on Nitro.
- `Wx.unbind(1)` instead of `Wx[:, t]`. Per-step slicing allocated a full-size gradient
  every timestep, so backward was 15× slower.
- Mac MPS stalled in background runs: variable T fragmented the MPS cache up to 21 GB and
  the Mac swapped. Use Nitro.
- Sampling: the move and the pen-lift flag are drawn independently, which sometimes gave
  a long jump with the pen down. Strokes are resampled at one unit per step, so any move
  longer than 2.5 units is treated as a lift (`engine.js`, `sample.py`).
