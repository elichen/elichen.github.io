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
   - Caveat: each feature set is z-scored on its own, so a shift shared by every generated
     line is invisible, and the test sentences are under 20 characters. Both hid the drift
     described under "Writing in chunks".
5. `python3 export.py runs/h1/latest.pt export --golden`, then `node web/parity.js export`:
   max |Δ| about 4e-4 against torch, ~2.4 ms/step in Node and ~3 ms in a browser worker.
6. Borrowed hands: `python3 gen_hands.py runs/h1/latest.pt` writes a contact sheet.
   Then `--pick 2,10,13,24,7` writes `../model/hands.json`, trimming stray trailing dots.
7. `calib.py` → `calib.json` (copied to `../model/calib.json`). It estimates baseline,
   x-height and tilt from a visitor's prompt line using ink-height quantiles. Writer-CV
   error: x-height 12.6%, baseline 0.08 x-height.

## Writing in chunks
Primed on one line, the network keeps the hand for about a dozen characters and then drifts
toward a generic narrow, crowded hand as more of its context is its own ink (exposure bias).
On ~45-character lines by held-out writers, the advance per character fell from 0.89× of
the prime's over characters 0–12 to about 0.7× after that, and word gaps shrank to 0.78×.
Training writers drifted the same way. A two-line prime didn't help, and neither did any
neatness setting.

So `worker.js` writes each line in chunks of up to 13 characters (whole words). Each chunk is
freshly primed from the cached prime state, and it goes where the prime would put it if the
ink so far were the prime: shift = (rightmost ink so far) − (prime's rightmost ink right of
its last point). The network's own jump sets the gap. Two fixes come with it:
- `sampleJump` (`engine.js`): the prime ends with a lift that the network only learns from
  its next input. About 1 chunk in 5 began with a tiny in-stroke step and overlapped the
  previous chunk, so the first move is redrawn until it is longer than 2.5 units.
- A chunk whose ink is more than 7 x-heights tall or 3 x-heights wide per character is
  redrawn, up to 4 tries. Runaway strokes went from 8 of 252 test lines to 0.

Same eval set as eval_style.py, run through the page's worker
(`python3 export_evalset.py` on the data machine, then
`node web/eval_page.js evalset.json out.json` and `python3 eval_page.py out.json`):

|                               | whole lines | chunks |
|-------------------------------|------------:|-------:|
| pairwise, primed              |        77% |    86% |
| top-3 of 12                   |        42% |    75% |
| pairwise, swapped primes      |        82% |    89% |
| r slant / width               | 0.81 / 0.82 | 0.89 / 0.88 |
| r strokes per letter          |       0.45 |   0.29 |
| r ascender / descender        | 0.07 / 0.16 | 0.63 / 0.47 |

Measured against the writers' own lines of the same lowercase text, letter width went from
0.83× to 0.91× and word gaps from 0.62× to 0.82×.

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
