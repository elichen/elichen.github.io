"""Fetch human doodles from Google's Quick, Draw! dataset (CC BY 4.0) and keep
the ones drawn essentially as one stroke: the kind of input Kempe's Machine
gets. Closed strokes stay closed; open strokes become out-and-back paths.

usage: python qd_fetch.py OUT.npz [--per 3000]
"""

import argparse
import json
import urllib.parse
import urllib.request
import numpy as np
from linkage import resample, to_complex, normalise

CATS = ['circle', 'square', 'triangle', 'star', 'hexagon', 'octagon', 'diamond', 'moon',
        'cloud', 'fish', 'leaf', 'lightning', 'apple', 'pear', 'banana', 'boomerang',
        'peanut', 'potato', 'bread', 'light bulb', 'mushroom', 'tooth', 'sock', 'shoe',
        'foot', 'hand', 't-shirt', 'mountain', 'zigzag', 'squiggle', 'line', 'snail',
        'hat', 'bowtie', 'crown', 'hockey stick', 'hourglass', 'lollipop', 'cactus',
        'whale', 'duck', 'rabbit', 'bird', 'snake', 'strawberry', 'ice cream', 'sailboat',
        'mouth', 'ear', 'string bean', 'rainbow', 'feather', 'tornado', 'swan']
URL = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson'


def extract(drawing, n_raw=256):
    """Main stroke of a doodle as a closed polyline, or None."""
    strokes = [np.array(s, float).T for s in drawing]           # each (k, 2)
    lens = [np.linalg.norm(np.diff(s, axis=0), axis=1).sum() if len(s) > 1 else 0 for s in strokes]
    total = sum(lens)
    i = int(np.argmax(lens))
    s = strokes[i]
    if total <= 0 or lens[i] < 0.8 * total or len(s) < 6:
        return None, None
    s = s * [1, -1]                                              # y up
    diag = np.linalg.norm(s.max(0) - s.min(0))
    if diag < 20:
        return None, None
    gap = np.linalg.norm(s[0] - s[-1])
    closed = gap < 0.2 * diag
    if not closed:
        s = np.vstack([s, s[-2:0:-1]])
    # densify so arc-length resampling is well behaved
    seg = np.linalg.norm(np.diff(np.vstack([s, s[:1]]), axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, cum[-1], n_raw, endpoint=False)
    ss = np.vstack([s, s[:1]])
    dense = np.stack([np.interp(u, cum, ss[:, 0]), np.interp(u, cum, ss[:, 1])], 1)
    return dense, closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--per', type=int, default=3000)
    args = ap.parse_args()
    raws, cats, closed_f = [], [], []
    for ci, cat in enumerate(CATS):
        url = URL.format(urllib.parse.quote(cat))
        for attempt in range(4):   # the bucket occasionally times out mid-stream
            try:
                got = fetch_category(url, args.per)
                break
            except OSError as e:
                print(f'{cat}: {e}, retrying', flush=True)
        else:
            continue
        for dense, closed in got:
            raws.append(dense)
            cats.append(ci)
            closed_f.append(closed)
        print(f'{cat:14s} kept {len(got)}', flush=True)
    raws = np.stack(raws)
    z, _ = normalise(to_complex(resample(raws)))
    np.savez(args.out, raw=raws.astype(np.float32), z=z.astype(np.complex64),
             cat=np.array(cats, np.int16), closed=np.array(closed_f), cats=np.array(CATS))


def fetch_category(url, per):
    got, seen = [], 0
    with urllib.request.urlopen(url, timeout=60) as f:
        for line in f:
            seen += 1
            d = json.loads(line)
            if not d.get('recognized', False):
                continue
            dense, closed = extract(d['drawing'])
            if dense is None:
                continue
            got.append((dense, closed))
            if len(got) >= per or seen >= 12 * per:
                break
    return got


if __name__ == '__main__':
    main()
