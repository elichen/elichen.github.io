"""Mystery drums for the page's quiz: held-out doodles (seed % 10 == 0, never
trained on), a few per category, as quiz.json.

usage: python export_quiz.py SPECTRA.npz QD.npz OUT.json [--per 4]
"""

import argparse
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spectra')
    ap.add_argument('qd')
    ap.add_argument('out')
    ap.add_argument('--per', type=int, default=4)
    args = ap.parse_args()
    d = np.load(args.spectra)
    q = np.load(args.qd)
    cats, names = q['cat'], q['cats']
    keep = np.flatnonzero((d['kind'] == 'doodle') & (d['seed'] % 10 == 0))
    rng = np.random.default_rng(4)
    rng.shuffle(keep)
    out, count = [], {}
    for i in keep:
        c = str(names[cats[d['seed'][i]]])
        if count.get(c, 0) >= args.per:
            continue
        count[c] = count.get(c, 0) + 1
        out.append(dict(cat=c, outline=np.round(d['outline'][i].astype(float).ravel(), 4).tolist()))
    json.dump(out, open(args.out, 'w'), separators=(',', ':'))
    print(len(out), 'mystery drums from', len(count), 'categories')


if __name__ == '__main__':
    main()
