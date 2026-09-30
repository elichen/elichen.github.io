"""Merge machines tuned to training doodles (tools/tune_doodles.mjs output) into
a selected pool (select_index.py output), in the pool's storage format.

usage: python merge_tuned.py SEL.npz TUNED.jsonl OUT.npz [--max-err 0.15]
"""

import argparse
import json
import numpy as np

JP = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sel')
    ap.add_argument('tuned')
    ap.add_argument('out')
    ap.add_argument('--max-err', type=float, default=0.15)
    args = ap.parse_args()
    d = np.load(args.sel)
    M0 = len(d['n'])
    rows = [json.loads(l) for l in open(args.tuned) if l.strip()]
    rows = [r for r in rows if r['err'] < args.max_err and len(r['kind']) <= JP
            and np.all(np.abs(r['pos']) < 6.4)]
    M = len(rows)
    pos = np.zeros((M, JP, 2), np.float32)
    kind = -np.ones((M, JP), np.int8)
    par = -np.ones((M, JP, 2), np.int8)
    gear = np.zeros((M, JP), np.int8)
    n = np.zeros(M, np.int8)
    for i, r in enumerate(rows):
        k = len(r['kind'])
        n[i] = k
        kind[i, :k] = r['kind']
        par[i, :k, 0] = r['a']
        par[i, :k, 1] = r['b']
        gear[i, :k] = r['gear']
        pos[i, :k] = np.array(r['pos']).reshape(k, 2)
    out = dict(pos=np.concatenate([d['pos'], pos]), kind=np.concatenate([d['kind'], kind]),
               par=np.concatenate([d['par'], par]), gear=np.concatenate([d['gear'], gear]),
               n=np.concatenate([d['n'], n]),
               tuned=np.concatenate([np.zeros(M0, bool), np.ones(M, bool)]))
    np.savez(args.out, **out, pool_size=d['pool_size'] if 'pool_size' in d.files else M0)
    errs = np.array([r['err'] for r in rows])
    print(f'pool {M0} + tuned {M} (median tuned err {np.median(errs):.4f}) = {M0 + M}')


if __name__ == '__main__':
    main()
