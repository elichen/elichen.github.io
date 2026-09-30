"""Choose which machines ship with the web app.

1. Embed every curve in the pool (Fourier descriptors, or a trained encoder),
   cached per shard.
2. For each selection query (Quick, Draw! doodles except a held-out set, and
   stretched and sheared test shapes), find its nearest curves in the whole pool
   by embedding, re-rank them by exact alignment error, and keep the best few.
3. Fill the budget in rounds (every query's best, then every query's
   second best, ...) and add a random sample of the pool for coverage.

usage: python select_index.py --pool POOLDIR --doodles qd.npz --out sel.npz [--encoder encoder.pt]
"""

import argparse
import glob
import os
import time
import numpy as np
import torch
from encoder import Encoder, exact_err, normalise, fourier_descriptor
from train_encoder import augment, embed_all
from targets import targets
from linkage import resample as np_resample, to_complex, normalise as np_norm

KEYS = ('pos', 'kind', 'par', 'gear', 'n')


def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    a = ck['args']
    m = Encoder(a['width'], a['dim']).to(dev)
    m.load_state_dict(ck['model'])
    m.eval()
    return m


def shard_z(f, dev):
    z = np.load(f)['z'].astype(np.float32)
    return torch.from_numpy(z[..., 0] + 1j * z[..., 1]).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--encoder', default='', help='default: Fourier descriptors')
    ap.add_argument('--doodles', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget', type=int, default=300000)
    ap.add_argument('--random', type=int, default=30000)
    ap.add_argument('--k', type=int, default=600, help='embedding neighbours re-ranked per query')
    ap.add_argument('--keep', type=int, default=3, help='best machines remembered per query')
    ap.add_argument('--holdout', type=int, default=4000)
    args = ap.parse_args()
    dev = 'cuda'
    model = load_model(args.encoder, dev) if args.encoder else None
    embed = (lambda z: embed_all(model, z)) if model else (lambda z: fourier_descriptor(z).half())
    files = sorted(glob.glob(os.path.join(args.pool, 'shard_*.npz')))
    emb_dir = os.path.join(os.path.dirname(args.out), 'emb_' + ('enc' if model else 'fourier'))
    os.makedirs(emb_dir, exist_ok=True)

    # 1. embeddings of the whole pool
    t0 = time.time()
    E, sizes = [], []
    for f in files:
        ef = os.path.join(emb_dir, os.path.basename(f).replace('.npz', '.npy'))
        if not os.path.exists(ef):
            np.save(ef, embed(shard_z(f, dev)).cpu().numpy())
        e = np.load(ef)
        E.append(torch.from_numpy(e).to(dev))
        sizes.append(len(e))
    offs = np.concatenate([[0], np.cumsum(sizes)])
    print(f'embedded {offs[-1]} curves in {time.time() - t0:.0f}s', flush=True)

    # 2. queries: doodles (minus the held-out ones used for evaluation) + test shapes
    qd = np.load(args.doodles)
    Zd = torch.from_numpy(qd['z']).to(dev)
    perm = torch.randperm(len(Zd), generator=torch.Generator().manual_seed(1)).to(dev)
    Q = [Zd[perm[args.holdout:]]]
    T = targets()
    Zs = torch.from_numpy(np.concatenate([np_norm(to_complex(np_resample(T[k][None])))[0] for k in T]).astype(np.complex64)).to(dev)
    g = torch.Generator(device=dev).manual_seed(7)
    Q.append(Zs)
    for _ in range(40):
        Q.append(augment(Zs, g, jitter=0.02, affine=0.35))
    Q = torch.cat(Q)
    print(f'{len(Q)} selection queries', flush=True)

    # pass A: nearest curves in embedding space, over every shard
    K = args.k
    cand = torch.zeros(len(Q), K, dtype=torch.int32, device=dev)
    B = 256
    with torch.no_grad():
        Eq = torch.cat([embed(Q[i:i + 4096]).half() for i in range(0, len(Q), 4096)])
        for i in range(0, len(Q), B):
            eq = Eq[i:i + B]
            vals, ids = [], []
            for s, e in enumerate(E):
                v, ix = (eq @ e.T).float().topk(K, dim=1)
                vals.append(v)
                ids.append(ix + int(offs[s]))
            v, ix = torch.cat(vals, 1), torch.cat(ids, 1)
            cand[i:i + B] = ix.gather(1, v.topk(K, dim=1).indices).int()
            if (i // B) % 100 == 0:
                print(f'pass A {i}/{len(Q)} {time.time() - t0:.0f}s', flush=True)
    del E
    torch.cuda.empty_cache()
    print(f'pass A done {time.time() - t0:.0f}s', flush=True)

    # pass B: exact alignment error of every candidate, a block of queries at
    # a time, streaming each shard's curves through the GPU
    err = torch.full((len(Q), K), 9.0, device=dev)
    QB = 40000
    bounds = torch.tensor(offs[1:], device=dev)
    for qb in range(0, len(Q), QB):
        flat = cand[qb:qb + QB].reshape(-1).long()
        shard_of = torch.bucketize(flat, bounds, right=True)
        qidx = torch.arange(qb, qb + len(flat) // K, device=dev).repeat_interleave(K)
        eflat = err[qb:qb + QB].view(-1)
        for s, f in enumerate(files):
            m = torch.nonzero(shard_of == s).squeeze(1)
            if m.numel() == 0:
                continue
            z = shard_z(f, dev)
            for j in range(0, m.numel(), 65536):
                mm = m[j:j + 65536]
                eflat[mm] = exact_err(Q[qidx[mm]], z[flat[mm] - int(offs[s])][:, None, :])[:, 0]
            del z
        print(f'pass B {qb + QB}/{len(Q)} {time.time() - t0:.0f}s', flush=True)
    best_err, o = err.topk(args.keep, dim=1, largest=False)
    best_ids = cand.gather(1, o).long()
    print(f'pass B done {time.time() - t0:.0f}s; mean best err {best_err[:, 0].mean():.4f}', flush=True)

    # 3. fill the budget round by round, then add random coverage
    chosen, seen = [], set()
    for r in range(args.keep):
        for qi in best_err[:, r].argsort().tolist():
            gid = int(best_ids[qi, r])
            if gid >= 0 and gid not in seen:
                seen.add(gid)
                chosen.append(gid)
            if len(chosen) >= args.budget - args.random:
                break
        if len(chosen) >= args.budget - args.random:
            break
    rng = np.random.default_rng(0)
    while len(chosen) < args.budget:
        gid = int(rng.integers(0, offs[-1]))
        if gid not in seen:
            seen.add(gid)
            chosen.append(gid)
    chosen = np.array(sorted(chosen))
    print(f'chose {len(chosen)} machines; query best err mean {best_err[:, 0].mean():.4f}', flush=True)

    # gather the machines
    out = {k: [] for k in KEYS}
    shard_of = np.searchsorted(offs[1:], chosen, side='right')
    for s in np.unique(shard_of):
        d = np.load(files[s])
        loc = chosen[shard_of == s] - offs[s]
        for k in KEYS:
            out[k].append(d[k][loc])
    np.savez(args.out, **{k: np.concatenate(v) for k, v in out.items()}, gid=chosen,
             pool_size=offs[-1], query_best=best_err[:, 0].cpu().numpy())


if __name__ == '__main__':
    main()
