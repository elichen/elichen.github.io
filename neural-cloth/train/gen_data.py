"""Generate teacher trajectories in chunks (one cloth size per chunk so scenes batch on the GPU).

python gen_data.py --out data/train --chunks 100 --seed 1
python gen_data.py --out data/valid --chunks 4 --per-chunk 8 --seed 999
"""
import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

import cloth as C
import scenes

jax.config.update("jax_default_matmul_precision", "highest")  # see cloth.py

_ROLLOUTS = {}


def rollout_fn(size, frames):
    key = (size, frames)
    if key not in _ROLLOUTS:
        mesh = C.grid_mesh(*size)
        _ROLLOUTS[key] = (mesh, jax.jit(jax.vmap(C.make_rollout(mesh, frames))))
    return _ROLLOUTS[key]


def make_chunk(rng, size, per_chunk, frames, kind=None):
    mesh, fn = rollout_fn(size, frames)
    scs, kinds = [], []
    for _ in range(per_chunk):
        _, sc, k = scenes.random_scene(rng, frames, size, kind)
        scs.append(sc)
        kinds.append(k)
    batch = {k: jnp.asarray(np.stack([s[k] for s in scs])) for k in scs[0]}
    xs, its, cgs = fn(batch)
    out = {k: np.stack([s[k] for s in scs]) for k in scs[0]}
    out.update(pos=np.asarray(xs, np.float32), newton=np.asarray(its), cg=np.asarray(cgs),
               size=np.array(size), kinds=np.array(kinds))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--per-chunk", type=int, default=16)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--size", default=None, help="force a cloth size, e.g. 31x31")
    ap.add_argument("--kind", default=None, help="force a scenario, e.g. lift_release")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for c in range(args.start, args.start + args.chunks):
        path = os.path.join(args.out, f"chunk_{c:04d}.npz")
        if os.path.exists(path):
            continue
        rng = np.random.default_rng([args.seed, c])
        size = tuple(int(v) for v in args.size.split("x")) if args.size else scenes.SIZES[rng.integers(len(scenes.SIZES))]
        t0 = time.time()
        out = make_chunk(rng, size, args.per_chunk, args.frames, args.kind)
        ok = np.isfinite(out["pos"]).all()
        if not ok:
            print(f"chunk {c}: non-finite positions, skipping", flush=True)
            continue
        tmp = path + ".tmp.npz"
        np.savez(tmp, **out)
        os.replace(tmp, path)
        print(f"chunk {c}: size {size} {time.time() - t0:.1f}s newton/frame {out['newton'].mean():.1f} "
              f"(max {out['newton'].max()}) cg/frame {out['cg'].mean():.0f} min y {out['pos'][..., 1].min():.4f}",
              flush=True)


if __name__ == "__main__":
    main()
