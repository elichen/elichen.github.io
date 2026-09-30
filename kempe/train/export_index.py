"""Pack a set of pruned machines into the web app's search index.

Each machine is quantised (int16 joint positions in its curve's frame), then
re-simulated: machines that lock after quantisation are dropped, and the
embedding is computed from the quantised machine's own curve so the index
describes exactly what the browser will build.

Binary layout of index.bin (sections 4-byte aligned, offsets in index.json):
  pos     int16  [2 * joints]    joint x, y * POS_SCALE, machine by machine
  emb     int8   [count * dim]   unit embedding * 127
  n       uint8  [count]         joints per machine (the pen is the last)
  struct  uint8  [2 * joints]    kind | (gear + 4) << 2,  a | b << 4 (15 = none)

usage: python export_index.py SEL.npz OUTDIR [--encoder encoder.pt] [--limit N]
SEL.npz holds pruned machines as written by gen_torch.py (pos, kind, par, gear, n).
"""

import argparse
import gzip
import json
import os
import numpy as np
import torch
from refine import Topo, resample_offset, normalise
from encoder import Encoder, fourier_descriptor

POS_SCALE = 5000


def quantise(sel):
    q = np.clip(np.round(sel['pos'] * POS_SCALE), -32767, 32767).astype(np.int16)
    return q


def simulate_curves(sel, q):
    """Curves of the quantised machines; returns z (M, 64) and ok mask."""
    M = len(q)
    z = np.zeros((M, 64), complex)
    ok = np.zeros(M, bool)
    for i in range(M):
        n = int(sel['n'][i])
        pos = q[i, :n].astype(float) / POS_SCALE
        par = sel['par'][i, :n].astype(int)
        topo = Topo(pos, par, sel['kind'][i, :n], sel['gear'][i, :n], n - 1)
        traj, ms = topo.simulate(topo.params_from_pos(pos)[None])
        if not np.isfinite(traj).all() or ms[0] < 0.15:
            continue
        z[i] = normalise(resample_offset(traj[:, n - 1], [0.0]))[0]
        ok[i] = True
    return z, ok


def embed(z, enc_path, dim_fourier=10):
    zt = torch.from_numpy(z.astype(np.complex64))
    if enc_path:
        ck = torch.load(enc_path, map_location='cpu')
        a = ck['args']
        model = Encoder(a['width'], a['dim'])
        model.load_state_dict(ck['model'])
        model.eval()
        with torch.no_grad():
            e = torch.cat([model(zt[i:i + 8192]) for i in range(0, len(zt), 8192)])
        return e.numpy(), 'encoder'
    return fourier_descriptor(zt, dim_fourier).numpy(), 'fourier'


def pack(sel, q, emb, outdir, kind_name, extra):
    M = len(q)
    n = sel['n'].astype(np.uint8)
    pos = np.concatenate([q[i, :n[i]].ravel() for i in range(M)]).astype(np.int16)
    struct = []
    for i in range(M):
        k = sel['kind'][i, :n[i]].astype(int)
        g = sel['gear'][i, :n[i]].astype(int)
        a = sel['par'][i, :n[i], 0].astype(int)
        b = sel['par'][i, :n[i], 1].astype(int)
        a = np.where(a < 0, 15, a)
        b = np.where(b < 0, 15, b)
        s = np.stack([k | ((g + 4) << 2), a | (b << 4)], 1).astype(np.uint8)
        struct.append(s.ravel())
    struct = np.concatenate(struct)
    e8 = np.clip(np.round(emb * 127), -127, 127).astype(np.int8)
    sections, blobs, off = {}, [], 0
    for name, arr in [('pos', pos), ('emb', e8.ravel()), ('n', n), ('struct', struct)]:
        b = arr.tobytes()
        sections[name] = [off, len(b)]
        pad = (-len(b)) % 4
        blobs.append(b + b'\0' * pad)
        off += len(b) + pad
    raw = b''.join(blobs)
    os.makedirs(outdir, exist_ok=True)
    with gzip.open(os.path.join(outdir, 'index.bin.gz'), 'wb', compresslevel=9) as f:
        f.write(raw)
    meta = dict(version=1, count=int(M), dim=int(emb.shape[1]), embedding=kind_name,
                posScale=POS_SCALE, joints=int(len(pos) // 2), sections=sections,
                file='index.bin.gz', bytes=len(raw), **extra)
    json.dump(meta, open(os.path.join(outdir, 'index.json'), 'w'), indent=1)
    gz = os.path.getsize(os.path.join(outdir, 'index.bin.gz'))
    print(f'{M} machines, {len(pos) // 2} joints, raw {len(raw) / 1e6:.1f} MB, gzip {gz / 1e6:.1f} MB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sel')
    ap.add_argument('outdir')
    ap.add_argument('--encoder', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pool-size', type=int, default=0, help='machines the selection was drawn from')
    args = ap.parse_args()
    d = np.load(args.sel)
    sel = {k: d[k] for k in ('pos', 'kind', 'par', 'gear', 'n')}
    if args.limit and args.limit < len(sel['n']):
        idx = np.random.default_rng(args.seed).choice(len(sel['n']), args.limit, replace=False)
        sel = {k: v[idx] for k, v in sel.items()}
    q = quantise(sel)
    z, ok = simulate_curves(sel, q)
    print(f'dropped {int((~ok).sum())} machines that lock after quantisation')
    sel = {k: v[ok] for k, v in sel.items()}
    q, z = q[ok], z[ok]
    emb, kind_name = embed(z, args.encoder)
    pack(sel, q, emb, args.outdir, kind_name, dict(poolSize=args.pool_size or int(len(q))))


if __name__ == '__main__':
    main()
