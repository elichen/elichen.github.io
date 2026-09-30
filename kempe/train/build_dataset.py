"""Generate random dyad linkages in parallel and store their tracer curves.

Every moving dyad joint of every mechanism is a candidate pen. Stored per curve:
the normalised arc-length samples (complex64, N_CURVE) and which mechanism /
joint it came from; per mechanism: joint kinds, parents and initial positions.

usage: python build_dataset.py OUT.npz --mechs 300000 [--seed 0]
"""

import argparse
import multiprocessing as mp
import numpy as np
from linkage import generate, resample, to_complex, normalise, N_CURVE

SIZES = [(g, c, d) for g in (2, 3, 4) for c in (1, 2, 3) for d in range(2, 11) if c <= g]


def mech_size_weights():
    # favour mid-sized machines; small ones are few in number of distinct curves
    w = np.array([(1.0 + 0.3 * d) * (1.0, 1.5, 1.2)[c - 1] for g, c, d in SIZES])
    return w / w.sum()


def work(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    g, c, d = SIZES[rng.choice(len(SIZES), p=mech_size_weights())]
    m = generate(rng, n, g, c, d)
    M, J = m['pos'].shape[:2]
    kind, par, traj = m['kind'], m['parents'], m['traj']
    # ancestor sets, vectorised over the batch: anc[m, j, k] = j depends on k
    anc = np.zeros((M, J, J), bool)
    rows = np.arange(M)
    for k in range(J):
        anc[:, k, k] = True
        if kind[k] == 1:
            anc[rows, k] |= anc[rows, par[:, k, 0]]
        elif kind[k] == 2:
            anc[rows, k] |= anc[rows, par[:, k, 0]] | anc[rows, par[:, k, 1]]
    dy = np.flatnonzero(kind == 2)
    curves = traj[:, dy]                                       # (M, nd, T, 2)
    rs = resample(curves.reshape(-1, *curves.shape[2:]))       # (M*nd, n, 2)
    z, scale = normalise(to_complex(rs))
    mu = to_complex(rs).mean(-1).reshape(M, len(dy))
    sc = scale[:, 0].reshape(M, len(dy))
    tz = to_complex(traj)                                      # (M, J, T)
    extent = np.zeros((M, len(dy)), np.float32)
    for i, j in enumerate(dy):
        r = np.abs(tz - mu[:, i, None, None]).max(-1)          # (M, J)
        extent[:, i] = np.where(anc[:, j], r, 0).max(-1) / sc[:, i]
    motor = anc[:, dy, g]                                      # depends on the motor crank
    n_pruned = anc[:, dy].sum(-1).astype(np.int8)
    return dict(kind=kind, pos=m['pos'].astype(np.float32), parents=par, gear=m['ratio'],
                z=z.astype(np.complex64).reshape(M, len(dy), -1),
                extent=extent, motor=motor, n_pruned=n_pruned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--mechs', type=int, default=300000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=2000)
    ap.add_argument('--max-extent', type=float, default=5.0)
    args = ap.parse_args()
    jobs = [(args.seed * 1000003 + i, args.chunk) for i in range(args.mechs // args.chunk)]
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        parts = pool.map(work, jobs, chunksize=1)
    # pack: mechanisms padded to max joints
    Jmax = max(p['pos'].shape[1] for p in parts)
    pos, par, kind, nj, rat = [], [], [], [], []
    z, mech_id, joint_id, extent, n_pruned = [], [], [], [], []
    base = 0
    for p in parts:
        M, J = p['pos'].shape[:2]
        pp = np.zeros((M, Jmax, 2), np.float32); pp[:, :J] = p['pos']
        pr = -np.ones((M, Jmax, 2), np.int16); pr[:, :J] = p['parents']
        kd = -np.ones((M, Jmax), np.int8); kd[:, :J] = p['kind']
        ra = np.zeros((M, Jmax), np.int8); ra[:, :J] = p['gear']; rat.append(ra)
        pos.append(pp); par.append(pr); kind.append(kd); nj.append(np.full(M, J, np.int8))
        dy = np.flatnonzero(p['kind'] == 2)
        keep = p['motor'].ravel() & (p['extent'].ravel() < args.max_extent)
        z.append(p['z'].reshape(-1, p['z'].shape[-1])[keep])
        mech_id.append((base + np.repeat(np.arange(M), len(dy))).astype(np.int32)[keep])
        joint_id.append(np.tile(dy, M).astype(np.int8)[keep])
        extent.append(p['extent'].ravel()[keep])
        n_pruned.append(p['n_pruned'].ravel()[keep])
        base += M
    np.savez(args.out, pos=np.concatenate(pos), parents=np.concatenate(par),
             kind=np.concatenate(kind), gear=np.concatenate(rat), n_joints=np.concatenate(nj),
             z=np.concatenate(z), mech=np.concatenate(mech_id),
             joint=np.concatenate(joint_id), extent=np.concatenate(extent),
             n_pruned=np.concatenate(n_pruned))
    print('mechanisms', base, 'curves', sum(len(a) for a in mech_id))


if __name__ == '__main__':
    main()
