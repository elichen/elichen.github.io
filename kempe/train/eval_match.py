"""Brute-force exact retrieval + LM refinement on test drawings; plots results.

usage: python eval_match.py DATASET.npz OUT.png [--topk 12] [--iters 80]
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from linkage import align_dist, resample, to_complex, normalise, prune
from refine import Topo, fit, resample_offset
from targets import targets, jitter


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def topo_of(ds, mech, joint):
    """Pruned topology for one curve, and its initial design vector."""
    J = int(ds['n_joints'][mech])
    pos, par, kind, gear, tr = prune(ds['pos'][mech, :J].astype(float), ds['parents'][mech, :J],
                                     ds['kind'][mech, :J], ds['gear'][mech, :J], joint)
    topo = Topo(pos, par, kind, gear, tr)
    return topo, topo.params_from_pos(pos)


def run(ds, target_xy, topk=12, iters=80, pool=400, max_extent=4.5):
    t, _ = normalise(to_complex(resample(target_xy[None]))[0])
    t0 = time.time()
    dist, _, _ = align_dist(t, ds['z'])
    dist = np.where(ds['extent'] < max_extent, dist, 1.0)
    t_search = time.time() - t0
    order = np.argsort(dist)[:pool]
    # one candidate per mechanism
    seen, cands = set(), []
    for i in order:
        m = int(ds['mech'][i])
        if m in seen:
            continue
        seen.add(m)
        cands.append(i)
        if len(cands) == topk:
            break
    best = None
    t0 = time.time()
    for i in cands:
        topo, p0 = topo_of(ds, int(ds['mech'][i]), int(ds['joint'][i]))
        p, err, hist, _ = fit(topo, p0, t, iters=iters)
        if best is None or err < best[1]:
            best = (i, err, np.sqrt(dist[i]), topo, p, hist)
    t_fit = time.time() - t0
    return best, np.sqrt(dist[order[0]]), t_search, t_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('out')
    ap.add_argument('--topk', type=int, default=12)
    ap.add_argument('--iters', type=int, default=80)
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    ds = load(args.dataset)
    rng = np.random.default_rng(0)
    T = targets()
    names = [k for k in T if not args.only or k in args.only.split(',')]
    cols = 4
    rows = (len(names) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    for ax, name in zip(axs.ravel(), names):
        xy = jitter(T[name], rng, 0.015)
        best, retr, ts, tf = run(ds, xy, args.topk, args.iters)
        i, err, d0, topo, p, hist = best
        print(f'{name:10s} retrieval {retr:.3f}  chosen-start {d0:.3f}  refined {err:.3f}  '
              f'joints {topo.J}  search {ts:.1f}s fit {tf:.1f}s', flush=True)
        # draw the fitted curve aligned onto the target
        traj, _ = topo.simulate(p[None])
        cz = to_complex(traj[0, topo.tracer])
        rs = normalise(to_complex(resample(traj[:, topo.tracer]))[0:1])[0][0]
        tz = normalise(to_complex(resample(xy[None]))[0:1])[0][0]
        # similarity that maps the raw curve onto the target frame
        tn = tz
        from refine import best_alignment, target_frame
        _, v, s = best_alignment(tn, rs)
        tt = target_frame(tn, v, s)
        cc = rs
        alpha = (np.conj(cc) * tt).sum() / len(tt)
        # apply to all joints: normalise with the curve's mean / rms
        raw = to_complex(resample(traj[:, topo.tracer]))[0]
        mu = raw.mean()
        sc = np.sqrt((np.abs(raw - mu) ** 2).mean())
        def tf_(w):
            w = alpha * (w - mu) / sc
            if v in (2, 3):
                w = np.conj(w)
            return w
        ax.plot(tn.real, tn.imag, color='#bbb', lw=6)
        cw = tf_(cz)
        ax.plot(cw.real, cw.imag, color='#c22', lw=1.5)
        P = tf_(to_complex(traj[0, :, 0]))
        ext_ = np.abs(tf_(to_complex(traj[0]))).max()
        for j in topo.dyads:
            for q in topo.parents[j]:
                ax.plot([P[j].real, P[q].real], [P[j].imag, P[q].imag], 'k-', lw=1, alpha=0.5)
        for c in topo.cranks:
            q = topo.parents[c, 0]
            ax.plot([P[q].real, P[c].real], [P[q].imag, P[c].imag], 'b-', lw=2)
            ax.text(P[q].real, P[q].imag, f'x{topo.ratio[c]}', color='b')
        g = topo.grounds
        ax.plot(P[g].real, P[g].imag, 'k^')
        ax.set_aspect('equal')
        ax.set_title(f'{name}: {err*100:.1f}% (retr {retr*100:.1f}%) J={topo.J} C={len(topo.cranks)} ext={ext_:.1f}')
    plt.tight_layout()
    plt.savefig(args.out, dpi=55)


if __name__ == '__main__':
    main()
