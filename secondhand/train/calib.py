"""Fit a baseline / x-height / tilt estimator for a known prompt line.
Features: quantiles of ink height (uniform along arc length). Scale-equivariant
linear fit on BRUSH writers who wrote the prompt, evaluated by writer CV."""
import pickle, numpy as np, json, sys
D = pickle.load(open('data/brush_norm.pkl', 'rb'))
QS = np.array([2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]) / 100

def dense(pts, lift, step):
    out = []; st = 0
    for e in np.where(lift)[0]:
        s = pts[st:e + 1]; st = e + 1
        if len(s) < 2: out.append(s); continue
        d = np.hypot(*np.diff(s, axis=0).T); c = np.concatenate([[0], np.cumsum(d)])
        u = np.arange(0, c[-1] + 1e-9, step)
        out.append(np.stack([np.interp(u, c, s[:, 0]), np.interp(u, c, s[:, 1])], 1))
    return np.concatenate(out)

def feats(pts, lift):
    h = np.ptp(pts[:, 1]); q = dense(pts, lift, h / 150)
    # tilt: least squares y on x
    A = np.stack([q[:, 0], np.ones(len(q))], 1)
    slope = np.linalg.lstsq(A, q[:, 1], rcond=None)[0][0]
    y = q[:, 1] - slope * (q[:, 0] - q[:, 0].mean())
    qq = np.quantile(y, QS)
    return qq, slope

def fit(prompt):
    S = [s for s in D['samples'] if s['text'] == prompt]
    F, slopes = [], []
    for s in S:
        qq, sl = feats(s['pts'].astype(np.float64), s['lift']); F.append(qq); slopes.append(sl)
    F = np.array(F); slopes = np.array(slopes)
    med = F[:, 6:7]; Z = F - med        # relative to median
    rng = np.random.default_rng(0); idx = rng.permutation(len(S)); folds = np.array_split(idx, 5)
    exh, eb = [], []
    for k in range(5):
        te = folds[k]; tr = np.setdiff1d(idx, te)
        w = np.linalg.lstsq(Z[tr], np.ones(len(tr)), rcond=None)[0]       # xh = Z.w
        xh = Z[te] @ w
        v = np.linalg.lstsq(Z[tr], -med[tr, 0], rcond=None)[0]           # baseline = med + Z.v
        b = med[te, 0] + Z[te] @ v
        exh += list(xh - 1); eb += list(b)
    w = np.linalg.lstsq(Z, np.ones(len(S)), rcond=None)[0]
    v = np.linalg.lstsq(Z, -med[:, 0], rcond=None)[0]
    print(f'{prompt!r}: n={len(S)}  x-height err  rms {np.sqrt(np.mean(np.square(exh))):.3f}  baseline err rms {np.sqrt(np.mean(np.square(eb))):.3f} (x-heights)  slope mean {slopes.mean():.4f} sd {slopes.std():.4f}')
    return dict(q=QS.tolist(), w=w.tolist(), v=v.tolist(), slope=float(slopes.mean()))

out = {p: fit(p) for p in ['The quick brown fox', 'love my big sphinx', 'jogging Norway defy']}
json.dump(out, open('data/calib.json', 'w'))
