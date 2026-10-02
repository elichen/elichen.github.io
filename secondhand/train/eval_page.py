"""Score web/eval_page.js output with eval_style.py's measures: python3 eval_page.py out*.json"""
import json, sys, numpy as np
E = json.load(open('evalset.json')); vw = list(E['writers']); S = E['sentences']
def strokes_of(pts, pen):
    out, cur = [], [pts[0]]
    for i in range(1, len(pts)):
        if pen[i] > 0.5: out.append(np.array(cur)); cur = [pts[i]]
        else: cur.append(pts[i])
    out.append(np.array(cur)); return out
def features(pts, pen, text):
    S_ = strokes_of(pts, pen)
    d = np.concatenate([np.diff(s, axis=0) for s in S_ if len(s) > 1])
    down = d[(d[:, 1] > 0) & (np.abs(d[:, 1]) > 1.5 * np.abs(d[:, 0]))]
    slant = float(np.sum(down[:, 0]) / max(np.sum(down[:, 1]), 1e-6)) if len(down) else 0.0
    letters = max(1, sum(c != ' ' for c in text))
    width = float(np.ptp(pts[:, 0]) / letters)
    lifts = float(sum(len(s) > 1 for s in S_) / letters)
    y = pts[:, 1]
    return np.array([slant, width, lifts, float(np.quantile(y, 0.03)), float(np.quantile(y, 0.97))])
NAMES = ['slant', 'width/letter', 'strokes/letter', 'ascender reach', 'descender reach']
R = np.array([np.mean([features(np.array(r['pts']), np.concatenate([[1], r['lift'][:-1]]), t) for r, t in zip(E['writers'][w]['real'], S)], 0) for w in vw])
z = lambda X: (np.asarray(X) - np.mean(X, 0)) / (np.std(X, 0) + 1e-9)
Rz = z(R)
def ranks(F, targets):
    Fz = z(F); out = []
    for f, t in zip(Fz, targets):
        order = np.argsort(np.linalg.norm(Rz - f, axis=1)); out.append([vw[i] for i in order].index(t))
    return np.array(out)
def pairwise(F, targets):
    Fz = z(F); idx = {w: i for i, w in enumerate(vw)}; wins = tot = 0
    for f, t in zip(Fz, targets):
        da = np.linalg.norm(Rz[idx[t]] - f)
        for b in vw:
            if b != t: wins += da < np.linalg.norm(Rz[idx[b]] - f); tot += 1
    return wins / tot
res = {}
for f in sys.argv[1:]: res.update(json.load(open(f)))
def feats(key):
    fs = []
    for g in res[key]['gens']:
        for q, t in zip(g, S):
            q = np.array(q)
            if len(q) > 3: fs.append(features(q[:, :2], q[:, 2], t))
    return np.mean(fs, 0)
G = np.array([feats(f'{w}:self') for w in vw])
others = [vw[(i + 5) % 12] for i in range(12)]
GO = np.array([feats(f'{w}:other') for w in vw])
for name, F, T in [('self', G, vw), ('other', GO, others)]:
    rk = ranks(F, T)
    print(f'  {name:5s} top-1 {np.mean(rk == 0):.2f}  top-3 {np.mean(rk < 3):.2f}  pairwise {pairwise(F, T):.2f}')
for j, n in enumerate(NAMES):
    print(f'  {n:16s} r = {np.corrcoef(G[:, j], R[:, j])[0, 1]:.2f}   gen/real median {np.median(G[:, j] / R[:, j]) if j in (1, 2) else float("nan"):.2f}')
