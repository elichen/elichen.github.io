"""Does priming carry a writer's hand? Measured on held-out writers.

For each held-out writer, prime with their "The quick brown fox" and write
sentences they really wrote. Measure scale-free style features on the
generated and the real lines, then ask which writer's real lines the generated
ones are closest to (12-way, chance = 1/12). Controls: no prime, and a prime
from a different held-out writer (which should then point at *that* writer).
"""
import argparse, sys, math, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('ckpt')
ap.add_argument('--bias', type=float, default=0.75)
ap.add_argument('--n', type=int, default=4, help='sentences per writer')
ap.add_argument('--seeds', type=int, default=2)
ev = ap.parse_args()
sys.argv = ['sample.py', ev.ckpt]
exec(open('sample.py').read().split("if __name__ == '__main__':")[0])

PROMPT = 'The quick brown fox'


def strokes_of(pts, pen):
    out, cur = [], [pts[0]]
    for i in range(1, len(pts)):
        if pen[i] > 0.5:
            out.append(np.array(cur)); cur = [pts[i]]
        else:
            cur.append(pts[i])
    out.append(np.array(cur))
    return out


def features(pts, pen, text):
    """slant, width per letter, pen lifts per letter, ascender reach, descender reach"""
    S = strokes_of(pts, pen)
    d = np.concatenate([np.diff(s, axis=0) for s in S if len(s) > 1])
    down = d[(d[:, 1] > 0) & (np.abs(d[:, 1]) > 1.5 * np.abs(d[:, 0]))]   # downward near-vertical moves
    slant = float(np.sum(down[:, 0]) / max(np.sum(down[:, 1]), 1e-6)) if len(down) else 0.0
    letters = max(1, sum(c != ' ' for c in text))
    width = float(np.ptp(pts[:, 0]) / letters)
    lifts = float(sum(len(s) > 1 for s in S) / letters)
    y = pts[:, 1]
    return np.array([slant, width, lifts, float(np.quantile(y, 0.03)), float(np.quantile(y, 0.97))])


NAMES = ['slant', 'width/letter', 'strokes/letter', 'ascender reach', 'descender reach']
vw = meta['val_writers']
mine = {w: [s for s in samples if s['w'] == w] for w in vw}
common = None
for w in vw:
    texts = set(s['text'] for s in mine[w] if s['text'] != PROMPT and all(c in CI for c in s['text']))
    common = texts if common is None else common & texts
rng = np.random.default_rng(0)
sentences = sorted(common)
rng.shuffle(sentences)
sentences = [t for t in sentences if len(t) >= 12][:ev.n]
print('sentences:', sentences, flush=True)

real = {}
for w in vw:
    fs = []
    for t in sentences:
        s = next(x for x in mine[w] if x['text'] == t)
        pen = np.concatenate([[1], s['lift'][:-1]])
        fs.append(features(s['pts'].astype(float), pen, t))
    real[w] = np.mean(fs, 0)
R = np.array([real[w] for w in vw])
mu, sd = R.mean(0), R.std(0) + 1e-9


def primed_feats(prime_w, seed0):
    s = next(x for x in mine[prime_w] if x['text'] == PROMPT)
    px = to_input(s['pts'], s['lift'])
    fs = []
    for k, t in enumerate(sentences):
        for j in range(ev.seeds):
            pts, pen = generate(px, PROMPT, t, ev.bias, seed0 + 100 * k + j)
            pts = pts + s['pts'][-1]
            fs.append(features(pts, pen, t))
    return np.mean(fs, 0)


def z(X):
    X = np.asarray(X, float)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


Rz = z(R)


def ranks(F, targets):
    """Rank of each target writer among real writers, comparing standardized
    features (each set standardized on its own, so a shift shared by all
    generated lines doesn't decide the answer)."""
    Fz = z(F)
    out = []
    for f, t in zip(Fz, targets):
        order = np.argsort(np.linalg.norm(Rz - f, axis=1))
        out.append([vw[i] for i in order].index(t))
    return out


G, GO, others = [], [], []
for i, w in enumerate(vw):
    G.append(primed_feats(w, 1))
    other = vw[(i + 5) % len(vw)]
    others.append(other)
    GO.append(primed_feats(other, 2))
    print(f'writer {w} done', flush=True)
fu = []
for j in range(len(vw)):
    fs = []
    for k, t in enumerate(sentences):
        pts, pen = generate(None, '', t, ev.bias, 500 + 100 * k + j)
        fs.append(features(pts, pen, t))
    fu.append(np.mean(fs, 0))
res = {'self': ranks(G, vw), 'other': ranks(GO, others), 'none': ranks(fu, vw)}


def pairwise(F, targets):
    """Over all ordered pairs (A, B) of writers: is the line primed by A closer to A's
    real hand than to B's? Chance is 50%."""
    Fz = z(F); idx = {w: i for i, w in enumerate(vw)}
    wins = tot = 0
    for f, t in zip(Fz, targets):
        da = np.linalg.norm(Rz[idx[t]] - f)
        for b in vw:
            if b == t: continue
            wins += da < np.linalg.norm(Rz[idx[b]] - f); tot += 1
    return wins / tot


pair = {'self': pairwise(G, vw), 'other': pairwise(GO, others), 'none': pairwise(fu, vw)}
G = np.array(G)
out = {}
for k, v in res.items():
    v = np.array(v)
    out[k] = dict(top1=float(np.mean(v == 0)), top3=float(np.mean(v < 3)), mean_rank=float(np.mean(v) + 1), pairwise=float(pair[k]))
    print(f'{k:6s} top-1 {out[k]["top1"]:.2f}  top-3 {out[k]["top3"]:.2f}  mean rank {out[k]["mean_rank"]:.1f} of {len(vw)}  pairwise {pair[k]:.2f}')
corr = {}
for j, n in enumerate(NAMES):
    r = float(np.corrcoef(G[:, j], R[:, j])[0, 1])
    corr[n] = r
    print(f'  {n:16s} generated vs real across writers: r = {r:.2f}')
out['corr'] = corr; out['sentences'] = sentences; out['bias'] = ev.bias
json.dump(out, open(ev.ckpt.rsplit('/', 1)[0] + f'/style_eval_b{ev.bias}.json', 'w'), indent=1)
