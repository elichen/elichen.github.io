"""BRUSH -> normalized, arc-length-resampled samples.

Each sample: absolute points in x-height units (baseline at y=0, y down),
resampled to fixed spacing along each pen stroke, plus a pen-lift flag.
"""
import pickle, numpy as np, os, json, sys
from multiprocessing import Pool

ROOT = 'data/BRUSH'
SPACING = 0.2          # resample step, in x-heights
XCHARS = set('acemnorsuvwxz')
DESC = set('gjpqyQJ,;()[]{}/')
NOSIT = DESC | set('\'"-=*+^~`_<>#$%&@|\\ ')

def resample(stroke, step):
    if len(stroke) == 1:
        return stroke
    d = np.hypot(*np.diff(stroke, axis=0).T)
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < step * 0.5:
        return stroke[[0, -1]] if s[-1] > 1e-6 else stroke[:1]
    n = max(2, int(round(s[-1] / step)) + 1)
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, stroke[:, 0]), np.interp(u, s, stroke[:, 1])], 1)

def load(args):
    w, f = args
    t, p, c = pickle.load(open(f'{ROOT}/{w}/{f}', 'rb'))
    p = np.asarray(p, np.float64); lab = np.asarray(c).argmax(1)
    bottoms, tops, xs = [], [], []
    for i, ch in enumerate(t):
        m = lab == i
        if not m.any():
            continue
        if ch not in NOSIT:
            bottoms.append(p[m, 1].max())
        if ch in XCHARS:
            tops.append((p[m, 1].min(), p[m, 1].max()))
    base = np.median(bottoms) if bottoms else np.nan
    xh = np.median([b - a for a, b in tops]) if len(tops) >= 2 else np.nan
    # word gaps: horizontal gap between ink of consecutive words
    gaps = []
    words, cur = [], []
    for i, ch in enumerate(t):
        if ch == ' ':
            if cur: words.append(cur); cur = []
        else:
            cur.append(i)
    if cur: words.append(cur)
    ext = []
    for wd in words:
        m = np.isin(lab, wd)
        if m.any(): ext.append((p[m, 0].min(), p[m, 0].max()))
    for (a0, a1), (b0, b1) in zip(ext, ext[1:]):
        gaps.append(b0 - a1)
    return dict(w=int(w), f=int(f), text=t, pts=p, base=base, xh=xh, gaps=gaps)

def main():
    jobs = [(w, f) for w in os.listdir(ROOT) if w.isdigit()
            for f in os.listdir(f'{ROOT}/{w}') if f.isdigit()]
    with Pool(12) as pool:
        raw = pool.map(load, jobs, chunksize=64)
    by_w = {}
    for r in raw:
        by_w.setdefault(r['w'], []).append(r)
    out = []
    stats = {}
    for w, rs in by_w.items():
        wxh = np.nanmedian([r['xh'] for r in rs])
        g = [x for r in rs for x in r['gaps']]
        stats[w] = dict(xh=float(wxh), gap=float(np.median(g) / wxh))
        for r in rs:
            xh = r['xh'] if np.isfinite(r['xh']) and 0.6 < r['xh'] / wxh < 1.6 else wxh
            base = r['base']
            if not np.isfinite(base):
                continue
            p = r['pts']
            xy = np.stack([(p[:, 0] - p[:, 0].min()) / xh, (p[:, 1] - base) / xh], 1)
            ends = np.where(p[:, 2] > 0.5)[0]
            if len(ends) == 0 or ends[-1] != len(p) - 1:
                ends = np.append(ends, len(p) - 1)
            strokes, st = [], 0
            for e in ends:
                s = resample(xy[st:e + 1], SPACING)
                strokes.append(s); st = e + 1
            pts = np.concatenate(strokes).astype(np.float32)
            lift = np.zeros(len(pts), np.uint8)
            lift[np.cumsum([len(s) for s in strokes]) - 1] = 1
            out.append(dict(w=w, f=r['f'], text=r['text'], pts=pts, lift=lift, xh=float(xh)))
    print('samples', len(out), 'writers', len(by_w))
    L = np.array([len(o['pts']) for o in out])
    print('points/sample pct', np.percentile(L, [1, 50, 99]))
    print('points/char', np.median([len(o['pts']) / len(o['text']) for o in out]))
    gaps = np.array([s['gap'] for s in stats.values()]); print('word gap (xh) pct', np.percentile(gaps, [5, 50, 95]))
    pickle.dump(dict(samples=out, writers=stats, spacing=SPACING), open('data/brush_norm.pkl', 'wb'))

if __name__ == '__main__':
    main()
