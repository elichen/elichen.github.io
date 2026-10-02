"""Write evalset.json for web/eval_page.js: the sentences, primes and real lines that
eval_style.py uses, so the page's own worker can be scored the same way."""
import pickle, json, numpy as np
D = pickle.load(open("data/brush_norm.pkl", "rb"))
meta = json.load(open("runs/h1/meta.json"))
chars = sorted(set(''.join(s['text'] for s in D['samples'])))
CI = set(chars)
PROMPT = 'The quick brown fox'
vw = meta['val_writers']
mine = {w: [s for s in D['samples'] if s['w'] == w] for w in vw}
common = None
for w in vw:
    texts = set(s['text'] for s in mine[w] if s['text'] != PROMPT and all(c in CI for c in s['text']))
    common = texts if common is None else common & texts
rng = np.random.default_rng(0)
sentences = sorted(common); rng.shuffle(sentences)
sentences = [t for t in sentences if len(t) >= 12][:6]
out = dict(sentences=sentences, writers={})
for w in vw:
    p = next(x for x in mine[w] if x['text'] == PROMPT)
    out['writers'][w] = dict(prime=dict(pts=p['pts'].tolist(), lift=p['lift'].tolist()),
        real=[dict(pts=(s := next(x for x in mine[w] if x['text'] == t))['pts'].tolist(), lift=s['lift'].tolist()) for t in sentences])
json.dump(out, open("evalset.json", "w"))
