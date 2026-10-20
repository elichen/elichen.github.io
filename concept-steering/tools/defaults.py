"""Default strength (1x) per concept, written into data/stats.bin (rows 2 and 3).

Reads the per-strength measurements from the Nitro calibration passes (calib_all.py output dirs: base.pt +
chunk_*.pt, strengths 0.5-4 x best on 4 prompts) and the fluency table (table.json, used for concepts that
never show up). A concept's 1x is the smallest strength where it clearly shows up in what the model writes
(mean act/best over the continuation on an unsteered re-read rises by >= 0.08) while the text stays readable
(judge PPL <= 3x unsteered and distinct-3 >= 0.8). Row 3 holds that rise; 0 = steering with it doesn't make
the model write about it, and the page gives it no pin (about 93% of concepts).

  python3 tools/defaults.py <table.json> <calib dir> [<calib dir> ...]
"""
import glob, json, math, sys
import numpy as np
import torch

T, PPL, D3, FLOOR = 0.08, 3.0, 0.8, 0.25
ROOT = __file__.rsplit("/tools/", 1)[0]


def iso(y):
    """pool-adjacent-violators: non-decreasing fit"""
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return [v for v, n in blocks for _ in range(n)]


table = json.load(open(sys.argv[1]))
acc = {}
for d in sys.argv[2:]:
    base = torch.load(f"{d}/base.pt")
    nll0, d30, p0 = base["nll"].mean().item(), base["d3"].mean().item(), base["pres"].mean((0, 1))
    for f in sorted(glob.glob(f"{d}/chunk_*.pt")):
        c = torch.load(f)
        S = c["strengths"]
        for i, j in enumerate(c["concepts"]):
            acc.setdefault(j, []).append(([nll0] + c["nll"][i].mean((0, 2)).tolist(),
                                          [d30] + c["d3"][i].mean((0, 2)).tolist(),
                                          [p0[j].item()] + c["pres"][i].mean((0, 2)).tolist()))
xs = [0.0] + list(S)
g = np.linspace(0, xs[-1], 801)
stats = np.fromfile(f"{ROOT}/data/stats.bin", np.float32).reshape(4, -1)
for j, runs in acc.items():
    nll, d3, pres = (np.mean([r[k] for r in runs], 0) for k in range(3))
    bad = iso([max(math.exp(n - nll[0]) / PPL, (1 - d) / (1 - D3)) for n, d in zip(nll, d3)])   # 1 = at the limit
    pr = iso(list(pres))
    bad, pr = np.interp(g, xs, bad), np.interp(g, xs, np.array(pr) - pr[0])
    hit = g[(pr >= T) & (bad <= 1)]
    s_fluent = table[str(j)]["s"]
    if len(hit) and s_fluent > 0:
        s = max(FLOOR, hit.min())
        stats[2, j], stats[3, j] = s, np.interp(s, g, pr)
    else:
        stats[2, j], stats[3, j] = 0.8 * s_fluent, 0
stats.tofile(f"{ROOT}/data/stats.bin")
print(f"{(stats[3] > 0).sum()} of {len(acc)} concepts show up when steered")
