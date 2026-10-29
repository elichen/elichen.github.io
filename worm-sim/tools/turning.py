"""Mean turning rate of seek trials (deg/s, + = counterclockwise in x-z), from the
center of mass heading over whole gait cycles (2 s), forward crawling only.
  python tools/turning.py <trial.csv>..."""
import sys

import numpy as np

for f in sys.argv[1:]:
    d = np.genfromtxt(f, delimiter=",", names=True)
    dt = d["t"][1] - d["t"][0]
    n = int(round(2.0 / dt))
    x, z, m = d["com_x"][::n], d["com_z"][::n], d["mode"][::n]
    h = np.unwrap(np.arctan2(np.diff(z), np.diff(x)))
    w = np.diff(h) / 2.0
    ok = (m[1:-1] == 0) & (m[2:] == 0)
    sp = np.hypot(np.diff(x), np.diff(z)).mean() / 2.0
    print(f"{f}: turning {np.degrees(w[ok][5:]).mean():+.2f} deg/s, speed {sp / 0.63:.3f} L/s")
