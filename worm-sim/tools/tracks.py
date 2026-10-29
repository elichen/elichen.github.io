"""Plot seek trial tracks (seek's CSV output) on the plate, colored by mode.
  python tools/tracks.py <out.png> <trial.csv>... [--food d,angle]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

args = [a for a in sys.argv[2:] if not a.startswith("--")]
food = [a for a in sys.argv[2:] if a.startswith("--food")]
fig, ax = plt.subplots(figsize=(7, 7))
for f in args:
    d = np.genfromtxt(f, delimiter=",", names=True)
    x, z, m = d["head_x"], d["head_z"], d["mode"]
    ax.plot(x, z, lw=0.6, color="0.3")
    for k, c in [(1, "tab:red"), (2, "tab:blue")]:
        s = m == k
        ax.plot(x[s], z[s], ".", ms=1.5, color=c)
    ax.plot(x[0], z[0], "go", ms=4)
if food:
    dist, ang = map(float, food[0].split("=")[1].split(","))
    a = np.radians(ang)
    fx, fz = dist * np.sin(a), -dist * np.cos(a)
    ax.add_patch(plt.Circle((fx, fz), 0.6, color="goldenrod", alpha=0.5))
ax.set_aspect("equal")
ax.set_xlabel("x (0.63 = 1 mm)")
ax.set_title("head tracks (red reversal, blue omega)")
fig.savefig(sys.argv[1], dpi=110)
