"""Food-seeking statistics of seek trials, in the forms the papers report:

  reach        share of trials whose center of mass comes within R_MM (default 5 mm, as
               Pierce-Shimomura et al. 1999's peak zone) of the food, and the time
  weathervane  (Iino & Yoshida 2009) curving rate (°/mm, + = counterclockwise) per gradient
               across the heading (mM/mm), from a regression over forward runs; their 12.7;
               and the SD of curving, their 32.3 °/mm
  bearing      curving rate toward the food by bearing (their Fig. 2A)
  pirouettes   turn rate (per min) by smoothed dC/dt (mM/s); Iino & Yoshida's fit

  python tools/seek_stats.py <food distance> <trial.csv>...   (food angle from the name: *_<angle>.csv)
  env: SIGMA, C_PEAK (the hill, as in seek), R_MM
"""
import os
import re
import sys

import numpy as np

MM = 0.63  # body units per mm
SIGMA = float(os.environ.get("SIGMA", 6.3))
C_PEAK = float(os.environ.get("C_PEAK", 4.0))
R = float(os.environ.get("R_MM", 5.0)) * MM
WIN = 2.0  # s: one gait cycle


def food_of(path, dist):
    a = np.radians(float(re.search(r"_(-?[\d.]+)\.csv$", path).group(1)))
    return np.array([dist * np.sin(a), -dist * np.cos(a)])


def grad(p, food):
    """∇C in mM per body unit"""
    d = p - food
    return -C_PEAK * np.exp(-(d**2).sum(-1) / (2 * SIGMA**2))[:, None] * d / SIGMA**2


def main():
    dist = float(sys.argv[1])
    files = sys.argv[2:]
    reach, G, H, B, HB, tdc, rdc = [], [], [], [], [], [], []
    total_t = 0
    for f in files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        food = food_of(f, dist)
        t = d["t"]
        dt = t[1] - t[0]
        c = np.c_[d["com_x"], d["com_z"]]
        r = np.linalg.norm(c - food, axis=1)
        hit = np.nonzero(r < R)[0]
        reach.append(t[hit[0]] if len(hit) else np.nan)
        total_t = t[-1]
        n = int(round(WIN / dt))
        cs, ms = c[::n], d["mode"][::n]
        v = np.diff(cs, axis=0)
        head = np.arctan2(v[:, 1], v[:, 0])
        dh = np.angle(np.exp(1j * np.diff(head)))  # turn from window i to i+1 (+ = ccw)
        step = (np.linalg.norm(v[:-1], axis=1) + np.linalg.norm(v[1:], axis=1)) / 2 / MM
        mid = cs[1:-1]
        u = v[:-1] / np.linalg.norm(v[:-1], axis=1, keepdims=True)
        left = np.c_[-u[:, 1], u[:, 0]]  # ccw normal of the heading
        g = (grad(mid, food) * left).sum(1) * MM  # mM/mm across the heading, + = to the left
        to = food - mid
        bear = np.angle(np.exp(1j * (np.arctan2(to[:, 1], to[:, 0]) - head[:-1])))
        ok = (ms[:-2] == 0) & (ms[1:-1] == 0) & (ms[2:] == 0) & (step > 0.05)
        ok[:3] = False
        if len(hit):
            ok &= np.arange(len(ok)) * n < hit[0]
        h = np.degrees(dh) / step
        G += list(g[ok])
        H += list(h[ok])
        B += list(np.degrees(np.abs(bear[ok])))
        HB += list(h[ok] * np.sign(bear[ok]))
        m = d["mode"]
        k = int(round(2.0 / dt))
        dc = np.convolve(d["dcdt"], np.ones(k) / k, mode="full")[: len(m)]
        start = np.nonzero((m[1:] == 1) & (m[:-1] == 0))[0] + 1
        before = hit[0] if len(hit) else len(m)
        tdc += list(dc[start[start < before]])
        rdc += list(dc[:before][m[:before] == 0])
    reach = np.array(reach)
    ok = np.isfinite(reach)
    print(f"{len(files)} trials of {total_t:.0f} s, food {dist / MM:.1f} mm away")
    print(f"  reached the {R / MM:.0f} mm zone: {ok.mean() * 100:.0f}%" + (f", median {np.median(reach[ok]):.0f} s" if ok.any() else ""))
    G, H = np.array(G), np.array(H)
    if len(G) > 10:
        slope = np.polyfit(G, H, 1)[0]
        print(f"  weathervane: {slope:+.1f} °/mm per mM/mm across the heading (Iino & Yoshida 12.7); curving SD {H.std():.1f} °/mm (32.3); n {len(G)}")
        B, HB = np.array(B), np.array(HB)
        print("  curving toward the food by bearing: " + "  ".join(f"{lo}-{lo + 45}°: {HB[(B >= lo) & (B < lo + 45)].mean():+.1f}" for lo in range(0, 180, 45)))
    if rdc:
        td, rd = np.array(tdc), np.array(rdc)
        edges = np.quantile(rd, [0, 0.2, 0.4, 0.6, 0.8, 1])
        print("  turns per min by dC/dt (mM/s, 2 s mean): " + "  ".join(
            f"[{a:+.4f},{b:+.4f}) {((td >= a) & (td < b)).sum() / max(((rd >= a) & (rd < b)).sum() * dt / 60, 1e-9):.2f}" for a, b in zip(edges[:-1], edges[1:])))


if __name__ == "__main__":
    main()
