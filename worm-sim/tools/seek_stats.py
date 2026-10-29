"""Food-seeking statistics of seek trials, in the forms the papers report:

  reach      first time the head is within R of the food (s), and the share of trials
  approach   mean speed toward the food over the trial / crawl speed
  weathervane (Iino & Yoshida 2009, Fig. 3): curving rate (deg/mm, + = toward the
             food) against bearing (angle between heading and the food), forward runs
  pirouettes (Pierce-Shimomura et al. 1999, Fig. 4): sharp-turn rate (per min)
             against d ln C/dt in the second before

  python tools/seek_stats.py <food distance> <trial.csv>...   (food angle from the name: *_<angle>.csv)
"""
import re
import sys

import numpy as np

MM = 0.63  # body units per mm
R = 0.6  # "reached" radius (units, about 1 mm)


def food_of(path, dist):
    a = np.radians(float(re.search(r"_(-?[\d.]+)\.csv$", path).group(1)))
    return np.array([dist * np.sin(a), -dist * np.cos(a)])


def main():
    dist = float(sys.argv[1])
    files = sys.argv[2:]
    reach, appr, B, C, turns_dc = [], [], [], [], []
    run_dc = []
    for f in files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        food = food_of(f, dist)
        t = d["t"]
        dt = t[1] - t[0]
        h = np.c_[d["head_x"], d["head_z"]]
        c = np.c_[d["com_x"], d["com_z"]]
        r = np.linalg.norm(h - food, axis=1)
        hit = np.nonzero(r < R)[0]
        reach.append(t[hit[0]] if len(hit) else np.nan)
        end = hit[0] if len(hit) else len(t) - 1
        appr.append((np.linalg.norm(c[0] - food) - np.linalg.norm(c[end] - food)) / max(t[end], 1))
        # heading and curving over half-cycle-free windows (2 s = one gait cycle)
        n = int(round(2.0 / dt))
        cs, ms = c[::n], d["mode"][::n]
        v = np.diff(cs, axis=0)
        head = np.arctan2(v[:, 1], v[:, 0])
        to = food - cs[:-1]
        bear = np.angle(np.exp(1j * (np.arctan2(to[:, 1], to[:, 0]) - head)))  # + = food to the left (ccw)
        dh = np.angle(np.exp(1j * np.diff(head)))
        step = np.linalg.norm(v, axis=1)[1:] / MM  # mm
        ok = (ms[1:-1] == 0) & (ms[2:] == 0) & (step > 0.05) & (np.linalg.norm(to[1:], axis=1) > 2 * R)
        ok[: 3] = False
        # curving toward the food is + : sign of the turn matches the side of the food
        B += list(np.degrees(np.abs(bear[:-1][ok])))
        C += list(np.degrees(dh[ok] * np.sign(bear[:-1][ok])) / step[ok])
        # turns vs d ln C/dt over the preceding second
        m = d["mode"]
        k = int(round(1.0 / dt))
        dc = np.convolve(d["dlogc"], np.ones(k) / k, mode="full")[: len(m)]
        start = np.nonzero((m[1:] == 1) & (m[:-1] == 0))[0] + 1
        turns_dc += list(dc[start])
        fwd = (m == 0) & (r > 2 * R)
        run_dc += list(dc[fwd])
    reach = np.array(reach)
    print(f"{len(files)} trials, food {dist / MM:.1f} mm away")
    print(f"  reached (head within {R / MM:.1f} mm): {np.isfinite(reach).mean() * 100:.0f}%, median time {np.nanmedian(reach) if np.isfinite(reach).any() else float('nan'):.0f} s")
    print(f"  approach speed {np.mean(appr) / MM * 1000:.1f} µm/s")
    B, C = np.array(B), np.array(C)
    if len(B):
        print("  weathervane: curving rate toward food (deg/mm) by bearing")
        for lo in range(0, 180, 30):
            s = (B >= lo) & (B < lo + 30)
            if s.sum() > 3:
                print(f"    {lo:3d}-{lo + 30:3d} deg: {C[s].mean():+6.1f} ± {C[s].std() / np.sqrt(s.sum()):.1f}  (n {s.sum()})")
    if turns_dc:
        td, rd = np.array(turns_dc), np.array(run_dc)
        print("  pirouettes: turn rate (per min) by d ln C/dt (per s) in the second before")
        edges = np.quantile(rd, [0, 0.2, 0.4, 0.6, 0.8, 1])
        for a, b in zip(edges[:-1], edges[1:]):
            nt = ((td >= a) & (td < b)).sum()
            tt = ((rd >= a) & (rd < b)).sum() * (dt) / 60
            print(f"    [{a:+.4f}, {b:+.4f}): {nt / tt:.2f}/min (turns {nt})")


if __name__ == "__main__":
    main()
