"""Export held-out teacher clips for the web app's side-by-side replay.

python export_replays.py data/valid --out ../replays --pick 0:3,1:5

Each clip: <name>.json (cloth size, fabric, wind, gripper script, obstacle tracks) and
<name>.bin (teacher positions as int16, quantized over the clip's bounding box).
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("valid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pick", default="", help="chunk:traj pairs, comma separated")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.valid, "chunk_*.npz")))
    picks = [tuple(int(v) for v in p.split(":")) for p in args.pick.split(",") if p]
    index = []
    for c, b in picks:
        z = np.load(files[c])
        size = [int(v) for v in z["size"]]
        n = size[0] * size[1]
        pos = z["pos"][b][:, :n]
        lo, hi = pos.min((0, 1)), pos.max((0, 1))
        span = np.maximum(hi - lo, 1e-6)
        q = np.round((pos - lo) / span * 65535 - 32768).astype(np.int16)
        name = f"clip_{c}_{b}_{z['kinds'][b]}"
        q.tofile(os.path.join(args.out, name + ".bin"))
        grips = []
        for k in range(len(z["h_vid"][b])):
            v = int(z["h_vid"][b][k])
            if v >= 0:
                grips.append({"vertex": v, "t0": int(z["h_t0"][b][k]), "t1": int(z["h_t1"][b][k])})
        meta = {
            "name": name, "kind": str(z["kinds"][b]), "size": size, "frames": int(pos.shape[0]),
            "lo": lo.tolist(), "span": span.tolist(), "kb": float(z["kb"][b]), "mu": float(z["mu"][b]),
            "wind": z["wind"][b].tolist(), "grips": grips,
            "spheres": [{"r": float(z["sph_r"][b][k]), "c": z["sph_c"][b][:, k].round(5).tolist()}
                        for k in range(len(z["sph_on"][b])) if z["sph_on"][b][k] > 0],
            "capsules": [{"r": float(z["cap_r"][b][k]), "a": z["cap_a"][b][:, k].round(5).tolist(),
                          "b": z["cap_b"][b][:, k].round(5).tolist()}
                         for k in range(len(z["cap_on"][b])) if z["cap_on"][b][k] > 0],
            "solver_ms_per_frame": None,
        }
        with open(os.path.join(args.out, name + ".json"), "w") as f:
            json.dump(meta, f)
        index.append({"name": name, "kind": meta["kind"], "size": size})
        print(name, size, meta["kind"], f"{q.nbytes / 1e6:.2f} MB")
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
