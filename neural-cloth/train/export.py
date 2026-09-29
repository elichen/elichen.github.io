"""Export trained cloth-network params for the web app, plus a golden one-step test case.

python export.py runs/c1/params_01000000.pkl --out ../model --valid data/valid
python export.py --random --out ../model --valid data/valid     # untrained weights, for engine tests
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp

import cloth as C
import mgn


def flatten(params):
    out = []

    def mlp(prefix, p):
        for i, lyr in enumerate(p["layers"]):
            out.append((f"{prefix}.l{i}.w", lyr["w"]))
            out.append((f"{prefix}.l{i}.b", lyr["b"]))
        if "ln" in p:
            out.append((f"{prefix}.ln.g", p["ln"]["g"]))
            out.append((f"{prefix}.ln.b", p["ln"]["b"]))

    mlp("enc_node", params["enc_node"])
    for l, p in enumerate(params["enc_edge"]):
        mlp(f"enc_edge{l}", p)
    mlp("enc_world", params["enc_world"])
    for i, blk in enumerate(params["proc"]):
        for k in ("edge", "world", "node"):
            if k in blk:
                mlp(f"proc{i}.{k}", blk[k])
    mlp("dec", params["dec"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--valid", default=None)
    ap.add_argument("--golden-traj", type=int, default=5)
    ap.add_argument("--golden-frame", type=int, default=150)
    args = ap.parse_args()
    if args.random:
        cfg = mgn.default_config()
        params = mgn.init_params(jax.random.PRNGKey(1), cfg)
        stats = {"vel_mean": np.zeros(3, np.float32), "vel_std": np.full(3, 0.005, np.float32),
                 "acc_mean": np.zeros(3, np.float32), "acc_std": np.full(3, 0.003, np.float32)}
        source = "random"
    else:
        with open(args.ckpt, "rb") as f:
            ck = pickle.load(f)
        cfg, stats, params = ck["cfg"], ck["stats"], ck["params"]
        source = os.path.basename(args.ckpt)
    params = jax.tree.map(jnp.asarray, params)
    os.makedirs(args.out, exist_ok=True)

    tensors, blobs, offset = [], [], 0
    for name, arr in flatten(params):
        arr = np.asarray(arr, np.float32)
        tensors.append({"name": name, "shape": list(arr.shape), "offset": offset})
        blobs.append(arr.ravel())
        offset += arr.size
    np.concatenate(blobs).astype("<f4").tofile(os.path.join(args.out, "cloth.bin"))
    meta = {
        "config": {k: v for k, v in cfg.items()},
        "strides": list(mgn.STRIDES), "nbr_k": mgn.NBR_K, "world_k": mgn.WORLD_K, "world_r": mgn.WORLD_R,
        "d_clip": mgn.D_CLIP, "kb_log_mid": mgn.KB_LOG_MID, "kb_log_half": mgn.KB_LOG_HALF,
        "wind_scale": mgn.WIND_SCALE, "grip_k": mgn.GRIP_K, "grip_scale": mgn.GRIP_SCALE, "contact_d": mgn.CONTACT_D,
        "spacing": C.SPACING, "obs_offset": C.OBS_OFFSET,
        "self_exclude": C.SELF_EXCLUDE, "frame_dt": C.FRAME_DT, "dhat": C.DHAT,
        "stats": {k: [float(x) for x in np.asarray(v)] for k, v in stats.items()},
        "params": int(offset), "tensors": tensors, "source": source,
    }
    with open(os.path.join(args.out, "cloth.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"exported {offset} floats")

    if args.valid:
        from train import load_chunk, held_at, obs_at, graph_of, tables
        z = load_chunk(sorted(glob.glob(os.path.join(args.valid, "chunk_*.npz")))[0])
        b, t = args.golden_traj, args.golden_frame
        sc = {k: jnp.asarray(v[b]) for k, v in z.items()}
        g = graph_of(sc)
        n = int(np.asarray(g["valid"]).sum())
        st = {k: jnp.asarray(v) for k, v in stats.items()}
        pos = sc["pos"]
        held = held_at(sc, t)
        obs, obs_next = obs_at(sc, t), obs_at(sc, t + 1)
        a = mgn.predict(params, cfg, st, pos[t - 1], pos[t], pos[t + 1], held, obs, sc["kb"], sc["mu"], sc["wind"][t], g)
        nxt = mgn.integrate(st, a, pos[t - 1], pos[t], pos[t + 1], held, obs_next)
        wn, wm = mgn.world_neighbors(pos[t], g["rest"], g["valid"])
        size = next(s for s in [(31, 31), (31, 26), (31, 21), (26, 26), (26, 21), (21, 21), (21, 16), (16, 16)]
                    if s[0] * s[1] == n)
        golden = {
            "size": list(size), "n": n,
            "x1": np.asarray(pos[t - 1][:n]).ravel().tolist(), "x0": np.asarray(pos[t][:n]).ravel().tolist(),
            "kin": np.asarray(pos[t + 1][:n]).ravel().tolist(), "held": np.asarray(held[:n]).astype(int).tolist(),
            "obs": {k: np.asarray(v).tolist() for k, v in obs.items()},
            "obs_next": {k: np.asarray(v).tolist() for k, v in obs_next.items()},
            "kb": float(sc["kb"]), "mu": float(sc["mu"]), "wind": np.asarray(sc["wind"][t]).tolist(),
            "acc_norm": np.asarray(a[:n]).ravel().tolist(), "next": np.asarray(nxt[:n]).ravel().tolist(),
            "world_edges": int(np.asarray(wm[:n]).sum()),
        }
        with open(os.path.join(args.out, "golden.json"), "w") as f:
            json.dump(golden, f)
        print(f"golden: {size}, {n} vertices, {int(held.sum())} held, {golden['world_edges']} world edges")


if __name__ == "__main__":
    main()
