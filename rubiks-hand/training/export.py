"""Export a trained PyTorch PPO checkpoint to browser-ready policy.json + policy.bin + test.json."""
import json
import os
import sys
import numpy as np
import torch


def export_checkpoint(ckpt_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    mean = np.asarray(ckpt["obs_mean"], dtype=np.float64)
    std = np.asarray(ckpt["obs_std"], dtype=np.float64)
    obs_clip = float(ckpt.get("obs_clip", 10.0))
    activation = str(ckpt.get("activation", "elu"))
    tanh_out = bool(ckpt.get("tanh_out", True))

    sd = ckpt["actor_state_dict"]
    # Collect linear layers in order
    w_keys = [k for k in sd.keys() if k.endswith(".weight")]
    layers = []
    blobs = []
    for wk in w_keys:
        bk = wk[:-7] + ".bias"
        w = sd[wk].detach().cpu().numpy().astype(np.float32)  # [out, in] row-major!
        b = sd[bk].detach().cpu().numpy().astype(np.float32)  # [out]
        layers.append({"in": int(w.shape[1]), "out": int(w.shape[0])})
        blobs.append(w.ravel())
        blobs.append(b.ravel())

    meta = {
        "obs": obs_dim,
        "act": act_dim,
        "obsClip": obs_clip,
        "activation": activation,
        "tanhOut": tanh_out,
        "layers": layers,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    np.concatenate(blobs).astype("<f4").tofile(os.path.join(out_dir, "policy.bin"))
    with open(os.path.join(out_dir, "policy.json"), "w") as f:
        json.dump(meta, f)

    # Generate 16 reference input/output pairs for JS parity verification
    rng = np.random.default_rng(0)
    obs = (mean + std * rng.normal(size=(16, obs_dim))).astype(np.float32)
    x = np.clip((obs - mean.astype(np.float32)) / std.astype(np.float32), -obs_clip, obs_clip)
    for l_idx, wk in enumerate(w_keys):
        bk = wk[:-7] + ".bias"
        w = sd[wk].detach().cpu().numpy().astype(np.float32)
        b = sd[bk].detach().cpu().numpy().astype(np.float32)
        x = x @ w.T + b
        if l_idx < len(w_keys) - 1:
            if activation == "swish":
                x = x / (1.0 + np.exp(-x))
            elif activation == "tanh":
                x = np.tanh(x)
            else:
                x = np.where(x >= 0.0, x, np.exp(x) - 1.0)
        else:
            x = np.tanh(x) if tanh_out else np.clip(x, -1.0, 1.0)

    with open(os.path.join(out_dir, "test.json"), "w") as f:
        json.dump({"obs": obs.tolist(), "act": x.tolist()}, f)
    print(f"Exported {ckpt_path} -> {out_dir} ({layers}, {sum(b.size for b in blobs)*4/1024:.1f} KB)")


if __name__ == "__main__":
    export_checkpoint(sys.argv[1], sys.argv[2])
