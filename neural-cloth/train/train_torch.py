"""PyTorch trainer for the cloth network (same model and recipe as train.py).

python train_torch.py --data data/train --valid data/valid --run runs/c1 --init runs/c1/params_00082000.pkl

With --energy-weight > 0, part of each batch continues the network's own long rollouts, which are trained
on the frame energy alone (energy_torch.py), and the teacher-frame samples add the same energy to their
supervised loss.

Checkpoints: <run>/params_XXXXXXXX.pkl in mgn.py's parameter layout (for export.py) and
<run>/latest_torch.pkl (params + Adam state) for resuming.
"""
import argparse
import glob
import os
import pickle
import random
import time

import numpy as np
import torch

import energy_torch as E
import mgn_torch as M

SCHEDULE = [0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0]
SCENE_KEYS = ["h_vid", "h_t0", "h_t1", "sph_c", "sph_r", "sph_on", "cap_a", "cap_b", "cap_r", "cap_on", "wind", "kb", "mu"]
HUBER = 3.0
ACC_CLIP = 50.0
_TABLES = {}


def tables(size):
    if size not in _TABLES:
        _TABLES[size] = {**M.graph_tables(*size), **E.energy_tables(*size)}
    return _TABLES[size]


def load_chunk(path):
    with np.load(path) as z:
        size = tuple(int(v) for v in z["size"])
        pos = z["pos"]
        B, T1, n, _ = pos.shape
        out = {"pos": np.zeros((B, T1, M.N_MAX, 3), np.float32)}
        out["pos"][:, :, :n] = pos
        for k in SCENE_KEYS:
            out[k] = z[k]
    for k, v in tables(size).items():
        out[k] = np.broadcast_to(v, (B,) + v.shape).copy()
    ok = pos[..., 1].min((1, 2)) > 0.0
    if not ok.all():
        print(f"{os.path.basename(path)}: dropping {int((~ok).sum())} clip(s) with table penetration", flush=True)
        out = {k: v[ok] for k, v in out.items()}
    out["size"] = np.array([size] * out["pos"].shape[0])
    return out


def to_torch(chunk, device):
    out = {}
    for k, v in chunk.items():
        if k == "size":
            continue
        t = torch.from_numpy(np.ascontiguousarray(v))
        if t.dtype == torch.float64:
            t = t.float()
        if t.dtype in (torch.int32, torch.int8, torch.int16):
            t = t.long()
        out[k] = t.to(device)
    return out


GRAPH_KEYS = ["nodes0", "nbr0", "mask0", "nodes1", "nbr1", "mask1", "nodes2", "nbr2", "mask2", "rest", "valid"]


def held_at(sc, t):
    """[B, N_MAX] vertices a gripper holds during step t -> t+1 (t: [B])."""
    B = t.shape[0]
    active = (sc["h_vid"] >= 0) & (sc["h_t0"] <= t[:, None]) & (t[:, None] < sc["h_t1"])
    idx = torch.where(active, sc["h_vid"], torch.full_like(sc["h_vid"], M.N_MAX))
    held = torch.zeros(B, M.N_MAX + 1, dtype=torch.bool, device=t.device)
    held[torch.arange(B, device=t.device)[:, None].expand_as(idx), idx] = True
    return held[:, :-1]


def obs_at(sc, t):
    B = t.shape[0]
    bi = torch.arange(B, device=t.device)
    nxt = torch.clamp(t + 1, max=sc["sph_c"].shape[1] - 1)
    at = lambda a, i: a[bi, i]
    return {"sph_c": at(sc["sph_c"], t), "sph_v": (at(sc["sph_c"], nxt) - at(sc["sph_c"], t)) / M.FRAME_DT,
            "sph_r": sc["sph_r"], "sph_on": sc["sph_on"],
            "cap_a": at(sc["cap_a"], t), "cap_b": at(sc["cap_b"], t),
            "cap_va": (at(sc["cap_a"], nxt) - at(sc["cap_a"], t)) / M.FRAME_DT,
            "cap_vb": (at(sc["cap_b"], nxt) - at(sc["cap_b"], t)) / M.FRAME_DT,
            "cap_r": sc["cap_r"], "cap_on": sc["cap_on"]}


def frame(sc, t):
    return sc["pos"][torch.arange(t.shape[0], device=t.device), t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", default=None, help="params pickle (mgn.py layout) to start from")
    ap.add_argument("--start-step", type=int, default=None)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--swap-every", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--lr-decay-steps", type=float, default=700_000)
    ap.add_argument("--anneal", type=float, nargs=3, default=None, metavar=("START", "STEPS", "LR_END"),
                    help="from step START, decay the learning rate geometrically to LR_END over STEPS steps")
    ap.add_argument("--noise", type=float, default=5e-4)
    ap.add_argument("--noise-gamma", type=float, default=0.1)
    ap.add_argument("--strain-weight", type=float, default=0.05)
    ap.add_argument("--unroll-start", type=int, default=10_000)
    ap.add_argument("--unroll-every", type=int, default=30_000)
    ap.add_argument("--unroll-max", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--energy-weight", type=float, default=0.0, help="weight of the frame-energy loss (0 = off)")
    ap.add_argument("--energy-cap", type=float, default=100.0, help="energies above this grow logarithmically")
    ap.add_argument("--rollout-frac", type=float, default=0.5, help="share of the batch spent on own rollouts")
    ap.add_argument("--rollout-len", type=int, nargs=2, default=(20, 250), help="rollout length range (frames)")
    ap.add_argument("--release-frac", type=float, default=0.0,
                    help="share of own rollouts that start from a held frame with the grippers let go")
    ap.add_argument("--release-len", type=int, nargs=2, default=(5, 40),
                    help="release rollout length range: short, so they dwell on the moment of letting go")
    ap.add_argument("--rest-frac", type=float, default=0.0, help="share of own rollouts that start at rest")
    ap.add_argument("--release-energy-weight", type=float, default=None,
                    help="energy weight on release rollouts (default: --energy-weight); the teacher's many "
                         "towels hanging still from grippers otherwise outvote the few it shows falling")
    ap.add_argument("--com-weight", type=float, default=0.0,
                    help="weight of the error in the free vertices' mean acceleration (teacher rows). Hovering "
                         "after a release and creeping on the table are errors in this one mode, which the "
                         "per-vertex loss weighs like any other")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="decay of an exponential moving average of the weights, used for evaluation and the "
                         "params_*.pkl exports (0 = off); smooths out checkpoint-to-checkpoint swings")
    ap.add_argument("--free-com-weight", type=float, default=0.0,
                    help="weight of the net-force residual on own-rollout rows with nothing held: internal forces "
                         "cancel over the whole cloth, so its mean acceleration must match gravity, contact and "
                         "friction alone (a label-free check that a released towel falls at the right rate)")
    ap.add_argument("--eval-only", action="store_true", help="evaluate the starting weights and exit")
    ap.add_argument("--release-focus", type=float, default=0.0,
                    help="share of teacher rows drawn from just after a gripper lets go (-3..+25 frames); the "
                         "network fits the moment of release but slows the fall too early after it")
    ap.add_argument("--release-weight", type=float, default=1.0,
                    help="supervised-loss weight on the rows --release-focus draws from just after a release: the "
                         "under-predicted fall there is small per frame but compounds into a towel that hangs")
    ap.add_argument("--extra-data", default=None, help="a second chunk directory mixed into the pool")
    ap.add_argument("--extra-frac", type=float, default=0.5, help="share of pool loads drawn from --extra-data")
    ap.add_argument("--release-valid", default=None,
                    help="held-out lift-and-release chunk: each evaluation compares the fall after release")
    ap.add_argument("--rotate", action="store_true",
                    help="turn every clip entering the pool by a random angle about the vertical axis")
    ap.add_argument("--new-rows-lr-mult", type=float, default=1.0,
                    help="learning-rate multiplier for the node encoder's gripper/contact input rows")
    ap.add_argument("--sync-every", type=int, default=0,
                    help="torch.cuda.synchronize() every N steps; keeps the launch queue shallow (0 = never)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.run, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    resume = os.path.join(args.run, "latest_torch.pkl")
    opt_state = None
    if os.path.exists(resume):
        with open(resume, "rb") as f:
            ck = pickle.load(f)
        cfg, stats_np, params_np, opt_state, start = ck["cfg"], ck["stats"], ck["params"], ck["opt_state"], ck["step"]
        ema_np = ck.get("ema")
        print(f"resumed at step {start}", flush=True)
    else:
        with open(args.init, "rb") as f:
            ck = pickle.load(f)
        cfg, stats_np, params_np = ck["cfg"], ck["stats"], ck["params"]
        ema_np = None
        start = args.start_step or 0
        w = np.asarray(params_np["enc_node"]["layers"][0]["w"])
        node_in = M.NODE_IN
        if w.shape[0] < node_in:  # a checkpoint from before the gripper/contact features: they start at zero weight
            params_np["enc_node"]["layers"][0]["w"] = np.concatenate([w, np.zeros((node_in - w.shape[0], w.shape[1]), w.dtype)])
            print(f"node encoder widened from {w.shape[0]} to {node_in} inputs", flush=True)
        print(f"initialised from {args.init} at step {start}", flush=True)
    schedule = cfg["schedule"]
    stats = {k: torch.tensor(np.asarray(v), dtype=torch.float32, device=dev) for k, v in stats_np.items()}
    P = M.Params(params_np, dev)
    params = P.tree
    # The gripper/contact inputs (node encoder rows 21+) joined an already trained network at zero weight.
    # Adam moves every weight by about lr per step whatever its gradient's size, so they get their own
    # parameter with a larger learning rate; join_enc() rebuilds the layer after each update.
    w_enc = params["enc_node"]["layers"][0]["w"]
    w_base = w_enc[:21].detach().clone().requires_grad_(True)
    w_new = w_enc[21:].detach().clone().requires_grad_(True)
    tensors = [t if t is not w_enc else w_base for t in P.tensors()] + [w_new]

    def join_enc():
        params["enc_node"]["layers"][0]["w"] = torch.cat([w_base, w_new])

    join_enc()
    opt = torch.optim.Adam([{"params": tensors[:-1]}, {"params": [w_new], "lr_mult": args.new_rows_lr_mult}], lr=args.lr)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    ema = None
    if args.ema > 0:
        ema = [t.detach().clone() for t in tensors]
        if opt_state is not None and ema_np is not None:
            for e, v in zip(ema, ema_np):
                e.copy_(torch.as_tensor(v, device=dev))

    def swap_ema():
        """Exchange the live weights with their moving average (call again to swap back)."""
        if ema is None:
            return
        with torch.no_grad():
            for e, t in zip(ema, tensors):
                tmp = t.detach().clone()
                t.copy_(e)
                e.copy_(tmp)
        join_enc()
    base_sched = lambda s: args.lr_min + (args.lr - args.lr_min) * 0.1 ** (s / args.lr_decay_steps)

    def sched(s):
        if args.anneal is None or s < args.anneal[0]:
            return base_sched(s)
        s0, n, lr_end = args.anneal
        lr0 = base_sched(s0)
        return lr0 * (lr_end / lr0) ** min(1.0, (s - s0) / n)

    files = sorted(glob.glob(os.path.join(args.data, "chunk_*.npz")))
    first = load_chunk(files[0])
    T = first["pos"].shape[1] - 1
    Pn = args.pool
    pool = None
    filled = 0

    def add_chunk(chunk):
        nonlocal pool, filled
        c = to_torch(chunk, dev)
        k = c["pos"].shape[0]
        if args.rotate:
            # The physics has no preferred direction on the table, but the network's inputs are in world
            # coordinates; random turns keep it from learning a fixed-direction bias (a slow sideways creep).
            th = torch.rand(k, device=dev) * 2 * np.pi
            cs, sn = torch.cos(th), torch.sin(th)
            for key in ("pos", "sph_c", "cap_a", "cap_b", "wind"):
                v = c[key]
                shp = (k,) + (1,) * (v.dim() - 2)
                x, z = v[..., 0].clone(), v[..., 2].clone()
                v[..., 0] = cs.view(shp) * x + sn.view(shp) * z
                v[..., 2] = -sn.view(shp) * x + cs.view(shp) * z
        if pool is None:
            pool = {key: torch.zeros((Pn,) + v.shape[1:], dtype=v.dtype, device=dev) for key, v in c.items()}
        idx = torch.arange(filled, min(Pn, filled + k), device=dev) if filled < Pn else \
            torch.randperm(Pn, device=dev)[:k]
        for key in pool:
            pool[key][idx] = c[key][:len(idx)]
        filled = min(Pn, filled + len(idx)) if filled < Pn else filled

    def pick_chunk():
        extra = sorted(glob.glob(os.path.join(args.extra_data, "chunk_*.npz"))) if args.extra_data else []
        if extra and random.random() < args.extra_frac:
            return random.choice(extra)
        return random.choice(sorted(glob.glob(os.path.join(args.data, "chunk_*.npz"))))

    add_chunk(first)
    while filled < Pn:
        add_chunk(load_chunk(pick_chunk()))
    print(f"pool filled with {filled} trajectories", flush=True)

    B = args.batch
    gamma = args.noise_gamma

    def graph(batch):
        return {k: batch[k] for k in GRAPH_KEYS}

    def wind_at(batch, ts):
        return batch["wind"][torch.arange(ts.shape[0], device=dev), ts.clamp(max=T - 1)]

    acc_scale = float(stats["acc_std"].mean())

    def energy_loss(batch, ts, x1, x0, y, free):
        """Frame energy of the step x0 -> y in supervised-loss units, log-compressed above --energy-cap."""
        terms = E.frame_energy(y, x1, x0, free, obs_at(batch, ts), obs_at(batch, ts + 1), batch["kb"], batch["mu"],
                               wind_at(batch, ts), batch, batch["rest"], batch["valid"])
        e = sum(terms.values()) / E.energy_scale(free, batch, acc_scale).clamp(min=1e-12)
        cap = args.energy_cap
        return torch.where(e < cap, e, cap * (1 + torch.log(e.clamp(min=cap) / cap))), terms

    def net_force_residual(terms, y, free, mass):
        """[B] squared mean-acceleration residual, in acc_std units. Membrane, bending and self-contact forces
        sum to zero over a cloth with no held vertices, so the gradient of inertia + obstacle + friction summed
        over the vertices is the net force the step leaves unbalanced."""
        ext = terms["inertia"] + terms["obstacle"] + terms["friction"]
        grad, = torch.autograd.grad(ext.sum(), y, create_graph=True)
        res = (grad * free[..., None]).sum(1) / (mass * free).sum(1, keepdim=True).clamp(min=1e-12)
        return (res * M.FRAME_DT ** 2 / acc_scale).pow(2).sum(-1)

    def strain_penalty(x, g):
        xp, rp = M.pad(x), M.pad(g["rest"])
        dx = M.bgather(xp, g["nbr0"]) - M.bgather(xp, g["nodes0"])[:, :, None]
        du = M.bgather(rp, g["nbr0"]) - M.bgather(rp, g["nodes0"])[:, :, None]
        strain = torch.sqrt((dx * dx).sum(-1) + 1e-12) / torch.sqrt((du * du).sum(-1) + 1e-12) - 1.0
        m = g["mask0"]
        return (torch.where(m, strain ** 2, torch.zeros_like(strain)).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)) / 0.01 ** 2

    def one_step_loss(batch, ts, x1, x0, sup, drop, kin=None, row_w=None):
        """sup [B]: rows with a teacher target (the rest are own rollouts, trained on the energy alone).
        drop [B]: rows whose grippers have been let go. kin: gripper targets for step ts -> ts+1 (default:
        the teacher's next frame). Returns the loss, per-row supervised loss and energy, and the predicted
        accelerations."""
        g = graph(batch)
        held = held_at(batch, ts) & ~drop[:, None]
        free = g["valid"] & ~held
        noise = torch.randn_like(x0) * args.noise * (free & sup[:, None])[..., None]
        x0 = x0 + noise
        nxt = frame(batch, ts + 1) if kin is None else kin
        a = M.predict(params, schedule, stats, x1, x0, nxt, held, obs_at(batch, ts), batch["kb"], batch["mu"],
                      wind_at(batch, ts), g)
        a_t = ((nxt + (1 - gamma) * noise) - 2 * x0 + x1 - stats["acc_mean"]) / stats["acc_std"]
        err = (a - a_t).abs()
        hub = torch.where(err < HUBER, 0.5 * err ** 2, HUBER * (err - 0.5 * HUBER)) * 2
        loss = (hub.sum(-1) * free).sum(1) / free.sum(1).clamp(min=1)
        com = (((a - a_t) * free[..., None]).sum(1) / free.sum(1, keepdim=True).clamp(min=1)).pow(2).sum(-1)
        x_next = M.integrate(stats, a.clamp(-ACC_CLIP, ACC_CLIP), x1, x0, nxt, held, obs_at(batch, ts + 1))
        total = torch.where(sup, loss + args.strain_weight * strain_penalty(x_next, g) + args.com_weight * com,
                            torch.zeros_like(loss))
        if row_w is not None:
            total = total * row_w
        en = torch.zeros_like(loss)
        if args.energy_weight > 0:
            y = torch.where(held[..., None], nxt, 2 * x0 - x1 + a * stats["acc_std"] + stats["acc_mean"])
            en, terms = energy_loss(batch, ts, x1, x0, y, free)
            w_rel = args.energy_weight if args.release_energy_weight is None else args.release_energy_weight
            total = total + torch.where(drop, w_rel, args.energy_weight) * en
            if args.free_com_weight > 0:
                unheld = ~sup & (held.sum(1) == 0)
                if unheld.any():
                    total = total + torch.where(unheld, args.free_com_weight * net_force_residual(terms, y, free, batch["e_mass"]),
                                                torch.zeros_like(loss))
        return total.mean(), loss, en, a.detach()

    @torch.no_grad()
    def push_step(batch, ts, x1, x0):
        g = graph(batch)
        held = held_at(batch, ts)
        nxt = frame(batch, ts + 1)
        a = M.predict(params, schedule, stats, x1, x0, nxt, held, obs_at(batch, ts), batch["kb"], batch["mu"],
                      wind_at(batch, ts), g)
        x = M.integrate(stats, a, x1, x0, nxt, held, obs_at(batch, ts + 1))
        return torch.nan_to_num(x).clamp(-3.0, 3.0)

    # ---- evaluation: every validation clip, one batched rollout per chunk (a chunk shares one cloth size)
    vchunks = [load_chunk(f) for f in sorted(glob.glob(os.path.join(args.valid, "chunk_*.npz")))]
    GIFS = {(0, 0), (1, 4)}  # (chunk, clip): a fling, and a fold that starts in free fall

    def grip_kin(sc, t, x0, grip):
        """Gripper targets for step t -> t+1 in a network rollout. Each gripper replays the teacher's motion
        from the moment it closes, starting from wherever the network's vertex is then; snapping to the
        teacher's absolute positions would yank the cloth across any gap between the two (and did, in a
        rollout that blew up). grip = {"off": [B,H,3], "on": [B,H]} is updated in place."""
        Bv, H = sc["h_vid"].shape
        bi = torch.arange(Bv, device=dev)[:, None].expand(Bv, H)
        vid = sc["h_vid"].clamp(min=0)
        active = (sc["h_vid"] >= 0) & (sc["h_t0"] <= t[:, None]) & (t[:, None] < sc["h_t1"])
        new = active & ~grip["on"]
        teacher = sc["pos"][bi, t[:, None].expand(Bv, H), vid]
        grip["off"] = torch.where(new[..., None], x0[bi, vid] - teacher, grip["off"])
        grip["on"] = grip["on"] | new
        nxt = frame(sc, t + 1).clone()
        nxt.index_put_((bi, vid), torch.where(active[..., None], grip["off"], torch.zeros_like(grip["off"])),
                       accumulate=True)
        return nxt

    def new_grip(sc):
        Bv, H = sc["h_vid"].shape
        return {"off": torch.zeros(Bv, H, 3, device=dev), "on": torch.zeros(Bv, H, dtype=torch.bool, device=dev)}
    h2g = 9.81 * M.FRAME_DT ** 2

    @torch.no_grad()
    def probes(step_i):
        """Three failure modes the clip averages hide: a released towel that hangs in the air, a towel
        let go from rest that doesn't fall, and a towel lying still that creeps sideways."""
        out = []
        c_np = vchunks[0]
        sc = to_torch({k: v[[1, 4]] for k, v in c_np.items()}, dev)  # the two lift-and-drop clips
        g = graph(sc)
        rel = torch.where(sc["h_vid"] >= 0, sc["h_t1"], torch.zeros_like(sc["h_t1"])).amax(1)
        corner = sc["h_vid"][:, 0]
        t = rel - 5
        bi = torch.arange(2, device=dev)
        x1, x0 = sc["pos"][bi, t - 1], sc["pos"][bi, t]
        grip = new_grip(sc)
        for _ in range(25):
            held = held_at(sc, t)
            nxt = grip_kin(sc, t, x0, grip)
            a = M.predict(params, schedule, stats, x1, x0, nxt, held, obs_at(sc, t), sc["kb"], sc["mu"],
                          sc["wind"][bi, t.clamp(max=T - 1)], g)
            x1, x0, t = x0, M.integrate(stats, a, x1, x0, nxt, held, obs_at(sc, t + 1)), t + 1
        net = x0[bi, corner, 1].tolist()
        teach = sc["pos"][bi, t, corner, 1].tolist()
        out.append("released corner +20f net " + "/".join(f"{v:.2f}" for v in net) +
                   " teacher " + "/".join(f"{v:.2f}" for v in teach) + " m")
        empty = {k: torch.zeros((1,) + v.shape[1:], device=dev) for k, v in obs_at(sc, t).items()}
        nx, ny = 26, 26
        gt = {k: torch.tensor(v, device=dev)[None] for k, v in M.graph_tables(nx, ny).items()}
        n = nx * ny
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        none = torch.zeros(1, M.N_MAX, dtype=torch.bool, device=dev)
        for name, xyz in [("hanging sheet let go from rest", ((jj - ny / 2) * M.SPACING, 0.9 - ii * M.SPACING, 0 * ii)),
                          ("towel lying still", ((ii - nx / 2) * M.SPACING, 0 * ii + 0.0045, (jj - ny / 2) * M.SPACING))]:
            x = torch.zeros(1, M.N_MAX, 3, device=dev)
            x[0, :n] = torch.tensor(np.stack([np.ravel(c) for c in xyz], 1), dtype=torch.float32, device=dev)
            a = M.predict(params, schedule, stats, x, x, x, none, empty, torch.tensor([3e-6], device=dev),
                          torch.tensor([0.5], device=dev), torch.zeros(1, 3, device=dev), gt)
            acc = (a * stats["acc_std"] + stats["acc_mean"])[0, :n].mean(0) / h2g
            out.append(f"{name}: a = ({acc[0]:+.2f}, {acc[1]:+.2f}, {acc[2]:+.2f}) g")
        print(f"probes step {step_i}: " + " | ".join(out), flush=True)
        if rel_valid is not None:
            release_probe(step_i)

    rel_valid = load_chunk(sorted(glob.glob(os.path.join(args.release_valid, "chunk_*.npz")))[0]) \
        if args.release_valid else None

    @torch.no_grad()
    def release_probe(step_i):
        """Held-out lift-and-release clips: from the teacher's state 5 frames before each release, how high
        the towel's top is (and how far from the teacher's towel) 10, 20 and 40 frames after it."""
        sc = to_torch(rel_valid, dev)
        g = graph(sc)
        Bv = sc["pos"].shape[0]
        bi = torch.arange(Bv, device=dev)
        rel = torch.where(sc["h_vid"] >= 0, sc["h_t1"], torch.zeros_like(sc["h_t1"])).amax(1)
        t = rel - 5
        x1, x0 = sc["pos"][bi, t - 1], sc["pos"][bi, t]
        grip = new_grip(sc)
        top = lambda x: torch.where(g["valid"], x[..., 1], torch.full_like(x[..., 1], -1.0)).amax(1).mean().item()
        rows = []
        for k in range(1, 46):
            held = held_at(sc, t)
            nxt = grip_kin(sc, t, x0, grip)
            a = M.predict(params, schedule, stats, x1, x0, nxt, held, obs_at(sc, t), sc["kb"], sc["mu"],
                          sc["wind"][bi, t.clamp(max=T - 1)], g)
            x1, x0, t = x0, M.integrate(stats, a, x1, x0, nxt, held, obs_at(sc, t + 1)), t + 1
            if k - 5 in (10, 20, 40):
                teach = sc["pos"][bi, t]
                dist = (((x0 - teach) ** 2).sum(-1).sqrt() * g["valid"]).sum(1) / g["valid"].sum(1)
                rows.append(f"+{k - 5}f top {top(x0):.2f} (teacher {top(teach):.2f}) apart {dist.mean().item() * 100:.0f} cm")
        print(f"release probe step {step_i} ({Bv} clips): " + " | ".join(rows), flush=True)

    @torch.no_grad()
    def evaluate(step_i):
        errs, stretch = [], []
        for ci, c_np in enumerate(vchunks):
            size = tuple(int(v) for v in c_np["size"][0])
            n = size[0] * size[1]
            sc = to_torch(c_np, dev)
            g = graph(sc)
            Bv = sc["pos"].shape[0]
            Tv = sc["pos"].shape[1] - 1
            x1, x0 = sc["pos"][:, 0], sc["pos"][:, 1]
            out = [x1, x0]
            grip = new_grip(sc)
            for t in range(1, Tv):
                tt = torch.full((Bv,), t, device=dev)
                held = held_at(sc, tt)
                nxt = grip_kin(sc, tt, x0, grip)
                a = M.predict(params, schedule, stats, x1, x0, nxt, held, obs_at(sc, tt), sc["kb"], sc["mu"],
                              sc["wind"][:, min(t, Tv - 1)], g)
                x = M.integrate(stats, a, x1, x0, nxt, held, obs_at(sc, tt + 1))
                x = torch.where(g["valid"][..., None], x, torch.zeros_like(x))
                out.append(x)
                x1, x0 = x0, x
            pred = torch.stack(out, 1)[:, :, :n].cpu().numpy()  # [B, T, n, 3]
            gt = c_np["pos"][:, :, :n]
            errs.append(np.sqrt(np.mean(np.sum((pred - gt) ** 2, -1), 2)))
            ed = np.array(M.grid_edges(*size))
            rl = M.SPACING * np.linalg.norm(np.array(np.unravel_index(ed[:, 0], size)) -
                                            np.array(np.unravel_index(ed[:, 1], size)), axis=0)
            ln = np.linalg.norm(pred[:, :, ed[:, 0]] - pred[:, :, ed[:, 1]], axis=-1)
            stretch.append(np.abs(ln / rl - 1).mean(-1))
            for (gc, gb) in GIFS:
                if gc == ci and gb < Bv:
                    from render3d import render_pair
                    render_pair(os.path.join(args.run, f"rollout_{step_i:08d}_{ci}_{gb}.gif"), gt[gb], pred[gb],
                                M.grid_tris(*size), {k: c_np[k][gb] for k in SCENE_KEYS})
        errs, stretch = np.concatenate(errs), np.concatenate(stretch)
        probes(step_i)
        print(f"eval step {step_i} ({len(errs)} clips): rmse@30 {errs[:, 30].mean():.4f} @100 {errs[:, 100].mean():.4f} "
              f"@end {errs[:, -1].mean():.4f} m  |strain| @30 {stretch[:, 30].mean():.3f} @100 "
              f"{stretch[:, 100].mean():.3f} @end {stretch[:, -1].mean():.3f}  worst@end {errs[:, -1].max():.3f} m",
              flush=True)

    def save(step_i):
        join_enc()
        tmp = resume + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"cfg": cfg, "stats": stats_np, "params": P.to_numpy(), "opt_state": opt.state_dict(),
                         "step": step_i, "ema": None if ema is None else [e.cpu().numpy() for e in ema]}, f)
        os.replace(tmp, resume)

    # ---- own rollouts: rows of the batch that keep stepping the network's predictions from a teacher
    # frame for a random number of frames, so the energy loss sees the states the network drifts into.
    Br = int(round(B * args.rollout_frac)) if args.energy_weight > 0 else 0
    Bf = B - Br
    ro = {"p": torch.zeros(Br, dtype=torch.long, device=dev), "t": torch.zeros(Br, dtype=torch.long, device=dev),
          "x1": None, "x0": None, "age": torch.zeros(Br, dtype=torch.long, device=dev),
          "len": torch.zeros(Br, dtype=torch.long, device=dev), "drop": torch.zeros(Br, dtype=torch.bool, device=dev),
          "grip": {"off": torch.zeros(Br, pool["h_vid"].shape[1], 3, device=dev),
                   "on": torch.zeros(Br, pool["h_vid"].shape[1], dtype=torch.bool, device=dev)}}

    def reset_rollouts(mask):
        k = int(mask.sum())
        if k == 0:
            return
        lo, hi = args.rollout_len
        ro["p"][mask] = torch.randint(0, filled, (k,), device=dev)
        ro["len"][mask] = torch.randint(lo, hi + 1, (k,), device=dev)
        ro["t"][mask] = (torch.rand(k, device=dev) * (T - 1 - ro["len"][mask]).clamp(min=1)).long() + 1
        ro["age"][mask] = 0
        # Virtual releases: start inside a gripper hold and let go. The teacher only shows a few frames
        # after each release, so without these the network learns that a hanging towel stays put.
        sub = {key: pool[key][ro["p"][mask]] for key in ("pos", "h_vid", "h_t0", "h_t1")}
        on = sub["h_vid"] >= 0
        lo = torch.where(on, sub["h_t0"], torch.full_like(sub["h_t0"], T)).amin(1).clamp(min=1)
        hi = torch.where(on, sub["h_t1"], torch.zeros_like(sub["h_t1"])).amax(1).clamp(max=T - 2)
        drop = (torch.rand(k, device=dev) < args.release_frac) & (lo < hi)
        t_rel = lo + (torch.rand(k, device=dev) * (hi - lo).clamp(min=1)).long()
        tm = torch.where(drop, t_rel, ro["t"][mask])
        ro["t"][mask] = tm
        rlo, rhi = args.release_len
        short = torch.randint(rlo, rhi + 1, (k,), device=dev)
        ro["len"][mask] = torch.minimum(torch.where(drop, short, ro["len"][mask]), T - 1 - tm)
        ro["drop"][mask] = drop
        ro["grip"]["off"][mask] = 0.0
        ro["grip"]["on"][mask] = False
        x1n, x0n = frame(sub, tm - 1), frame(sub, tm)
        still = (torch.rand(k, device=dev) < args.rest_frac)[:, None, None]
        x1n = torch.where(still, x0n, x1n)
        if ro["x1"] is None:
            ro["x1"], ro["x0"] = torch.zeros((Br,) + x1n.shape[1:], device=dev), torch.zeros((Br,) + x0n.shape[1:], device=dev)
        ro["x1"][mask], ro["x0"][mask] = x1n, x0n

    if Br:
        reset_rollouts(torch.ones(Br, dtype=torch.bool, device=dev))
        print(f"batch: {Bf} teacher-frame rows + {Br} own-rollout rows, energy weight {args.energy_weight}, "
              f"releases {args.release_frac}, starts at rest {args.rest_frac}", flush=True)

    if args.eval_only:
        swap_ema()
        evaluate(start)
        return
    if opt_state is None:
        evaluate(start)  # baseline for a new run (the moving average starts equal to the weights)
    t_log = time.time()
    losses, ens_f, ens_r, ages = [], [], [], []
    for step_i in range(start, args.steps):
        if step_i % args.swap_every == 0 and step_i > start:
            add_chunk(load_chunk(pick_chunk()))
            if Br:
                reset_rollouts(torch.ones(Br, dtype=torch.bool, device=dev))  # their pool rows may be gone
        for grp in opt.param_groups:
            grp["lr"] = sched(step_i) * grp.get("lr_mult", 1.0)
        join_enc()
        K = int(min(args.unroll_max, 1 + max(0, step_i - args.unroll_start) // args.unroll_every))
        push = int(np.random.randint(K))
        p = torch.randint(0, filled, (Bf,), device=dev)
        ts = torch.randint(1, T - push, (Bf,), device=dev)
        near = torch.zeros(Bf, dtype=torch.bool, device=dev)
        if args.release_focus > 0:
            on = pool["h_vid"][p] >= 0
            rel = torch.where(on & (pool["h_t1"][p] < T - 1), pool["h_t1"][p], torch.zeros_like(pool["h_t1"][p])).amax(1)
            near = (rel > 0) & (torch.rand(Bf, device=dev) < args.release_focus)
            t_rel = (rel + torch.randint(-3, 26, (Bf,), device=dev)).clamp(1, T - push - 1)
            ts = torch.where(near, t_rel, ts)
        batch = {k: v[p] for k, v in pool.items()}
        x1, x0 = frame(batch, ts - 1), frame(batch, ts)
        for _ in range(push):
            x1, x0 = x0, push_step(batch, ts, x1, x0)
            ts = ts + 1
        if Br:
            rb = {k: v[ro["p"]] for k, v in pool.items()}
            batch = {k: torch.cat([batch[k], rb[k]]) for k in batch}
            ts, x1, x0 = torch.cat([ts, ro["t"]]), torch.cat([x1, ro["x1"]]), torch.cat([x0, ro["x0"]])
        sup = torch.arange(B, device=dev) < Bf
        drop = torch.cat([torch.zeros(Bf, dtype=torch.bool, device=dev), ro["drop"]])
        kin = None
        if Br:  # own rollouts replay the teacher's gripper motion from where their cloth is
            with torch.no_grad():
                kin_r = grip_kin({k: v[Bf:] for k, v in batch.items()}, ro["t"], ro["x0"], ro["grip"])
            kin = torch.cat([frame({k: v[:Bf] for k, v in batch.items()}, ts[:Bf] + 1), kin_r])
        opt.zero_grad(set_to_none=True)
        row_w = torch.cat([torch.where(near, args.release_weight, 1.0), torch.ones(B - Bf, device=dev)])
        total, acc_loss, en, a = one_step_loss(batch, ts, x1, x0, sup, drop, kin, row_w)
        total.backward()
        torch.nn.utils.clip_grad_norm_(tensors, 1.0)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for e, t in zip(ema, tensors):
                    e.mul_(args.ema).add_(t.detach(), alpha=1 - args.ema)
        if Br:
            with torch.no_grad():
                r = slice(Bf, B)
                rbt = {k: v[r] for k, v in batch.items()}
                held = held_at(rbt, ro["t"]) & ~ro["drop"][:, None]
                x = M.integrate(stats, a[r].clamp(-ACC_CLIP, ACC_CLIP), ro["x1"], ro["x0"], kin_r,
                                held, obs_at(rbt, ro["t"] + 1))
                x = torch.where(rbt["valid"][..., None], x, torch.zeros_like(x))
                bad = ~torch.isfinite(x).all(-1).all(-1) | (x.abs().amax((1, 2)) > 3.0)
                ro["x1"], ro["x0"] = ro["x0"], torch.nan_to_num(x).clamp(-3.0, 3.0)
                ro["t"] += 1
                ro["age"] += 1
                reset_rollouts(bad | (ro["age"] >= ro["len"]) | (ro["t"] >= T - 1))
        if args.sync_every and (step_i + 1) % args.sync_every == 0 and dev == "cuda":
            torch.cuda.synchronize()
        if (step_i + 1) % max(1, args.log_every // 20) == 0:
            losses.append(acc_loss[:Bf].mean().item())
            if Br:
                ens_f.append(en[:Bf].mean().item())
                ens_r.append(en[Bf:].mean().item())
                ages.append(ro["age"].float().mean().item())
        if (step_i + 1) % args.log_every == 0:
            dt = time.time() - t_log
            t_log = time.time()
            extra = f" energy {np.mean(ens_f):.2f} rollout-energy {np.mean(ens_r):.2f} age {np.mean(ages):.0f}" if Br else ""
            print(f"step {step_i + 1} loss {np.mean(losses):.4f}{extra} unroll {K} lr {sched(step_i):.2e} "
                  f"{args.log_every / dt:.1f} it/s", flush=True)
            losses, ens_f, ens_r, ages = [], [], [], []
        if (step_i + 1) % args.save_every == 0:
            save(step_i + 1)
        if (step_i + 1) % args.eval_every == 0:
            swap_ema()
            join_enc()
            with open(os.path.join(args.run, f"params_{step_i + 1:08d}.pkl"), "wb") as f:
                pickle.dump({"cfg": cfg, "stats": stats_np, "params": P.to_numpy()}, f)
            evaluate(step_i + 1)
            swap_ema()
    save(args.steps)


if __name__ == "__main__":
    main()
