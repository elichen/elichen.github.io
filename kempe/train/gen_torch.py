"""GPU generator for the big pool: random dyad linkages -> compact tracer curves.

Same construction as linkage.generate (grounds, geared cranks, dyads with
rejection of locking, stubby or crank-rigid joints), vectorised in PyTorch.
Each kept curve is stored with its pruned mechanism (only the joints the pen
depends on, in construction order, so the pen is the last joint) expressed in
the curve's own frame: curve centroid at the origin, curve RMS radius 1.

usage: python gen_torch.py OUTDIR --mechs 10000000 [--shard 500000] [--seed 0]
"""

import argparse
import math
import os
import time
import numpy as np
import torch

T_SIM = 200
N_CURVE = 64
MIN_SIN = 0.2
MAX_EXTENT = 5.0
JP = 16                     # max pruned joints stored per curve
RATIOS = torch.tensor([-4, -3, -2, -1, 1, 2, 3, 4])
RATIO_W = torch.tensor([1, 2, 3, 3, 2, 3, 2, 1], dtype=torch.float)
SIZES = [(g, c, d) for g in (2, 3, 4) for c in (1, 2, 3) for d in range(2, 11) if c <= g]
SIZE_W = torch.tensor([(1.0 + 0.3 * d) * (1.0, 1.5, 1.2)[c - 1] for g, c, d in SIZES])


def circle_meet(pa, pb, la, lb, side):
    """pa, pb (B, T, 2); la, lb, side (B,) -> point (B, T, 2), sin (B, T)."""
    dv = pb - pa
    d = dv.norm(dim=-1).clamp_min(1e-9)
    la_, lb_ = la[:, None], lb[:, None]
    a = (la_ ** 2 - lb_ ** 2 + d ** 2) / (2 * d)
    h2 = la_ ** 2 - a ** 2
    h = torch.sqrt(h2.clamp_min(0))
    u = dv / d[..., None]
    perp = torch.stack([-u[..., 1], u[..., 0]], -1)
    p = pa + a[..., None] * u + (side[:, None, None] * h[..., None]) * perp
    sin_g = torch.where(h2 > 0, d * h / (la_ * lb_), torch.full_like(d, -1.0))
    return p, sin_g


def generate(gen, dev, B, G, C, D, tries=24):
    J = G + C + D
    theta = torch.linspace(0, 2 * math.pi, T_SIM + 1, device=dev)[:-1]
    kind = torch.zeros(J, dtype=torch.int8)
    kind[G:G + C] = 1
    kind[G + C:] = 2
    U = lambda *s: torch.rand(*s, generator=gen, device=dev)
    pos = torch.zeros(B, J, 2, device=dev)
    par = torch.full((B, J, 2), -1, dtype=torch.long, device=dev)
    gear = torch.zeros(B, J, dtype=torch.long, device=dev)
    traj = torch.zeros(B, J, T_SIM, 2, device=dev)
    pos[:, 0] = U(B, 2) - 0.5
    if G > 1:
        pos[:, 1:G] = U(B, G - 1, 2) * 2 - 1
    traj[:, :G] = pos[:, :G, None, :]
    for c in range(C):
        k = G + c
        r = 0.12 + 0.33 * U(B) if c == 0 else 0.08 + 0.42 * U(B)
        phi = 2 * math.pi * U(B)
        if c == 0:
            rho = torch.ones(B, dtype=torch.long, device=dev)
        else:
            rho = RATIOS.to(dev)[torch.multinomial(RATIO_W.to(dev), B, replacement=True, generator=gen)]
        gear[:, k] = rho
        par[:, k, 0] = c
        pos[:, k] = pos[:, c] + r[:, None] * torch.stack([phi.cos(), phi.sin()], -1)
        ang = rho[:, None] * theta[None] + phi[:, None]
        traj[:, k] = pos[:, c, None, :] + r[:, None, None] * torch.stack([ang.cos(), ang.sin()], -1)

    alive = torch.ones(B, dtype=torch.bool, device=dev)
    moving = list(range(G, G + C))
    for k in range(G + C, J):
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(tries):
            todo = torch.nonzero(alive & ~done).squeeze(1)
            n = todo.numel()
            if n == 0:
                break
            w = torch.arange(1, len(moving) + 1, dtype=torch.float, device=dev) ** 1.5
            a = torch.tensor(moving, device=dev)[torch.multinomial(w, n, replacement=True, generator=gen)]
            b = (U(n) * (k - 1)).long().clamp_max(k - 2)
            b = torch.where(b >= a, b + 1, b)
            if C >= 2 and k == G + C:
                five = U(n) < 0.7
                a = torch.where(five, torch.full_like(a, G), a)
                b = torch.where(five, torch.full_like(b, G + 1), b)
            pa0, pb0 = pos[todo, a], pos[todo, b]
            span = (pb0 - pa0).norm(dim=-1)
            pk = 0.5 * (pa0 + pb0) + torch.randn(n, 2, generator=gen, device=dev) * (0.25 + 0.6 * span)[:, None]
            pk = pk.clamp(-1.6, 1.6)
            la, lb = (pk - pa0).norm(dim=-1), (pk - pb0).norm(dim=-1)
            cross = (pb0[:, 0] - pa0[:, 0]) * (pk[:, 1] - pa0[:, 1]) - (pb0[:, 1] - pa0[:, 1]) * (pk[:, 0] - pa0[:, 0])
            side = torch.sign(cross)
            ok = (la > 0.08) & (lb > 0.08) & (side != 0)
            p, sin_g = circle_meet(traj[todo, a], traj[todo, b], la, lb, side)
            ok &= sin_g.min(-1).values > MIN_SIN
            ok &= (p[..., 0].amax(-1) - p[..., 0].amin(-1) + p[..., 1].amax(-1) - p[..., 1].amin(-1)) > 0.05
            for c in range(C):
                r0 = (p - traj[todo, c]).norm(dim=-1)
                r1 = (p - traj[todo, G + c]).norm(dim=-1)
                rigid = ((r0.amax(-1) - r0.amin(-1)) < 1e-5) & ((r1.amax(-1) - r1.amin(-1)) < 1e-5)
                ok &= ~rigid
            idx = todo[ok]
            pos[idx, k] = pk[ok]
            par[idx, k, 0] = a[ok]
            par[idx, k, 1] = b[ok]
            traj[idx, k] = p[ok]
            done[idx] = True
        alive &= done
        moving.append(k)
    keep = torch.nonzero(alive).squeeze(1)
    return kind.to(dev), pos[keep], par[keep], gear[keep], traj[keep]


def resample(curves, n=N_CURVE):
    """Closed curves (K, T, 2) -> complex arc-length samples (K, n)."""
    K, T, _ = curves.shape
    closed = torch.cat([curves, curves[:, :1]], 1)
    seg = (closed[:, 1:] - closed[:, :-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(K, 1, device=curves.device), seg.cumsum(1)], 1)
    total = cum[:, -1:].clamp_min(1e-12)
    s = torch.arange(n, device=curves.device)[None] / n * total
    idx = (torch.searchsorted(cum.contiguous(), s.contiguous(), right=True) - 1).clamp(0, T - 1)
    c0, c1 = cum.gather(1, idx), cum.gather(1, idx + 1)
    w = ((s - c0) / (c1 - c0).clamp_min(1e-12))[..., None]
    pts = closed.gather(1, idx[..., None].expand(-1, -1, 2)) * (1 - w) + \
        closed.gather(1, (idx + 1)[..., None].expand(-1, -1, 2)) * w
    return torch.complex(pts[..., 0], pts[..., 1])


def curves_and_mechs(kind, pos, par, gear, traj, G):
    """Keep compact motor-driven tracer curves; return their data (on CPU)."""
    B, J = pos.shape[:2]
    dev = pos.device
    kind_l = kind.long()
    anc = torch.zeros(B, J, J, dtype=torch.bool, device=dev)
    rows = torch.arange(B, device=dev)
    for k in range(J):
        anc[:, k, k] = True
        if kind_l[k] >= 1:
            anc[:, k] |= anc[rows, par[:, k, 0]]
        if kind_l[k] == 2:
            anc[:, k] |= anc[rows, par[:, k, 1]]
    out = []
    for j in torch.nonzero(kind_l == 2).squeeze(1).tolist():
        A = anc[:, j]                                            # (B, J)
        ok = A[:, G] & (A.sum(-1) <= JP)                         # needs the motor crank
        z = resample(traj[:, j])
        mu = z.mean(-1)
        sc = (z - mu[:, None]).abs().pow(2).mean(-1).sqrt()
        tz = torch.complex(traj[..., 0], traj[..., 1])           # (B, J, T)
        reach = (tz - mu[:, None, None]).abs().amax(-1)          # (B, J)
        extent = torch.where(A, reach, torch.zeros_like(reach)).amax(-1) / sc
        ok &= (extent < MAX_EXTENT) & (sc > 1e-3)
        sel = torch.nonzero(ok).squeeze(1)
        if sel.numel() == 0:
            continue
        A = A[sel]
        # pruned order: ancestors in increasing index; stable sort puts them first
        order = torch.sort((~A).to(torch.int8), dim=1, stable=True).indices   # (K, J)
        newidx = A.long().cumsum(1) - 1                                       # (K, J)
        npr = A.sum(1)
        slot = torch.arange(J, device=dev)[None] < npr[:, None]
        jj = order[:, :JP] if J >= JP else torch.cat([order, order[:, :1].expand(-1, JP - J)], 1)
        slot = slot[:, :JP] if J >= JP else torch.cat([slot, torch.zeros(len(sel), JP - J, dtype=torch.bool, device=dev)], 1)
        b = sel[:, None]
        m_pos = torch.view_as_real((torch.complex(pos[b, jj, 0], pos[b, jj, 1]) - mu[sel, None]) / sc[sel, None])
        m_kind = torch.where(slot, kind_l[jj], torch.full_like(jj, -1))
        m_gear = torch.where(slot, gear[b, jj], torch.zeros_like(jj))
        pp = par[b, jj]                                                       # (K, JP, 2)
        remap = torch.gather(newidx, 1, pp.clamp_min(0).reshape(len(sel), -1)).reshape(pp.shape)
        m_par = torch.where((pp >= 0) & slot[..., None], remap, torch.full_like(pp, -1))
        zc = (z[sel] - mu[sel, None]) / sc[sel, None]
        out.append(dict(z=torch.view_as_real(zc).half().cpu(), pos=m_pos.float().cpu(),
                        kind=m_kind.to(torch.int8).cpu(), par=m_par.to(torch.int8).cpu(),
                        gear=m_gear.to(torch.int8).cpu(), n=npr.to(torch.int8).cpu(),
                        extent=extent[sel].half().cpu()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--mechs', type=int, default=10_000_000)
    ap.add_argument('--shard', type=int, default=500_000)
    ap.add_argument('--batch', type=int, default=20_000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    cpu_gen = torch.Generator().manual_seed(args.seed + 1)
    done_mechs, shard_id, buf, t0 = 0, 0, [], time.time()
    buf_mechs = 0
    while done_mechs < args.mechs:
        g, c, d = SIZES[torch.multinomial(SIZE_W, 1, generator=cpu_gen).item()]
        kind, pos, par, gear, traj = generate(gen, dev, args.batch, g, c, d)
        buf += curves_and_mechs(kind, pos, par, gear, traj, g)
        done_mechs += args.batch
        buf_mechs += args.batch
        if buf_mechs >= args.shard or done_mechs >= args.mechs:
            cat = {k: torch.cat([x[k] for x in buf]).numpy() for k in buf[0]}
            path = os.path.join(args.outdir, f'shard_{args.seed:03d}_{shard_id:04d}.npz')
            np.savez(path, **cat)
            print(f'{path}: {len(cat["z"])} curves, {done_mechs} mechs, {time.time() - t0:.0f}s', flush=True)
            shard_id += 1
            buf, buf_mechs = [], 0


if __name__ == '__main__':
    main()
