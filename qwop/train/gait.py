"""Gait diagnostics in the float64 reference simulator.

  python gait.py rl.pt [more.npz ...]

Reports, per policy, over 256 greedy episodes: 100 m time, knee-down fraction,
how evenly the two legs share the lead (thigh split sign), whether foot strikes
alternate, and stride coverage: the share of the distance covered by valid
strides as jppo.gait_update defines them (a foot landing ahead of the other,
not the foot of the last valid stride, no knee down in between). A gallop keeps
the same lead leg, so it scores near zero; a clean run scores near 100%.
"""
import sys
import numpy as np
import sim
from train import load_actor, greedy

O = sim.NJ * 5


def analyse(path, n=256, seed=11, max_steps=3000, margin=0.2, cap=3.0):
    np.random.seed(seed)
    sim.seed_numba(seed)
    env = sim.QwopVec(n, noise=0.05, max_steps=max_steps)
    pol = greedy(load_actor(path))
    active = np.ones(n, bool)
    finished = np.zeros(n, bool)
    steps = np.zeros(n)
    knee = np.zeros(n)
    r_lead = np.zeros(n)          # steps with the right thigh ahead of the left
    flight = np.zeros(n)          # steps with no foot or knee on the ground
    prev_foot = np.zeros((n, 2), bool)
    last_strike = -np.ones(n, int)
    strikes = np.zeros(n)
    alt = np.zeros(n)             # strikes by the other foot than last time
    swaps = np.zeros(n)           # sign changes of the thigh split
    prev_sign = np.zeros(n)
    last_valid = -np.ones(n, int)
    x0 = np.zeros(n)
    kflag = np.zeros(n, bool)
    credit = np.zeros(n)
    for _ in range(max_steps):
        dx, fell, goal, _to = env.step(pol(env))
        c = env.ACC[:, O:O + 12:2] > 0          # footR heel/toe, footL heel/toe, kneeR, kneeL
        foot = np.stack([c[:, 0] | c[:, 1], c[:, 2] | c[:, 3]], 1)
        steps += active
        knee += active & (c[:, 4] | c[:, 5])
        flight += active & ~c.any(1)
        split = env.S[:, sim.THIGH_R, 2] - env.S[:, sim.THIGH_L, 2]
        sign = np.sign(np.where(np.abs(split) > 0.15, split, 0.0))
        r_lead += active & (split > 0)
        ch = active & (sign != 0) & (prev_sign != 0) & (sign != prev_sign)
        swaps += ch
        prev_sign = np.where(sign != 0, sign, prev_sign)
        for f in (0, 1):
            hit = active & foot[:, f] & ~prev_foot[:, f]
            strikes += hit
            alt += hit & (last_strike == 1 - f)
            last_strike = np.where(hit, f, last_strike)
        kflag |= c[:, 4] | c[:, 5]
        lead = env.S[:, sim.FOOT_R, 0] - env.S[:, sim.FOOT_L, 0]
        front = np.stack([lead > margin, -lead > margin], 1)
        hit = foot & ~prev_foot & front & (last_valid[:, None] != np.arange(2)[None, :])
        valid = hit.any(1)
        xt = env.S[:, sim.TORSO, 0]
        credit += np.where(active & valid & ~kflag, np.clip(xt - x0, 0.0, cap), 0.0)
        last_valid = np.where(hit[:, 0], 0, np.where(hit[:, 1], 1, last_valid))
        x0 = np.where(valid, xt, x0)
        kflag &= ~valid
        prev_foot = foot
        finished |= active & goal
        active &= ~(fell | goal)
        if not active.any():
            break
    t = steps * sim.DT
    m = finished
    dist = np.maximum(np.minimum(env.S[:, sim.TORSO, 0], sim.GOAL_X), 1.0)
    return dict(stride=(credit / dist).mean(), finish=m.mean(), time=t[m].mean() if m.any() else float('nan'),
                knee=(knee / steps).mean(), flight=(flight / steps).mean(),
                r_lead=(r_lead / steps).mean(), alt=(alt / np.maximum(strikes - 1, 1)).mean(),
                strikes_per_s=(strikes / t).mean(), swaps_per_s=(swaps / t).mean())


if __name__ == '__main__':
    for p in sys.argv[1:]:
        r = analyse(p)
        print(f"{p:40s} finish {r['finish'] * 100:5.1f}%  100m {r['time']:6.2f}s  knee {r['knee'] * 100:5.2f}%  "
              f"stride coverage {r['stride'] * 100:5.1f}%  flight {r['flight'] * 100:4.1f}%  R-leads {r['r_lead'] * 100:4.1f}%  "
              f"alternating strikes {r['alt'] * 100:5.1f}%  strikes/s {r['strikes_per_s']:.2f}  "
              f"leg swaps/s {r['swaps_per_s']:.2f}", flush=True)
