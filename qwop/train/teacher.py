"""Deterministic scripted teacher: a cyclic finite-state machine over key combos.

Each of K phases holds one key combo until the hip split angle crosses a
threshold (or a timeout hits); two torso-lean reflexes override the cycle.
Its parameters are found with the cross-entropy method, evaluated in the
batched simulator (one candidate per env). `python teacher.py` runs the search
and writes teacher.json.
"""
import json, sys, time
import numpy as np
import sim

K = 4


class Teacher:
    """Vectorised FSM. Param arrays are (n, ...) or broadcastable (1, ...)."""

    def __init__(self, A, SG, D, TH, PHI, n):
        self.A, self.SG, self.D, self.TH, self.PHI = A, SG, D, TH, PHI
        self.n = n
        self.phase = np.zeros(n, dtype=np.int64)
        self.count = np.zeros(n, dtype=np.int64)
        self.rows = np.arange(n) if A.shape[0] == n else np.zeros(n, dtype=np.int64)

    @classmethod
    def from_json(cls, path, n):
        p = json.load(open(path))
        return cls(np.array([p['A']]), np.array([p['SG']]), np.array([p['D']]),
                   np.array([p['TH']]), np.array([p['PHI']]), n)

    def reset(self, idx):
        self.phase[idx] = 0
        self.count[idx] = 0

    def act(self, S):
        """Advance the FSM on the observed state and return the teacher's action."""
        r = self.rows
        split = (S[:, sim.THIGH_R, 2] - S[:, sim.THIGH_L, 2])
        self.count += 1
        p = self.phase
        cond = (self.SG[r, p] * (split - self.TH[r, p]) > 0) | (self.count >= self.D[r, p])
        self.phase = np.where(cond, (p + 1) % K, p)
        self.count = np.where(cond, 0, self.count)
        act = self.A[r, self.phase]
        ta = S[:, sim.TORSO, 2]
        act = np.where(ta > self.PHI[r, 0], self.A[r, K], act)
        act = np.where(ta < -self.PHI[r, 1], self.A[r, K + 1], act)
        return act


def evaluate(A, SG, D, TH, PHI, reps, steps, noise, eps, rng):
    """Mean distance (m) per candidate; a fall ends the run and costs 3 m."""
    npop = A.shape[0]
    rep = lambda x: np.repeat(x, reps, axis=0)
    n = npop * reps
    t = Teacher(rep(A), rep(SG), rep(D), rep(TH), rep(PHI), n)
    env = sim.QwopVec(n, noise=noise, max_steps=steps + 1)
    alive = np.ones(n, bool)
    dist = np.zeros(n)
    for _ in range(steps):
        a = t.act(env.S)
        a = np.where(rng.random(n) < eps, rng.integers(0, sim.NACT, n), a)
        dx, fell, goal, _ = env.step(a)
        dist += np.where(alive, dx, 0.0)
        dist -= np.where(alive & fell, 3.0, 0.0)
        alive &= ~fell
    return dist.reshape(npop, reps).mean(1), alive.reshape(npop, reps).mean(1)


def search(npop=2048, reps=4, gens=40, steps=600, seed=0):
    rng = np.random.default_rng(seed)
    nd = K + 2
    probs = np.full((nd, sim.NACT), 1.0 / sim.NACT)
    sg_p = np.full(K, 0.5)
    mu = np.concatenate([np.full(K, 10.0), np.zeros(K), [0.4, 0.6]])
    sd = np.concatenate([np.full(K, 6.0), np.full(K, 0.6), [0.3, 0.3]])
    hof = []
    for g in range(gens):
        A = np.stack([rng.choice(sim.NACT, npop, p=probs[i]) for i in range(nd)], 1)
        SG = np.where(rng.random((npop, K)) < sg_p, 1.0, -1.0)
        X = mu + sd * rng.standard_normal((npop, mu.size))
        D = np.clip(np.round(X[:, :K]), 2, 40)
        TH = X[:, K:2 * K]
        PHI = np.clip(X[:, 2 * K:], 0.05, 1.5)
        fit, surv = evaluate(A, SG, D, TH, PHI, reps, steps, 0.05, 0.03, rng)
        order = np.argsort(-fit)
        el = order[:npop // 20]
        hof.append([x[order[:4]] for x in (A, SG, D, TH, PHI)])
        for i in range(nd):
            cnt = np.bincount(A[el, i], minlength=sim.NACT) + 0.5
            probs[i] = 0.5 * probs[i] + 0.5 * cnt / cnt.sum()
        sg_p = np.clip(0.5 * sg_p + 0.5 * (SG[el] > 0).mean(0), 0.05, 0.95)
        Xe = np.concatenate([D[el], TH[el], PHI[el]], 1)
        mu = 0.5 * mu + 0.5 * Xe.mean(0)
        sd = 0.5 * sd + 0.5 * (Xe.std(0) + np.concatenate([np.full(K, 0.5), np.full(K + 2, 0.02)]))
        print(f"gen {g:2d} best {fit[order[0]]:6.2f} m  elite mean {fit[el].mean():6.2f}  "
              f"pop mean {fit.mean():6.2f}  best surv {surv[order[0]]:.2f}", flush=True)
    # single-generation scores are noisy: re-score the hall of fame with many reps
    A, SG, D, TH, PHI = [np.concatenate([h[i] for h in hof]) for i in range(5)]
    fit, surv = evaluate(A, SG, D, TH, PHI, 64, steps, 0.05, 0.03, rng)
    i = int(np.argmax(fit))
    print(f"hall of fame winner: {fit[i]:.2f} m, survival {surv[i]:.3f}")
    return fit[i], dict(A=A[i].tolist(), SG=SG[i].tolist(), D=D[i].tolist(),
                        TH=TH[i].tolist(), PHI=PHI[i].tolist())


if __name__ == '__main__':
    t0 = time.time()
    fit, params = search(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print("best fitness", fit, params, f"{time.time() - t0:.0f}s")
    json.dump(dict(params, fitness=fit), open(sys.argv[2] if len(sys.argv) > 2 else 'teacher.json', 'w'), indent=1)
