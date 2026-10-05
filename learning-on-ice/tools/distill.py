# Copy Unitree's G1 walking controller into a Stream-AC actor (DAgger), as the
# starting point for streaming RL.
#
#   git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym /tmp/unitree_rl_gym
#   python3 tools/distill.py /tmp/unitree_rl_gym tools/runs/student.json      (needs mujoco, torch)
#   node tools/pretrain.mjs --wasm --student tools/runs/student.json --lr-policy 0 --steps 300000 --out warm
#
# The teacher is unitree_rl_gym's deploy/pre_train/g1/motion.pt (BSD-3-Clause): a
# small LSTM that drives the 12 leg joints on Unitree's PD gains. G1 below is a Python
# copy of g1-env.js (same model, gains, action mapping and observation); keep them in sync.
# Round 0 lets the teacher drive; every later round the student drives, the teacher
# labels each state it visits, and the student is refit on everything so far.
import json, sys
import numpy as np, mujoco, torch

SCENE = __file__.rsplit("/", 2)[0] + "/robot/scene.xml"
KP = np.array([100, 100, 100, 150, 40, 40] * 2, float)
KD = np.array([2, 2, 2, 4, 2, 2] * 2, float)
LEG_DEFAULT = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2)
SCALE, LIMIT, PERIOD = 0.25, 4.0, 0.8
COMMAND = 1.0          # teacher's speed command (it walks ~0.75 m/s)
NOISE = 0.3            # exploration std while the student drives, and its initial std head
ENVS, ROUND_STEPS, ROUNDS = 16, 120_000, 8


def to_body(q, v):
    """Rotate world vector v into the frame of quaternion q = (w, x, y, z)."""
    w, qv = q[0], -np.asarray(q[1:])
    t = 2 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


class G1:
    def __init__(self):
        m = self.m = mujoco.MjModel.from_xml_path(SCENE)
        m.opt.timestep = 0.004
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != 0:
                m.geom_contype[g], m.geom_conaffinity[g] = 1, 0
        for i in range(12):
            m.actuator_gainprm[i, 0] = KP[i]
            m.actuator_biasprm[i, 1:3] = [-KP[i], -KD[i]]
        m.geom_friction[:, 0] = 0.6
        self.d = mujoco.MjData(m)
        self.center = m.key_qpos[0][7:].copy()
        self.center[:12] = LEG_DEFAULT
        self.lo, self.hi = m.actuator_ctrlrange.T
        self.t = 0

    def reset(self, rng):
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[7:] += rng.uniform(-0.02, 0.02, m.nq - 7)
        d.qvel[:] = rng.uniform(-0.02, 0.02, m.nv)
        mujoco.mj_forward(m, d)
        self.t = 0
        return self.obs()

    def clock(self):
        p = 2 * np.pi * ((self.t * 0.02) % PERIOD) / PERIOD
        return np.array([np.sin(p), np.cos(p)])

    def gravity(self):
        return to_body(self.d.qpos[3:7], np.array([0, 0, -1.0]))

    def obs(self):
        d = self.d
        return np.concatenate([[d.qpos[2]], self.gravity(), to_body(d.qpos[3:7], d.qvel[:3]), d.qvel[3:6],
                               d.qpos[7:19], d.qvel[6:18], self.clock()])

    def step(self, action):
        """Returns (obs, fell, applied action)."""
        a = np.clip(action, -LIMIT, LIMIT)
        self.d.ctrl[:] = np.clip(self.center + SCALE * np.concatenate([a, np.zeros(17)]), self.lo, self.hi)
        for _ in range(5):
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        q = self.d.qpos
        fell = not (0.5 <= q[2] <= 1.0 and 1 - 2 * (q[4] ** 2 + q[5] ** 2) >= 0.5)
        return self.obs(), fell or self.t >= 1000, a


class Teacher:
    """motion.pt run in a batch, one LSTM memory per environment."""
    def __init__(self, path, n):
        sd = torch.jit.load(path).state_dict()
        self.lstm = torch.nn.LSTM(47, 64)
        self.lstm.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith("memory.")})
        self.mlp = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ELU(), torch.nn.Linear(32, 12))
        self.mlp.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("actor.")})
        self.h, self.c = torch.zeros(1, n, 64), torch.zeros(1, n, 64)
        self.last = np.zeros((n, 12))

    def reset(self, i):
        self.h[0, i] = self.c[0, i] = 0
        self.last[i] = 0

    @torch.no_grad()
    def act(self, envs):
        obs = np.zeros((len(envs), 47), np.float32)
        for i, e in enumerate(envs):
            d = e.d
            obs[i] = np.concatenate([d.qvel[3:6] * 0.25, e.gravity(), [COMMAND * 2.0, 0, 0], d.qpos[7:19] - LEG_DEFAULT,
                                     d.qvel[6:18] * 0.05, self.last[i], e.clock()])
        out, (self.h, self.c) = self.lstm(torch.from_numpy(obs)[None], (self.h, self.c))
        return self.mlp(out[0]).numpy().astype(float)


class Student(torch.nn.Module):
    """stream-ac.js's actor: in -> 128 -> LayerNorm -> LeakyReLU -> 128 -> LayerNorm -> LeakyReLU -> mean, std."""
    def __init__(self, n_in, n_out, h=128):
        super().__init__()
        self.l1, self.l2 = torch.nn.Linear(n_in, h), torch.nn.Linear(h, h)
        self.mu, self.std = torch.nn.Linear(h, n_out), torch.nn.Linear(h, n_out)
        self.ln = torch.nn.LayerNorm(h, elementwise_affine=False, eps=1e-5)
        self.act = torch.nn.LeakyReLU(0.01)
        torch.nn.init.zeros_(self.std.weight)
        torch.nn.init.constant_(self.std.bias, float(np.log(np.expm1(NOISE))))   # softplus(bias) = NOISE

    def forward(self, x):
        return self.mu(self.act(self.ln(self.l2(self.act(self.ln(self.l1(x)))))))


def collect(teacher_path, student, norm, rng):
    envs = [G1() for _ in range(ENVS)]
    teacher = Teacher(teacher_path, ENVS)
    obs = np.array([e.reset(rng) for e in envs])
    X, Y, falls = [], [], 0
    for _ in range(ROUND_STEPS // ENVS):
        label = teacher.act(envs)
        if student is None:
            mean = label
        else:
            with torch.no_grad():
                mean = student(torch.from_numpy(((obs - norm[0]) / norm[1]).astype(np.float32))).numpy()
        action = mean + NOISE * rng.standard_normal(mean.shape)
        X.append(obs.copy())
        Y.append(label)
        for i, e in enumerate(envs):
            obs[i], done, applied = e.step(action[i])
            teacher.last[i] = applied   # the teacher sees the action actually applied
            if done:
                falls += e.t < 1000
                obs[i] = e.reset(rng)
                teacher.reset(i)
    return np.concatenate(X), np.concatenate(Y), falls


def fit(student, norm, X, Y, epochs):
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    Xn = torch.from_numpy(((X - norm[0]) / norm[1]).astype(np.float32))
    Yt = torch.from_numpy(Y.astype(np.float32))
    for _ in range(epochs):
        perm = torch.randperm(len(Xn))
        for k in range(0, len(Xn), 2048):
            idx = perm[k:k + 2048]
            loss = ((student(Xn[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return loss.item()


def main(rl_gym, out):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    teacher_path = f"{rl_gym}/deploy/pre_train/g1/motion.pt"
    X, Y, falls = collect(teacher_path, None, None, rng)
    norm = (X.mean(0), np.maximum(X.std(0), 0.05))   # floor keeps still joints from dominating
    student = Student(X.shape[1], 12)
    loss = fit(student, norm, X, Y, 30)
    print(f"round 0, teacher drives: {falls} falls, fit loss {loss:.4f}", flush=True)
    for k in range(1, ROUNDS + 1):
        Xk, Yk, falls = collect(teacher_path, student, norm, rng)
        X, Y = np.concatenate([X, Xk]), np.concatenate([Y, Yk])
        loss = fit(student, norm, X, Y, 15)
        print(f"round {k}, student drives: {falls} falls in {ROUND_STEPS:,} steps, fit loss {loss:.4f}", flush=True)
    # stream-ac.js Trunk layout: W1, b1, W2, b2, W_mean, b_mean, W_std, b_std (weights [out, in], row-major)
    sd = student.state_dict()
    actor = np.concatenate([sd[k].numpy().ravel() for k in
                            ["l1.weight", "l1.bias", "l2.weight", "l2.bias", "mu.weight", "mu.bias", "std.weight", "std.bias"]])
    json.dump({"actor": actor.tolist(), "obsMean": norm[0].tolist(), "obsVar": (norm[1] ** 2).tolist(), "count": len(X)},
              open(out, "w"))
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
