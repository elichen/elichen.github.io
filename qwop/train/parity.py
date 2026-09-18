"""Check that physics.js reproduces sim.py: same actions -> same states/obs."""
import json, subprocess, sys, os
import numpy as np
import sim

T = 300
rng = np.random.default_rng(0)
actions = np.repeat(rng.integers(0, sim.NACT, T // 5), 5)
S = np.zeros((1, sim.NB, 6)); ACC = np.zeros((1, sim.NACC))
sim.set_pose(S[0], 0.0, sim.POSE0_TORSO, sim.POSE0)
fell = np.zeros(1, dtype=np.int64); dx = np.zeros(1); obs = np.zeros((1, sim.NOBS))
states, observations, fells = [], [], []
for t in range(T):
    sim.step_batch(S, ACC, actions[t:t + 1].astype(np.int64), fell, dx)
    sim.obs_batch(S, ACC, actions[t:t + 1].astype(np.int64), obs)
    states.append(S[0].ravel().tolist()); observations.append(obs[0].tolist()); fells.append(int(fell[0]))

js = """
const P = require('../physics.js');
const actions = %s;
const w = P.create(); const out = []; const o = new Float64Array(P.NOBS);
for (const a of actions) { const f = P.step(w.s, w.acc, a); P.observe(w.s, w.acc, a, o);
  out.push({s: Array.from(w.s), o: Array.from(o), f: f ? 1 : 0}); }
console.log(JSON.stringify(out));
""" % json.dumps(actions.tolist())
res = json.loads(subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout)
for t in [0, 9, 29, 99, 199, 299]:
    ds = np.abs(np.array(res[t]['s']) - np.array(states[t])).max()
    do = np.abs(np.array(res[t]['o']) - np.array(observations[t])).max()
    print(f"t={t:3d} max|ds|={ds:.3e} max|dobs|={do:.3e} fell py/js={fells[t]}/{res[t]['f']}")
ok = np.abs(np.array(res[29]['s']) - np.array(states[29])).max() < 1e-9
print("PARITY OK" if ok else "PARITY FAIL"); sys.exit(0 if ok else 1)
