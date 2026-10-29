# 10 food directions, 2 units from the head end, 60 s: BAAIWorm's own code vs worm-sim's engine
import subprocess, os, sys, numpy as np
from concurrent.futures import ThreadPoolExecutor
# usage: BAAIWORM=<checkout> PY=<venv python> sweep.py <out dir>; then summary.py <out dir>
B = os.path.abspath(os.environ['BAAIWORM']); PY = os.environ['PY']
E = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); secs = 60
dirs = [(0,0,-1),(.7,0,-.7),(-.7,0,-.7),(0,.7,-.7),(0,-.7,-.7),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1)]
jobs = []
for i, d in enumerate(dirs):
    d = np.array(d) / np.linalg.norm(d); f = [f'{x:.5f}' for x in d * 2.0 + np.array([0, 0, -0.313])]
    jobs.append(([PY, f'{HERE}/run.py', 'online', str(secs * 30), f'{OUT}/ref_{i}.npz'] + f, f'{OUT}/ref_{i}.log', {'LOCAL': '1'}))
    jobs.append(([f'{E}/engine/target/release/loop_run', f'{E}/model', 'online', str(secs)] + f, f'{OUT}/mine_{i}.log', {'WARM': '0', 'DUMPX': f'{OUT}/mine_{i}.x'}))
def run(j):
    with open(j[1], 'w') as o: subprocess.run(j[0], stdout=o, stderr=subprocess.STDOUT, env={**os.environ, **j[2]})
with ThreadPoolExecutor(12) as ex: list(ex.map(run, jobs))
print('done')
