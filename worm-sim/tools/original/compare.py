# run.py's npz vs loop_run's DUMPX (positions) and DUMP (muscles) from the same start (WARM=0)
# usage: compare.py ref.npz mine.x mine_muscles.txt
import sys, numpy as np
ref = np.load(sys.argv[1]); mx = np.fromfile(sys.argv[2]).reshape(-1, 2952); mm = np.loadtxt(sys.argv[3])
n = min(len(ref['x']), len(mx))
rx = ref['x'][:n].astype(np.float64); mx = mx[:n]; rm = ref['muscle'][:n]; mm = mm[:n]
print('step0 vertex diff max', np.abs(rx[0] - mx[0]).max())
for s in list(range(0, 30, 5)) + list(range(29, n, 30)):
    a, b = rx[s].reshape(-1, 3), mx[s].reshape(-1, 3)
    print(f'{(s+1)/30:5.2f}s muscle |d|max {np.abs(rm[s]-mm[s]).max():.2e}  vertex rms {np.sqrt(((a-b)**2).sum(1).mean()):.2e}  com ref {a.mean(0).round(3)} mine {b.mean(0).round(3)}')
