# Drives BAAIWorm's own WormInEnv (NEURON brain) against its own SoftCon body via the
# headless interact module: wormInEnv.step() then 8 body steps per 1/30 s tick, as
# SimulationWindow::Timer (MC_C302) does.
# usage: BAAIWORM=<checkout> run.py offline|online steps out.npz [tx ty tz]   (LOCAL=1: target in body coordinates)
import os, sys, time
import numpy as np
# their code targets numpy 1.21 / Python 3.8: restore removed aliases instead of editing it
np.float, np.int, np.object = float, int, object
import xml.etree.ElementTree as ET
ET.ElementTree.getiterator = ET.ElementTree.iter
ROOT = os.path.abspath(os.environ['BAAIWORM'])
sys.path.insert(0, os.environ.get('INTERACT', os.path.dirname(os.path.abspath(__file__)) + '/build'))
sys.path.insert(1, ROOT + '/eworm/ghost_in_mesh_sim')
sys.path.insert(1, ROOT)
os.chdir(ROOT + '/Metaworm')  # Environment loads args/ relative to cwd
import interact
from worm_in_env import WormInEnv

mode, steps, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
group = 'video_' + mode
G = ROOT + '/eworm/ghost_in_mesh_sim/data/tuned/' + group
env_kw = {}
if len(sys.argv) > 6 and not os.environ.get('LOCAL'):
    env_kw['target_location'] = [np.array([float(v) for v in sys.argv[4:7]], dtype=np.float32)]
w = WormInEnv(**env_kw)
if os.environ.get('LOCAL') and len(sys.argv) > 6:  # target given in body coordinates instead
    w.worm_0.target_location = np.array([float(v) for v in sys.argv[4:7]], dtype=np.float32)
    w.worm_0.concentration = w.worm_0._get_concentration(w.worm_0.head_location)
w.config['interaction_mode'] = mode
if mode == 'online':  # the online group's readout, as worm_in_env.py would load with group_name = "video_online"
    import pickle
    w.worm_0.wout = pickle.load(open(G + f'/{group}_wout.pkl', 'rb'))
rec = {k: [] for k in ['input', 'mnv', 'muscle', 'head', 'x', 'states']}
t0 = time.time()
init = int(os.environ.get('INIT', 0))  # first INIT ticks: body driven by output/test.npy (step_init), brain still senses
for s in range(steps):
    if s == init and init:  # step() only sets up its (record-only) arrays on step 1, which step_init took
        for a in ('rel_x', 'rel_y', 'rel_z', 'rel_vx', 'rel_vy', 'rel_vz'): setattr(w, a, np.zeros((0, 17)))
    w.step_init() if s < init else w.step()
    interact.tick()
    rec['input'].append(w.test_i[-1]); rec['mnv'].append(w.mnv_0.copy()); rec['muscle'].append(np.array(w.init_muscle_file[s] if s < init else w.muscle_signal_list[0]))
    rec['head'].append(interact.get_head_location(0)); rec['x'].append(interact.positions(0)); rec['states'].append(interact.states(0))
    if s % 30 == 29:
        print(f'{(s+1)/30:.1f} s  head {rec["head"][-1]}  {time.time()-t0:.0f} s wall', flush=True)
np.savez_compressed(out, **{k: np.array(v) for k, v in rec.items()},
                    target_local=w.worm_0.target_location, init=init, init_pos=interact.get_init_position(0), init_rot=interact.get_init_rotation(0))
