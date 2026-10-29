"""The readouts (into the model dir) and the reference data the check tools compare
against (into a ref dir, not committed).
usage: python export_refs.py <BAAIWorm checkout> <model dir> <ref dir>

model/wout_{offline,online}.bin   80 x 96 readouts (video_offline is the page's)
ref/ref_soma.bin      NEURON's 80 motor-neuron somas, 18,000 steps (video_offline)
ref/input300.bin      the input it was given (build2/forfigure_input.txt, first 300 values;
                      from value 238 on the file holds a later run, see brain_parity)
ref/ref_muscles.bin   SoftCon's recorded muscle activations (worm_muscles_300_220428-152601)
ref/ref_states.bin    SoftCon's states from the same run (worm_states_300_220428-152601)"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_pickle import load
R, M, O = sys.argv[1:4]
os.makedirs(O, exist_ok=True)
T = R + '/eworm/ghost_in_mesh_sim/data/tuned/'
for g in ('offline', 'online'):
    load(T + f'video_{g}/video_{g}_wout.pkl').astype('<f8').tofile(f'{M}/wout_{g}.bin')
load(T + 'video_offline/video_offline_model_output_soma.pkl').astype('<f8').tofile(f'{O}/ref_soma.bin')
inp = np.array([float(x) for x in open(R + '/build2/forfigure_input.txt').read().split()])[:300]
inp.astype('<f8').tofile(f'{O}/input300.bin')
S = R + '/eworm/ghost_in_mesh_sim/data/'
np.array(json.load(open(S + 'muscle/worm_muscles_300_220428-152601.json'))['Frames'], '<f8').tofile(f'{O}/ref_muscles.bin')
np.array(json.load(open(S + 'state/worm_states_300_220428-152601.json'))['Frames'], '<f8').tofile(f'{O}/ref_states.bin')
print('wrote', M, O)
