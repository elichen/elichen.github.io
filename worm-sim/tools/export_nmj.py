"""BAAIWorm's anatomical neuromuscular map and proprioceptive map -> <model dir>/nmj.bin.

BAAIWorm ships both as spreadsheets (eworm/components/param/connection) but its loop
never uses them; it reads muscles out of the 80 motor neurons with a dense fitted matrix.
  neuron_muscle.xlsx Sheet1: 80 motor neurons (worm.rs OUTPUT order) x 96 muscles, 1 = synapse
  muscle_neuron.xlsx Sheet1: 96 muscles x 43 motor neurons, 1 = the neuron senses that
      muscle's bending (Wen et al. 2012, Yeon et al. 2018, Krieg et al. 2022)

nmj.bin, little-endian float64: 80 x 96 NMJ (row = neuron), then 96 x 80 proprioception
(row = muscle, columns in OUTPUT order, 0 for neurons not in the sheet). Muscle columns are
DR01-24, VR01-24, DL01-24, VL01-24, the body's order.

usage: python export_nmj.py <BAAIWorm checkout> <model dir>
"""
import sys

import numpy as np
import pandas as pd


OUTPUT = ["RIML", "RIMR", "RMEL", "RMER", "RMED", "RMEV", "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR", "RIVL", "RIVR", "SMDDL", "SMDDR", "SMDVL", "SMDVR", "SMBDL", "SMBDR", "SMBVL", "SMBVR",
          "DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "DA08", "DA09", "DB01", "DB02", "DB03", "DB04", "DB05", "DB06", "DB07", "DD01", "DD02", "DD03", "DD04", "DD05", "DD06",
          "VA01", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11", "VA12", "VB01", "VB02", "VB03", "VB04", "VB05", "VB06", "VB07", "VB08", "VB09", "VB10", "VB11",
          "VD01", "VD02", "VD03", "VD04", "VD05", "VD06", "VD07", "VD08", "VD09", "VD10", "VD11", "VD12", "VD13"]
MUSCLES = [f"{q}{s:02d}" for q in ("DR", "VR", "DL", "VL") for s in range(1, 25)]

d = sys.argv[1] + "/eworm/components/param/connection/"
nm = pd.read_excel(d + "neuron_muscle.xlsx", sheet_name="Sheet1").set_index("neuron/muscle")
assert list(nm.index) == OUTPUT
nmj = np.nan_to_num(nm[MUSCLES].values.astype(float))
mn = pd.read_excel(d + "muscle_neuron.xlsx", sheet_name="Sheet1").set_index("muscle/neuron")
prop = np.zeros((96, 80))
for n in mn.columns:
    prop[:, OUTPUT.index(n)] = np.nan_to_num(mn.loc[MUSCLES, n].values.astype(float))
print(f"NMJ synapses {int(nmj.sum())} ({(nmj.sum(1) > 0).sum()} neurons), proprioceptive inputs {int(prop.sum())} ({(prop.sum(0) > 0).sum()} neurons)")
np.concatenate([nmj.ravel(), prop.ravel()]).astype("<f8").tofile(sys.argv[2] + "/nmj.bin")
