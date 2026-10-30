"""Neuropil surfaces for the page's outline -> model/neuropils.json.

    python tools/export_neuropils.py <dir of JFRC2NP.surf.fw .ply files> model/neuropils.json

The surfaces are Ito et al. 2014's JFRC2 neuropils as transformed to FlyWire (FAFB14.1) space
for fafbseg (file fafbseg/data/JFRC2NP.surf.fw.zip). Vertices in µm, rounded to 0.1 µm.
"""
import glob
import json
import os
import sys
import numpy as np
import trimesh

src, out = sys.argv[1:3]
names, ranges, verts, faces = [], [], [], []
nv = 0
for f in sorted(glob.glob(os.path.join(src, '*.ply'))):
    m = trimesh.load(f)
    names.append(os.path.basename(f)[:-4])
    ranges.append([len(faces) and sum(len(x) for x in faces), len(m.faces)])
    verts.append(np.round(m.vertices / 1000, 1))
    faces.append(m.faces + nv)
    nv += len(m.vertices)
v = np.concatenate(verts)
f = np.concatenate(faces)
json.dump({'names': names, 'ranges': ranges, 'vertices': v.ravel().tolist(), 'faces': f.ravel().tolist()},
          open(out, 'w'), separators=(',', ':'))
print(len(names), 'neuropils', len(v), 'vertices', len(f), 'faces')
