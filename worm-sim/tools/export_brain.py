"""geom.json (extract_geom.py) -> <model dir>/brain.bin and brain.json.
usage: python export_brain.py <geom.json> <model dir>

brain.bin: one little-endian float64 array the Rust solver reads.
Layout: [n_nodes, n_cells, n_conns,
         per node: parent, area(um2), ri(MOhm), cm, gpas, epas,
         per cell: soma node, 16 soma gbars (CHANNELS order),
         per conn: kind (0 gap, 1 exc, 2 inh), pre node, post node, |weight|]"""
import json, sys, struct, numpy as np
CHANNELS = ['gbshl1','gbshk1','gbkvs1','gbegl2','gbegl36','gbkqt3','gbegl19','gbunc2','gbcca1',
            'gbslo1_egl19','gbslo1_unc2','gbslo2_egl19','gbslo2_unc2','gbkcnl','gbnca','gbirk']
g = json.load(open(sys.argv[1]))
out = []
nn = sum(c['n'] for c in g['cells'])
out += [nn, len(g['cells']), len(g['conns'])]
node = []
for c in g['cells']:
    for k in range(c['n']):
        p = c['param']['neurite' if c['neurite'][k] else 'soma']
        node.append([c['parent'][k], c['area'][k], c['ri'][k], p['cm'], p['gpas'], p['epas']])
out += list(np.array(node).ravel())
for c in g['cells']:
    out += [c['soma']] + [c['param']['soma'][k] for k in CHANNELS]
for x in g['conns']:
    kind = 0 if x['cat'] == 'gj' else (1 if x['w'] >= 0 else 2)
    out += [kind, x['pre'], x['post'], abs(x['w'])]
a = np.array(out, dtype='<f8')
a.tofile(sys.argv[2] + '/brain.bin')
names = [c['name'] for c in g['cells']]
# the 15 sensory and 80 motor neurons of worm_in_env.py, as cell indices
INPUT = ["AWAL","AWAR","AWCL","AWCR","ASKL","ASKR","ALNL","ALNR","PLML","PHAL","PHAR","URYDL","URYDR","URYVL","URYVR"]
OUTPUT = ["RIML","RIMR","RMEL","RMER","RMED","RMEV","RMDDL","RMDDR","RMDL","RMDR","RMDVL","RMDVR","RIVL","RIVR","SMDDL","SMDDR","SMDVL","SMDVR","SMBDL","SMBDR","SMBVL","SMBVR","DA01","DA02","DA03","DA04","DA05","DA06","DA07","DA08","DA09","DB01","DB02","DB03","DB04","DB05","DB06","DB07","DD01","DD02","DD03","DD04","DD05","DD06","VA01","VA02","VA03","VA04","VA05","VA06","VA07","VA08","VA09","VA10","VA11","VA12","VB01","VB02","VB03","VB04","VB05","VB06","VB07","VB08","VB09","VB10","VB11","VD01","VD02","VD03","VD04","VD05","VD06","VD07","VD08","VD09","VD10","VD11","VD12","VD13"]
json.dump(dict(names=names, soma=[c['soma'] for c in g['cells']], input=[names.index(n) for n in INPUT], output=[names.index(n) for n in OUTPUT]),
          open(sys.argv[2] + '/brain.json', 'w'))
print(sys.argv[2], a.size * 8, 'bytes; zero-area membrane nodes:', sum(1 for n in node if n[1] == 0))
