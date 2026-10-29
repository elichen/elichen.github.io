"""Build each BAAIWorm cell in NEURON from the hoc text (parsed here, never executed)
and export the node tree NEURON solves: area, ri, parent, 3D position, plus the
circuit's connections. NEURON only computes geometry (segment area and axial
resistance from the pt3d points); no BAAIWorm code or .mod file runs.
usage: python extract_geom.py <BAAIWorm checkout> <out geom.json>   (needs: pip install neuron numpy)"""
import sys, os, re, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_pickle import load
from neuron import h
import numpy as np

B = sys.argv[1] + '/eworm/'
group = 'video_offline'  # video_online has the same circuit; only its readout differs
circ = load(B + f'ghost_in_mesh_sim/data/tuned/{group}/{group}_abscircuit.pkl')

def parse_hoc(name):
    txt = open(B + f'components/model/{name}.hoc').read()
    creates = re.findall(r'^\s*create\s+(\w+)\s*$', txt, re.M)
    sub = re.search(r'all = new SectionList\(\)(.*?)\n\}', txt, re.S).group(1)
    all_order = re.findall(r'^\s*(\w+)\s+all\.append\(\)', sub, re.M)
    conns = re.findall(r'connect\s+(\w+)\(0\),\s*(\w+)\(1\.0\)', txt)
    pts = {}
    for sec, args in re.findall(r'^\s*(\w+)\s*\{\s*pt3dadd\(([^)]*)\)\s*\}', txt, re.M):
        pts.setdefault(sec, []).append([float(a) for a in args.split(',')])
    return creates, all_order, conns, pts

cells = []
nodes_total = 0
seg_node = {}  # (cell index, segment index) -> global node
for ci, cell in enumerate(circ.cells):
    name = cell.name
    param = json.load(open(B + f'components/param/cell/{name}.json'))
    creates, all_order, conns, pts = parse_hoc(name)
    assert sorted(creates) == sorted(all_order), name
    secs = {s: h.Section(name=f'{name}.{s}') for s in creates}
    for s, p in pts.items():
        for x, y, z, d in p:
            h.pt3dadd(x, y, z, d, sec=secs[s])
    parent = {}
    for child, par in conns:
        secs[child].connect(secs[par](1.0), 0)
        parent[child] = par
    roots = [s for s in creates if s not in parent]
    assert roots == ['Soma'], (name, roots)
    for s in all_order:
        cat = 'soma' if 'Soma' in s else 'neurite'
        sec = secs[s]
        sec.Ra = param[cat]['Ra']; sec.cm = param[cat]['cm']
        sec.nseg = 1 if 'Soma' in s else int(math.ceil(sec.L / 2))
    # node list in NEURON order is irrelevant; we use our own order: per section in all order
    nodes = []  # dicts: area, ri, parent(local), cm, gpas, epas, soma
    endnode = {}
    def where(sec, x):
        arc = x * sec.L
        n3 = sec.n3d()
        for k in range(n3 - 1):
            a0, a1 = sec.arc3d(k), sec.arc3d(k + 1)
            if arc <= a1 or k == n3 - 2:
                t = 0.0 if a1 == a0 else min(max((arc - a0) / (a1 - a0), 0.0), 1.0)
                p0 = np.array([sec.x3d(k), sec.y3d(k), sec.z3d(k)]); p1 = np.array([sec.x3d(k + 1), sec.y3d(k + 1), sec.z3d(k + 1)])
                return list(p0 + t * (p1 - p0))
    def add(area, ri, par, cat, sec_name, pos):
        nodes.append(dict(area=area, ri=ri, parent=par, cat=cat, pos=pos))
        return len(nodes) - 1
    root = add(0.0, 0.0, -1, 'soma', 'Soma', where(secs['Soma'], 0.0))  # Soma(0)
    seg_idx = 0
    seg_local = []
    order = []
    # sections must be added after their parent: walk all_order but defer children
    pending = list(all_order)
    while pending:
        for s in pending:
            if s == 'Soma' or parent[s] in endnode:
                break
        pending.remove(s)
        sec = secs[s]
        cat = 'soma' if 'Soma' in s else 'neurite'
        prev = root if s == 'Soma' else endnode[parent[s]]
        local = []
        for seg in sec:
            prev = add(seg.area(), seg.ri(), prev, cat, s, where(sec, seg.x))
            local.append(prev)
        endnode[s] = add(0.0, sec(1.0).ri(), prev, cat, s, where(sec, 1.0))
        order.append((s, local))
    # segment index in abs circuit follows all_order
    by_sec = dict(order)
    k = 0
    for s in all_order:
        for n in by_sec[s]:
            seg_node[(ci, k)] = nodes_total + n
            k += 1
    assert k == len(cell.segments), (name, k, len(cell.segments))
    for s in all_order:
        assert all(seg.name.split('.')[-1] == s for seg in cell.segments if False) or True
    soma_node = by_sec['Soma'][0]
    cells.append(dict(name=name, index=int(cell.index), offset=nodes_total, n=len(nodes),
                      soma=nodes_total + soma_node, param=param,
                      area=[n['area'] for n in nodes], ri=[n['ri'] for n in nodes],
                      parent=[(-1 if n['parent'] < 0 else n['parent'] + nodes_total) for n in nodes],
                      neurite=[n['cat'] == 'neurite' for n in nodes], pos=[n['pos'] for n in nodes]))
    nodes_total += len(nodes)
    # check segment names line up
    k = 0
    for s in all_order:
        for _ in by_sec[s]:
            assert cell.segments[k].name.endswith('.' + s), (name, k, cell.segments[k].name, s)
            k += 1
    for sec in secs.values():
        pass
    secs.clear()

conns = []
for c in circ.connections:
    pre = seg_node[(circ.cells.index(c.pre_cell), c.pre_segment.index)]
    post = seg_node[(circ.cells.index(c.post_cell), c.post_segment.index)]
    conns.append(dict(cat=c.category, w=float(c.weight), pre=pre, post=post))
print(group, 'cells', len(cells), 'nodes', nodes_total, 'conns', len(conns))
json.dump(dict(cells=cells, conns=conns), open(sys.argv[2], 'w'))
