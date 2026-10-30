"""Brian 2 reference runs of Shiu's model with recorded Poisson input, for engine/src/bin/parity.rs.

    python tools/parity_ref.py <Completeness_783.csv> <Connectivity_783.parquet> <annotations.tsv> <model/neurons.json> <out dir> [seconds]

Cases: the page's experiment buttons (Shiu et al.'s neuron sets from neurons.json) and sets
taken from the annotations, three of which run away (thousands of neurons firing).

Each case draws the Poisson kicks with numpy, runs Brian with exactly those kicks (bm.forced:
same v += w_syn * f_poi in the same step as PoissonInput), and writes <case>.kicks,
<case>.spikes (u32 count, then (step, neuron) pairs) and <case>.silence / <case>.steps.
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from brian2 import Network, SpikeMonitor, defaultclock, Hz, prefs
import brian_model as bm

comp_path, con_path, ann_path, neurons_path, out = sys.argv[1:6]
seconds = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
prefs.codegen.target = 'cython'

comp, con = bm.load(comp_path, con_path)
index = {f: i for i, f in enumerate(comp.index)}
ann = pd.read_csv(ann_path, sep='\t', low_memory=False)
ann = ann[ann.root_id.isin(index)]


def group(**kw):
    m = np.ones(len(ann), bool)
    for k, v in kw.items():
        m &= (ann[k] == v).to_numpy()
    return sorted(index[r] for r in ann.root_id[m])


sugar_r = sorted(group(cell_sub_class='sugar', side='right') + group(cell_sub_class='sugar/low_salt', side='right'))
bitter_r = group(cell_sub_class='bitter', side='right')
jo = ann.cell_type.astype(str)
jo_ce = sorted(index[r] for r in ann.root_id[(jo.str.startswith('JO-C') | jo.str.startswith('JO-E')) & (ann.side == 'right')])
mn9 = index[720575940660219265]

page = json.load(open(neurons_path))
ps, named = page['sets'], page['named']
cases = {
    'page_sugar150': ([(i, 150) for i in ps['sugar']['index']], []),
    'page_sugar_bitter': ([(i, 150) for i in ps['sugar']['index'] + ps['bitter']['index']], []),
    'page_water200': ([(i, 200) for i in ps['water']['index']], []),
    'page_jo_ce150': ([(i, 150) for i in ps['JO-CE']['index']], []),
    'page_silence': ([(i, 100) for i in ps['sugar']['index']], [named['CB0553']]),
    'page_runaway': ([(i, 150) for i in ps['LB3b-right']['index']], []),
    'sugar150': ([(i, 150) for i in sugar_r], []),
    'sugar_bitter': ([(i, 100) for i in sugar_r] + [(i, 200) for i in bitter_r], []),
    'jo_ce200': ([(i, 200) for i in jo_ce], []),
}
# silence the three most active non-sugar neurons of sugar150 once it has run
dt = float(defaultclock.dt)
steps = int(round(seconds / dt))
rng = np.random.default_rng(783)


def write(path, ev):
    ev = np.asarray(ev, np.uint32).reshape(-1, 2)
    with open(path, 'wb') as f:
        f.write(np.uint32(len(ev)).tobytes())
        f.write(ev.tobytes())


def run(name, exc, silence):
    print(name, 'excited', len(exc), 'silenced', len(silence), flush=True)
    ev = []
    for i, hz in exc:
        hits = np.nonzero(rng.random(steps) < hz * dt)[0]
        ev += [(s, i) for s in hits]
    ev.sort()
    ev = np.array(ev, np.int64)
    neu, syn = bm.build(len(comp), con, silence=silence)
    objs = bm.forced(neu, [i for i, _ in exc], ev[:, 1], ev[:, 0], defaultclock.dt)
    mon = SpikeMonitor(neu)
    Network(neu, syn, mon, *objs).run(seconds * 1e3 * bm.ms)
    sp = np.stack([np.round(np.asarray(mon.t / bm.ms) / (dt * 1e3)).astype(np.int64), np.asarray(mon.i)], 1)
    sp = sp[np.lexsort((sp[:, 1], sp[:, 0]))]
    write(out / f'{name}.kicks', ev)
    write(out / f'{name}.spikes', sp)
    (out / f'{name}.steps').write_text(str(steps))
    (out / f'{name}.silence').write_text(' '.join(map(str, silence)))
    print(f'  {len(sp)} spikes in {len(set(sp[:, 1]))} neurons; MN9 {np.sum(sp[:, 1] == mn9) / seconds:.0f} Hz', flush=True)
    return sp


for name, (exc, silence) in cases.items():
    sp = run(name, exc, silence)
    if name == 'sugar150':
        ids, n = np.unique(sp[:, 1], return_counts=True)
        top = [int(i) for i in ids[np.argsort(-n)] if i not in set(sugar_r)][:3]
        run('sugar150_silenced', exc, top)
