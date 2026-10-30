# Shiu et al. 2024's whole-brain LIF model (philshiu/Drosophila_brain_model, model.py, MIT),
# copied so our tools never import from the cloned repo. Same parameters and equations.
from textwrap import dedent
from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network, SpikeGeneratorGroup
from brian2 import mV, ms, Hz
import numpy as np
import pandas as pd

params = {
    'v_0': -52 * mV, 'v_rst': -52 * mV, 'v_th': -45 * mV, 't_mbr': 20 * ms,
    'tau': 5 * ms, 't_rfc': 2.2 * ms, 't_dly': 1.8 * ms, 'w_syn': .275 * mV,
    'r_poi': 150 * Hz, 'r_poi2': 0 * Hz, 'f_poi': 250,
    'eqs': dedent('''
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau               : volt (unless refractory)
        rfc                            : second
        '''),
    'eq_th': 'v > v_th',
    'eq_rst': 'v = v_rst; w = 0; g = 0 * mV',
}


def load(path_comp, path_con):
    comp = pd.read_csv(path_comp, index_col=0)
    con = pd.read_parquet(path_con)
    return comp, con


def build(n, con, p=params, silence=()):
    neu = NeuronGroup(n, model=p['eqs'], method='linear', threshold=p['eq_th'],
                      reset=p['eq_rst'], refractory='rfc', name='default_neurons', namespace=p)
    neu.v = p['v_0']
    neu.g = 0
    neu.rfc = p['t_rfc']
    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=p['t_dly'], name='default_synapses')
    syn.connect(i=con['Presynaptic_Index'].values, j=con['Postsynaptic_Index'].values)
    syn.w = con['Excitatory x Connectivity'].values * p['w_syn']
    for i in silence:
        syn.w['{} == i'.format(i)] = 0 * mV
    return neu, syn


def poisson(neu, exc, rate, p=params):
    """Shiu's PoissonInput: one input per excited neuron, which also loses its refractory period."""
    pois = []
    for i in exc:
        pois.append(PoissonInput(target=neu[i], target_var='v', N=1, rate=rate, weight=p['w_syn'] * p['f_poi']))
        neu[i].rfc = 0 * ms
    return pois


def forced(neu, exc, idx, steps, dt, p=params):
    """The same kick as PoissonInput (v += w_syn * f_poi in the 'synapses' slot of the same step),
    but at given steps, so another simulator can be fed identical input."""
    for i in exc:
        neu[i].rfc = 0 * ms
    gen = SpikeGeneratorGroup(neu.N, idx, steps * dt, dt=dt, name='kicks')
    kick = Synapses(gen, neu, 'w : volt', on_pre='v += w', name='kick_syn')
    kick.connect(j='i')
    kick.w = p['w_syn'] * p['f_poi']
    return [gen, kick]
