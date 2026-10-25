"""Export a BBF snapshot (params_<step>.pkl from the training job) for the page.

  python export.py <params.pkl> <out dir> [--states test/states.npy]

Writes <out>/policy.json + policy.bin: the target network (the one BBF acts with), conv kernels as float16,
dense kernels as int8 with one scale per output column. Also writes <out>/test.json: Q values of a few
states from both the original float32 weights and the exported ones, for parity.mjs.
"""
import argparse, json, os, pickle
import numpy as np
import jax
import flax
from bbf_net import network, q_values

ap = argparse.ArgumentParser()
ap.add_argument('params')
ap.add_argument('out')
ap.add_argument('--states', help='uint8 [N,84,84,4] .npy; random states if omitted')
args = ap.parse_args()

snap = pickle.load(open(args.params, 'rb'))
params = flax.core.unfreeze(snap['target'])
p = params['params']
num_actions = p['head']['advantage']['net']['kernel'].shape[1] // 51

order = []  # (name, array, kind)
for s in range(3):
    stage = p['encoder'][f'ResidualStage_{s}']
    for c in range(5):
        conv = stage[f'Conv_{c}']
        order.append((f'stage{s}/conv{c}/kernel', conv['kernel'], 'f16'))
        order.append((f'stage{s}/conv{c}/bias', conv['bias'], 'f32'))
for name, layer in [('projection', p['projection']['net']), ('advantage', p['head']['advantage']['net']),
                    ('value', p['head']['value']['net'])]:
    order.append((f'{name}/kernel', layer['kernel'], 'i8'))
    order.append((f'{name}/bias', layer['bias'], 'f32'))

os.makedirs(args.out, exist_ok=True)
blob, tensors = bytearray(), []
exported = jax.tree.map(np.asarray, params)
for name, a, kind in order:
    a = np.asarray(a, np.float32)
    entry = {'name': name, 'shape': list(a.shape), 'dtype': kind, 'offset': len(blob)}
    if kind == 'f16':
        data, back = a.astype(np.float16), a.astype(np.float16).astype(np.float32)
    elif kind == 'f32':
        data, back = a, a
    else:
        scale = np.maximum(np.abs(a).max(0), 1e-12) / 127
        data = np.clip(np.round(a / scale), -127, 127).astype(np.int8)
        back = data.astype(np.float32) * scale
        while len(blob) % 4: blob.append(0)
        entry['scaleOffset'] = len(blob)
        blob += scale.astype(np.float32).tobytes()
        entry['offset'] = len(blob)
    blob += data.tobytes()
    while len(blob) % 4: blob.append(0)
    # same tensor, dequantized, for the reference Q values
    *path, leaf = name.split('/')
    if path[0].startswith('stage'):
        node = exported['params']['encoder'][f'ResidualStage_{path[0][5:]}'][f'Conv_{path[1][4:]}']
    else:
        node = {'projection': exported['params']['projection']['net'],
                'advantage': exported['params']['head']['advantage']['net'],
                'value': exported['params']['head']['value']['net']}[path[0]]
    node[leaf] = back
    tensors.append(entry)

open(os.path.join(args.out, 'policy.bin'), 'wb').write(bytes(blob))
json.dump({'numActions': num_actions, 'numAtoms': 51, 'vmax': 10.0, 'trainingSteps': int(snap['training_steps']),
           'tensors': tensors}, open(os.path.join(args.out, 'policy.json'), 'w'))

if args.states:
    states = np.load(args.states)
else:
    states = np.random.default_rng(0).integers(0, 256, (4, 84, 84, 4), dtype=np.uint8)
net = network(num_actions)
q32 = q_values(net, params, states)
qx = q_values(net, exported, states)
json.dump({'states': [s.reshape(-1).tolist() for s in states], 'q': qx.tolist(), 'qFloat32': q32.tolist()},
          open(os.path.join(args.out, 'test.json'), 'w'))
print(f'{len(blob) / 1e6:.1f} MB, {num_actions} actions; '
      f'max |Q export - Q float32| = {np.abs(qx - q32).max():.4f}, argmax agree {np.mean(qx.argmax(1) == q32.argmax(1)):.2f}')
