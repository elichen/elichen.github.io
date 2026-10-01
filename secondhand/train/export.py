"""Export a checkpoint to hand.bin (fp16, concatenated per-layer matrices) + hand.json.

Layer k's gate matrix is [W_ih | W_hh] so the browser does one matvec per layer
on the concatenated input [x, w, h_below, h_self]. Gate order is torch's i,f,g,o.
Also writes golden.json: a short teacher-forced run for the JS parity test.
"""
import json, sys, numpy as np, torch

ck = torch.load(sys.argv[1], map_location='cpu')
out = sys.argv[2] if len(sys.argv) > 2 else 'export'
import os; os.makedirs(out, exist_ok=True)
sd = {k: v.float().numpy() for k, v in ck['model'].items()}
meta = ck['meta']
H, M, K = meta['hidden'], meta['mix'], meta['win']
V = len(meta['vocab'])

mats = {
    'W1': np.concatenate([sd['cell1.weight_ih'], sd['cell1.weight_hh']], 1),
    'b1': sd['cell1.bias_ih'] + sd['cell1.bias_hh'],
    'Wwin': sd['win.weight'], 'bwin': sd['win.bias'],
    'W2': np.concatenate([sd['lstm2.weight_ih_l0'], sd['lstm2.weight_hh_l0']], 1),
    'b2': sd['lstm2.bias_ih_l0'] + sd['lstm2.bias_hh_l0'],
    'W3': np.concatenate([sd['lstm3.weight_ih_l0'], sd['lstm3.weight_hh_l0']], 1),
    'b3': sd['lstm3.bias_ih_l0'] + sd['lstm3.bias_hh_l0'],
    'Wout': sd['out.weight'], 'bout': sd['out.bias'],
}
layout, blobs, off = {}, [], 0
for k, a in mats.items():
    a16 = a.astype(np.float16)
    layout[k] = dict(offset=off, shape=list(a.shape))
    blobs.append(a16.tobytes()); off += a16.size
open(f'{out}/hand.bin', 'wb').write(b''.join(blobs))
json.dump(dict(vocab=''.join(meta['vocab'][1:]), spacing=meta['spacing'], H=H, M=M, K=K, V=V,
               layout=layout, step=ck.get('step')), open(f'{out}/hand.json', 'w'))
print('params', off, 'bytes', off * 2)

# golden: run the fp16-rounded weights in torch on a fixed input
if '--golden' in sys.argv:
    sys.argv = ['train.py', '--steps', '0', '--device', 'cpu', '--compile', '0', '--out', '/tmp/_unused_hand',
                '--hidden', str(H), '--mix', str(M), '--win', str(K)]
    src = open('train.py').read(); src = src[:src.index('model = Synth(')]
    exec(src)
    model = Synth(V, H, M, K)
    model.load_state_dict({k: v.half().float() for k, v in ck['model'].items()}); model.eval()
    s = samples[0]
    off_ = np.diff(s['pts'], axis=0, prepend=s['pts'][:1]) / SPACING
    pen = np.concatenate([[1], s['lift'][:-1]])
    X = np.concatenate([off_, pen[:, None]], 1).astype(np.float32)[:60]
    text = s['text'] + ' hello'
    C = torch.tensor([encode_text(text)]); CM = torch.ones_like(C, dtype=torch.float32)
    with torch.no_grad():
        Y, st, phi = model(torch.from_numpy(X)[None], C, CM, None, return_phi=True)
    json.dump(dict(text=text, x=X.tolist(), y_last=Y[0, -1].tolist(), y_10=Y[0, 10].tolist(),
                   phi_last=phi[0, -1].tolist(), kappa=st[1][0].tolist()), open(f'{out}/golden.json', 'w'))
    print('golden written')
