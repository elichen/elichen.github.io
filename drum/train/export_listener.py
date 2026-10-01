"""Write the listener for the web app: listener.json (tensor offsets) and
listener.bin (float16), plus reference outputs for drum/tools/listener_test.mjs.

usage: python export_listener.py listener.pt SPECTRA.npz OUTDIR [--golden golden.json]
"""

import argparse
import json
import os
import numpy as np
import torch
from train_hear import Listener, features, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('data')
    ap.add_argument('outdir')
    ap.add_argument('--golden', default='')
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location='cpu')
    a = ck['args']
    model = Listener(a['width'], a.get('depth', 4))
    model.load_state_dict(ck['model'])
    model.eval()
    tensors, blobs, off = {}, [], 0
    for name, t in model.state_dict().items():
        if name in ('basis', 'circle'):
            continue
        arr = t.detach().float().numpy().ravel().astype('<f2')
        tensors[name] = [off, int(arr.size)]
        blobs.append(arr.tobytes())
        off += arr.size
    os.makedirs(args.outdir, exist_ok=True)
    open(os.path.join(args.outdir, 'listener.bin'), 'wb').write(b''.join(blobs))
    json.dump(dict(width=a['width'], depth=a.get('depth', 4), M=M, file='listener.bin', tensors=tensors),
              open(os.path.join(args.outdir, 'listener.json'), 'w'))
    print(f'{off} weights, {2 * off / 1e6:.1f} MB')
    if args.golden:
        # reference outputs with the float16-rounded weights the page uses
        sd = {k: (v.half().float() if k not in ('basis', 'circle') else v) for k, v in model.state_dict().items()}
        model.load_state_dict(sd)
        d = np.load(args.data)
        rng = np.random.default_rng(0)
        cases = []
        for i in rng.choice(len(d['lam']), 4, replace=False):
            lam = torch.tensor(d['lam'][i:i + 1], dtype=torch.float32)
            for K in (5, 40):
                with torch.no_grad():
                    z, conf = model(features(lam, torch.tensor([K])))
                cases.append(dict(lam=d['lam'][i].astype(float).tolist(), K=K,
                                  outlines=[np.stack([z[0, m].real, z[0, m].imag], 1).ravel().tolist() for m in range(M)],
                                  confidence=torch.softmax(conf[0], 0).tolist()))
        json.dump(cases, open(args.golden, 'w'))


if __name__ == '__main__':
    main()
