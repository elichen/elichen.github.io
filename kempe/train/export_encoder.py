"""Write the trained encoder for the web app: encoder.json (shapes, offsets)
and encoder.bin (float32), plus a few reference embeddings for checking the
JavaScript port (kempe/tools/encoder_test.mjs).

usage: python export_encoder.py encoder.pt OUTDIR [--golden golden.json]
"""

import argparse
import json
import os
import numpy as np
import torch
from encoder import Encoder
from targets import targets
from linkage import resample, to_complex, normalise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('outdir')
    ap.add_argument('--golden', default='')
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location='cpu')
    a = ck['args']
    model = Encoder(a['width'], a['dim'])
    model.load_state_dict(ck['model'])
    model.eval()
    tensors, blobs, off = {}, [], 0
    for name, t in model.state_dict().items():
        arr = t.detach().float().numpy().ravel()
        tensors[name] = [off, int(arr.size)]
        blobs.append(arr.astype('<f4').tobytes())
        off += arr.nbytes
    os.makedirs(args.outdir, exist_ok=True)
    open(os.path.join(args.outdir, 'encoder.bin'), 'wb').write(b''.join(blobs))
    dils = [b.c1.dil for b in model.blocks]
    json.dump(dict(width=a['width'], dim=a['dim'], dils=dils, file='encoder.bin', tensors=tensors),
              open(os.path.join(args.outdir, 'encoder.json'), 'w'))
    print(f'{sum(v[1] for v in tensors.values())} weights')
    if args.golden:
        T = targets()
        cases = []
        for k in ['heart', 'star', 'wave', 'fish']:
            z = normalise(to_complex(resample(T[k][None])))[0][0]
            with torch.no_grad():
                e = model(torch.from_numpy(z.astype(np.complex64))[None])[0].numpy()
            cases.append(dict(name=k, re=z.real.tolist(), im=z.imag.tolist(), emb=e.tolist()))
        json.dump(cases, open(args.golden, 'w'))


if __name__ == '__main__':
    main()
