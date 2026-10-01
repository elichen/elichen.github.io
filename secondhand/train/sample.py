"""Prime the synthesis net with one line of a writer, then write new text."""
import argparse, json, math, pickle, sys
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('ckpt')
ap.add_argument('--out', default='samples.png')
ap.add_argument('--bias', type=float, default=0.75)
ap.add_argument('--writers', default='')
ap.add_argument('--texts', default='hello from dreamfield|a line in your own hand|Sphinx of black quartz, judge my vow')
ap.add_argument('--prime', default='The quick brown fox')
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location='cpu')
meta = ck['meta']
sys.argv = ['train.py', '--steps', '0', '--device', 'cpu', '--compile', '0', '--out', '/tmp/_unused_hand',
            '--hidden', str(meta['hidden']), '--mix', str(meta['mix']), '--win', str(meta['win'])]
_src = open('train.py').read()
_src = _src[:_src.index('model = Synth(')]
exec(_src)
model = Synth(V, meta['hidden'], meta['mix'], meta['win'])
model.load_state_dict(ck['model']); model.eval()


def to_input(pts, lift):
    off = np.diff(pts, axis=0, prepend=pts[:1]) / SPACING
    pen = np.concatenate([[1], lift[:-1]])
    return np.concatenate([off, pen[:, None]], 1).astype(np.float32)


@torch.no_grad()
def generate(prime_x, prime_text, text, bias, seed, max_per_char=40):
    g = torch.Generator().manual_seed(seed)
    full = (prime_text + ' ' + text) if prime_text else text
    C = torch.tensor([encode_text(full)]); CM = torch.ones_like(C, dtype=torch.float32)
    U = C.shape[1]
    state = None
    if prime_x is not None:
        Y, state = model(torch.from_numpy(prime_x)[None], C, CM, None)
        y = Y[:, -1]
        force_lift = True
    else:
        x = torch.tensor([[[0.0, 0.0, 1.0]]])
        Y, state = model(x, C, CM, None); y = Y[:, -1]
        force_lift = False
    out = []
    done_at = None
    for t in range(max_per_char * len(text) + 50):
        e, pi, mx, my, sx, sy, rho = model.split(y, bias)
        k = torch.multinomial(pi.exp(), 1, generator=g).item()
        mux, muy = mx[0, k].item(), my[0, k].item()
        sgx, sgy, r = sx[0, k].exp().item(), sy[0, k].exp().item(), rho[0, k].item()
        z1, z2 = torch.randn(2, generator=g).tolist()
        dx = mux + sgx * z1
        dy = muy + sgy * (r * z1 + math.sqrt(1 - r * r) * z2)
        lift = 1.0 if force_lift else float(torch.rand(1, generator=g).item() < torch.sigmoid(e[0]).item())
        if math.hypot(dx, dy) > 2.5:   # longer than any in-stroke step: a jump
            lift = 1.0
        force_lift = False
        out.append((dx, dy, lift))
        x = torch.tensor([[[dx, dy, lift]]])
        Y, state, phi = model(x, C, CM, state, return_phi=True)
        y = Y[:, -1]
        # stop once the window has moved past the last character
        (h, c), kappa, w, s2, s3 = state
        ab = model.win(h)
        al, be, _ = ab.chunk(3, -1)
        phi_end = (al.exp() * torch.exp(-be.exp() * (kappa - U) ** 2)).sum().item()
        if done_at is None and phi_end > phi[0, -1].max().item():
            done_at = t
        if done_at is not None and (t - done_at > 1 and lift):
            break
    o = np.array(out)
    pts = np.cumsum(o[:, :2] * SPACING, 0)
    pen = o[:, 2]  # pen[t]=1: move into point t was a jump
    return pts, pen


def draw(ax, pts, pen, x0=0, y0=0, color='k'):
    segs, cur = [], [pts[0]]
    for i in range(1, len(pts)):
        if pen[i] > 0.5:
            segs.append(np.array(cur)); cur = [pts[i]]
        else:
            cur.append(pts[i])
    segs.append(np.array(cur))
    for s in segs:
        if len(s) == 1:
            ax.plot(s[:, 0] + x0, -(s[:, 1] + y0), '.', color=color, ms=2)
        else:
            ax.plot(s[:, 0] + x0, -(s[:, 1] + y0), '-', color=color, lw=1.1)


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    texts = a.texts.split('|')
    ws = [int(w) for w in a.writers.split(',')] if a.writers else meta['val_writers'][:6]
    rows = []
    for w in ws:
        mine = [s for s in samples if s['w'] == w and s['text'] == a.prime] or [s for s in samples if s['w'] == w]
        rows.append(mine[0])
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(rows) + 1, 1, figsize=(14, 2.1 * (len(rows) + 1)))
    for ax in axs: ax.set_aspect('equal'); ax.axis('off')
    for r, s in enumerate(rows):
        ax = axs[r]
        px = to_input(s['pts'], s['lift'])
        draw(ax, s['pts'], np.concatenate([[1], s['lift'][:-1]]), 0, 0, '#1a4fa0')
        for i, t in enumerate(texts):
            pts, pen = generate(px, s['text'], t, a.bias, a.seed + i)
            pts = pts + s['pts'][-1]
            draw(ax, pts, pen, -pts[0, 0], 3.2 * (i + 1))
        ax.set_title(f"writer {s['w']}: prime '{s['text']}'", fontsize=8, loc='left')
        ax.set_xlim(-1, 40); ax.set_ylim(-3.2 * (len(texts) + 0.6), 2)
    ax = axs[-1]
    for i, t in enumerate(texts):
        pts, pen = generate(None, '', t, a.bias, a.seed + 10 + i)
        draw(ax, pts, pen, -pts[0, 0], 3.2 * i)
    ax.set_title('unprimed', fontsize=8, loc='left'); ax.set_xlim(-1, 40); ax.set_ylim(-3.2 * (len(texts) + 0.2), 2)
    plt.tight_layout(); plt.savefig(a.out, dpi=60)
    print('wrote', a.out)
