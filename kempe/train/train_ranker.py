"""Train the tuning predictor (kempe/ranker.js): an MLP that, for a drawing and
a shortlisted machine, predicts the error the machine reaches after tuning.

Data comes from tools/ranker_data.mjs (features exactly as the page computes
them). Each doodle contributes a group of 48 machines: the 32 closest by raw
alignment error and 16 random ones from deeper in the shortlist. Loss: listwise
KL between the softmax of the (negated) tuned errors and of the predictions,
plus a small regression term on log error. Doodles are split by index into
training and validation groups.

usage: python train_ranker.py DATA_PREFIX OUT.json [--epochs 30]
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = 48


def load(prefix):
    meta = json.load(open(prefix + '.json'))
    x = np.fromfile(prefix + '.x.f32', np.float32).reshape(meta['rows'], meta['features'])
    y = np.fromfile(prefix + '.y.f32', np.float32)
    q = np.fromfile(prefix + '.q.i32', np.int32)
    # keep only complete groups of G rows (a doodle's rows are contiguous)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    lens = np.diff(np.r_[starts, len(q)])
    keep = starts[lens == G]
    idx = (keep[:, None] + np.arange(G)[None]).ravel()
    return x[idx].reshape(-1, G, x.shape[1]), y[idx].reshape(-1, G), q[keep]


class MLP(nn.Module):
    def __init__(self, f, h=128):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(f, h), nn.Linear(h, h), nn.Linear(h, 1)

    def forward(self, x):
        x = F.gelu(self.l1(x), approximate='tanh')
        x = F.gelu(self.l2(x), approximate='tanh')
        return self.l3(x).squeeze(-1)


def pick_eval(pred, raw, tuned, k=8):
    """Mean best tuned error among the top-k picked by pred, and by raw error."""
    by_pred = tuned.gather(1, pred.topk(k, dim=1, largest=False).indices).min(1).values
    by_raw = tuned.gather(1, raw.topk(k, dim=1, largest=False).indices).min(1).values
    return by_pred.mean().item(), by_raw.mean().item(), tuned.min(1).values.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix')
    ap.add_argument('out')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--tau', type=float, default=0.01)
    args = ap.parse_args()
    X, Y, Q = load(args.prefix)
    val = (Q % 10) == 0
    Xt, Yt, Xv, Yv = map(torch.from_numpy, (X[~val], Y[~val], X[val], Y[val]))
    print(f'{len(Xt)} training groups, {len(Xv)} validation groups, {X.shape[2]} features')
    flat = Xt.reshape(-1, Xt.shape[2])
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    norm = lambda x: (x - mean) / std
    model = MLP(X.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * (len(Xt) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    g = torch.Generator().manual_seed(0)
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xt), generator=g)
        model.train()
        for i in range(0, len(Xt) - 255, 256):
            b = perm[i:i + 256]
            pred = model(norm(Xt[b]))
            t = torch.log(Yt[b] + 1e-3)
            pt = F.softmax(-Yt[b] / args.tau, 1)
            kl = (pt * (torch.log(pt + 1e-12) - F.log_softmax(-pred / 0.05, 1))).sum(1).mean()
            loss = kl + 0.1 * F.mse_loss(pred, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(norm(Xv))
            a, b_, o = pick_eval(pv, Xv[..., 0], Yv)
        print(f'epoch {ep}: loss {loss.item():.4f}  best of top-8 by model {a:.4f}  by raw error {b_:.4f}  oracle {o:.4f}', flush=True)
    layers = [dict(w=l.weight.detach().tolist(), b=l.bias.detach().tolist()) for l in (model.l1, model.l2, model.l3)]
    json.dump(dict(mean=mean.tolist(), std=std.tolist(), layers=layers), open(args.out, 'w'))


if __name__ == '__main__':
    main()
