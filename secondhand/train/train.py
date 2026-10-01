"""Handwriting synthesis (Graves 2013) trained on BRUSH, built for priming.

Training sequences concatenate 1-3 samples by the same writer on one line, so
the network learns that a hand persists: shown one line, it should keep
writing in that hand. Sizes are in x-heights; offsets are divided by the
resampling step so an in-stroke step has length ~1.
"""
import argparse, math, os, pickle, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--data', default='data/brush_norm.pkl')
p.add_argument('--out', default='runs/dev')
p.add_argument('--hidden', type=int, default=400)
p.add_argument('--mix', type=int, default=20)
p.add_argument('--win', type=int, default=10)
p.add_argument('--batch', type=int, default=48)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--steps', type=int, default=60000)
p.add_argument('--maxlen', type=int, default=1100)
p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps')
p.add_argument('--resume', default='')
p.add_argument('--log_every', type=int, default=100)
p.add_argument('--eval_every', type=int, default=2000)
p.add_argument('--val_writers', type=int, default=12)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--compile', type=int, default=0)
p.add_argument('--fixT', type=int, default=1)
p.add_argument('--graph', type=int, default=0)
p.add_argument('--upad', type=int, default=72)
p.add_argument('--nparts', default='0.15,0.45,0.40')
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = torch.device(args.device)

D = pickle.load(open(args.data, 'rb'))
SPACING = D['spacing']
samples = D['samples']
chars = sorted(set(''.join(s['text'] for s in samples)))
VOCAB = ['\0'] + chars                       # index 0 = unknown/pad
CI = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)
writers = sorted(D['writers'])
vrng = np.random.default_rng(1234)
val_w = set(vrng.choice(writers, args.val_writers, replace=False).tolist())
by_w = {}
for s in samples:
    by_w.setdefault(s['w'], []).append(s)
train_ids = [i for i, s in enumerate(samples) if s['w'] not in val_w]
val_ids = [i for i, s in enumerate(samples) if s['w'] in val_w]
print(f'vocab {V}  train samples {len(train_ids)}  val samples {len(val_ids)}  val writers {sorted(val_w)}', flush=True)


def make_sequence(first, r, augment=True, nparts=None):
    """Concatenate 1-3 samples by one writer on a shared baseline."""
    s0 = samples[first]
    mine = by_w[s0['w']]
    if nparts is None:
        nparts = r.choice([1, 2, 3], p=[float(v) for v in args.nparts.split(',')])
    parts = [s0] + [mine[r.integers(len(mine))] for _ in range(nparts - 1)]
    gap = D['writers'][s0['w']]['gap']
    pts, lift, text, x0 = [], [], [], 0.0
    total = 0
    for k, s in enumerate(parts):
        if k and (total + len(s['pts']) > args.maxlen or len(' '.join(text + [s['text']])) > args.upad):
            break
        q = s['pts'].copy()
        q[:, 0] += x0 - q[:, 0].min()
        pts.append(q); lift.append(s['lift']); text.append(s['text'])
        x0 = q[:, 0].max() + gap * math.exp(r.normal(0, 0.12))
        total += len(q)
    pts = np.concatenate(pts).astype(np.float64)
    lift = np.concatenate(lift)
    if augment:
        shear = r.uniform(-0.15, 0.15)
        sc = math.exp(r.normal(0, 0.06))
        sx = math.exp(r.normal(0, 0.05))
        pts[:, 0] = (pts[:, 0] + shear * (-pts[:, 1])) * sc * sx
        pts[:, 1] *= sc
    pts = pts[:args.maxlen]; lift = lift[:args.maxlen].copy(); lift[-1] = 1
    off = np.diff(pts, axis=0, prepend=pts[:1]) / SPACING
    # input t = (offset into point t, lifted before point t); target t = input t+1
    pen = np.concatenate([[1], lift[:-1]])
    x = np.concatenate([off, pen[:, None]], 1).astype(np.float32)
    return x, ' '.join(text)


def encode_text(t):
    return [CI.get(c, 0) for c in t]


def batch(ids, r, augment=True, nparts=None):
    seqs = [make_sequence(i, r, augment, nparts) for i in ids]
    T = args.maxlen if args.fixT else max(len(x) for x, _ in seqs)
    U = max(args.upad, max(len(t) for _, t in seqs))
    assert not args.fixT or U == args.upad
    X = np.zeros((len(seqs), T, 3), np.float32)
    M = np.zeros((len(seqs), T), np.float32)
    C = np.zeros((len(seqs), U), np.int64)
    CM = np.zeros((len(seqs), U), np.float32)
    for b, (x, t) in enumerate(seqs):
        X[b, :len(x)] = x; M[b, :len(x) - 1] = 1
        e = encode_text(t); C[b, :len(e)] = e; CM[b, :len(e)] = 1
    return (torch.from_numpy(X).to(dev), torch.from_numpy(M).to(dev),
            torch.from_numpy(C).to(dev), torch.from_numpy(CM).to(dev))


def step_fn(xt, w, h, c, kappa, Wwh, winW, winb, C1, CM, upos):
    g = xt + F.linear(torch.cat([w, h], 1), Wwh)
    i, f, gg, o = g.chunk(4, 1)
    c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
    h = torch.sigmoid(o) * torch.tanh(c)
    a, b, k = F.linear(h, winW, winb).chunk(3, -1)
    kappa = kappa + k.exp()
    phi = (a.exp().unsqueeze(2) * torch.exp(-b.exp().unsqueeze(2) * (kappa.unsqueeze(2) - upos) ** 2)).sum(1) * CM
    w = torch.bmm(phi.unsqueeze(1), C1).squeeze(1)
    return h, c, w, kappa, phi


STEP = torch.compile(step_fn, dynamic=False) if args.compile else step_fn


class Synth(nn.Module):
    def __init__(self, V, H, M, K):
        super().__init__()
        self.V, self.H, self.M, self.K = V, H, M, K
        self.cell1 = nn.LSTMCell(3 + V, H)
        self.win = nn.Linear(H, 3 * K)
        self.lstm2 = nn.LSTM(3 + V + H, H, batch_first=True)
        self.lstm3 = nn.LSTM(3 + V + H, H, batch_first=True)
        self.out = nn.Linear(3 * H, 1 + 6 * M)
        with torch.no_grad():
            self.win.bias[2 * K:].fill_(math.log(1 / 16.0))
            self.win.bias[:K].fill_(0.0)
            self.win.bias[K:2 * K].fill_(0.0)

    def window(self, h, kappa, C1, CM, upos):
        a, b, k = self.win(h).chunk(3, -1)
        a, b = a.exp(), b.exp()
        kappa = kappa + k.exp()
        phi = (a.unsqueeze(2) * torch.exp(-b.unsqueeze(2) * (kappa.unsqueeze(2) - upos) ** 2)).sum(1)
        phi = phi * CM
        w = torch.bmm(phi.unsqueeze(1), C1).squeeze(1)
        return w, kappa, phi

    def forward(self, X, C, CM, state=None, return_phi=False):
        B, T, _ = X.shape
        C1 = (C.unsqueeze(-1) == torch.arange(self.V, device=C.device)).float() * CM.unsqueeze(-1)
        upos = torch.arange(C.shape[1], device=X.device).float().view(1, 1, -1)
        if state is None:
            h = X.new_zeros(B, self.H); c = X.new_zeros(B, self.H)
            kappa = X.new_zeros(B, self.K); w = X.new_zeros(B, self.V)
            s2 = s3 = None
        else:
            (h, c), kappa, w, s2, s3 = state
        Wx = F.linear(X, self.cell1.weight_ih[:, :3], self.cell1.bias_ih + self.cell1.bias_hh)
        Wwh = torch.cat([self.cell1.weight_ih[:, 3:], self.cell1.weight_hh], 1)
        hs, ws, phis = [], [], []
        wb = self.win.weight, self.win.bias
        xs = Wx.unbind(1)
        for t in range(T):
            h, c, w, kappa, phi = STEP(xs[t], w, h, c, kappa, Wwh, wb[0], wb[1], C1, CM, upos)
            hs.append(h); ws.append(w)
            if return_phi:
                phis.append(phi)
        H1 = torch.stack(hs, 1); W = torch.stack(ws, 1)
        H2, s2 = self.lstm2(torch.cat([X, W, H1], -1), s2)
        H3, s3 = self.lstm3(torch.cat([X, W, H2], -1), s3)
        Y = self.out(torch.cat([H1, H2, H3], -1))
        st = ((h, c), kappa, w, s2, s3)
        if return_phi:
            return Y, st, torch.stack(phis, 1)
        return Y, st

    def split(self, Y, bias=0.0):
        M = self.M
        e = Y[..., 0]
        pi, mx, my, sx, sy, rho = Y[..., 1:].split(M, -1)
        pi = F.log_softmax(pi * (1 + bias), -1)
        sx = (sx - bias).clamp(-6, 6); sy = (sy - bias).clamp(-6, 6)
        rho = torch.tanh(rho) * 0.999
        return e, pi, mx, my, sx, sy, rho


def nll(model, Y, X, Mask):
    tgt = X[:, 1:]; Y = Y[:, :-1]; Mk = Mask[:, :-1]
    e, pi, mx, my, lsx, lsy, rho = model.split(Y)
    dx = tgt[..., :1]; dy = tgt[..., 1:2]
    zx = (dx - mx) / lsx.exp(); zy = (dy - my) / lsy.exp()
    om = 1 - rho ** 2
    logN = -(zx ** 2 + zy ** 2 - 2 * rho * zx * zy) / (2 * om) - lsx - lsy - 0.5 * om.log() - math.log(2 * math.pi)
    lp = torch.logsumexp(pi + logN, -1)
    le = -F.binary_cross_entropy_with_logits(e, tgt[..., 2], reduction='none')
    n = Mk.sum()
    return -(lp * Mk).sum() / n, -(le * Mk).sum() / n


model = Synth(V, args.hidden, args.mix, args.win).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
nparams = sum(p.numel() for p in model.parameters())
print(f'params {nparams/1e6:.2f}M on {dev}', flush=True)
step = 0
if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); step = ck['step']
meta = dict(vocab=VOCAB, spacing=SPACING, hidden=args.hidden, mix=args.mix, win=args.win, val_writers=sorted(val_w))
json.dump(meta, open(f'{args.out}/meta.json', 'w'))

vr = np.random.default_rng(99)
val_batches = [batch(vr.choice(val_ids, args.batch), vr, augment=False, nparts=2) for _ in range(6)]


def evaluate():
    model.eval(); tot = [0, 0]
    with torch.no_grad():
        for X, Mk, C, CM in val_batches:
            Y, _ = model(X, C, CM)
            a, b = nll(model, Y, X, Mk); tot[0] += a.item(); tot[1] += b.item()
    model.train()
    return tot[0] / len(val_batches), tot[1] / len(val_batches)


sched = lambda s: args.lr * min(1.0, s / 500) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))) * 0.95 + 0.05)
params = list(model.parameters())


def train_step(X, Mk, C, CM):
    Y, _ = model(X, C, CM)
    lxy, le = nll(model, Y, X, Mk)
    (lxy + le).backward()
    gn = torch.nn.utils.clip_grad_norm_(params, 3.0, foreach=True)
    # skip the update (zero grads) if anything went non-finite, without a host sync
    fin = torch.isfinite(gn).float()
    for p_ in params:
        p_.grad.nan_to_num_(0.0, 0.0, 0.0).mul_(fin)
    opt.step()
    return lxy.detach(), le.detach(), gn.detach()


if args.graph:
    lr_t = torch.tensor(sched(step), device=dev)
    opt = torch.optim.Adam(params, lr=lr_t, capturable=True)
    if args.resume:
        opt.load_state_dict(ck['opt'])
        for g in opt.param_groups:
            g['lr'] = lr_t
    static = batch(rng.choice(train_ids, args.batch), rng)
    lr_t.fill_(0.0)                      # warm-up replays must not move the weights
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            train_step(*static)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    t_cap = time.time()
    with torch.cuda.graph(graph):
        out_static = train_step(*static)
    torch.cuda.synchronize()
    print(f'captured training step in {time.time() - t_cap:.1f}s', flush=True)

t0 = time.time(); run = torch.zeros(3, device=dev)
while step < args.steps:
    ids = rng.choice(train_ids, args.batch)
    X, Mk, C, CM = batch(ids, rng)
    if args.graph:
        for dst, src_ in zip(static, (X, Mk, C, CM)):
            dst.copy_(src_)
        lr_t.fill_(sched(step))
        graph.replay()
        lxy, le, gn = out_static
    else:
        for g in opt.param_groups:
            g['lr'] = sched(step)
        opt.zero_grad(set_to_none=True)
        lxy, le, gn = train_step(X, Mk, C, CM)
    step += 1
    run[0] += lxy; run[1] += le; run[2] += 1
    if step % args.log_every == 0:
        r = run.tolist(); el = time.time() - t0
        print(f'step {step} xy {r[0]/r[2]:.4f} pen {r[1]/r[2]:.4f} gn {gn.item():.2f} lr {sched(step):.2e} T {X.shape[1]} {el/args.log_every:.2f}s/step', flush=True)
        run.zero_(); t0 = time.time()
    if step % args.eval_every == 0 or step == args.steps:
        vx, ve = evaluate()
        print(f'EVAL step {step} val_xy {vx:.4f} val_pen {ve:.4f}', flush=True)
        ck = dict(model=model.state_dict(), opt=opt.state_dict(), step=step, meta=meta)
        torch.save(ck, f'{args.out}/latest.pt')
        if step % (args.eval_every * 5) == 0:
            torch.save(dict(model=model.state_dict(), step=step, meta=meta), f'{args.out}/step{step}.pt')
        t0 = time.time()
