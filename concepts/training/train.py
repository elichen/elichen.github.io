# Train TopK SAEs on robust ResNet-50 layers at once, streaming activations from augmented images.
# Evaluation always uses the same 5,000 held-out validation images.
#   python train.py run1 --epochs 12                                   # val only (45k train images)
#   python train.py run2 --data train --layers layer4.1:16384,layer4.2:16384 --epochs 10 --lr 2.8e-4
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn.functional as F

from common import LAYERS, ROOT, TopKSAE, augment, batches, center, load_model, tokens

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--data', default='val', help="'val' (minus the held-out 5,000) or 'train' (train256.npy)")
ap.add_argument('--layers', default='layer3.5:8192,layer4.2:8192', help='layer:latents,...')
ap.add_argument('--epochs', type=int, default=12)
ap.add_argument('--k', type=int, default=32)
ap.add_argument('--lr', type=float, default=4e-4)
ap.add_argument('--batch', type=int, default=4096, help='SAE batch in tokens (approximate)')
ap.add_argument('--dead_tokens', type=float, default=10e6, help='a latent is dead after this many tokens without firing')
args = ap.parse_args()
SIZES = {l: int(m) for l, m in (s.split(':') for s in args.layers.split(','))}
CFG = dict(vars(args), layers=SIZES, img_batch=256, warmup=200, k_aux=512, aux_coef=1 / 32, held=5000, seed=0)
OUT = f'{ROOT}/runs/{args.run}'
os.makedirs(OUT, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(CFG['seed'])
rng = np.random.default_rng(CFG['seed'])

val_imgs = np.load(f'{ROOT}/imgs256.npy', mmap_mode='r')
val_labels = np.load(f'{ROOT}/labels.npy')
perm = np.random.default_rng(0).permutation(len(val_imgs))
held = np.sort(perm[:CFG['held']])
np.save(f'{OUT}/held.npy', held)
if args.data == 'val':
    imgs, labels, train_idx = val_imgs, val_labels, perm[CFG['held']:]
else:
    imgs = np.load(f'{ROOT}/{args.data}256.npy', mmap_mode='r')
    labels = np.load(f'{ROOT}/{args.data}_labels.npy')
    train_idx = np.arange(len(imgs))
model = load_model()
saes = {n: TopKSAE(LAYERS[n], m, args.k).cuda() for n, m in SIZES.items()}
opts = {n: torch.optim.Adam(s.parameters(), lr=args.lr) for n, s in saes.items()}
since = {n: torch.zeros(m, device='cuda') for n, m in SIZES.items()}  # tokens since each latent last fired
GRID_TOKENS = {n: 224 // 16 if n.startswith('layer3') else 224 // 32 for n in SIZES}
chunks = max(1, round(CFG['img_batch'] * min(GRID_TOKENS.values()) ** 2 / args.batch))
total_steps = args.epochs * (len(train_idx) // CFG['img_batch']) * chunks
print('train images', len(train_idx), 'chunks per image batch', chunks, 'total steps', total_steps, flush=True)


def acts(x, logits=False):
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        return model(x, taps=set(SIZES), logits=logits)


def fvu(x, r):
    return (x - r).pow(2).sum() / (x - x.mean(0)).pow(2).sum()


def train_step(n, x, step):
    sae, opt = saes[n], opts[n]
    lr = args.lr * min(1, (step + 1) / CFG['warmup'])
    lr *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total_steps))
    for g in opt.param_groups:
        g['lr'] = lr
    v, i, p = sae.encode(x)
    r = sae.decode(v, i)
    loss = main = fvu(x, r)
    dead = since[n] > args.dead_tokens
    if dead.any():
        e = (x - r).detach()
        va, ia = p.masked_fill(~dead, -1e9).topk(min(CFG['k_aux'], int(dead.sum())), dim=-1)
        loss = loss + CFG['aux_coef'] * fvu(e, sae.dense(F.relu(va), ia) @ sae.W_dec)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    W, g = sae.W_dec.data, sae.W_dec.grad
    g -= (g * W).sum(1, keepdim=True) * W
    opt.step()
    sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)
    since[n] += len(x)
    since[n][i[v > 0]] = 0
    return main.item(), dead.float().mean().item()


@torch.no_grad()
def evaluate():
    err = {n: torch.zeros(2, device='cuda') for n in SIZES}
    fired = {n: torch.zeros(m, dtype=torch.bool, device='cuda') for n, m in SIZES.items()}
    correct = {k: 0 for k in ['orig', *SIZES]}
    agree = {n: 0 for n in SIZES}
    for u8, y, _ in batches(val_imgs, val_labels, held, 250):
        y = y.cuda()
        logits, a = acts(center(u8), logits=True)
        pred = logits.argmax(1)
        correct['orig'] += (pred == y).sum().item()
        for n, t in a.items():
            B, C, H, W = t.shape
            sae = saes[n]
            x = tokens(t).reshape(-1, C) / sae.scale
            v, i, _ = sae.encode(x)
            r = sae.decode(v, i)
            err[n] += torch.stack([(x - r).pow(2).sum(), (x - x.mean(0)).pow(2).sum()])
            fired[n][i[v > 0]] = True
            h = (r * sae.scale).reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', torch.bfloat16):
                pn = model(start=n, h=h)[0].argmax(1)
            correct[n] += (pn == y).sum().item()
            agree[n] += (pn == pred).sum().item()
    N = len(held)
    return {'acc': correct['orig'] / N,
            **{n: {'fvu': (err[n][0] / err[n][1]).item(), 'alive': fired[n].float().mean().item(),
                   'acc': correct[n] / N, 'agree': agree[n] / N} for n in SIZES}}


def save():
    torch.save({'cfg': CFG, 'state': {n: s.state_dict() for n, s in saes.items()}}, f'{OUT}/sae.pt')


# Scale each layer so E|x|^2 = d and start b_dec at the mean, from a few batches.
with torch.no_grad():
    sample = {n: [] for n in SIZES}
    for b, (u8, _, _) in enumerate(batches(imgs, labels, train_idx, 256, rng)):
        for n, t in acts(augment(u8))[1].items():
            sample[n].append(tokens(t).reshape(-1, t.shape[1])[::7])
        if b == 3:
            break
    for n, s in sample.items():
        s = torch.cat(s)
        saes[n].scale.fill_(s.pow(2).sum(1).mean().div(s.shape[1]).sqrt().item())
        saes[n].b_dec.data = (s / saes[n].scale).mean(0)
        print(n, 'scale', saes[n].scale.item(), 'tokens sampled', len(s))

log = open(f'{OUT}/log.jsonl', 'a')
m = evaluate()
print('epoch 0', json.dumps(m))
log.write(json.dumps({'epoch': 0, **m}) + '\n')
step, t0 = 0, time.time()
for epoch in range(1, args.epochs + 1):
    run = {n: [] for n in SIZES}
    for b, (u8, _, _) in enumerate(batches(imgs, labels, train_idx, CFG['img_batch'], rng)):
        _, a = acts(augment(u8))
        xs = {}
        for n, t in a.items():
            x = tokens(t).reshape(-1, t.shape[1]) / saes[n].scale
            xs[n] = x[torch.randperm(len(x), device='cuda')].chunk(chunks)
        for c in range(chunks):
            for n in SIZES:
                run[n].append(train_step(n, xs[n][c], step))
            step += 1
        if b % 100 == 0:
            msg = ' '.join(f'{n} fvu {np.mean([r[0] for r in run[n][-300:]]):.4f} dead {run[n][-1][1]:.3f}' for n in SIZES)
            print(f'epoch {epoch} batch {b} step {step} {time.time() - t0:.0f}s {msg}', flush=True)
    m = evaluate()
    print(f'epoch {epoch}', json.dumps(m), flush=True)
    log.write(json.dumps({'epoch': epoch, 'step': step, 'time': time.time() - t0, **m}) + '\n')
    log.flush()
    save()
print('done')
