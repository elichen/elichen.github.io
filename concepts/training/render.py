# Feature visualization of randomly chosen SAE latents and neurons, in several variants for comparison.
# Every variant renders the same targets (renders/targets.json) and saves renders/{layer}_{kind}_{j}{suffix}.png.
#   python render.py run1 [latents_per_layer] [neurons_per_layer] [variant]
#   python render.py run2 all:layer4.2:1e-4 0 dircos   every latent of a layer with validation frequency above the
#                                                      threshold, into renders_all/ (skips ones already there)
# Variants:
#   center   Lucid: Fourier parameterization, color decorrelation, jitter/scale/rotate; activation at the
#            center position (suffix '')
#   mean     the same, mean activation over the whole image (_mean)
#   dircos   the same over the whole image, but Lucid's direction objective with cossim_pow=3 along the
#            concept's decoder row (a one-hot channel for neurons): mean of dot * max(cos, 0.1)^3 (_dircos)
#   diverse  'mean' for 4 images at once, minus Lucid's diversity penalty on their Gram matrices (_diverse,
#            a 2x2 grid)
#   maco     MACO (Fel et al. 2023): only the Fourier phase is optimized, the magnitude is fixed to the mean
#            spectrum of validation images; random crops and noise; whole-image objective (_maco). _macoalpha
#            fades the render toward white where the summed |gradient| (MACO's transparency) is low.
# SAE latents maximize the encoder pre-activation (except dircos); neurons use the pre-ReLU block output,
# which has a gradient even where the neuron is off.
import json, math, os, random, sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import LAYERS, ROOT, load_model, load_saes

RUN = sys.argv[1] if len(sys.argv) > 1 else 'run1'
N_LAT = sys.argv[2] if len(sys.argv) > 2 else '40'
ALL = N_LAT.startswith('all:')
N_NEU = int(sys.argv[3]) if len(sys.argv) > 3 else 8
VARIANT = sys.argv[4] if len(sys.argv) > 4 else 'center'
SUFFIX = {'center': '', 'mean': '_mean', 'dircos': '_dircos', 'diverse': '_diverse', 'maco': '_maco'}[VARIANT]
OUT = f'{ROOT}/runs/{RUN}'
SIZE = 224
STEPS = 256 if VARIANT == 'maco' else 128  # 128 as in the Lucid app; 256 is MACO's default
BATCH = {'maco': 8, 'diverse': 6}.get(VARIANT, 24)  # targets per optimization
COPIES = 4 if VARIANT == 'diverse' else 1
MACO_SIZE, MACO_CROPS = 320, 6
RDIR = f'{OUT}/renders_all' if ALL else f'{OUT}/renders'
os.makedirs(RDIR, exist_ok=True)
model = load_model()
saes = load_saes(f'{OUT}/sae.pt')
mine = np.load(f'{OUT}/mine.npz')

COLOR = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])
COLOR = (COLOR / COLOR.norm(dim=0).max()).cuda()
fy = torch.fft.fftfreq(SIZE)[:, None]
fx = torch.fft.rfftfreq(SIZE)[None, :]
SPECTRUM_SCALE = (1.0 / (fx ** 2 + fy ** 2).sqrt().clamp(min=1.0 / SIZE)).cuda()
SCALES = [1 + (i - 5) / 50 for i in range(11)]
ANGLES = list(range(-10, 11)) + [0] * 5


def image(spec):
    x = torch.fft.irfft2(torch.view_as_complex(spec) * SPECTRUM_SCALE, s=(SIZE, SIZE), norm='ortho') / 4
    x = torch.einsum('bchw,dc->bdhw', x, COLOR)
    return torch.sigmoid(x)


def jitter(x, d):
    dy, dx = random.randrange(d), random.randrange(d)
    return x[:, :, dy:dy + x.shape[2] - d, dx:dx + x.shape[3] - d]


def transform(x):
    x = jitter(F.pad(x, (12,) * 4, value=0.5), 8)
    s = random.choice(SCALES)
    x = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
    a = math.radians(random.choice(ANGLES))
    theta = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0]], device=x.device)
    grid = F.affine_grid(theta[None].expand(x.shape[0], 2, 3), x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode='border', align_corners=False)
    return jitter(x, 4)


def maco_magnitude(n=2000):
    """Mean Fourier magnitude of validation images in decorrelated color, at MACO_SIZE."""
    imgs = np.load(f'{ROOT}/imgs256.npy', mmap_mode='r')
    inv = torch.linalg.inv(COLOR)
    acc = 0
    for s in range(0, n, 100):
        x = torch.from_numpy(np.array(imgs[s:s + 100])).cuda().permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=MACO_SIZE, mode='bilinear', align_corners=False)
        acc = acc + torch.fft.rfft2(torch.einsum('bchw,dc->bdhw', x, inv), norm='ortho').abs().sum(0)
    return acc / n


def maco_image(phase, mag):
    p = (phase - phase.mean((1, 2, 3), keepdim=True)) / (phase.std((1, 2, 3), keepdim=True) + 1e-5)
    x = torch.fft.irfft2(torch.polar(mag.expand_as(p), p), s=(MACO_SIZE, MACO_SIZE), norm='ortho')
    x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-5)
    return torch.sigmoid(torch.einsum('bchw,dc->bdhw', x, COLOR))


def maco_crops(x):
    """MACO_CROPS random square crops per image (50-100% of the side), resized to SIZE, plus noise."""
    B = x.shape[0]
    x = x.repeat_interleave(MACO_CROPS, 0)
    n = x.shape[0]
    s = torch.empty(n, device='cuda').uniform_(0.5, 1.0)
    theta = torch.zeros(n, 2, 3, device='cuda')
    theta[:, 0, 0] = theta[:, 1, 1] = s
    theta[:, :, 2] = (torch.rand(n, 2, device='cuda') * 2 - 1) * (1 - s)[:, None]
    x = F.grid_sample(x, F.affine_grid(theta, (n, 3, SIZE, SIZE), align_corners=False), align_corners=False)
    return x + 0.04 * torch.randn_like(x)


def tap_of(l, kind):
    return l if kind == 'sae' else f'{l}/pre_relu'


def objectives(a, targets, per):
    """Objective of every image in the batch ([B]); image b belongs to target b // per."""
    out = []
    for b in range(next(iter(a.values())).shape[0]):
        l, kind, j = targets[b // per]
        t = a[tap_of(l, kind)][b].float()
        H, W = t.shape[1:]
        x = t.flatten(1).T  # [positions, channels]
        if kind == 'sae':
            sae = saes[l]
            x = x / sae.scale - sae.b_dec
            d = sae.W_dec[j]
        else:
            d = F.one_hot(torch.tensor(j, device='cuda'), x.shape[1]).float()
        if VARIANT == 'dircos':
            dot = x @ d
            v = dot * (dot / (x.norm(dim=1) * d.norm() + 1e-4)).clamp(min=0.1) ** 3
        else:
            v = x @ sae.W_enc[:, j] + sae.b_enc[j] if kind == 'sae' else x[:, j]
        out.append(v[(H // 2) * W + W // 2] if VARIANT == 'center' else v.mean())
    return torch.stack(out)


def diversity(t):
    """Lucid's diversity: summed cosine similarity between the Gram matrices of different images in each
    group of COPIES, divided by COPIES. <G_i, G_j> = |X_i X_j^T|_F^2 avoids the C x C Gram matrices."""
    X = t.float().flatten(2).transpose(1, 2)  # [B, positions, C]
    total = []
    for g in range(0, X.shape[0], COPIES):
        Xg = X[g:g + COPIES]
        K = torch.einsum('ipc,jqc->ijpq', Xg, Xg).pow(2).sum((2, 3))
        n = K.diagonal().sqrt()
        S = K / (n[:, None] * n[None, :] + 1e-10)
        total.append((S.sum() - S.diagonal().sum()) / COPIES)
    return torch.stack(total)


def run(x, targets, per):
    with torch.autocast('cuda', torch.bfloat16):
        _, a = model(x.contiguous(memory_format=torch.channels_last),
                     taps={tap_of(l, kind) for l, kind, _ in targets}, logits=False)
    return a, objectives(a, targets, per)


def to_u8(x):
    return (x.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).round().astype(np.uint8)


def render_lucid(targets):
    spec = (torch.randn(len(targets) * COPIES, 3, SIZE, SIZE // 2 + 1, 2, device='cuda') * 0.01).requires_grad_(True)
    opt = torch.optim.Adam([spec], lr=0.05)
    for step in range(STEPS):
        a, obj = run(transform(image(spec)), targets, COPIES)
        loss = -obj.sum()
        if VARIANT == 'diverse':
            # Lucid weights diversity by a constant; here it is in units of the group's mean activation.
            groups = obj.view(-1, COPIES)
            div = torch.zeros(len(targets), device='cuda')
            for l, kind in {(l, kind) for l, kind, _ in targets}:
                rows = [g for g, t in enumerate(targets) if (t[0], t[1]) == (l, kind)]
                idx = torch.tensor([g * COPIES + c for g in rows for c in range(COPIES)], device='cuda')
                div[rows] = diversity(a[tap_of(l, kind)][idx])
            loss = loss + (groups.mean(1).detach().abs() * div).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        ims = to_u8(image(spec))
    if COPIES == 1:
        return [Image.fromarray(im) for im in ims]
    out = []
    for g in range(len(targets)):
        grid = Image.new('RGB', (2 * SIZE, 2 * SIZE))
        for c in range(COPIES):
            grid.paste(Image.fromarray(ims[g * COPIES + c]), ((c % 2) * SIZE, (c // 2) * SIZE))
        out.append(grid)
    return out


def render_maco(targets, mag):
    phase = torch.randn(len(targets), 3, MACO_SIZE, MACO_SIZE // 2 + 1, device='cuda').requires_grad_(True)
    opt = torch.optim.NAdam([phase], lr=1.0)
    alpha = torch.zeros(len(targets), 1, MACO_SIZE, MACO_SIZE, device='cuda')
    for step in range(STEPS):
        img = maco_image(phase, mag)
        img.retain_grad()
        _, obj = run(maco_crops(img), targets, MACO_CROPS)
        opt.zero_grad(set_to_none=True)
        (-obj.sum()).backward()
        opt.step()
        alpha += img.grad.abs().sum(1, keepdim=True)
    with torch.no_grad():
        img = maco_image(phase, mag)
        a = F.avg_pool2d(alpha, 9, 1, 4)
        a = a / torch.quantile(a.flatten(1), 0.98, dim=1)[:, None, None, None]
        faded = img * a.clamp(max=1) + (1 - a.clamp(max=1))
    return [(Image.fromarray(i), Image.fromarray(f)) for i, f in zip(to_u8(img), to_u8(faded))]


rng = np.random.default_rng(1)
targets = []
if ALL:
    _, l, th = N_LAT.split(':')
    targets = [(l, 'sae', int(j)) for j in np.flatnonzero(mine[f'{l}/freq'] > float(th))
               if not os.path.exists(f'{RDIR}/{l}_sae_{j}{SUFFIX}.png')]
for l in saes if not ALL and not os.path.exists(f'{RDIR}/targets.json') else []:
    alive = np.flatnonzero(mine[f'{l}/freq'] > 0)
    targets += [(l, 'sae', int(j)) for j in sorted(rng.choice(alive, int(N_LAT), replace=False))]
    targets += [(l, 'neuron', int(j)) for j in sorted(rng.choice(LAYERS[l], N_NEU, replace=False))]
if targets and not ALL:
    json.dump(targets, open(f'{RDIR}/targets.json', 'w'))
if not ALL:
    targets = [tuple(t) for t in json.load(open(f'{RDIR}/targets.json'))]
mag = maco_magnitude() if VARIANT == 'maco' else None
for b in range(0, len(targets), BATCH):
    chunk = targets[b:b + BATCH]
    if VARIANT == 'maco':
        for (l, kind, j), (im, faded) in zip(chunk, render_maco(chunk, mag)):
            im.save(f'{RDIR}/{l}_{kind}_{j}_maco.png')
            faded.save(f'{RDIR}/{l}_{kind}_{j}_macoalpha.png')
    else:
        for (l, kind, j), im in zip(chunk, render_lucid(chunk)):
            im.save(f'{RDIR}/{l}_{kind}_{j}{SUFFIX}.png')
    print(VARIANT, 'rendered', b + len(chunk), '/', len(targets), flush=True)
