# Painting with a concept, compared across ways of parameterizing the change. Inside a soft box on a
# validation image, raise (or lower) one SAE latent's pre-activation, averaged over the positions under the box.
#   pixel     normalized gradient steps on the raw pixel change (paint.py; leaves rainbow fringes)
#   fourier   the change is Lucid's image parameterization without the sigmoid: 1/f-scaled Fourier
#             coefficients, decorrelated colors; Adam on the coefficients
#   fourier+  the same, through random jitter/scale like Lucid's renders, so the change can't hide in exact pixels
#   python paint2.py run2
import json, math, random, sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from common import ROOT, load_model, load_saes, tokens

RUN = sys.argv[1]
OUT = f'{ROOT}/runs/{RUN}'
LAYER = 'layer4.2'
SNAPS = [8, 16, 32, 64]
T, SIZE = 160, 224
imgs = np.load(f'{ROOT}/imgs256.npy', mmap_mode='r')
labels = np.load(f'{ROOT}/labels.npy')
model = load_model()
sae = load_saes(f'{OUT}/sae.pt')[LAYER]

COLOR = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])
COLOR = (COLOR / COLOR.norm(dim=0).max()).cuda()
fy = torch.fft.fftfreq(SIZE)[:, None]
fx = torch.fft.rfftfreq(SIZE)[None, :]
SPECTRUM_SCALE = (1.0 / (fx ** 2 + fy ** 2).sqrt().clamp(min=1.0 / SIZE)).cuda()


def fourier(spec):
    x = torch.fft.irfft2(torch.view_as_complex(spec) * SPECTRUM_SCALE, s=(SIZE, SIZE), norm='ortho')
    return torch.einsum('bchw,dc->bdhw', x, COLOR)


def jitter(x):
    """Random shift of up to 8 px and scale 0.95-1.05, applied the same to the image and the weights."""
    s = random.uniform(0.95, 1.05)
    theta = torch.tensor([[[s, 0, random.uniform(-8, 8) / 112], [0, s, random.uniform(-8, 8) / 112]]], device='cuda')
    return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode='reflection',
                         align_corners=False)


def latent(x, j):
    with torch.autocast('cuda', torch.bfloat16):
        _, a = model(x.contiguous(memory_format=torch.channels_last), taps={LAYER}, logits=False)
    t = a[LAYER]
    p = (tokens(t)[0] / sae.scale - sae.b_dec) @ sae.W_enc[:, j] + sae.b_enc[j]
    return p.reshape(t.shape[2], t.shape[3])


def heat(z):
    z = z / max(z.max(), 1e-6)
    h = Image.fromarray((np.clip(z, 0, 1) * 255).astype(np.uint8)).resize((T, T), Image.BILINEAR)
    return Image.merge('RGB', (h, h.point(lambda v: v // 3), Image.new('L', (T, T), 0)))


def to_im(x):
    return Image.fromarray((x[0].permute(1, 2, 0).detach().cpu().numpy() * 255).round().astype(np.uint8)).resize((T, T))


def paint(x0, j, mask, sign, variant):
    w = F.adaptive_avg_pool2d(mask, (7, 7))[0, 0]
    frames, norms = [], []
    if variant == 'pixel':
        delta = torch.zeros_like(x0, requires_grad=True)
    else:
        spec = torch.zeros(1, 3, SIZE, SIZE // 2 + 1, 2, device='cuda', requires_grad=True)
        opt = torch.optim.Adam([spec], lr=0.002)
    for it in range(1, SNAPS[-1] + 1):
        d = delta if variant == 'pixel' else fourier(spec)
        x = (x0 + d * mask).clamp(0, 1)
        if variant == 'fourier+':
            xm = jitter(torch.cat([x, mask], 1))
            xt, wt = xm[:, :3], F.adaptive_avg_pool2d(xm[:, 3:], (7, 7))[0, 0]
        else:
            xt, wt = x, w
        obj = sign * (latent(xt, j) * wt).sum() / wt.sum()
        if variant == 'pixel':
            g, = torch.autograd.grad(obj, delta)
            with torch.no_grad():
                g = g * mask
                delta += 0.5 * g / g.norm().clamp(min=1e-12)
        else:
            opt.zero_grad(set_to_none=True)
            (-obj).backward()
            opt.step()
        if it in SNAPS:
            frames.append(x.detach())
            norms.append(((x.detach() - x0) * 255).norm().item() / 255)
    return frames, norms


# Cases: (image class, concept, box, sign). Images are the first validation image of each class.
first = {int(c): int(np.flatnonzero(labels == c)[0]) for c in [207, 978, 532, 954, 953, 340]}
CASES = [
    (207, 4150, [40, 40, 184, 184], 1),    # golden retriever + pineapple
    (978, 12399, [60, 30, 200, 190], 1),   # seashore + accordion pleats
    (532, 5108, [0, 0, 112, 224], 1),      # dining table + clouds (top half)
    (954, 10096, [40, 40, 184, 184], 1),   # banana + gecko skin
    (953, 4150, [0, 0, 224, 224], -1),     # pineapple, erase pineapple
    (340, 11783, [30, 30, 194, 194], 1),   # zebra + fluffy fur
]
rows = []
for cls, j, box, sign in CASES:
    x0 = torch.from_numpy(np.asarray(imgs[first[cls]])[16:240, 16:240].copy()).cuda().permute(2, 0, 1)[None].float() / 255
    mask = torch.zeros(1, 1, SIZE, SIZE, device='cuda')
    mask[..., box[0]:box[2], box[1]:box[3]] = 1
    mask = F.avg_pool2d(F.pad(mask, (4,) * 4, mode='replicate'), 9, 1)
    with torch.no_grad():
        before = F.relu(latent(x0, j)).cpu().numpy()
    for variant in ['pixel', 'fourier', 'fourier+']:
        random.seed(0)
        frames, norms = paint(x0, j, mask, sign, variant)
        with torch.no_grad():
            after = F.relu(latent(frames[-1], j)).cpu().numpy()
        row = Image.new('RGB', ((len(SNAPS) + 3) * (T + 4), T + 16), 'white')
        for q, p in enumerate([x0, *frames]):
            row.paste(to_im(p), (q * (T + 4), 16))
        row.paste(heat(before), ((len(SNAPS) + 1) * (T + 4), 16))
        row.paste(heat(after), ((len(SNAPS) + 2) * (T + 4), 16))
        d = ImageDraw.Draw(row)
        d.rectangle([box[1] * T / 224, 16 + box[0] * T / 224, box[3] * T / 224, 16 + box[2] * T / 224], outline='yellow')
        d.text((2, 2), f'img {first[cls]} latent {j} sign {sign} {variant}: steps {SNAPS} L2 '
                       f'{[round(n, 1) for n in norms]}; max before {before.max():.1f} after {after.max():.1f}', fill='black')
        rows.append(row)
        print(cls, j, variant, [round(n, 1) for n in norms], before.max(), after.max(), flush=True)
sheet = Image.new('RGB', (rows[0].width, len(rows) * (rows[0].height + 6)), 'white')
for q, r in enumerate(rows):
    sheet.paste(r, (0, q * (r.height + 6)))
sheet.save(f'{OUT}/paint2.jpg', quality=88)
