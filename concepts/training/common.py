# Shared pieces for the robust ResNet-50 SAE spike: the model, a TopK SAE, and image batches.
import queue, threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = '/home/aif_eng/play/sae'
LAYERS = {'layer3.5': 1024, 'layer4.1': 2048, 'layer4.2': 2048}  # channels of the layers SAEs are trained on
GRID = {'layer3.5': 14, 'layer4.1': 7, 'layer4.2': 7}  # positions per side at 224 px


class Bottleneck(nn.Module):
    def __init__(self, cin, width, stride, down):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, width * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.downsample = nn.Sequential(nn.Conv2d(cin, width * 4, 1, stride, bias=False),
                                        nn.BatchNorm2d(width * 4)) if down else None

    def pre_relu(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return y + (x if self.downsample is None else self.downsample(x))


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        cin = 64
        for L, (n, w, s) in enumerate([(3, 64, 1), (4, 128, 2), (6, 256, 2), (3, 512, 2)], 1):
            blocks = []
            for b in range(n):
                blocks.append(Bottleneck(cin, w, s if b == 0 else 1, b == 0))
                cin = w * 4
            setattr(self, f'layer{L}', nn.Sequential(*blocks))
        self.fc = nn.Linear(2048, 1000)
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))

    def named_blocks(self):
        for L in range(1, 5):
            for b, blk in enumerate(getattr(self, f'layer{L}')):
                yield f'layer{L}.{b}', blk

    def forward(self, x=None, taps=(), logits=True, start=None, h=None):
        """x: [B, 3, H, W] in [0, 1]. Returns (logits or None, {tap: block output}); a tap 'layerL.B/pre_relu'
        is the residual sum before the block's ReLU.
        With start='layerL.B', h stands in for that block's output and the network runs on from there."""
        out = {}
        if start is None:
            h = (x - self.mean) / self.std
            h = F.max_pool2d(F.relu(self.bn1(self.conv1(h))), 3, 2, 1)
        live = start is None
        for name, blk in self.named_blocks():
            if live:
                pre = blk.pre_relu(h)
                h = F.relu(pre)
                if name in taps:
                    out[name] = h
                if f'{name}/pre_relu' in taps:
                    out[f'{name}/pre_relu'] = pre
                if not logits and taps and len(out) == len(taps):
                    return None, out
            live = live or name == start
        return self.fc(h.mean((2, 3))), out


def load_model(device='cuda'):
    ck = torch.load(f'{ROOT}/resnet50_l2_eps3.ckpt', map_location='cpu', weights_only=False)['model']
    sd = {k[len('module.model.'):]: v for k, v in ck.items() if k.startswith('module.model.')}
    m = ResNet50()
    res = m.load_state_dict(sd, strict=False)
    assert set(res.missing_keys) == {'mean', 'std'} and not res.unexpected_keys, res
    m.mean.copy_(ck['module.normalizer.new_mean'].view(1, 3, 1, 1))
    m.std.copy_(ck['module.normalizer.new_std'].view(1, 3, 1, 1))
    return m.eval().to(device).to(memory_format=torch.channels_last).requires_grad_(False)


class TopKSAE(nn.Module):
    """Activations are divided by `scale` (so E|x|^2 = d), then z = ReLU(TopK(W_enc (x - b_dec) + b_enc)),
    x_hat = W_dec z + b_dec. Decoder rows are kept at unit norm."""

    def __init__(self, d, m, k):
        super().__init__()
        W = F.normalize(torch.randn(m, d), dim=1)
        self.W_dec = nn.Parameter(W.clone())
        self.W_enc = nn.Parameter(W.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(m))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer('scale', torch.tensor(1.0))
        self.k = k

    def pre(self, xn):
        return (xn - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, xn):
        p = self.pre(xn)
        v, i = p.topk(self.k, dim=-1)
        return F.relu(v), i, p

    def dense(self, v, i):
        return torch.zeros(*v.shape[:-1], self.W_dec.shape[0], device=v.device, dtype=v.dtype).scatter_(-1, i, v)

    def decode(self, v, i):
        return self.dense(v, i) @ self.W_dec + self.b_dec


def load_saes(path, device='cuda'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    saes = {}
    for n, sd in ck['state'].items():
        m, d = sd['W_dec'].shape
        sae = TopKSAE(d, m, ck['cfg']['k'])
        sae.load_state_dict(sd)
        saes[n] = sae.to(device).eval().requires_grad_(False)
    return saes


def batches(imgs, labels, idx, bs, rng=None):
    """Yield (uint8 [B, 256, 256, 3], labels) from the memmap, read on a background thread.
    With rng, the order is shuffled and the last partial batch dropped."""
    q = queue.Queue(maxsize=4)

    def work():
        order = rng.permutation(idx) if rng is not None else idx
        stop = len(order) - (bs - 1 if rng is not None else 0)
        for b in range(0, max(stop, 0), bs):
            j = np.sort(order[b:b + bs])
            q.put((torch.from_numpy(imgs[j]).pin_memory(), torch.from_numpy(labels[j]), j))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def center(u8):
    x = u8.cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    return x[:, :, 16:240, 16:240].contiguous(memory_format=torch.channels_last)


def augment(u8):
    """Random crop of 70-100% of the 256 side, resized to 224, random horizontal flip."""
    x = u8.cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    B = x.shape[0]
    s = torch.empty(B, device='cuda').uniform_(0.7, 1.0)
    theta = torch.zeros(B, 2, 3, device='cuda')
    theta[:, 0, 0] = s * torch.where(torch.rand(B, device='cuda') < 0.5, -1.0, 1.0)
    theta[:, 1, 1] = s
    theta[:, 0, 2] = (torch.rand(B, device='cuda') * 2 - 1) * (1 - s)
    theta[:, 1, 2] = (torch.rand(B, device='cuda') * 2 - 1) * (1 - s)
    grid = F.affine_grid(theta, (B, 3, 224, 224), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', align_corners=False).contiguous(memory_format=torch.channels_last)


def tokens(a):
    """[B, C, H, W] -> [B, H*W, C] float32."""
    return a.float().permute(0, 2, 3, 1).reshape(a.shape[0], -1, a.shape[1])
