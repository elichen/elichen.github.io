"""Curve encoder: a closed curve -> a unit vector whose dot products rank
mechanism curves the way the exact similarity-alignment error does.

Invariances: translation and scale by normalisation; rotation by using only
products z_a * conj(z_b); start point by circular convolutions and global
pooling; traversal direction and mirror image by summing the network over the
four variants of the curve. The web app runs the same network in JavaScript,
so it sticks to plain ops: circular conv1d, per-position LayerNorm, tanh-GELU,
mean/max pooling, linear layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N = 64
LAGS = (1, 2, 4, 8, 16, 32)
TLAGS = (1, 2, 4)
C_IN = 1 + 2 * len(LAGS) + 2 * len(TLAGS) + 1


def variants(z):
    """(B, n) complex -> list of 4 (B, n): as is, reversed, mirrored, both."""
    rev = torch.roll(torch.flip(z, [-1]), 1, -1)
    return [z, rev, z.conj(), rev.conj()]


def normalise(z):
    z = z - z.mean(-1, keepdim=True)
    return z / (z.abs().pow(2).mean(-1, keepdim=True).sqrt() + 1e-9)


def features(z):
    """Rotation-invariant per-sample features, (B, C_IN, n)."""
    ch = [z.abs()]
    for k in LAGS:
        p = torch.roll(z, -k, -1) * z.conj()
        ch += [p.real, p.imag]
    t = torch.roll(z, -1, -1) - z
    tn = t.abs().pow(2).mean(-1, keepdim=True) + 1e-9
    for k in TLAGS:
        p = torch.roll(t, -k, -1) * t.conj() / tn
        ch += [p.real, p.imag]
    perim = t.abs().sum(-1, keepdim=True) / 10.0
    ch.append(perim.expand_as(z.real))
    return torch.stack(ch, 1)


class CircConv(nn.Module):
    """Circular conv1d, kernel 3, dilation d, run channels-last as one matmul.
    Weight layout matches nn.Conv1d (out, in, 3): tap k reads x[i + (k-1) d]."""

    def __init__(self, cin, cout, dil=1):
        super().__init__()
        self.dil = dil
        self.weight = nn.Parameter(torch.empty(cout, cin, 3))
        self.bias = nn.Parameter(torch.zeros(cout))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        bound = 1 / (3 * cin) ** 0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):                       # x (N, L, C)
        d = self.dil
        xc = torch.cat([torch.roll(x, d, 1), x, torch.roll(x, -d, 1)], -1)
        w = self.weight.permute(0, 2, 1).reshape(self.weight.shape[0], -1)
        return F.linear(xc, w, self.bias)


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.c1 = CircConv(c, c, dil)
        self.c2 = CircConv(c, c, dil)

    def forward(self, x):
        return x + self.c2(F.gelu(self.c1(self.norm(x)), approximate='tanh'))


class Encoder(nn.Module):
    def __init__(self, width=64, dim=32, dils=(1, 2, 4, 8, 16)):
        super().__init__()
        self.inp = CircConv(C_IN, width, 1)
        self.blocks = nn.ModuleList([Block(width, d) for d in dils])
        self.norm = nn.LayerNorm(2 * width)
        self.fc1 = nn.Linear(2 * width, 2 * width)
        self.fc2 = nn.Linear(2 * width, dim)
        for b in self.blocks:
            nn.init.zeros_(b.c2.weight)
            nn.init.zeros_(b.c2.bias)

    def branch(self, z):
        x = self.inp(features(z).transpose(1, 2))      # (N, L, C)
        for b in self.blocks:
            x = b(x)
        h = torch.cat([x.mean(1), x.amax(1)], -1)
        return self.fc2(F.gelu(self.fc1(self.norm(h)), approximate='tanh'))

    def forward(self, z):
        """z (B, n) complex, normalised -> (B, dim) unit vectors."""
        B = z.shape[0]
        zz = torch.cat(variants(z), 0)
        e = self.branch(zz).reshape(4, B, -1).sum(0)
        return F.normalize(e, dim=-1)


def exact_err(q, C):
    """Relative RMS error after the best similarity alignment over start
    shift, direction and mirror. q (B, n), C (B, K, n) complex, normalised.
    Returns (B, K) = sqrt(1 - max |<c_s, q>|^2 / n^2)."""
    n = q.shape[-1]
    Fq = torch.fft.fft(q)[:, None, :]
    best = None
    for Cv in variants(C.reshape(-1, n)):
        Fc = torch.fft.fft(Cv).reshape(C.shape)
        m = torch.fft.ifft(Fq * Fc.conj()).abs().amax(-1) / n
        best = m if best is None else torch.maximum(best, m)
    return (1 - best.clamp(max=1) ** 2).clamp_min(0).sqrt()


def fourier_descriptor(z, K=10):
    """Non-learned baseline: symmetrised Fourier magnitudes, unit-normalised."""
    c = torch.fft.fft(z) / z.shape[-1]
    pos = c[:, 1:K + 1].abs()
    neg = torch.flip(c[:, -K:], [-1]).abs()
    d = torch.cat([pos + neg, (pos - neg).abs()], -1)
    return F.normalize(d, dim=-1)
