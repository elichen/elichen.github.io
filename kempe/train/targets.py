"""Test drawings: closed shapes people are likely to draw, as polylines."""

import numpy as np


def _smooth_poly(pts, n=400, r=2):
    """Densify a closed polygon and round its corners slightly."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], n, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    out = np.stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])], 1)
    for _ in range(r):
        out = (np.roll(out, 1, 0) + 2 * out + np.roll(out, -1, 0)) / 4
    return out


def out_and_back(pts):
    """An open stroke becomes the closed path that runs along it and back."""
    pts = np.asarray(pts, float)
    return np.vstack([pts, pts[-2:0:-1]])


def targets():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    T = {}
    T['circle'] = np.stack([np.cos(t), np.sin(t)], 1)
    T['heart'] = np.stack([16 * np.sin(t) ** 3,
                           13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)], 1)
    star = [(np.cos(np.pi / 2 + k * np.pi / 5) * (1 if k % 2 == 0 else 0.42),
             np.sin(np.pi / 2 + k * np.pi / 5) * (1 if k % 2 == 0 else 0.42)) for k in range(10)]
    T['star'] = _smooth_poly(star)
    T['square'] = _smooth_poly([(-1, -1), (1, -1), (1, 1), (-1, 1)], r=6)
    T['triangle'] = _smooth_poly([(0, 1), (-0.9, -0.6), (0.9, -0.6)], r=6)
    T['figure8'] = np.stack([np.sin(t), np.sin(t) * np.cos(t)], 1)
    moon_o = np.stack([np.cos(t), np.sin(t)], 1)
    T['crescent'] = _smooth_poly(np.vstack([
        np.stack([np.cos(np.linspace(-2.2, 2.2, 60)), np.sin(np.linspace(-2.2, 2.2, 60))], 1),
        np.stack([0.55 + 0.8 * np.cos(np.linspace(2.0, -2.0, 60)),
                  0.95 * np.sin(np.linspace(2.0, -2.0, 60))], 1)]), r=3)
    T['fish'] = np.stack([np.cos(t) - np.sin(t) ** 2 / np.sqrt(2), np.cos(t) * np.sin(t)], 1)
    T['trefoil'] = np.stack([(1 + 0.3 * np.cos(3 * t)) * np.cos(t), (1 + 0.3 * np.cos(3 * t)) * np.sin(t)], 1)
    T['cloud'] = np.stack([(1 + 0.12 * np.abs(np.sin(2.5 * t))) * 1.4 * np.cos(t),
                           (1 + 0.12 * np.abs(np.sin(2.5 * t))) * 0.8 * np.sin(t)], 1)
    T['D'] = _smooth_poly(np.vstack([[[-0.6, -1], [-0.6, 1]],
                                     np.stack([-0.6 + 1.2 * np.sin(np.linspace(0, np.pi, 40)),
                                               np.cos(np.linspace(0, np.pi, 40))], 1)]), r=3)
    T['bean'] = np.stack([np.cos(t) * (1.2 + 0.25 * np.sin(t)), np.sin(t) * (0.7 - 0.35 * np.cos(2 * t) * 0)
                          - 0.25 * np.cos(t) ** 2], 1)
    x = np.linspace(-1, 1, 200)
    T['line'] = out_and_back(np.stack([x, 0 * x], 1))
    T['wave'] = out_and_back(np.stack([x * 1.5, 0.5 * np.sin(np.pi * x)], 1))
    T['check'] = out_and_back(_open([(-1, 0.1), (-0.4, -0.6), (1, 0.9)]))
    return T


def _open(pts, n=200):
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1)


def jitter(pts, rng, amp=0.02):
    """Hand-drawn wobble: low-frequency noise along the path."""
    n = len(pts)
    scale = np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean())
    k = np.fft.rfftfreq(n) * n
    noise = np.zeros((n, 2))
    for d in range(2):
        spec = (rng.normal(size=len(k)) + 1j * rng.normal(size=len(k))) / (1 + (k / 4) ** 2)
        spec[0] = 0
        noise[:, d] = np.fft.irfft(spec, n)
    noise *= amp * scale / np.sqrt((noise ** 2).sum(1).mean())
    return pts + noise
