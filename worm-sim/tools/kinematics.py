"""Gait of a run from its centerlines (loop_run with CENTER=<file>): undulation frequency,
body wavelength, curvature amplitude and speed, in the units papers use (Hz, body
lengths L, 1/L, L/s), so they can be set against measured C. elegans gaits.

  python tools/kinematics.py <center.bin> [skip seconds]

Curvature is the signed turning of the centerline in its bending plane (the plane of
least spread, re-fit every frame), per unit arc length, times the body length. The
centerline is resampled at even arc length and smoothed along the body first, so the
slice binning does not show up as noise. Frequency is the peak of the mean power
spectrum over the mid-body; wavelength is 2π over the slope of the wave's phase along
the body (Hilbert transform in time); amplitude is √2 × rms curvature."""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

FPS = float(os.environ.get("FPS", 30))  # control steps per body second (240 / BODY_STEPS)
N = 40  # slices
M = 50  # even arc-length samples


def load(path):
    a = np.fromfile(path, dtype="<f8")
    return a.reshape(-1, N, 3)


def resample(c):
    """centerline (N, 3) -> M points at even arc length, and the length"""
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    u = np.linspace(0, s[-1], M)
    return np.stack([np.interp(u, s, c[:, k]) for k in range(3)], 1), s[-1]


def curvature(frames):
    """(T, M) curvature in 1/L, head first; plus body length per frame"""
    out, length, prev_n = [], [], None
    for c in frames:
        p, L = resample(c)
        p = gaussian_filter1d(p, 1.5, axis=0, mode="nearest")
        q = p - p.mean(0)
        n = np.linalg.svd(q, full_matrices=False)[2][2]  # normal of the bending plane
        if prev_n is not None and n @ prev_n < 0:
            n = -n
        prev_n = n
        t = np.diff(p, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        ang = np.arctan2(np.cross(t[:-1], t[1:]) @ n, np.sum(t[:-1] * t[1:], 1))
        k = ang / (L / (M - 1)) * L  # per unit length, times L
        out.append(np.concatenate([[k[0]], k, [k[-1]]])[:M])
        length.append(L)
    return np.array(out), np.array(length)


def main():
    path = sys.argv[1]
    skip = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    c = load(path)[int(skip * FPS):]
    k, L = curvature(c)
    mid = slice(int(0.2 * M), int(0.8 * M))
    km = k[:, mid] - k[:, mid].mean(0)
    f = np.fft.rfftfreq(len(km), 1 / FPS)
    pw = (np.abs(np.fft.rfft(km * np.hanning(len(km))[:, None], axis=0)) ** 2).mean(1)
    band = (f > 0.1) & (f < 5)
    fpk = f[band][np.argmax(pw[band])]
    ph = np.unwrap(np.angle(hilbert(km, axis=0)), axis=1)  # phase along the body
    slope = np.median(np.diff(ph, axis=1)[int(0.1 * len(ph)):-int(0.1 * len(ph))]) * (M - 1)  # rad per L
    lam = 2 * np.pi / abs(slope) if slope else np.inf
    amp = np.sqrt(2) * k[:, mid].std(0).mean()
    com = c.mean(1)
    path_len = np.linalg.norm(np.diff(com, axis=0), axis=1).sum()
    net = np.linalg.norm(com[-1] - com[0])
    T = len(c) / FPS
    Lm = L.mean()
    head = c[:, 0] - c[:, 1]
    print(f"{path}: {T:.0f} s after {skip:.0f} s, body length {Lm:.3f}")
    print(f"  frequency {fpk:.2f} Hz   wavelength {lam:.2f} L   curvature amplitude {amp:.1f} /L")
    print(f"  wave travels {'head to tail' if slope < 0 else 'tail to head'}   speed {path_len / T / Lm:.2f} L/s along the path, {net / T / Lm:.2f} L/s net")
    print(f"  wave speed {fpk * lam:.2f} L/s, slip ratio (speed / wave speed) {path_len / T / Lm / (fpk * lam):.2f}")


if __name__ == "__main__":
    main()
