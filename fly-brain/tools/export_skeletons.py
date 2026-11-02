#!/usr/bin/env python3
"""Real neuron shapes for the page: BANC v888 SWC skeletons -> model/skeletons.bin (gzip).

    python tools/export_skeletons.py <picked.json: {cell index: swc file}> <swc dir> model/skeletons.bin [twig um] [tolerance um]

Skeletons: gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_banc_space_swc (public, CC BY 4.0).
Twigs shorter than `twig` are cut, then each unbranched run is simplified (Douglas-Peucker) to `tolerance`.

File: uint32 N, V; uint32 cell[N]; uint32 verts[N]; int16 xyz[3V] (1/20 um, BANC space as model/neurons.json pos);
uint16 back[V] (how many vertices back the parent is; 0 = root of its neuron). Each vertex but a root makes one segment.
"""
import gzip, json, os, sys
from multiprocessing import Pool
import numpy as np


def simplify(path, xyz, tol):
    """Douglas-Peucker on a list of node ids; keeps both ends."""
    keep, stack = {path[0], path[-1]}, [(0, len(path) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2: continue
        p, q = xyz[path[a]], xyz[path[b]]
        pts = xyz[path[a + 1:b]]
        d = q - p; L = np.dot(d, d)
        t = np.clip((pts - p) @ d / L, 0, 1)[:, None] if L > 0 else np.zeros((len(pts), 1))
        e = np.linalg.norm(pts - (p + t * d), axis=1)
        i = int(e.argmax())
        if e[i] > tol:
            keep.add(path[a + 1 + i]); stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return [n for n in path if n in keep]


def one(job):
    cell, file, twig, tol = job
    try:
        a = np.loadtxt(file, comments='#', usecols=(0, 2, 3, 4, 6), ndmin=2)
    except Exception:
        return None
    if len(a) < 2: return None
    ids = {int(v): i for i, v in enumerate(a[:, 0])}
    xyz = a[:, 1:4].copy()
    if np.abs(xyz).max() > 5000: xyz /= 1000   # some files are in nm, some in um
    parent = np.array([ids.get(int(p), -1) for p in a[:, 4]])
    kids = [[] for _ in a]
    roots = []
    for i, p in enumerate(parent):
        if p >= 0: kids[p].append(i)
        else: roots.append(i)
    root = roots[0]   # fragments of a broken skeleton are dropped
    order, stack = [], [root]
    while stack:
        n = stack.pop(); order.append(n); stack += kids[n]
    # length of the deepest path below each node, to cut twigs
    reach = np.zeros(len(a))
    edge = np.array([np.linalg.norm(xyz[i] - xyz[parent[i]]) if parent[i] >= 0 else 0 for i in range(len(a))])
    for n in reversed(order):
        p = parent[n]
        if p >= 0: reach[p] = max(reach[p], reach[n] + edge[n])
    live = [[k for k in kids[n] if reach[k] + edge[k] >= twig] for n in range(len(a))]
    # walk unbranched runs, depth first, so a child usually sits right after its parent
    verts, back, where = [], [], {}
    stack = [(root, c) for c in reversed(live[root])]
    where[root] = 0; verts.append(root); back.append(0)
    while stack:
        start, n = stack.pop()
        path = [start, n]
        while len(live[n]) == 1:
            n = live[n][0]; path.append(n)
        prev = start
        for m in simplify(path, xyz, tol)[1:]:
            where[m] = len(verts); back.append(len(verts) - where[prev]); verts.append(m); prev = m
        stack += [(n, c) for c in reversed(live[n])]
    if len(verts) < 2 or max(back) > 65535: return None
    q = np.round(xyz[verts] * 20)
    if np.abs(q).max() > 32767: return None
    return cell, q.astype(np.int16), np.array(back, np.uint16)


def main():
    picked, swc, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
    twig = float(sys.argv[4]) if len(sys.argv) > 4 else 6
    tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1.2
    jobs = [(int(c), os.path.join(swc, f), twig, tol) for c, f in picked.items()]
    with Pool() as p:
        got = [r for r in p.imap(one, jobs, chunksize=16) if r]
    got.sort(key=lambda r: r[0])
    N, V = len(got), sum(len(r[2]) for r in got)
    with gzip.open(out, 'wb', 9) as f:
        f.write(np.array([N, V], np.uint32).tobytes())
        f.write(np.array([r[0] for r in got], np.uint32).tobytes())
        f.write(np.array([len(r[2]) for r in got], np.uint32).tobytes())
        for r in got: f.write(r[1].tobytes())
        for r in got: f.write(r[2].tobytes())
    print(f'{N} neurons of {len(jobs)}, {V} vertices, {os.path.getsize(out) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
