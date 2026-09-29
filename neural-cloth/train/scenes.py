"""Random robot-manipulation scenarios for the cloth teacher.

Every scene has a table (the plane y = 0) and a cloth that starts flat, either
lying on the table or dropped from a height. Grippers grasp vertices and move
them along scripted paths (pick-and-place, folds, flings, drags, shakes).
Optional obstacles: a sphere on the table, a rod (towel rack) or a moving
capsule (a sweeping robot link). Bending stiffness, friction and wind vary.
"""
import numpy as np

import cloth as C

SIZES = [(16, 16), (21, 16), (21, 21), (26, 21), (26, 26), (31, 21), (31, 26), (31, 31)]
KB_RANGE = (3e-7, 3e-5)
MU_RANGE = (0.2, 0.8)
REST_Y = C.OBS_OFFSET + 0.5 * C.DHAT


def min_jerk(t):
    t = np.clip(t, 0, 1)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def path_from_waypoints(waypoints, durations, t0, frames):
    """Displacements [frames+1, 3] relative to the anchor, starting at frame t0 (zero before)."""
    disp = np.zeros((frames + 1, 3))
    f = t0
    pts = [np.zeros(3)] + [np.asarray(w, float) for w in waypoints]
    for k, dur in enumerate(durations):
        n = max(1, int(round(dur / C.FRAME_DT)))
        for i in range(n):
            if f + i + 1 > frames:
                break
            s = min_jerk((i + 1) / n)
            disp[f + i + 1] = pts[k] + (pts[k + 1] - pts[k]) * s
        f += n
        if f >= frames:
            break
    if f < frames:
        disp[f + 1:] = pts[-1]
    return disp, min(f, frames)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_axis(axis, a):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def random_scene(rng, frames, size=None, kind=None):
    """kind: force a scenario (e.g. "lift_release"); by default one is drawn from the usual mix."""
    nx, ny = size or SIZES[rng.integers(len(SIZES))]
    mesh = C.grid_mesh(nx, ny)
    rest = mesh["rest"].astype(np.float64)
    rest = rest - rest.mean(0)
    flat = np.stack([rest[:, 0], np.zeros(len(rest)), rest[:, 1]], 1)

    on_table = rng.random() < (0.85 if kind == "lift_release" else 0.45)
    R = rot_y(rng.uniform(0, 2 * np.pi))
    if not on_table:
        tilt = rng.uniform(0, np.pi / 2) if rng.random() < 0.7 else rng.uniform(np.pi / 2, np.pi * 0.95)
        R = rot_axis(np.array([rng.normal(), 0, rng.normal()]), tilt) @ R
    x0 = flat @ R.T
    center = np.array([rng.uniform(-0.15, 0.15), 0, rng.uniform(-0.15, 0.15)])
    x0 = x0 + center
    if on_table:
        x0[:, 1] = REST_Y
    else:
        x0[:, 1] += REST_Y - x0[:, 1].min() + rng.uniform(0.02, 0.45)

    # ---- obstacles
    S, Cc, H = C.MAX_SPHERES, C.MAX_CAPSULES, C.MAX_HANDLES
    sph_c = np.zeros((frames + 1, S, 3))
    sph_r = np.zeros(S)
    sph_on = np.zeros(S)
    cap_a = np.zeros((frames + 1, Cc, 3))
    cap_b = np.zeros((frames + 1, Cc, 3))
    cap_r = np.zeros(Cc)
    cap_on = np.zeros(Cc)

    def clear_of_cloth(sdf_fn):
        return np.all(sdf_fn(x0) > 0.01)

    if rng.random() < 0.35:
        for _ in range(20):
            r = rng.uniform(0.05, 0.14)
            c = np.array([center[0] + rng.uniform(-0.2, 0.2), r, center[2] + rng.uniform(-0.2, 0.2)])
            if clear_of_cloth(lambda x: np.linalg.norm(x - c, axis=1) - r - C.OBS_OFFSET):
                sph_c[:, 0] = c
                if rng.random() < 0.25:  # a sphere pushed across the table, riding clear of it
                    d = rng.normal(size=3)
                    d[1] = 0
                    d *= rng.uniform(0.1, 0.4) / np.linalg.norm(d)
                    lift = np.array([0.0, 0.025, 0.0])  # a kinematic ball on the table would pinch cloth to zero
                    sph_c[:, 0] = c + lift + np.outer(np.arange(frames + 1) * C.FRAME_DT, d)
                sph_r[0], sph_on[0] = r, 1
                break

    def capsule_sdf(a, b, r):
        def f(x):
            ab = b - a
            t = np.clip(((x - a) @ ab) / (ab @ ab), 0, 1)
            return np.linalg.norm(x - (a + t[:, None] * ab), axis=1) - r - C.OBS_OFFSET
        return f

    u = rng.random()
    if u < 0.2:  # static rod: towel rack
        for _ in range(20):
            yaw = rng.uniform(0, np.pi)
            half = rng.uniform(0.3, 0.5) * np.array([np.cos(yaw), 0, np.sin(yaw)])
            mid = np.array([center[0] + rng.uniform(-0.15, 0.15), rng.uniform(0.12, 0.4), center[2] + rng.uniform(-0.15, 0.15)])
            r = rng.uniform(0.008, 0.03)
            if clear_of_cloth(capsule_sdf(mid - half, mid + half, r)):
                cap_a[:, 0], cap_b[:, 0], cap_r[0], cap_on[0] = mid - half, mid + half, r, 1
                break
    elif u < 0.35:  # a robot link sweeping through the scene
        for _ in range(20):
            r = rng.uniform(0.02, 0.05)
            length = rng.uniform(0.15, 0.4)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            start = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.05, 0.4), rng.uniform(-0.5, 0.5)])
            vel = rng.normal(size=3)
            vel[1] *= 0.3
            vel *= rng.uniform(0.2, 0.9) / np.linalg.norm(vel)
            vel -= 0.6 * (start - np.array([center[0], 0.15, center[2]])) * (np.linalg.norm(vel) / 0.5)
            a0, b0 = start - 0.5 * length * direction, start + 0.5 * length * direction
            if a0[1] - r < 0.01 or b0[1] - r < 0.01 or not clear_of_cloth(capsule_sdf(a0, b0, r)):
                continue
            ts = np.arange(frames + 1)[:, None] * C.FRAME_DT
            a_t, b_t = a0 + ts * vel, b0 + ts * vel
            low = np.minimum(a_t[:, 1], b_t[:, 1]) - r
            lift = np.maximum(0.01 - low, 0)[:, None] * np.array([0, 1, 0])
            cap_a[:, 0], cap_b[:, 0] = a_t + lift, b_t + lift
            cap_r[0], cap_on[0] = r, 1
            break

    # ---- grippers
    h_vid = -np.ones(H, np.int64)
    h_t0 = np.full(H, frames + 1, np.int64)
    h_t1 = np.full(H, frames + 1, np.int64)
    h_disp = np.zeros((frames + 1, H, 3))
    corners = [0, ny - 1, (nx - 1) * ny, nx * ny - 1]

    def pick_vertex():
        r = rng.random()
        if r < 0.55:
            return int(rng.choice(corners))
        if r < 0.8:  # a point on the border
            i, j = rng.integers(nx), rng.integers(ny)
            return int(rng.choice([i * ny, i * ny + ny - 1, j, (nx - 1) * ny + j]))
        return int(rng.integers(nx * ny))

    settle = 0 if on_table else int(rng.integers(40, 90))
    t0 = settle + int(rng.integers(0, 30))
    # lift_drop is weighted up: a towel let go after hanging still (the demo's most common move) is the
    # case the network found hardest, and the teacher shows only a few frames of each release.
    if kind is None:
        kind = rng.choice(["none", "place", "fold", "fling", "drag", "shake", "lift_drop"],
                          p=[0.06, 0.18, 0.17, 0.12, 0.12, 0.08, 0.27])
    yaw = rng.uniform(0, 2 * np.pi)
    fwd = np.array([np.cos(yaw), 0, np.sin(yaw)])
    if kind == "place" or kind == "lift_drop":
        v = pick_vertex()
        lift = rng.uniform(0.15, 0.55)
        move = fwd * rng.uniform(0.0, 0.35)
        if kind == "place":
            wps = [[0, lift, 0], [move[0], lift, move[2]], [move[0], 0.02, move[2]]]
            durs = [lift / rng.uniform(0.3, 0.9), rng.uniform(0.4, 1.2), lift / rng.uniform(0.3, 0.9)]
        else:
            wps = [[0, lift, 0], [move[0], lift, move[2]]]
            durs = [lift / rng.uniform(0.3, 1.2), rng.uniform(0.3, 1.0)]
        disp, end = path_from_waypoints(wps, durs, t0, frames)
        h_vid[0], h_t0[0], h_disp[:, 0] = v, t0, disp
        # lift_drop sometimes holds long enough for the towel to hang still before it is let go
        h_t1[0] = end + int(rng.integers(5, 90 if kind == "lift_drop" else 40))
    elif kind in ("fold", "fling"):
        # two adjacent corners along one cloth edge
        pairs = [(corners[0], corners[1]), (corners[2], corners[3]), (corners[0], corners[2]), (corners[1], corners[3])]
        va, vb = pairs[rng.integers(4)]
        if kind == "fold":
            # carry the gripped edge across the cloth, past its centre (a half fold when ratio ~ 2)
            mid = 0.5 * (x0[va] + x0[vb])
            across = x0.mean(0) - mid
            across[1] = 0
            span = np.linalg.norm(across)
            fdir = across / span if span > 1e-6 else fwd
            dist = span * rng.uniform(1.3, 2.1)
            h = rng.uniform(0.06, 0.2)
            wps = [[0.5 * dist * fdir[0], h, 0.5 * dist * fdir[2]], [dist * fdir[0], 0.02, dist * fdir[2]]]
            durs = [rng.uniform(0.6, 1.3), rng.uniform(0.5, 1.0)]
        else:
            h = rng.uniform(0.35, 0.65)
            reach = rng.uniform(0.15, 0.35)
            wps = [[0, h, 0], [reach * fwd[0], h + 0.05, reach * fwd[2]], [-0.25 * fwd[0], 0.04, -0.25 * fwd[2]]]
            durs = [rng.uniform(0.8, 1.5), rng.uniform(0.15, 0.35), rng.uniform(0.3, 0.6)]
        disp, end = path_from_waypoints(wps, durs, t0, frames)
        rel = end + int(rng.integers(3, 30))
        for k, v in enumerate((va, vb)):
            h_vid[k], h_t0[k], h_t1[k], h_disp[:, k] = v, t0, rel, disp
    elif kind == "drag":
        v = pick_vertex()
        h = rng.uniform(0.02, 0.1)
        dist = rng.uniform(0.15, 0.45)
        wps = [[0, h, 0], [dist * fwd[0], h, dist * fwd[2]]]
        durs = [0.3, dist / rng.uniform(0.15, 0.5)]
        disp, end = path_from_waypoints(wps, durs, t0, frames)
        h_vid[0], h_t0[0], h_t1[0], h_disp[:, 0] = v, t0, end + int(rng.integers(5, 40)), disp
    elif kind == "shake":
        v = pick_vertex()
        lift = rng.uniform(0.3, 0.6)
        disp, end = path_from_waypoints([[0, lift, 0]], [lift / 0.6], t0, frames)
        amp = rng.uniform(0.04, 0.12)
        freq = rng.uniform(1.0, 3.0)
        dur = int(rng.uniform(0.8, 2.0) / C.FRAME_DT)
        for f in range(end, min(frames + 1, end + dur + 1)):
            disp[f] = disp[end] + amp * np.sin(2 * np.pi * freq * (f - end) * C.FRAME_DT) * fwd
        if end + dur + 1 <= frames:
            disp[end + dur + 1:] = disp[end + dur]
        h_vid[0], h_t0[0], h_t1[0], h_disp[:, 0] = v, t0, min(frames, end + dur) + int(rng.integers(0, 30)), disp

    elif kind == "lift_release":
        # Lift the towel by any point (often the middle, which makes a tent), hold it still, let go.
        # A towel let go this way has to fall at once; the network learned to leave tents standing.
        r = rng.random()
        if r < 0.3:
            v = int(rng.choice(corners))
        elif r < 0.55:
            i, j = rng.integers(nx), rng.integers(ny)
            v = int(rng.choice([i * ny, i * ny + ny - 1, j, (nx - 1) * ny + j]))
        else:
            v = int(rng.integers(2, nx - 2)) * ny + int(rng.integers(2, ny - 2))
        lift = rng.uniform(0.12, 0.5)
        move = fwd * rng.uniform(0.0, 0.2)
        wps = [[0, lift, 0], [move[0], lift, move[2]]]
        durs = [lift / rng.uniform(0.3, 1.0), max(0.2, float(np.linalg.norm(move)) / 0.3)]
        disp, end = path_from_waypoints(wps, durs, t0, frames)
        h_vid[0], h_t0[0], h_disp[:, 0] = v, t0, disp
        h_t1[0] = end + int(rng.integers(30, 100))

    wind = np.zeros((frames, 3))
    if rng.random() < 0.15:
        d = rng.normal(size=3)
        d[1] = 0
        wind[:] = d / np.linalg.norm(d) * rng.uniform(1.0, 4.0)

    kb = float(np.exp(rng.uniform(*np.log(KB_RANGE))))
    mu = float(rng.uniform(*MU_RANGE))
    scene = dict(
        x0=x0.astype(np.float32), h_vid=h_vid.astype(np.int32), h_t0=h_t0.astype(np.int32),
        h_t1=h_t1.astype(np.int32), h_disp=h_disp.astype(np.float32),
        sph_c=sph_c.astype(np.float32), sph_r=sph_r.astype(np.float32), sph_on=sph_on.astype(np.float32),
        cap_a=cap_a.astype(np.float32), cap_b=cap_b.astype(np.float32), cap_r=cap_r.astype(np.float32),
        cap_on=cap_on.astype(np.float32), wind=wind.astype(np.float32),
        kb=np.float32(kb), mu=np.float32(mu),
    )
    return mesh, scene, str(kind)
