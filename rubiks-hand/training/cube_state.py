"""Exact 3x3x3 physical Rubik's cube state math (Python mirror of cube-state.js)."""
import numpy as np

FACE_AXES = np.array([
    [ 1.0,  0.0,  0.0],  # 0: pX (R, red)
    [-1.0,  0.0,  0.0],  # 1: nX (L, orange)
    [ 0.0,  1.0,  0.0],  # 2: pY (B, blue)
    [ 0.0, -1.0,  0.0],  # 3: nY (F, green)
    [ 0.0,  0.0,  1.0],  # 4: pZ (U, white)
    [ 0.0,  0.0, -1.0],  # 5: nZ (D, yellow)
], dtype=np.float64)

FACE_LETTERS = ("R", "L", "B", "F", "U", "D")
LETTER_TO_FACE = {c: i for i, c in enumerate(FACE_LETTERS)}

CUBELET_HOME = np.array([
    # 12 edges
    [ 1,  1,  0], [ 1, -1,  0], [ 1,  0,  1], [ 1,  0, -1],
    [-1,  1,  0], [-1, -1,  0], [-1,  0,  1], [-1,  0, -1],
    [ 0,  1,  1], [ 0,  1, -1], [ 0, -1,  1], [ 0, -1, -1],
    # 8 corners
    [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
    [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
], dtype=np.float64)

QPOS_CUBE_POS = 24
QPOS_CUBE_QUAT = 27
QPOS_CENTERS = 31
QPOS_CUBELETS = 37

SOLVED_FACELETS = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"


def quat_mul(a, b):
    return np.array([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ], dtype=np.float64)


def quat_from_axis_angle(axis, angle):
    h = 0.5 * angle
    return np.array([np.cos(h), axis[0]*np.sin(h), axis[1]*np.sin(h), axis[2]*np.sin(h)], dtype=np.float64)


def quat_rotate_vec(q, v):
    w, x, y, z = q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def apply_move_to_qpos(qpos, move):
    face_id = LETTER_TO_FACE[move[0]]
    axis = FACE_AXES[face_id]
    turns = 2 if move.endswith("2") else 1
    direction = 1.0 if move.endswith("'") else -1.0
    angle = direction * turns * (np.pi / 2.0)
    q_rot = quat_from_axis_angle(axis, angle)

    qpos[QPOS_CENTERS + face_id] += angle
    for k in range(20):
        off = QPOS_CUBELETS + 4 * k
        qk = qpos[off:off+4]
        pos = quat_rotate_vec(qk, CUBELET_HOME[k])
        if np.dot(pos, axis) > 0.5:
            qn = quat_mul(q_rot, qk)
            qpos[off:off+4] = qn / (np.linalg.norm(qn) + 1e-12)


def _normal_to_letter(n):
    a = np.abs(n)
    i = int(np.argmax(a))
    if i == 0:
        return "R" if n[0] > 0 else "L"
    if i == 1:
        return "B" if n[1] > 0 else "F"
    return "U" if n[2] > 0 else "D"


def read_facelets_from_qpos(qpos):
    stickers = {}
    for i, letter in enumerate(FACE_LETTERS):
        ax = FACE_AXES[i].astype(int)
        stickers[(int(ax[0]), int(ax[1]), int(ax[2]), letter)] = letter
    for k in range(20):
        off = QPOS_CUBELETS + 4 * k
        qk = qpos[off:off+4]
        h = CUBELET_HOME[k]
        p = np.round(quat_rotate_vec(qk, h)).astype(int)
        gx, gy, gz = int(p[0]), int(p[1]), int(p[2])
        for a in range(3):
            if h[a] == 0:
                continue
            home_n = np.zeros(3)
            home_n[a] = h[a]
            color_letter = _normal_to_letter(home_n)
            face_letter = _normal_to_letter(quat_rotate_vec(qk, home_n))
            stickers[(gx, gy, gz, face_letter)] = color_letter

    layout = [
        ("U", lambda r, c: (c - 1, 1 - r,  1)),
        ("R", lambda r, c: ( 1, c - 1, 1 - r)),
        ("F", lambda r, c: (c - 1, -1, 1 - r)),
        ("D", lambda r, c: (c - 1, r - 1, -1)),
        ("L", lambda r, c: (-1, 1 - c, 1 - r)),
        ("B", lambda r, c: (1 - c,  1, 1 - r)),
    ]
    out = []
    for letter, fn in layout:
        for r in range(3):
            for c in range(3):
                gx, gy, gz = fn(r, c)
                out.append(stickers.get((gx, gy, gz, letter), "?"))
    return "".join(out)
