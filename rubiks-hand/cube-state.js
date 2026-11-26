// Exact 3x3x3 physical Rubik's cube math matching shadow_rubik.xml:
// - 6 center hinges qpos[31..36] for faces [pX(R), nX(L), pY(B), nY(F), pZ(U), nZ(D)]
// - 20 edge/corner ball joints qpos[37..116] (4-elem unit quats [w, x, y, z] in core frame)
// Standard WCA colors: U(+Z)=White, D(-Z)=Yellow, R(+X)=Red, L(-X)=Orange, F(-Y)=Green, B(+Y)=Blue.

export const FACE_INFO = [
    { id: 0, name: 'pX', letter: 'R', color: 'red',    axis: [ 1,  0,  0] },
    { id: 1, name: 'nX', letter: 'L', color: 'orange', axis: [-1,  0,  0] },
    { id: 2, name: 'pY', letter: 'B', color: 'blue',   axis: [ 0,  1,  0] },
    { id: 3, name: 'nY', letter: 'F', color: 'green',  axis: [ 0, -1,  0] },
    { id: 4, name: 'pZ', letter: 'U', color: 'white',  axis: [ 0,  0,  1] },
    { id: 5, name: 'nZ', letter: 'D', color: 'yellow', axis: [ 0,  0, -1] },
];

export const LETTER_TO_FACE = { R: 0, L: 1, B: 2, F: 3, U: 4, D: 5 };

// Order matches shadow_rubik.xml bodies after the 6 centers (indices 0..19 -> qpos[37 + 4*k])
export const CUBELET_HOME = [
    // 12 edges
    [ 1,  1,  0], [ 1, -1,  0], [ 1,  0,  1], [ 1,  0, -1],
    [-1,  1,  0], [-1, -1,  0], [-1,  0,  1], [-1,  0, -1],
    [ 0,  1,  1], [ 0,  1, -1], [ 0, -1,  1], [ 0, -1, -1],
    // 8 corners
    [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
    [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
];

// Kociemba 54-facelet layout: U(0..8), R(9..17), F(18..26), D(27..35), L(36..44), B(45..53)
const KOCIEMBA_FACES = [
    { letter: 'U', normal: [ 0,  0,  1], pos: (r, c) => [c - 1, 1 - r,  1] },
    { letter: 'R', normal: [ 1,  0,  0], pos: (r, c) => [ 1, c - 1, 1 - r] },
    { letter: 'F', normal: [ 0, -1,  0], pos: (r, c) => [c - 1, -1, 1 - r] },
    { letter: 'D', normal: [ 0,  0, -1], pos: (r, c) => [c - 1, r - 1, -1] },
    { letter: 'L', normal: [-1,  0,  0], pos: (r, c) => [-1, 1 - c, 1 - r] },
    { letter: 'B', normal: [ 0,  1,  0], pos: (r, c) => [1 - c,  1, 1 - r] },
];

export const QPOS_CUBE_POS = 24;
export const QPOS_CUBE_QUAT = 27;
export const QPOS_CENTERS = 31;
export const QPOS_CUBELETS = 37;

export function quatMul(a, b) {
    return [
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ];
}

export function quatNorm(q) {
    const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    return [q[0]/n, q[1]/n, q[2]/n, q[3]/n];
}

export function quatFromAxisAngle(axis, angle) {
    const h = angle * 0.5, s = Math.sin(h);
    return [Math.cos(h), axis[0]*s, axis[1]*s, axis[2]*s];
}

export function quatRotateVec(q, v) {
    const [w, x, y, z] = q;
    const tx = 2 * (y * v[2] - z * v[1]);
    const ty = 2 * (z * v[0] - x * v[2]);
    const tz = 2 * (x * v[1] - y * v[0]);
    return [
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    ];
}

// Wrap angle to [-pi, pi]
export function wrapPi(a) {
    let x = (a + Math.PI) % (2 * Math.PI);
    if (x < 0) x += 2 * Math.PI;
    return x - Math.PI;
}

// Signed deviation of angle from nearest multiple of pi/2, in [-pi/4, pi/4]
export function wrap90(a) {
    const h = Math.PI / 2;
    let x = ((a + Math.PI / 4) % h);
    if (x < 0) x += h;
    return x - Math.PI / 4;
}

export function round90(a) {
    return Math.round(a / (Math.PI / 2)) * (Math.PI / 2);
}

// Apply a Singmaster move token (e.g. "U", "R'", "F2") directly to MuJoCo qpos (in-place)
export function applyMoveToQpos(qpos, move) {
    const letter = move[0];
    const faceId = LETTER_TO_FACE[letter];
    if (faceId === undefined) throw new Error(`Unknown move: ${move}`);
    const info = FACE_INFO[faceId];
    const turns = move.endsWith('2') ? 2 : 1;
    // Outward normal axis: CW (-pi/2) for X, CCW (+pi/2) for X'
    const dir = move.endsWith("'") ? 1 : -1;
    const angle = dir * turns * (Math.PI / 2);
    const qRot = quatFromAxisAngle(info.axis, angle);

    qpos[QPOS_CENTERS + faceId] += angle;
    for (let k = 0; k < 20; k++) {
        const off = QPOS_CUBELETS + 4 * k;
        const qk = [qpos[off], qpos[off+1], qpos[off+2], qpos[off+3]];
        const pos = quatRotateVec(qk, CUBELET_HOME[k]);
        const dot = pos[0]*info.axis[0] + pos[1]*info.axis[1] + pos[2]*info.axis[2];
        if (dot > 0.5) {
            const qNew = quatNorm(quatMul(qRot, qk));
            qpos[off] = qNew[0]; qpos[off+1] = qNew[1]; qpos[off+2] = qNew[2]; qpos[off+3] = qNew[3];
        }
    }
}

// Map a 3D unit-ish normal vector to face letter (U, D, R, L, F, B)
function normalToLetter(n) {
    const ax = Math.abs(n[0]), ay = Math.abs(n[1]), az = Math.abs(n[2]);
    if (ax >= ay && ax >= az) return n[0] > 0 ? 'R' : 'L';
    if (ay >= ax && ay >= az) return n[1] > 0 ? 'B' : 'F';
    return n[2] > 0 ? 'U' : 'D';
}

// Read exact 54-char Kociemba facelet string directly from live MuJoCo qpos[37..116]!
export function readFaceletsFromQpos(qpos) {
    // Build a lookup from rounded (gx, gy, gz, normalLetter) -> sticker color letter
    const stickers = new Map();
    // 6 centers are fixed at their faces
    for (const f of FACE_INFO) {
        const [x, y, z] = f.axis;
        stickers.set(`${x},${y},${z},${f.letter}`, f.letter);
    }
    // 20 edges & corners
    for (let k = 0; k < 20; k++) {
        const off = QPOS_CUBELETS + 4 * k;
        const qk = [qpos[off], qpos[off+1], qpos[off+2], qpos[off+3]];
        const h = CUBELET_HOME[k];
        const p = quatRotateVec(qk, h);
        const gx = Math.round(p[0]), gy = Math.round(p[1]), gz = Math.round(p[2]);
        // Each nonzero coordinate of h corresponds to one colored sticker on this cubelet
        for (let a = 0; a < 3; a++) {
            if (h[a] === 0) continue;
            const homeN = [0, 0, 0];
            homeN[a] = h[a];
            const colorLetter = normalToLetter(homeN);
            const curN = quatRotateVec(qk, homeN);
            const faceLetter = normalToLetter(curN);
            stickers.set(`${gx},${gy},${gz},${faceLetter}`, colorLetter);
        }
    }
    let out = '';
    for (const face of KOCIEMBA_FACES) {
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                const [gx, gy, gz] = face.pos(r, c);
                out += stickers.get(`${gx},${gy},${gz},${face.letter}`) ?? '?';
            }
        }
    }
    return out;
}

export const SOLVED_FACELETS = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB';

export function isCubeSolvedQpos(qpos) {
    return readFaceletsFromQpos(qpos) === SOLVED_FACELETS;
}

// Return which face id (0..5) is currently pointing most upward (+Z in world frame),
// along with its dot product with +Z.
export function topFaceFromCoreQuat(coreQuat) {
    let bestId = 4, bestZ = -2;
    for (let i = 0; i < 6; i++) {
        const wAxis = quatRotateVec(coreQuat, FACE_INFO[i].axis);
        if (wAxis[2] > bestZ) {
            bestZ = wAxis[2];
            bestId = i;
        }
    }
    return { faceId: bestId, dotZ: bestZ };
}
