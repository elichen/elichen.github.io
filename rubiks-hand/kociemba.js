// Pure-JS Kociemba Two-Phase 3x3x3 Rubik's Cube Solver (zero dependencies).
// Takes a 54-char URFDLB facelet string and returns a short Singmaster move sequence
// (typically 18-22 moves in <20 ms after a ~30 ms one-time table build).
//
// Facelet order: U(0..8), R(9..17), F(18..26), D(27..35), L(36..44), B(45..53).

const U = 0, R = 1, F = 2, D = 3, L = 4, B = 5;
const FACE_NAMES = ['U', 'R', 'F', 'D', 'L', 'B'];
const MOVE_NAMES = [
    'U', 'U2', "U'", 'R', 'R2', "R'", 'F', 'F2', "F'",
    'D', 'D2', "D'", 'L', 'L2', "L'", 'B', 'B2', "B'",
];

// Corners: URF(0), UFL(1), ULB(2), UBR(3), DFR(4), DLF(5), DBL(6), DRB(7)
const CORNER_FACELETS = [
    [8, 9, 20],  [6, 18, 38], [0, 36, 47], [2, 45, 11],
    [29, 26, 15],[27, 44, 24],[33, 53, 42],[35, 17, 51],
];
const CORNER_COLORS = [
    [U, R, F], [U, F, L], [U, L, B], [U, B, R],
    [D, F, R], [D, L, F], [D, B, L], [D, R, B],
];

// Edges: UR(0), UF(1), UL(2), UB(3), DR(4), DF(5), DL(6), DB(7), FR(8), FL(9), BL(10), BR(11)
const EDGE_FACELETS = [
    [5, 10],  [7, 19],  [3, 37],  [1, 46],
    [32, 16], [28, 25], [30, 43], [34, 52],
    [23, 12], [21, 41], [50, 39], [48, 14],
];
const EDGE_COLORS = [
    [U, R], [U, F], [U, L], [U, B],
    [D, R], [D, F], [D, L], [D, B],
    [F, R], [F, L], [B, L], [B, R],
];

// Basic quarter-turn clockwise moves (U, R, F, D, L, B) on cubies
// Convention: new[i] = old[p[i]], newO[i] = (oldO[p[i]] + o[i]) mod (3 or 2)
const BASIC_MOVES = [
    // U
    { cp: [3,0,1,2,4,5,6,7], co: [0,0,0,0,0,0,0,0],
      ep: [3,0,1,2,4,5,6,7,8,9,10,11], eo: [0,0,0,0,0,0,0,0,0,0,0,0] },
    // R
    { cp: [4,1,2,0,7,5,6,3], co: [2,0,0,1,1,0,0,2],
      ep: [8,1,2,3,11,5,6,7,4,9,10,0], eo: [0,0,0,0,0,0,0,0,0,0,0,0] },
    // F
    { cp: [1,5,2,3,0,4,6,7], co: [1,2,0,0,2,1,0,0],
      ep: [0,9,2,3,4,8,6,7,1,5,10,11], eo: [0,1,0,0,0,1,0,0,1,1,0,0] },
    // D
    { cp: [0,1,2,3,5,6,7,4], co: [0,0,0,0,0,0,0,0],
      ep: [0,1,2,3,5,6,7,4,8,9,10,11], eo: [0,0,0,0,0,0,0,0,0,0,0,0] },
    // L
    { cp: [0,2,6,3,4,1,5,7], co: [0,1,2,0,0,2,1,0],
      ep: [0,1,10,3,4,5,9,7,8,2,6,11], eo: [0,0,0,0,0,0,0,0,0,0,0,0] },
    // B
    { cp: [0,1,3,7,4,5,2,6], co: [0,0,1,2,0,0,2,1],
      ep: [0,1,2,11,4,5,6,10,8,9,3,7], eo: [0,0,0,1,0,0,0,1,0,0,1,1] },
];

function mulCubie(a, b) {
    const cp = new Int8Array(8), co = new Int8Array(8);
    for (let i = 0; i < 8; i++) {
        cp[i] = a.cp[b.cp[i]];
        co[i] = (a.co[b.cp[i]] + b.co[i]) % 3;
    }
    const ep = new Int8Array(12), eo = new Int8Array(12);
    for (let i = 0; i < 12; i++) {
        ep[i] = a.ep[b.ep[i]];
        eo[i] = (a.eo[b.ep[i]] + b.eo[i]) & 1;
    }
    return { cp, co, ep, eo };
}

function idCubie() {
    return {
        cp: Int8Array.from([0,1,2,3,4,5,6,7]), co: new Int8Array(8),
        ep: Int8Array.from([0,1,2,3,4,5,6,7,8,9,10,11]), eo: new Int8Array(12),
    };
}

// All 18 moves (face*3 + power, power 0=CW, 1=180, 2=CCW)
const MOVES18 = [];
for (let f = 0; f < 6; f++) {
    let cur = idCubie();
    for (let p = 0; p < 3; p++) {
        cur = mulCubie(cur, BASIC_MOVES[f]);
        MOVES18.push(cur);
    }
}

// Phase 2 allowed move indices in MOVES18: U, U2, U', R2, F2, D, D2, D', L2, B2
const P2_MOVES = [0, 1, 2, 4, 7, 9, 10, 11, 13, 16];

// Combinatorial helpers
function C(n, k) {
    if (n < k || k < 0) return 0;
    if (k > n - k) k = n - k;
    let r = 1;
    for (let i = 1; i <= k; i++) r = (r * (n - i + 1)) / i;
    return r;
}

// Coordinate conversions
function getTwist(c) {
    let r = 0;
    for (let i = 0; i < 7; i++) r = r * 3 + c.co[i];
    return r;
}
function setTwist(c, t) {
    let sum = 0;
    for (let i = 6; i >= 0; i--) { c.co[i] = t % 3; sum += c.co[i]; t = (t / 3) | 0; }
    c.co[7] = (3 - (sum % 3)) % 3;
}

function getFlip(c) {
    let r = 0;
    for (let i = 0; i < 11; i++) r = (r << 1) | c.eo[i];
    return r;
}
function setFlip(c, f) {
    let sum = 0;
    for (let i = 10; i >= 0; i--) { c.eo[i] = f & 1; sum ^= c.eo[i]; f >>= 1; }
    c.eo[11] = sum;
}

// Slice coord (0..494): which 4 positions hold edges 8..11 (FR, FL, BL, BR). Solved = 0!
function getSlice(c) {
    let a = 0, x = 0;
    for (let j = 11; j >= 0; j--) {
        if (c.ep[j] >= 8) { a += C(11 - j, x + 1); x++; }
    }
    return a;
}
function setSlice(c, s) {
    let x = 4, other = 0, sliceE = 8;
    for (let j = 0; j < 12; j++) {
        if (s >= C(11 - j, x)) {
            s -= C(11 - j, x);
            c.ep[j] = sliceE++;
            x--;
        } else {
            c.ep[j] = other++;
        }
    }
}

// Permutation <-> Lehmer index for small arrays
function permToIdx(arr, offset, len) {
    let idx = 0;
    for (let i = len - 1; i > 0; i--) {
        let s = 0;
        for (let j = i - 1; j >= 0; j--) if (arr[offset + j] > arr[offset + i]) s++;
        idx = (idx + s) * i;
    }
    return idx;
}
function idxToPerm(idx, arr, offset, len, baseVal) {
    const used = new Uint8Array(len);
    const digits = new Int8Array(len);
    for (let i = 1; i < len; i++) {
        digits[i] = idx % (i + 1);
        idx = (idx / (i + 1)) | 0;
    }
    for (let i = len - 1; i >= 0; i--) {
        let k = len - 1;
        let cnt = digits[i];
        while (true) {
            if (!used[k]) {
                if (cnt === 0) break;
                cnt--;
            }
            k--;
        }
        used[k] = 1;
        arr[offset + i] = baseVal + k;
    }
}

function getCP(c) { return permToIdx(c.cp, 0, 8); }
function setCP(c, idx) { idxToPerm(idx, c.cp, 0, 8, 0); }

function getUDEP(c) { return permToIdx(c.ep, 0, 8); }
function setUDEP(c, idx) { idxToPerm(idx, c.ep, 0, 8, 0); }

function getSlicePerm(c) { return permToIdx(c.ep, 8, 4); }
function setSlicePerm(c, idx) { idxToPerm(idx, c.ep, 8, 4, 8); }

// Lazy-initialized move & pruning tables
let TABLES = null;

function initTables() {
    if (TABLES) return TABLES;
    const N_TWIST = 2187, N_FLIP = 2048, N_SLICE = 495;
    const N_CP = 40320, N_UDEP = 40320, N_SP = 24;

    const twistMove = new Uint16Array(N_TWIST * 18);
    const flipMove  = new Uint16Array(N_FLIP * 18);
    const sliceMove = new Uint16Array(N_SLICE * 18);

    let c = idCubie();
    for (let i = 0; i < N_TWIST; i++) {
        setTwist(c, i);
        for (let m = 0; m < 18; m++) twistMove[i * 18 + m] = getTwist(mulCubie(c, MOVES18[m]));
    }
    c = idCubie();
    for (let i = 0; i < N_FLIP; i++) {
        setFlip(c, i);
        for (let m = 0; m < 18; m++) flipMove[i * 18 + m] = getFlip(mulCubie(c, MOVES18[m]));
    }
    c = idCubie();
    for (let i = 0; i < N_SLICE; i++) {
        setSlice(c, i);
        for (let m = 0; m < 18; m++) sliceMove[i * 18 + m] = getSlice(mulCubie(c, MOVES18[m]));
    }

    // Phase 2 move tables (only 10 P2 moves)
    const cpMove   = new Uint16Array(N_CP * 10);
    const udepMove = new Uint16Array(N_UDEP * 10);
    const spMove   = new Uint8Array(N_SP * 10);

    c = idCubie();
    for (let i = 0; i < N_CP; i++) {
        setCP(c, i);
        for (let m = 0; m < 10; m++) cpMove[i * 10 + m] = getCP(mulCubie(c, MOVES18[P2_MOVES[m]]));
    }
    c = idCubie();
    for (let i = 0; i < N_UDEP; i++) {
        setUDEP(c, i);
        for (let m = 0; m < 10; m++) udepMove[i * 10 + m] = getUDEP(mulCubie(c, MOVES18[P2_MOVES[m]]));
    }
    c = idCubie();
    for (let i = 0; i < N_SP; i++) {
        setSlicePerm(c, i);
        for (let m = 0; m < 10; m++) spMove[i * 10 + m] = getSlicePerm(mulCubie(c, MOVES18[P2_MOVES[m]]));
    }

    // BFS pruning tables
    function buildPrune(sizeA, sizeB, moveA, moveB, nMoves) {
        const N = sizeA * sizeB;
        const dist = new Int8Array(N);
        dist.fill(-1);
        const queue = new Int32Array(N);
        let head = 0, tail = 0;
        dist[0] = 0;
        queue[tail++] = 0;
        while (head < tail) {
            const cur = queue[head++];
            const d = dist[cur] + 1;
            const a = (cur / sizeB) | 0;
            const b = cur - a * sizeB;
            const aOff = a * nMoves, bOff = b * nMoves;
            for (let m = 0; m < nMoves; m++) {
                const nxt = moveA[aOff + m] * sizeB + moveB[bOff + m];
                if (dist[nxt] === -1) {
                    dist[nxt] = d;
                    queue[tail++] = nxt;
                }
            }
        }
        return dist;
    }

    const pruneTwistSlice = buildPrune(N_TWIST, N_SLICE, twistMove, sliceMove, 18); // 1.08 MB
    const pruneFlipSlice  = buildPrune(N_FLIP,  N_SLICE, flipMove,  sliceMove, 18); // 1.01 MB
    const pruneCPSP       = buildPrune(N_CP,    N_SP,    cpMove,    spMove,    10); // 0.96 MB
    const pruneUDEPSP     = buildPrune(N_UDEP,  N_SP,    udepMove,  spMove,    10); // 0.96 MB

    TABLES = {
        twistMove, flipMove, sliceMove,
        cpMove, udepMove, spMove,
        pruneTwistSlice, pruneFlipSlice, pruneCPSP, pruneUDEPSP,
    };
    return TABLES;
}

// Convert 54-char URFDLB facelet string to CubieCube
export function faceletsToCubie(str) {
    if (str.length !== 54) throw new Error(`Expected 54 facelets, got ${str.length}`);
    const f = new Int8Array(54);
    for (let i = 0; i < 54; i++) {
        const idx = FACE_NAMES.indexOf(str[i]);
        if (idx < 0) throw new Error(`Invalid facelet '${str[i]}' at ${i}`);
        f[i] = idx;
    }
    const cp = new Int8Array(8), co = new Int8Array(8);
    for (let i = 0; i < 8; i++) {
        const [f0, f1, f2] = CORNER_FACELETS[i];
        let ori = 0;
        for (; ori < 3; ori++) if (f[CORNER_FACELETS[i][ori]] === U || f[CORNER_FACELETS[i][ori]] === D) break;
        const c1 = f[CORNER_FACELETS[i][(ori + 1) % 3]];
        const c2 = f[CORNER_FACELETS[i][(ori + 2) % 3]];
        let found = -1;
        for (let j = 0; j < 8; j++) {
            if (c1 === CORNER_COLORS[j][1] && c2 === CORNER_COLORS[j][2]) { found = j; break; }
        }
        if (found < 0) throw new Error(`Invalid corner at slot ${i}`);
        cp[i] = found;
        co[i] = ori;
    }
    const ep = new Int8Array(12), eo = new Int8Array(12);
    for (let i = 0; i < 12; i++) {
        const [f0, f1] = EDGE_FACELETS[i];
        let found = -1, ori = 0;
        for (let j = 0; j < 12; j++) {
            if (f[f0] === EDGE_COLORS[j][0] && f[f1] === EDGE_COLORS[j][1]) { found = j; ori = 0; break; }
            if (f[f0] === EDGE_COLORS[j][1] && f[f1] === EDGE_COLORS[j][0]) { found = j; ori = 1; break; }
        }
        if (found < 0) throw new Error(`Invalid edge at slot ${i}`);
        ep[i] = found;
        eo[i] = ori;
    }
    return { cp, co, ep, eo };
}

// Prune redundant consecutive moves on same or commuting opposite faces
function skipFace(prevFace, face) {
    if (prevFace === face) return true;
    // Canonicalize opposite faces: (U=0, D=3), (R=1, L=4), (F=2, B=5)
    if (prevFace === face + 3) return true;
    return false;
}

// 3-axis SO(3) cyclic conjugations S^0, S^1, S^2 so Kociemba searches all 3 slice axes (U-D, R-L, F-B)
// and picks the lowest robot-cost plan (minimizing 180-deg flips, total flips, and quarter turns).
const FACE_POS_3D = [
    { letter: 'U', pos: (r, c) => [c - 1, 1 - r,  1], n: [ 0,  0,  1] },
    { letter: 'R', pos: (r, c) => [ 1, c - 1, 1 - r], n: [ 1,  0,  0] },
    { letter: 'F', pos: (r, c) => [c - 1, -1, 1 - r], n: [ 0, -1,  0] },
    { letter: 'D', pos: (r, c) => [c - 1, r - 1, -1], n: [ 0,  0, -1] },
    { letter: 'L', pos: (r, c) => [-1, 1 - c, 1 - r], n: [-1,  0,  0] },
    { letter: 'B', pos: (r, c) => [1 - c,  1, 1 - r], n: [ 0,  1,  0] },
];

// Proper 120-deg SO(3) rotation S: (x, y, z) -> (y, z, x)
// Maps R(+X)->U(+Z), B(+Y)->R(+X), U(+Z)->B(+Y), L(-X)->D(-Z), F(-Y)->L(-X), D(-Z)->F(-Y)
function rotVec(v, k) {
    if (k === 1) return [v[1], v[2], v[0]];
    if (k === 2) return [v[2], v[0], v[1]];
    return v;
}
function vecToFaceLetter(v) {
    if (v[2] ===  1) return 'U';
    if (v[2] === -1) return 'D';
    if (v[0] ===  1) return 'R';
    if (v[0] === -1) return 'L';
    if (v[1] === -1) return 'F';
    return 'B';
}

const AXIS_CONJ = [0, 1, 2].map(k => {
    // Build sticker index lookup: "x,y,z,L" -> 0..53
    const idxOf = new Map();
    for (let f = 0; f < 6; f++) {
        const fd = FACE_POS_3D[f];
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                const [x, y, z] = fd.pos(r, c);
                idxOf.set(`${x},${y},${z},${fd.letter}`, f * 9 + r * 3 + c);
            }
        }
    }
    // Forward face map: oldLetter -> newLetter under S^k; inverse: newLetter -> oldLetter
    const fwdFace = {}, invFace = {};
    for (const fd of FACE_POS_3D) {
        const mapped = vecToFaceLetter(rotVec(fd.n, k));
        fwdFace[fd.letter] = mapped;
        invFace[mapped] = fd.letter;
    }
    // Sticker src index for each destination sticker i in rotated cube: rotStr[i] = fwdFace[origStr[srcIdx[i]]]
    const srcIdx = new Int8Array(54);
    for (let f = 0; f < 6; f++) {
        const fd = FACE_POS_3D[f];
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                const origP = fd.pos(r, c);
                const rotP = rotVec(origP, k);
                const rotL = fwdFace[fd.letter];
                const dst = idxOf.get(`${rotP[0]},${rotP[1]},${rotP[2]},${rotL}`);
                srcIdx[dst] = f * 9 + r * 3 + c;
            }
        }
    }
    return { fwdFace, invFace, srcIdx };
});

const OPPOSITE_FACE = { U: 'D', D: 'U', R: 'L', L: 'R', F: 'B', B: 'F' };

export function planRobotCost(moves, startTopLetter = null) {
    // Weighted by empirical Shadow Hand drop risk:
    // 100% of drops happen during reorients (0% in Twist), with 180-deg flips ~2x riskier than 90-deg flips;
    // same-face X2 double turns add zero reorients!
    let cost = 0;
    let top = startTopLetter;
    for (const mv of moves) {
        const f = mv[0];
        if (top !== null && f !== top) {
            cost += (f === OPPOSITE_FACE[top]) ? 3.8 : 2.0;
        } else if (top === null) {
            cost += 2.0;
        }
        cost += mv.endsWith('2') ? 0.55 : 0.35;
        top = f;
    }
    return cost;
}

function invCubie(c) {
    const cp = new Int8Array(8), co = new Int8Array(8);
    for (let i = 0; i < 8; i++) cp[c.cp[i]] = i;
    for (let i = 0; i < 8; i++) {
        const ori = c.co[cp[i]];
        co[i] = ori === 0 ? 0 : 3 - ori;
    }
    const ep = new Int8Array(12), eo = new Int8Array(12);
    for (let i = 0; i < 12; i++) ep[c.ep[i]] = i;
    for (let i = 0; i < 12; i++) eo[i] = c.eo[ep[i]];
    return { cp, co, ep, eo };
}

function reorderCommutingMoves(moves, startTopLetter = null) {
    // Opposite faces (U/D, R/L, F/B) commute geometrically! Reordering [A, B] -> [B, A]
    // whenever B matches the current top face eliminates a wasted reorient / 180-deg flip for free.
    if (moves.length < 2) return moves;
    const out = moves.slice();
    let top = startTopLetter;
    for (let i = 0; i < out.length - 1; i++) {
        const f0 = out[i][0], f1 = out[i + 1][0];
        if (top !== null && f1 === OPPOSITE_FACE[f0] && f1 === top) {
            const tmp = out[i];
            out[i] = out[i + 1];
            out[i + 1] = tmp;
        }
        top = out[i][0];
    }
    return out;
}

function invertMoves(moves) {
    const out = new Array(moves.length);
    for (let i = 0; i < moves.length; i++) {
        const mv = moves[moves.length - 1 - i];
        if (mv.endsWith('2')) out[i] = mv;
        else if (mv.endsWith("'")) out[i] = mv[0];
        else out[i] = mv + "'";
    }
    return out;
}

function solveSingleAxis(startCubie, { maxTotalMoves = 22, timeoutMs = 25, startTopLetter = null, invFace = null, isInverse = false } = {}) {
    const T = initTables();

    const twist0 = getTwist(startCubie);
    const flip0  = getFlip(startCubie);
    const slice0 = getSlice(startCubie);
    if (twist0 === 0 && flip0 === 0 && slice0 === 0 &&
        getCP(startCubie) === 0 && getUDEP(startCubie) === 0 && getSlicePerm(startCubie) === 0) {
        return { sol: [], cost: 0 };
    }

    const tStart = performance.now();
    let bestSol = null;
    let bestCost = 1e9;
    let bestQTM = 999;
    let bestHTM = maxTotalMoves + 1;
    const p1Moves = new Int8Array(32);
    const p2Moves = new Int8Array(32);

    function searchP2(cp, udep, sp, depth, maxDepth, prevFace) {
        const h1 = T.pruneCPSP[cp * 24 + sp];
        const h2 = T.pruneUDEPSP[udep * 24 + sp];
        const h = h1 > h2 ? h1 : h2;
        if (h > maxDepth - depth) return false;
        if (h === 0) return true;

        const cpOff = cp * 10, udOff = udep * 10, spOff = sp * 10;
        for (let m = 0; m < 10; m++) {
            const fullMove = P2_MOVES[m];
            const face = (fullMove / 3) | 0;
            if (skipFace(prevFace, face)) continue;
            p2Moves[depth] = fullMove;
            if (searchP2(T.cpMove[cpOff + m], T.udepMove[udOff + m], T.spMove[spOff + m],
                         depth + 1, maxDepth, face)) {
                return true;
            }
        }
        return false;
    }

    function tryPhase2(d1) {
        let p1Q = 0;
        let c = startCubie;
        for (let i = 0; i < d1; i++) {
            const mv = p1Moves[i];
            p1Q += (mv % 3 === 1) ? 2 : 1;
            c = mulCubie(c, MOVES18[mv]);
        }
        if (d1 > bestHTM) return false;

        const cp = getCP(c), udep = getUDEP(c), sp = getSlicePerm(c);
        const h1 = T.pruneCPSP[cp * 24 + sp];
        const h2 = T.pruneUDEPSP[udep * 24 + sp];
        const minP2 = h1 > h2 ? h1 : h2;
        // Allow up to bestHTM (not pruned by QTM) so X2-rich / zero-180-flip low-reorient plans are never discarded
        const maxP2 = Math.min(maxTotalMoves - d1, bestHTM - d1);
        if (minP2 > maxP2) return false;

        const lastFace = d1 > 0 ? ((p1Moves[d1 - 1] / 3) | 0) : -1;
        for (let d2 = minP2; d2 <= maxP2; d2++) {
            if (searchP2(cp, udep, sp, 0, d2, lastFace)) {
                let p2Q = 0;
                let cand = [];
                for (let i = 0; i < d1; i++) {
                    const nm = MOVE_NAMES[p1Moves[i]];
                    cand.push(invFace ? invFace[nm[0]] + nm.slice(1) : nm);
                }
                for (let i = 0; i < d2; i++) {
                    p2Q += (p2Moves[i] % 3 === 1) ? 2 : 1;
                    const nm = MOVE_NAMES[p2Moves[i]];
                    cand.push(invFace ? invFace[nm[0]] + nm.slice(1) : nm);
                }
                if (isInverse) cand = invertMoves(cand);
                // Reorder adjacent commuting opposite-face pairs (e.g. startTop=U, [D, U] -> [U, D]) to avoid a 180 flip!
                cand = reorderCommutingMoves(cand, startTopLetter);
                const totalQ = p1Q + p2Q;
                const cost = planRobotCost(cand, startTopLetter);
                if (cost < bestCost - 1e-6 || (Math.abs(cost - bestCost) <= 1e-6 && totalQ < bestQTM)) {
                    bestCost = cost;
                    bestQTM = totalQ;
                    bestHTM = d1 + d2;
                    bestSol = cand;
                }
                return true;
            }
        }
        return false;
    }

    let nodes = 0;
    let timedOut = false;

    function searchP1(twist, flip, slice, depth, maxDepth, prevFace) {
        if (timedOut) return false;
        if ((++nodes & 2047) === 0 && bestSol && performance.now() - tStart > timeoutMs) {
            timedOut = true;
            return false;
        }
        const h1 = T.pruneTwistSlice[twist * 495 + slice];
        const h2 = T.pruneFlipSlice[flip * 495 + slice];
        const h = h1 > h2 ? h1 : h2;
        if (h > maxDepth - depth) return false;
        if (depth === maxDepth) {
            if (h !== 0) return false;
            tryPhase2(maxDepth);
            return false;
        }
        if (depth > 0 && h === 0) return false;

        const tOff = twist * 18, fOff = flip * 18, sOff = slice * 18;
        for (let m = 0; m < 18; m++) {
            const face = (m / 3) | 0;
            if (skipFace(prevFace, face)) continue;
            p1Moves[depth] = m;
            searchP1(T.twistMove[tOff + m], T.flipMove[fOff + m], T.sliceMove[sOff + m],
                     depth + 1, maxDepth, face);
            if (timedOut) return false;
        }
        return false;
    }

    const minP1 = Math.max(
        T.pruneTwistSlice[twist0 * 495 + slice0],
        T.pruneFlipSlice[flip0 * 495 + slice0]
    );
    for (let d1 = minP1; d1 <= 12; d1++) {
        if (d1 > bestHTM) break;
        searchP1(twist0, flip0, slice0, 0, d1, -1);
        if (bestSol && (d1 >= bestHTM || performance.now() - tStart > timeoutMs)) break;
    }

    if (!bestSol) {
        maxTotalMoves = 35;
        bestHTM = 36;
        timedOut = false;
        for (let d1 = minP1; d1 <= 14; d1++) {
            searchP1(twist0, flip0, slice0, 0, d1, -1);
            if (bestSol) break;
        }
    }
    return { sol: bestSol || [], cost: bestCost };
}

// Solve a 54-char URFDLB cube string across all 6 cosets (3 symmetry axes × forward/inverse),
// returning the lowest robot-cost Singmaster move sequence (fewest flips, 180-deg flips & quarter turns).
export function solveCube(faceletStr, { maxTotalMoves = 22, timeoutMs = 120, startTopLetter = null } = {}) {
    const perDirMs = Math.max(12, Math.floor(timeoutMs / 6));
    const c0 = faceletsToCubie(faceletStr);
    let best = solveSingleAxis(c0, { maxTotalMoves, timeoutMs: perDirMs, startTopLetter, invFace: null, isInverse: false });
    if (best.sol.length <= 5) return best.sol;

    // Inverse on axis 0
    const resInv0 = solveSingleAxis(invCubie(c0), { maxTotalMoves, timeoutMs: perDirMs, startTopLetter, invFace: null, isInverse: true });
    if (resInv0.sol.length > 0 && resInv0.cost < best.cost) best = resInv0;

    // Axes 1 and 2 (both forward and inverse)
    for (let k = 1; k <= 2; k++) {
        const { fwdFace, invFace, srcIdx } = AXIS_CONJ[k];
        let rotStr = '';
        for (let i = 0; i < 54; i++) rotStr += fwdFace[faceletStr[srcIdx[i]]];
        const ck = faceletsToCubie(rotStr);
        const resFwd = solveSingleAxis(ck, { maxTotalMoves, timeoutMs: perDirMs, startTopLetter, invFace, isInverse: false });
        if (resFwd.sol.length > 0 && resFwd.cost < best.cost) best = resFwd;
        const resInv = solveSingleAxis(invCubie(ck), { maxTotalMoves, timeoutMs: perDirMs, startTopLetter, invFace, isInverse: true });
        if (resInv.sol.length > 0 && resInv.cost < best.cost) best = resInv;
    }
    return best.sol;
}

// Expand half-turns ("U2" -> ["U", "U"]) into atomic 90-degree moves for the robot hand
export function expandQuarterTurns(moves) {
    const out = [];
    for (const mv of moves) {
        if (mv.endsWith('2')) {
            out.push(mv[0], mv[0]);
        } else {
            out.push(mv);
        }
    }
    return out;
}
