// Army ant bridge simulation. Units are cm and seconds, y is up, and the forked
// twig lies flat with its axis in the y = 0 plane. Shared by the page (main.js)
// and the measurement tools (tools/), so both run the same model.
//
// Nothing here knows what a bridge is. Each ant follows local rules taken from
// field studies of Eciton army ants:
//   1. Walk the shortest way over whatever footing there is (this stands in for the
//      pheromone trail and the ants' sense of direction).
//   2. If your footing is poor (your body is stretched across a gap, or you have
//      walked off the edge of the bridge), stop and hold on for a moment. Each ant
//      that walks over you may lock you in place, less readily if the footing sags
//      (Garnier et al. 2013; Lutz et al. 2021; Reid et al. 2015).
//   3. Once locked, let go when ants stop walking over you, sooner if few ants hold
//      you, but never while others hang from you (Garnier et al. 2013).

export const CASTES = [
    // share of the column, body length (cm), tendency to lock into a bridge, chance of carrying brood
    { name: 'minor', share: 0.22, length: 0.46, join: 1.3, carry: 0.15 },
    { name: 'media', share: 0.745, length: 0.62, join: 1.0, carry: 0.35 },
    { name: 'submajor', share: 0.025, length: 0.86, join: 0.0, carry: 0.85 },
    { name: 'major', share: 0.01, length: 1.0, join: 0.0, carry: 0.0 },
];

export const PARAMS = {
    armLength: 18,       // crotch to the end of each tine
    twigRadius: 0.5,
    cell: 0.125,         // footing grid
    speed: 8,            // Eciton run at about 8 cm/s (Reid et al. 2015)
    speedSD: 0.12,
    reach: 0.36,         // how far an ant's centre can be over air and still hold on at both ends
    reachCost: 4,        // a walker takes a step over air only if it saves this much distance
    stretchSpeed: 1.2,   // walking speed with the body over air
    laneBias: 0.4,       // how strongly each ant keeps to its own lane
    barkMargin: 0.12,    // within this of bark an ant has good footing, even at the bridge's edge
    holdTime: 1.5,       // how long an ant with poor footing waits to be walked over
    lockChance: 0.7,     // per ant that walks over a holding ant (times the caste's tendency)
    trafficTau: 5,       // traffic over an ant's back is felt over about 5 s
    leaveRate: 0.6,      // per second, for a bridge ant with no traffic and few neighbours
    leaveTraffic: 5.0,   // leaving falls as exp(-leaveTraffic * ants crossing per second)
    leaveNeighbors: 0.25, // and with each ant linked to it beyond two
    footReach: 0.6,      // a walker's feet land this far from its centre (times its length)
    legReach: 0.5,       // and touch a bridge ant whose body or legs are within this (times its length)
    gripReach: 0.48,     // a locking ant grips bark or bodies this close to its head and tail
    navInterval: 0.4,    // how often the shortest paths are updated (s)
    sag: 0.008,          // gravity scale for the bridge: ants hold their structure stiffer than a rope
    gripLevel: 0.36,     // top of a bridge ant gripping the side of the twig
    sagTolerance: 0.035, // locking falls as exp(-sag / sagTolerance)
    dt: 1 / 120,
};

const G = 981;

export function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const AIR = 0, REACH = 1, SUPPORT = 2;

// 16-neighbourhood for the path search; knight moves keep paths from bending to 45°.
const NB = [];
for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1],
    [1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]]) {
    NB.push({ dx, dz, len: Math.hypot(dx, dz) });
}

export class Sim {
    constructor({ angle = 20, traffic = 150, seed = 1, params = {} } = {}) {
        this.p = { ...PARAMS, ...params };
        this.angle = angle;
        this.traffic = traffic;          // ants per minute, both directions together
        this.rng = mulberry32(seed);
        this.t = 0;
        this.nextId = 1;
        this.walkers = [];
        this.bridge = [];
        this.antById = new Map();
        this.constraints = [];
        this.arrivals = 0;
        this.joins = 0;
        this.leaves = 0;
        this.falls = 0;
        this.spawnDebt = [0, 0];
        this.navTimer = 0;
        this.buildFork();
        this.buildGrid();
        this.updateDynamic();
        this.updateNav();
        this.baseline = this.pathLength();
    }

    // --- geometry

    buildFork() {
        const p = this.p, half = this.angle / 2 * Math.PI / 180, L = p.armLength;
        this.half = half;
        this.crotch = { x: 0, z: L * Math.cos(half) };
        const r = p.twigRadius;
        const tine = (sx) => {
            const dx = sx * Math.sin(half), dz = -Math.cos(half);
            return { ax: 0, az: this.crotch.z, dx, dz, len: L + 2, r, tip: L };
        };
        // tine 0 runs to the left end, tine 1 to the right; the stem continues past the crotch
        this.twigs = [tine(-1), tine(1), { ax: 0, az: this.crotch.z, dx: 0, dz: 1, len: 5, r: r * 1.3, tip: Infinity }];
        this.walkHalf = 0.9;   // footing reaches 0.9 r out from the axis, onto the twig's sides
        // where the inner edges of the two tines meet on the centre line
        this.innerCorner = this.crotch.z - this.walkHalf * r / Math.sin(half);
    }

    // nearest twig to (x, z): index, distance from axis, distance along
    twigAt(x, z) {
        let best = -1, bu = Infinity, bs = 0;
        for (let k = 0; k < this.twigs.length; k++) {
            const t = this.twigs[k];
            const s = Math.max(0, Math.min(t.len, (x - t.ax) * t.dx + (z - t.az) * t.dz));
            const u = Math.hypot(x - t.ax - s * t.dx, z - t.az - s * t.dz) / t.r;
            if (u < bu) { bu = u; best = k; bs = s; }
        }
        return { k: best, u: bu, s: bs };
    }

    // top of the twigs at (x, z), or -Infinity off them
    twigTop(x, z) {
        let y = -Infinity;
        for (const t of this.twigs) {
            const s = Math.max(0, Math.min(t.len, (x - t.ax) * t.dx + (z - t.az) * t.dz));
            const u2 = ((x - t.ax - s * t.dx) ** 2 + (z - t.az - s * t.dz) ** 2);
            if (u2 < t.r * t.r) y = Math.max(y, Math.sqrt(t.r * t.r - u2));
        }
        return y;
    }

    buildGrid() {
        const p = this.p, h = p.cell;
        const xs = this.twigs[1].ax + this.twigs[1].dx * this.twigs[1].tip;
        this.x0 = -xs - 2; this.z0 = -1.5;
        this.nx = Math.ceil((2 * xs + 4) / h);
        this.nz = Math.ceil((this.crotch.z + 3.5 - this.z0) / h);
        const n = this.nx * this.nz;
        this.staticType = new Uint8Array(n);
        this.tineOf = new Int8Array(n).fill(-1);
        this.alongOf = new Float32Array(n);
        for (let j = 0; j < this.nz; j++) {
            for (let i = 0; i < this.nx; i++) {
                const c = j * this.nx + i, x = this.x0 + (i + 0.5) * h, z = this.z0 + (j + 0.5) * h;
                const tw = this.twigAt(x, z);
                if (tw.u <= this.walkHalf) {
                    this.staticType[c] = SUPPORT;
                    this.tineOf[c] = tw.k; this.alongOf[c] = tw.s;
                }
            }
        }
        // distance from each cell to bark, for telling a bark edge from an edge of the bridge
        this.barkDist = new Float32Array(n);
        this.chamfer(this.barkDist, (c) => this.staticType[c] === SUPPORT);
        this.type = new Uint8Array(n);
        this.dist = new Float32Array(n);      // distance to footing, for cells over air
        this.owner = new Int32Array(n);       // bridge ant under each cell
        this.dynTop = new Float32Array(n);
        this.nav = [new Float32Array(n), new Float32Array(n)];   // distance to the end of tine 0 / tine 1
        this.heapKey = new Float32Array(n * 8);
        this.heapIdx = new Int32Array(n * 8);
    }

    cellOf(x, z) {
        const i = Math.floor((x - this.x0) / this.p.cell), j = Math.floor((z - this.z0) / this.p.cell);
        if (i < 0 || j < 0 || i >= this.nx || j >= this.nz) return -1;
        return j * this.nx + i;
    }

    // Stamp the bridge ants into the footing grid. Each covers a capsule around
    // its body that includes its spread legs.
    updateDynamic() {
        const { nx, nz, p } = this, h = p.cell;
        this.owner.fill(-1);
        this.dynTop.fill(-Infinity);
        this.bridgeIndex = new Map();
        this.bridge.forEach((a, k) => this.bridgeIndex.set(a.id, k));
        for (let k = 0; k < this.bridge.length; k++) {
            const a = this.bridge[k];
            const P = this.px, F = a.f, B = a.b;
            const fx = P[F * 3], fy = P[F * 3 + 1], fz = P[F * 3 + 2];
            const bx = P[B * 3], by = P[B * 3 + 1], bz = P[B * 3 + 2];
            const rad = 0.42 * a.size, top = 0.09 * a.size;
            const i0 = Math.max(0, Math.floor((Math.min(fx, bx) - rad - this.x0) / h));
            const i1 = Math.min(nx - 1, Math.floor((Math.max(fx, bx) + rad - this.x0) / h));
            const j0 = Math.max(0, Math.floor((Math.min(fz, bz) - rad - this.z0) / h));
            const j1 = Math.min(nz - 1, Math.floor((Math.max(fz, bz) + rad - this.z0) / h));
            const ex = fx - bx, ez = fz - bz, el = ex * ex + ez * ez || 1e-9;
            for (let j = j0; j <= j1; j++) {
                const z = this.z0 + (j + 0.5) * h;
                for (let i = i0; i <= i1; i++) {
                    const x = this.x0 + (i + 0.5) * h;
                    const t = Math.max(0, Math.min(1, ((x - bx) * ex + (z - bz) * ez) / el));
                    const dx = x - bx - t * ex, dz = z - bz - t * ez, d2 = dx * dx + dz * dz;
                    if (d2 > rad * rad) continue;
                    const c = j * nx + i;
                    const y = by + t * (fy - by) + top * Math.sqrt(1 - d2 / (rad * rad));
                    if (y > this.dynTop[c]) { this.dynTop[c] = y; this.owner[c] = k; }
                }
            }
        }
        // the legs linking two bridge ants are footing too
        for (const c of this.constraints) {
            if (c.body || c.j < 0) continue;
            const k = this.bridgeIndex.get(c.owner);
            if (k === undefined) continue;
            const P = this.px, i = c.i * 3, j = c.j * 3, rad = 0.13;
            const ax = P[i], ay = P[i + 1], az = P[i + 2], ex = P[j] - ax, ey = P[j + 1] - ay, ez = P[j + 2] - az;
            const el = ex * ex + ez * ez || 1e-9;
            const i0 = Math.max(0, Math.floor((Math.min(ax, ax + ex) - rad - this.x0) / h));
            const i1 = Math.min(nx - 1, Math.floor((Math.max(ax, ax + ex) + rad - this.x0) / h));
            const j0 = Math.max(0, Math.floor((Math.min(az, az + ez) - rad - this.z0) / h));
            const j1 = Math.min(nz - 1, Math.floor((Math.max(az, az + ez) + rad - this.z0) / h));
            for (let jj = j0; jj <= j1; jj++) {
                const z = this.z0 + (jj + 0.5) * h;
                for (let ii = i0; ii <= i1; ii++) {
                    const x = this.x0 + (ii + 0.5) * h;
                    const t = Math.max(0, Math.min(1, ((x - ax) * ex + (z - az) * ez) / el));
                    if ((x - ax - t * ex) ** 2 + (z - az - t * ez) ** 2 > rad * rad) continue;
                    const cc = jj * nx + ii, y = ay + t * ey;
                    if (this.owner[cc] < 0 || y > this.dynTop[cc]) { if (this.owner[cc] < 0) this.owner[cc] = k; this.dynTop[cc] = Math.max(this.dynTop[cc], y); }
                }
            }
        }
        // footing type, and distance to footing over air
        const n = nx * nz, D = this.dist;
        this.chamfer(D, (c) => this.staticType[c] === SUPPORT || this.owner[c] >= 0);
        const reach = p.reach;
        for (let c = 0; c < n; c++) this.type[c] = D[c] === 0 ? SUPPORT : D[c] <= reach ? REACH : AIR;
    }

    // distance to the nearest cell where isSource(c), by a two-pass chamfer (exact enough at this range)
    chamfer(D, isSource) {
        const { nx, nz } = this, h = this.p.cell, n = nx * nz, BIG = 1e9;
        for (let c = 0; c < n; c++) D[c] = isSource(c) ? 0 : BIG;
        const d1 = h, d2 = h * Math.SQRT2;
        for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
            const c = j * nx + i; if (D[c] === 0) continue;
            let v = D[c];
            if (i > 0) v = Math.min(v, D[c - 1] + d1);
            if (j > 0) {
                v = Math.min(v, D[c - nx] + d1);
                if (i > 0) v = Math.min(v, D[c - nx - 1] + d2);
                if (i < nx - 1) v = Math.min(v, D[c - nx + 1] + d2);
            }
            D[c] = v;
        }
        for (let j = nz - 1; j >= 0; j--) for (let i = nx - 1; i >= 0; i--) {
            const c = j * nx + i; if (D[c] === 0) continue;
            let v = D[c];
            if (i < nx - 1) v = Math.min(v, D[c + 1] + d1);
            if (j < nz - 1) {
                v = Math.min(v, D[c + nx] + d1);
                if (i < nx - 1) v = Math.min(v, D[c + nx + 1] + d2);
                if (i > 0) v = Math.min(v, D[c + nx - 1] + d2);
            }
            D[c] = v;
        }
    }

    // Shortest-path distance to the end of each tine over footing, with steps over air
    // costing more the farther they are from footing.
    updateNav() {
        const { nx, nz, p } = this, n = nx * nz, h = p.cell;
        const cost = this.costBuf || (this.costBuf = new Float32Array(n));
        for (let c = 0; c < n; c++) {
            const t = this.type[c];
            cost[c] = t === SUPPORT ? 1 : t === REACH ? 1 + p.reachCost * (this.dist[c] / p.reach) ** 2 : Infinity;
        }
        for (let target = 0; target < 2; target++) {
            const D = this.nav[target];
            D.fill(Infinity);
            let size = 0;
            const K = this.heapKey, I = this.heapIdx;
            const push = (key, idx) => {
                let i = size++;
                while (i > 0) {
                    const par = (i - 1) >> 1;
                    if (K[par] <= key) break;
                    K[i] = K[par]; I[i] = I[par]; i = par;
                }
                K[i] = key; I[i] = idx;
            };
            const pop = () => {
                const top = I[0], last = --size, lk = K[last], li = I[last];
                let i = 0;
                for (; ;) {
                    let ch = 2 * i + 1;
                    if (ch >= size) break;
                    if (ch + 1 < size && K[ch + 1] < K[ch]) ch++;
                    if (K[ch] >= lk) break;
                    K[i] = K[ch]; I[i] = I[ch]; i = ch;
                }
                K[i] = lk; I[i] = li;
                return top;
            };
            const tip = this.twigs[target].tip;
            const done = this.doneBuf || (this.doneBuf = new Uint8Array(n));
            done.fill(0);
            for (let c = 0; c < n; c++) {
                if (this.tineOf[c] === target && this.alongOf[c] >= tip - 0.1) { D[c] = 0; push(0, c); }
            }
            while (size > 0) {
                const key = K[0], c = pop();
                if (done[c]) continue;
                done[c] = 1;
                const i = c % nx, j = (c - i) / nx;
                for (const nb of NB) {
                    const ii = i + nb.dx, jj = j + nb.dz;
                    if (ii < 0 || jj < 0 || ii >= nx || jj >= nz) continue;
                    const cc = jj * nx + ii;
                    if (cost[cc] === Infinity) continue;
                    if (nb.len > 2) {
                        // knight move: the cell it passes over must be passable too
                        const mi = i + Math.round(nb.dx / 2), mj = j + Math.round(nb.dz / 2);
                        if (cost[mj * nx + mi] === Infinity) continue;
                    }
                    const nd = key + nb.len * h * 0.5 * (cost[c] + cost[cc]);
                    if (!done[cc] && nd < D[cc] && size < K.length) { D[cc] = nd; push(nd, cc); }
                }
            }
        }
    }

    navAt(target, x, z) {
        const c = this.cellOf(x, z);
        return c < 0 ? Infinity : this.nav[target][c];
    }

    // walking distance from one end to the other, as the ants would take it
    pathLength() {
        const t = this.twigs[0], s = t.tip - 0.3;
        return this.navAt(1, t.ax + t.dx * s, t.az + t.dz * s) + (t.tip - s);
    }

    // surface the ants walk on at (x, z): top of the twig or of the bridge ants
    groundAt(x, z) {
        const c = this.cellOf(x, z);
        return Math.max(this.twigTop(x, z), c >= 0 ? this.dynTop[c] : -Infinity);
    }

    // --- ants

    newCaste() {
        let u = this.rng();
        for (let k = 0; k < CASTES.length; k++) { u -= CASTES[k].share; if (u <= 0) return k; }
        return 1;
    }

    spawn(from) {
        const t = this.twigs[from], s = t.tip - 0.15;
        const lat = (this.rng() - 0.5) * 0.5;
        const x = t.ax + t.dx * s - t.dz * lat, z = t.az + t.dz * s + t.dx * lat;
        for (const w of this.walkers) if ((w.x - x) ** 2 + (w.z - z) ** 2 < 0.35 * 0.35) return false;
        const caste = this.newCaste(), c = CASTES[caste];
        const size = c.length * (1 + (this.rng() - 0.5) * 0.16);
        // tine 1's end leads to the raid, tine 0's to the bivouac: ants heading home carry brood
        const carry = from === 1 && this.rng() < c.carry;
        const w = {
            id: this.nextId++, x, z, y: this.twigTop(x, z) + 0.13 * size,
            hd: Math.atan2(-t.dz, -t.dx), v: 0,
            target: 1 - from, caste, size, carry,
            vPref: this.p.speed * (1 + this.p.speedSD * this.gauss()) * (caste === 3 ? 0.85 : 1),
            state: 0, holdT: 0, trampled: null, phase: this.rng(), stuck: 0,
            age: 0, lane: this.gauss() * 0.5,
        };
        this.walkers.push(w);
        return true;
    }

    gauss() {
        let u = -6;
        for (let i = 0; i < 12; i++) u += this.rng();
        return u;
    }

    // direction that lowers the path distance fastest, softly averaged over a ring of samples
    navDir(w) {
        const D = this.nav[w.target], r = 0.3;
        let best = Infinity;
        const vals = this._ring || (this._ring = new Float32Array(16));
        for (let k = 0; k < 16; k++) {
            const a = k * Math.PI / 8;
            const c = this.cellOf(w.x + Math.cos(a) * r, w.z + Math.sin(a) * r);
            const v = c < 0 ? Infinity : D[c];
            vals[k] = v; if (v < best) best = v;
        }
        if (best === Infinity) return null;
        let sx = 0, sz = 0;
        for (let k = 0; k < 16; k++) {
            if (vals[k] === Infinity) continue;
            const wt = Math.exp(-(vals[k] - best) / 0.04);
            const a = k * Math.PI / 8;
            sx += wt * Math.cos(a); sz += wt * Math.sin(a);
        }
        return { x: sx, z: sz };
    }

    // how far the footing under a holding ant's head and tail hangs below where bridge ants
    // grip the bark: a long or loaded span sags, and ants are less willing to lock onto it
    sagUnder(w) {
        const r = 0.5 * w.size, cx = Math.cos(w.hd) * r, cz = Math.sin(w.hd) * r;
        const a = this.groundAt(w.x + cx, w.z + cz), b = this.groundAt(w.x - cx, w.z - cz);
        const low = a === -Infinity ? b : b === -Infinity ? a : Math.min(a, b);
        return low === -Infinity ? 0 : Math.max(0, this.p.gripLevel - low);
    }

    // Footing is poor when the ant's body is over air and it either spans a gap with its head
    // and tail on footing, or is half off the edge of the living bridge, away from the bark
    // (a bark edge is easy to hold; an edge made of moving ants is not).
    poorFooting(w) {
        const c = this.cellOf(w.x, w.z);
        if (c < 0 || this.type[c] !== REACH) return false;
        const r = 0.5 * w.size, cx = Math.cos(w.hd) * r, cz = Math.sin(w.hd) * r;
        const head = this.cellOf(w.x + cx, w.z + cz), tail = this.cellOf(w.x - cx, w.z - cz);
        if (head < 0 || tail < 0) return false;
        // walked off the edge of the bridge (its back end still stands on bridge ants), away from the bark
        if (this.owner[tail] >= 0 && this.barkDist[c] > this.p.barkMargin) return true;
        return this.type[head] === SUPPORT && this.type[tail] === SUPPORT && this.dist[c] >= 0.2;
    }

    stepWalkers(dt) {
        const p = this.p, W = this.walkers;
        for (let a = 0; a < W.length; a++) {
            const w = W[a];
            w.age += dt;
            const c = this.cellOf(w.x, w.z);
            const t = c >= 0 ? this.type[c] : AIR;

            if (w.state === 1) {
                // holding over air: ants that walk over it may lock it in place
                w.holdT += dt; w.v = 0;
                for (const o of W) {
                    if (o === w || o.state !== 0 || w.trampled.has(o.id)) continue;
                    const d2 = (o.x - w.x) ** 2 + (o.z - w.z) ** 2, r = 0.32 * w.size;
                    if (d2 < r * r) {
                        w.trampled.add(o.id);
                        if (this.rng() < p.lockChance * CASTES[w.caste].join * Math.exp(-this.sagUnder(w) / p.sagTolerance)) { w.lock = true; break; }
                    }
                }
                if (w.holdT > p.holdTime) { w.state = 0; w.cooldown = 0.6; }
                continue;
            }

            const nd = this.navDir(w);
            let dx = Math.cos(w.hd), dz = Math.sin(w.hd);
            if (nd) {
                // each ant keeps to its own lane a little: columns spread over the footing
                const nl = Math.hypot(nd.x, nd.z) || 1;
                let tx = nd.x / nl - nd.z / nl * w.lane * p.laneBias, tz = nd.z / nl + nd.x / nl * w.lane * p.laneBias;
                // keep apart from oncoming ants a little; Eciton pass by stepping aside or over
                for (const o of W) {
                    if (o === w) continue;
                    const ox = w.x - o.x, oz = w.z - o.z, d2 = ox * ox + oz * oz;
                    if (d2 > 0.5 * 0.5 || d2 < 1e-8) continue;
                    const d = Math.sqrt(d2), f = (0.5 - d) / 0.5 * 0.6;
                    tx += ox / d * f; tz += oz / d * f;
                }
                const ang = Math.atan2(tz, tx) + (this.rng() - 0.5) * 0.6;
                let diff = ang - w.hd;
                diff -= Math.round(diff / (2 * Math.PI)) * 2 * Math.PI;
                const maxTurn = 14 * dt;
                w.hd += Math.max(-maxTurn, Math.min(maxTurn, diff));
                dx = Math.cos(w.hd); dz = Math.sin(w.hd);
            }

            // speed: slow when stretched over air or right behind a slower ant
            let vt = w.vPref;
            if (t === REACH) vt = p.stretchSpeed;
            for (const o of W) {
                if (o === w) continue;
                const ox = o.x - w.x, oz = o.z - w.z, ahead = ox * dx + oz * dz;
                if (ahead <= 0 || ahead > 0.5) continue;
                if (Math.abs(ox * dz - oz * dx) < 0.18) vt = Math.min(vt, Math.max(o.v, w.vPref * 0.45));
            }
            w.v += (vt - w.v) * Math.min(1, dt * 12);

            const nx = w.x + dx * w.v * dt, nz = w.z + dz * w.v * dt;
            const nc = this.cellOf(nx, nz);
            if (nc >= 0 && this.type[nc] !== AIR) {
                w.x = nx; w.z = nz; w.phase += w.v * dt / (0.55 * w.size); w.stuck = 0;
            } else {
                w.v *= 0.3; w.stuck += dt;
                w.hd += (this.rng() - 0.5) * 2;
            }
            if (w.cooldown > 0) w.cooldown -= dt;

            // rule 2: an ant with poor footing holds still and waits to be walked over
            if (!(w.cooldown > 0) && w.state === 0 && this.poorFooting(w)) {
                // porters, soldiers and ants carrying brood keep going; the others stop and hold on
                if (CASTES[w.caste].join > 0 && !w.carry) { w.state = 1; w.holdT = 0; w.trampled = new Set(); w.v = 0; }
                else w.cooldown = 0.6;
            }

            // traffic: each bridge ant this ant's feet land on feels it pass, once per pass
            const oc = this.cellOf(w.x, w.z), own = oc >= 0 ? this.owner[oc] : -1;
            const under = own >= 0 ? this.bridge[own] : null;
            if (under) under.load += Math.pow(w.size / 0.62, 3) * (w.carry ? 1.6 : 1);
            if (under || w.touching) {
                const now = new Set(), reach = p.footReach * w.size, P = this.px;
                if (under) {
                    for (const a of this.bridge) {
                        const mx = (P[a.f * 3] + P[a.b * 3]) / 2, mz = (P[a.f * 3 + 2] + P[a.b * 3 + 2]) / 2;
                        const r = reach + p.legReach * a.size;
                        if ((mx - w.x) ** 2 + (mz - w.z) ** 2 < r * r) {
                            now.add(a.id);
                            if (!w.touching || !w.touching.has(a.id)) a.traffic += 1;
                        }
                    }
                }
                w.touching = now.size ? now : null;
            }

            // height: follow the surface underfoot
            const g = this.groundAt(w.x, w.z);
            if (g > -Infinity) {
                const target = g + 0.13 * w.size;
                w.y += (target - w.y) * Math.min(1, dt * 25);
            }
        }

        // arrivals and locks
        for (let a = W.length - 1; a >= 0; a--) {
            const w = W[a];
            if (w.lock) { W.splice(a, 1); this.lockIn(w); continue; }
            const tw = this.twigs[w.target];
            const s = (w.x - tw.ax) * tw.dx + (w.z - tw.az) * tw.dz;
            if (s >= tw.tip && Math.abs((w.x - tw.ax) * tw.dz - (w.z - tw.az) * tw.dx) < tw.r * 1.5) {
                W.splice(a, 1); this.arrivals++;
            } else if (w.stuck > 3 || w.age > 120) {
                W.splice(a, 1);
            }
        }
    }

    // --- the bridge: ants as two-particle bodies held by their legs

    ensureParticles() {
        if (this.px) return;
        const N = 4096;
        this.px = new Float32Array(N * 3); this.po = new Float32Array(N * 3);
        this.freeParticles = [];
        for (let i = N - 1; i >= 0; i--) this.freeParticles.push(i);
    }

    lockIn(w) {
        this.ensureParticles();
        const L = 0.88 * w.size, cx = Math.cos(w.hd), cz = Math.sin(w.hd);
        const f = this.freeParticles.pop(), b = this.freeParticles.pop();
        const y = w.y - 0.1 * w.size;
        const set = (i, x, yy, z) => {
            this.px[i * 3] = x; this.px[i * 3 + 1] = yy; this.px[i * 3 + 2] = z;
            this.po[i * 3] = x; this.po[i * 3 + 1] = yy; this.po[i * 3 + 2] = z;
        };
        set(f, w.x + cx * L / 2, y, w.z + cz * L / 2);
        set(b, w.x - cx * L / 2, y, w.z - cz * L / 2);
        const a = {
            id: w.id, f, b, size: w.size, caste: w.caste, len: L, target: w.target,
            traffic: 0, rate: 0, load: 0, grips: [], links: 0, roll: (this.rng() - 0.5) * 0.8,
            seed: this.rng(),
        };
        this.constraints.push({ i: f, j: b, rest: L, body: true, owner: a.id, other: a.id });
        // grip whatever is within reach of the front and back legs: twig bark or other ants
        for (const pi of [f, b]) {
            const x = this.px[pi * 3], yy = this.px[pi * 3 + 1], z = this.px[pi * 3 + 2];
            const cands = [];
            for (const tw of this.twigs) {
                const s = Math.max(0, Math.min(tw.len, (x - tw.ax) * tw.dx + (z - tw.az) * tw.dz));
                const axx = tw.ax + s * tw.dx, axz = tw.az + s * tw.dz;
                let ox = x - axx, oy = yy, oz = z - axz;
                const ol = Math.hypot(ox, oy, oz) || 1;
                const gx = axx + ox / ol * tw.r, gy = oy / ol * tw.r, gz = axz + oz / ol * tw.r;
                const d = Math.hypot(x - gx, yy - gy, z - gz);
                if (d < this.p.gripReach) cands.push({ d, anchor: [gx, gy, gz] });
            }
            for (const o of this.bridge) {
                // legs hook anywhere on the other ant's body; the pull goes to its nearer end
                const P = this.px, F = o.f * 3, B = o.b * 3;
                const ex = P[F] - P[B], ey = P[F + 1] - P[B + 1], ez = P[F + 2] - P[B + 2];
                const el = ex * ex + ey * ey + ez * ez || 1e-9;
                const t = Math.max(0, Math.min(1, ((x - P[B]) * ex + (yy - P[B + 1]) * ey + (z - P[B + 2]) * ez) / el));
                const d = Math.hypot(x - P[B] - t * ex, yy - P[B + 1] - t * ey, z - P[B + 2] - t * ez);
                if (d < this.p.gripReach) {
                    const q = t > 0.5 ? o.f : o.b;
                    cands.push({ d, q, other: o.id, rest: Math.hypot(x - P[q * 3], yy - P[q * 3 + 1], z - P[q * 3 + 2]) });
                }
            }
            cands.sort((u, v) => u.d - v.d);
            for (const cd of cands.slice(0, 3)) {
                const rest = Math.max(0.12, cd.rest ?? cd.d);
                if (cd.anchor) this.constraints.push({ i: pi, j: -1, anchor: cd.anchor, rest, owner: a.id, other: 0 });
                else this.constraints.push({ i: pi, j: cd.q, rest, owner: a.id, other: cd.other });
            }
        }
        this.bridge.push(a);
        this.antById.set(a.id, a);
        this.relink();
        if (!this.grounded().has(a.id)) {
            // nothing within reach to hold: it carries on walking instead
            this.bridge.pop(); this.antById.delete(a.id);
            this.constraints = this.constraints.filter(c => c.owner !== a.id);
            this.freeParticles.push(f, b);
            this.relink();
            w.state = 0; w.lock = false; w.cooldown = 0.6;
            this.walkers.push(w);
            return;
        }
        this.joins++;
        this.updateDynamic();
    }

    // which particles each ant's legs hold, for the renderer and the neighbour counts
    relink() {
        for (const a of this.bridge) { a.grips = []; a.links = 0; a.anchored = false; }
        for (const c of this.constraints) {
            if (c.body) continue;
            const a = this.antById.get(c.owner);
            if (!a) continue;
            if (c.j < 0) { a.grips.push({ from: c.i, anchor: c.anchor }); a.anchored = true; }
            else {
                a.grips.push({ from: c.i, to: c.j });
                a.links++;
                const o = this.antById.get(c.other);
                if (o) { o.links++; o.grips.push({ from: c.j, to: c.i, passive: true }); }
            }
        }
    }

    // ants reachable from the bark through other ants' legs, optionally ignoring one ant
    grounded(without = null) {
        const adj = new Map();
        for (const o of this.bridge) adj.set(o.id, []);
        for (const c of this.constraints) {
            if (c.body || c.j < 0) continue;
            if (without && (c.owner === without.id || c.other === without.id)) continue;
            adj.get(c.owner)?.push(c.other);
            adj.get(c.other)?.push(c.owner);
        }
        const seen = new Set(), queue = [];
        for (const o of this.bridge) if (o !== without && o.anchored) { seen.add(o.id); queue.push(o.id); }
        while (queue.length) {
            const id = queue.pop();
            for (const nb of adj.get(id)) if (!seen.has(nb)) { seen.add(nb); queue.push(nb); }
        }
        return seen;
    }

    // Can this ant let go without leaving others hanging from nothing?
    canLeave(a) {
        return this.grounded(a).size === this.bridge.length - 1;
    }

    removeAnt(a) {
        const k = this.bridge.indexOf(a);
        if (k < 0) return;
        this.bridge.splice(k, 1);
        this.antById.delete(a.id);
        this.constraints = this.constraints.filter(c => c.owner !== a.id && c.other !== a.id);
        this.freeParticles.push(a.f, a.b);
        this.relink();
        this.updateDynamic();
    }

    leave(a) {
        const P = this.px;
        const x = (P[a.f * 3] + P[a.b * 3]) / 2, z = (P[a.f * 3 + 2] + P[a.b * 3 + 2]) / 2;
        const y = (P[a.f * 3 + 1] + P[a.b * 3 + 1]) / 2;
        this.removeAnt(a);
        this.leaves++;
        const hd = Math.atan2(P[a.f * 3 + 2] - P[a.b * 3 + 2], P[a.f * 3] - P[a.b * 3]);
        this.walkers.push({
            id: a.id, x, z, y: y + 0.1 * a.size, hd, v: 0, target: this.rng() < 0.5 ? 0 : 1,
            caste: a.caste, size: a.size, carry: false, vPref: this.p.speed * (1 + this.p.speedSD * this.gauss()),
            state: 0, holdT: 0, trampled: null, phase: this.rng(), stuck: 0, cooldown: 2, age: 0, lane: this.gauss() * 0.5,
        });
    }

    stepBridge(dt) {
        if (!this.bridge.length) return;
        const p = this.p, P = this.px, O = this.po;
        const decay = Math.exp(-dt / p.trafficTau);
        // rule 3: leaving falls with traffic over the ant's back and with its neighbours
        for (let k = this.bridge.length - 1; k >= 0; k--) {
            const a = this.bridge[k];
            a.rate = a.rate * decay + a.traffic / p.trafficTau;   // ants per second, smoothed over ~5 s
            a.traffic = 0;
            const rate = p.leaveRate * Math.exp(-p.leaveTraffic * a.rate) * Math.exp(-p.leaveNeighbors * Math.max(0, a.links - 2));
            if (this.rng() < rate * dt && this.canLeave(a)) this.leave(a);
        }
        if (!this.bridge.length) return;

        // position-based dynamics: gravity (own weight plus walkers on top), then constraints
        const damp = 0.96;
        for (const a of this.bridge) {
            const extra = a.load * 0.5;
            for (const i of [a.f, a.b]) {
                const ix = i * 3;
                for (let d = 0; d < 3; d++) {
                    const v = (P[ix + d] - O[ix + d]) * damp;
                    O[ix + d] = P[ix + d];
                    P[ix + d] += v;
                }
                P[ix + 1] -= G * p.sag * dt * dt * (1 + extra);
            }
            a.load = 0;
        }
        const C = this.constraints;
        for (let it = 0; it < 8; it++) {
            for (const c of C) {
                const i = c.i * 3;
                let jx, jy, jz, wj;
                if (c.j < 0) { [jx, jy, jz] = c.anchor; wj = 0; }
                else { const j = c.j * 3; jx = P[j]; jy = P[j + 1]; jz = P[j + 2]; wj = 1; }
                const dx = P[i] - jx, dy = P[i + 1] - jy, dz = P[i + 2] - jz;
                const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-9;
                let err = d - c.rest;
                if (!c.body && err < 0) err *= 0.1;   // legs resist stretching more than pushing
                const k = (c.body ? 1 : 0.7) * err / d / (1 + wj);
                P[i] -= dx * k; P[i + 1] -= dy * k; P[i + 2] -= dz * k;
                if (wj) { const j = c.j * 3; P[j] += dx * k; P[j + 1] += dy * k; P[j + 2] += dz * k; }
            }
            // ants hold their bodies roughly level; they do not hang head-down
            for (const a of this.bridge) {
                const fy = a.f * 3 + 1, by = a.b * 3 + 1, dy = (P[fy] - P[by]) * 0.08;
                P[fy] -= dy; P[by] += dy;
            }
            // keep bodies outside the bark
            for (const a of this.bridge) {
                for (const q of [a.f, a.b]) {
                    const i = q * 3;
                    for (const tw of this.twigs) {
                        const s = Math.max(0, Math.min(tw.len, (P[i] - tw.ax) * tw.dx + (P[i + 2] - tw.az) * tw.dz));
                        const ox = P[i] - tw.ax - s * tw.dx, oy = P[i + 1], oz = P[i + 2] - tw.az - s * tw.dz;
                        const r = tw.r + 0.05 * a.size / 0.62, d = Math.hypot(ox, oy, oz);
                        if (d < r && d > 1e-6) {
                            P[i] += ox / d * (r - d); P[i + 1] += oy / d * (r - d); P[i + 2] += oz / d * (r - d);
                        }
                    }
                }
            }
        }
        // legs that are pulled too far lose their grip; an ant with no grip left falls
        let broke = false;
        for (let m = C.length - 1; m >= 0; m--) {
            const c = C[m];
            if (c.body) continue;
            const i = c.i * 3;
            const [jx, jy, jz] = c.j < 0 ? c.anchor : [P[c.j * 3], P[c.j * 3 + 1], P[c.j * 3 + 2]];
            const d = Math.hypot(P[i] - jx, P[i + 1] - jy, P[i + 2] - jz);
            if (d > c.rest + 0.35) { C.splice(m, 1); broke = true; }
        }
        if (broke) {
            this.relink();
            const held = this.grounded();
            for (const a of [...this.bridge]) {
                if (!held.has(a.id)) { this.removeAnt(a); this.falls++; }
            }
        }
    }

    step(dt = this.p.dt) {
        this.t += dt;
        // new ants arrive at both ends as a Poisson stream
        for (let from = 0; from < 2; from++) {
            if (this.rng() < this.traffic / 120 * dt) this.spawnDebt[from]++;
            if (this.spawnDebt[from] > 0 && this.spawn(from)) this.spawnDebt[from]--;
        }
        this.stepWalkers(dt);
        this.stepBridge(dt);
        this.dynTimer = (this.dynTimer || 0) + dt;
        if (this.dynTimer > 0.05) { this.dynTimer = 0; this.updateDynamic(); }
        this.navTimer += dt;
        if (this.navTimer >= this.p.navInterval) { this.navTimer = 0; this.updateNav(); }
    }

    // --- measurements, as Reid et al. (2015) took them

    // Distance from the junction of the tines to the inner edge of the bridge, and the
    // bridge's width, both along the centre line between the tines.
    measure() {
        const h = this.p.cell, i = Math.floor((0 - this.x0) / h);
        let inner = Infinity, outer = -Infinity;
        for (let j = 0; j < this.nz; j++) {
            const c = j * this.nx + i;
            if (this.owner[c] >= 0 && this.staticType[c] !== SUPPORT) {
                const z = this.z0 + (j + 0.5) * h;
                if (z < inner) inner = z;
                if (z > outer) outer = z;
            }
        }
        const spans = inner < Infinity;
        const d = spans ? Math.max(0, this.innerCorner - inner) : NaN;
        const width = spans ? Math.min(outer, this.innerCorner) - inner : NaN;
        return {
            t: this.t, ants: this.bridge.length, walkers: this.walkers.length,
            d, width, length: spans ? 2 * d * Math.tan(this.half) : NaN,
            saved: this.baseline - this.pathLength(), path: this.pathLength(),
            joins: this.joins, leaves: this.leaves, falls: this.falls, arrivals: this.arrivals,
        };
    }
}
