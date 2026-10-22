// Builds a circuit in three.js from the TUM centerline and widths.
// World frame: y up; a simulation point (x, y) is at (x, 0, -y).
import * as THREE from 'three';

export const toWorld = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

// ---- procedural textures --------------------------------------------------------
function canvasTexture(size, draw, repeat = true) {
    const c = document.createElement('canvas');
    c.width = c.height = size;
    draw(c.getContext('2d'), size);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 8;
    return t;
}

function noise(ctx, size, base, spread, count, alpha = 0.5) {
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < count; i++) {
        const v = (Math.random() - 0.5) * spread;
        ctx.fillStyle = `rgba(${v > 0 ? 255 : 0},${v > 0 ? 255 : 0},${v > 0 ? 255 : 0},${Math.abs(v) * alpha})`;
        const r = Math.random() * 1.6 + 0.4;
        ctx.fillRect(Math.random() * size, Math.random() * size, r, r);
    }
}

export function makeTextures(palette) {
    return {
        asphalt: canvasTexture(512, (ctx, s) => noise(ctx, s, palette.asphalt, 0.5, 60000, 0.35)),
        grass: canvasTexture(512, (ctx, s) => {
            noise(ctx, s, palette.grass, 0.6, 50000, 0.25);
            // mowing stripes: two bands across the tile
            ctx.fillStyle = 'rgba(255,255,255,0.045)';
            ctx.fillRect(0, 0, s, s / 2);
        }),
        kerb: canvasTexture(64, (ctx, s) => {
            ctx.fillStyle = palette.kerbA; ctx.fillRect(0, 0, s, s);
            ctx.fillStyle = palette.kerbB; ctx.fillRect(0, 0, s / 2, s);
        }),
        wall: canvasTexture(256, (ctx, s) => {
            ctx.fillStyle = palette.wall; ctx.fillRect(0, 0, s, s);
            ctx.fillStyle = palette.wallBand; ctx.fillRect(0, s * 0.18, s, s * 0.5);
            ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.fillRect(s - 3, 0, 3, s);
        }),
        checker: canvasTexture(64, (ctx, s) => {
            ctx.fillStyle = '#f4f4f0'; ctx.fillRect(0, 0, s, s);
            ctx.fillStyle = '#141414';
            for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) if ((i + j) % 2) ctx.fillRect(i * s / 8, j * s / 8, s / 8, s / 8);
        }),
    };
}

// ---- track geometry ---------------------------------------------------------------
// center: Float32Array [x..., y..., wr..., wl...] with n points, ~2 m apart.
export function centerline(data, n) {
    const x = data.subarray(0, n), y = data.subarray(n, 2 * n);
    const wr = data.subarray(2 * n, 3 * n), wl = data.subarray(3 * n, 4 * n);
    const tx = new Float32Array(n), ty = new Float32Array(n), s = new Float32Array(n), k = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const a = (i - 1 + n) % n, b = (i + 1) % n;
        const dx = x[b] - x[a], dy = y[b] - y[a], l = Math.hypot(dx, dy) || 1;
        tx[i] = dx / l; ty[i] = dy / l;
        if (i) s[i] = s[i - 1] + Math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]);
    }
    for (let i = 0; i < n; i++) {
        const a = (i - 4 + n) % n, b = (i + 4) % n;
        const h = Math.atan2(ty[b], tx[b]) - Math.atan2(ty[a], tx[a]);
        k[i] = Math.atan2(Math.sin(h), Math.cos(h)) / Math.max(s[b] - s[a], 8) || 0;
    }
    const length = s[n - 1] + Math.hypot(x[0] - x[n - 1], y[0] - y[n - 1]);
    return { n, x, y, wr, wl, tx, ty, s, k, length };
}

// Spatial hash of the centerline, to keep barriers and trees off other parts of the track.
function grid(c, cell = 40) {
    const map = new Map();
    const key = (gx, gy) => gx * 100003 + gy;
    for (let i = 0; i < c.n; i++) {
        const k = key(Math.floor(c.x[i] / cell), Math.floor(c.y[i] / cell));
        if (!map.has(k)) map.set(k, []);
        map.get(k).push(i);
    }
    // distance from (px, py) to the asphalt edge of the nearest centerline point,
    // ignoring points within `skip` samples of `self`
    return (px, py, self = -1, skip = 0, reach = 60) => {
        let best = Infinity;
        const r = Math.ceil(reach / cell);
        const gx = Math.floor(px / cell), gy = Math.floor(py / cell);
        for (let a = -r; a <= r; a++) for (let b = -r; b <= r; b++) {
            const list = map.get(key(gx + a, gy + b));
            if (!list) continue;
            for (const i of list) {
                if (self >= 0) {
                    const d = Math.abs(i - self);
                    if (Math.min(d, c.n - d) < skip) continue;
                }
                const d = Math.hypot(c.x[i] - px, c.y[i] - py) - Math.max(c.wl[i], c.wr[i]);
                if (d < best) best = d;
            }
        }
        return best;
    };
}

// A ribbon between two offsets from the centerline (positive = left), as a
// triangle strip closed into a loop. u runs along the track, v across.
function ribbon(c, inner, outer, h, uScale, { from = 0, to = c.n, closed = true, vScale = 1 } = {}) {
    const pos = [], uv = [], idx = [];
    const count = closed ? c.n + 1 : to - from + 1;
    for (let j = 0; j < count; j++) {
        const i = (from + j) % c.n;
        const nx = -c.ty[i], ny = c.tx[i];
        const s = closed && j === c.n ? c.length : c.s[i] + (i < from ? c.length : 0);
        const a = inner(i), b = outer(i);
        pos.push(c.x[i] + nx * a, h, -(c.y[i] + ny * a), c.x[i] + nx * b, h, -(c.y[i] + ny * b));
        uv.push(s / uScale, 0, s / uScale, vScale);
        if (j > 0) {
            const q = 2 * j;
            idx.push(q - 2, q, q - 1, q - 1, q, q + 1);
        }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
}

// Runs of indices where pred(i) holds, as [from, to) pairs (wrapping).
function runs(n, pred) {
    const out = [];
    let start = -1;
    for (let i = 0; i <= n; i++) {
        const on = i < n && pred(i);
        if (on && start < 0) start = i;
        if (!on && start >= 0) { out.push([start, i]); start = -1; }
    }
    if (out.length > 1 && out[0][0] === 0 && out[out.length - 1][1] === n) {
        const last = out.pop();
        out[0] = [last[0], out[0][1] + n];
    }
    return out;
}

export function buildCircuit(c, racing, palette, tex) {
    const world = new THREE.Group();
    const near = grid(c);
    const std = (opts) => new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, ...opts });
    const add = (geo, mat, receive = true) => {
        const m = new THREE.Mesh(geo, mat);
        m.receiveShadow = receive;
        world.add(m);
        return m;
    };

    // bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < c.n; i++) {
        minX = Math.min(minX, c.x[i]); maxX = Math.max(maxX, c.x[i]);
        minY = Math.min(minY, c.y[i]); maxY = Math.max(maxY, c.y[i]);
    }
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const span = Math.max(maxX - minX, maxY - minY) + 1600;

    // ground
    const grass = tex.grass.clone();
    grass.needsUpdate = true;
    grass.repeat.set(span / 48, span / 48);
    const ground = add(new THREE.PlaneGeometry(span * 3, span * 3), std({ map: grass, color: 0xffffff }));
    ground.material.map.repeat.set(span * 3 / 48, span * 3 / 48);
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(cx, -0.02, -cy);

    // verge: a lighter band of short grass along both edges
    const verge = std({ color: palette.verge, transparent: true, opacity: 0.55, depthWrite: false });
    add(ribbon(c, i => c.wl[i], i => c.wl[i] + 7, 0.004, 10), verge);
    add(ribbon(c, i => -c.wr[i] - 7, i => -c.wr[i], 0.004, 10), verge);

    // asphalt
    const asphalt = tex.asphalt.clone();
    asphalt.needsUpdate = true;
    add(ribbon(c, i => -c.wr[i] - 0.3, i => c.wl[i] + 0.3, 0.012, 14, { vScale: 1.4 }),
        std({ map: asphalt, color: 0xffffff, roughness: 0.95 }));

    // the rubbered-in racing line, darker where cars run
    const rubber = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.16, depthWrite: false });
    add(ribbon(racing, () => -1.1, () => 1.1, 0.016, 10), rubber);

    // white lines
    const white = std({ color: palette.line, roughness: 0.6 });
    add(ribbon(c, i => c.wl[i] - 0.05, i => c.wl[i] + 0.2, 0.02, 10), white);
    add(ribbon(c, i => -c.wr[i] - 0.2, i => -c.wr[i] + 0.05, 0.02, 10), white);

    // kerbs on corners, both sides
    const kerbMat = std({ map: tex.kerb, roughness: 0.7 });
    for (const [a, b] of runs(c.n, i => {
        for (let o = -6; o <= 6; o++) if (Math.abs(c.k[(i + o + c.n) % c.n]) > 1 / 280) return true;
        return false;
    })) {
        if (b - a < 6) continue;
        const opts = { from: a, to: b, closed: false };
        add(ribbon(c, i => c.wl[i] + 0.2, i => c.wl[i] + 1.5, 0.03, 8, opts), kerbMat);
        add(ribbon(c, i => -c.wr[i] - 1.5, i => -c.wr[i] - 0.2, 0.03, 8, opts), kerbMat);
    }

    // barriers: a low wall ~14 m beyond each edge, left out wherever another part
    // of the circuit (or its run-off) is close
    const wallMat = std({ map: tex.wall, roughness: 0.8, side: THREE.DoubleSide });
    const B = 14;
    for (const side of [1, -1]) {
        const off = i => side > 0 ? c.wl[i] + B : -(c.wr[i] + B);
        const clear = i => {
            const nx = -c.ty[i], ny = c.tx[i], o = off(i);
            return near(c.x[i] + nx * o, c.y[i] + ny * o, i, 40, 40) > 10;
        };
        for (const [a, b] of runs(c.n, clear)) {
            if (b - a < 10) continue;
            const pos = [], uv = [], idx = [];
            for (let j = 0; j <= b - a; j++) {
                const i = (a + j) % c.n, nx = -c.ty[i], ny = c.tx[i], o = off(i);
                const px = c.x[i] + nx * o, py = c.y[i] + ny * o, u = (c.s[i] + (i < a ? c.length : 0)) / 4;
                pos.push(px, 0, -py, px, 1.1, -py);
                uv.push(u, 0, u, 1);
                if (j) { const q = 2 * j; idx.push(q - 2, q, q - 1, q - 1, q, q + 1); }
            }
            const g = new THREE.BufferGeometry();
            g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
            g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
            g.setIndex(idx);
            g.computeVertexNormals();
            const m = add(g, wallMat);
            m.castShadow = true;
        }
    }

    // start line and grid, at the racing line's first point
    const i0 = nearestCenter(c, racing.x[0], racing.y[0]);
    const startFrame = frameAt(c, i0);
    const checker = tex.checker.clone();
    checker.needsUpdate = true;
    const W = c.wl[i0] + c.wr[i0];
    checker.repeat.set(1, W / 1.2);
    const lineFrame = new THREE.Group();
    const line = new THREE.Mesh(new THREE.PlaneGeometry(1.2, W), std({ map: checker, roughness: 0.7 }));
    line.rotation.x = -Math.PI / 2;
    line.receiveShadow = true;
    lineFrame.add(line);
    lineFrame.position.copy(startFrame.at(0, (c.wl[i0] - c.wr[i0]) / 2, 0.025));
    lineFrame.rotation.y = startFrame.yaw;
    world.add(lineFrame);

    // start gantry with five pairs of lights
    const gantry = new THREE.Group();
    const metal = std({ color: palette.gantry, roughness: 0.5, metalness: 0.4 });
    const wTotal = c.wl[i0] + c.wr[i0] + 4;
    const beam = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.9, wTotal), metal);
    beam.position.set(0, 7, 0);
    beam.castShadow = true;
    gantry.add(beam);
    for (const sgn of [-1, 1]) {
        const post = new THREE.Mesh(new THREE.BoxGeometry(0.5, 7.5, 0.5), metal);
        post.position.set(0, 3.75, sgn * wTotal / 2);
        post.castShadow = true;
        gantry.add(post);
    }
    const lamps = [];
    for (let k = 0; k < 5; k++) {
        const pod = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1.6, 0.7), std({ color: 0x0c0c0c }));
        const z = (k - 2) * 0.95;
        pod.position.set(-0.6, 6.4, z);
        gantry.add(pod);
        const pair = [];
        for (const yy of [6.85, 6.2]) {
            const lamp = new THREE.Mesh(new THREE.CircleGeometry(0.22, 16),
                new THREE.MeshBasicMaterial({ color: 0x2a0a0a }));
            lamp.position.set(-0.86, yy, z);
            lamp.rotation.y = -Math.PI / 2;
            gantry.add(lamp);
            pair.push(lamp);
        }
        lamps.push(pair);
    }
    gantry.position.copy(startFrame.at(0, (c.wl[i0] - c.wr[i0]) / 2, 0));
    gantry.rotation.y = startFrame.yaw;
    world.add(gantry);
    const setLights = (lit, color = 0xff2a1a) => lamps.forEach((pair, k) =>
        pair.forEach(l => l.material.color.set(k < lit ? color : 0x2a0a0a)));

    // grandstand on the left of the start straight, pit building on the right
    const stand = new THREE.Group();
    const seatMat = std({ color: palette.stand, roughness: 0.8 });
    const roofMat = std({ color: palette.roof, roughness: 0.5, metalness: 0.3, side: THREE.DoubleSide });
    const crowd = std({ color: 0xffffff, roughness: 1 });
    const L = 160;
    for (let r = 0; r < 9; r++) {
        const step = new THREE.Mesh(new THREE.BoxGeometry(L, 0.9 * (r + 1), 1.4), seatMat);
        step.position.set(0, 0.45 * (r + 1), -(r * 1.4 + 0.7));
        step.castShadow = step.receiveShadow = true;
        stand.add(step);
    }
    // spectators: little coloured blocks on each step
    const people = new THREE.InstancedMesh(new THREE.BoxGeometry(0.45, 0.75, 0.35), crowd, 9 * 220);
    const tmp = new THREE.Object3D(), col = new THREE.Color();
    let p = 0;
    for (let r = 0; r < 9; r++) for (let k = 0; k < 220; k++) {
        if (Math.random() < 0.25) continue;
        tmp.position.set(-L / 2 + (k + 0.5) * L / 220 + (Math.random() - 0.5) * 0.2, 0.9 * (r + 1) + 0.37, -(r * 1.4 + 0.6));
        tmp.updateMatrix();
        people.setMatrixAt(p, tmp.matrix);
        people.setColorAt(p, col.set(palette.crowd[Math.floor(Math.random() * palette.crowd.length)]));
        p++;
    }
    people.count = p;
    stand.add(people);
    const roof = new THREE.Mesh(new THREE.BoxGeometry(L + 4, 0.3, 15), roofMat);
    roof.position.set(0, 12.5, -6.5);
    roof.rotation.x = 0.08;
    roof.castShadow = true;
    stand.add(roof);
    for (let k = -2; k <= 2; k++) {
        const col2 = new THREE.Mesh(new THREE.BoxGeometry(0.4, 12.5, 0.4), metal);
        col2.position.set(k * L / 4.4, 6.25, -13);
        stand.add(col2);
    }
    const sideL = c.wl[i0] + B + 6;
    stand.position.copy(startFrame.at(20, sideL, 0));
    stand.rotation.y = startFrame.yaw;
    world.add(stand);

    const pits = new THREE.Group();
    const building = new THREE.Mesh(new THREE.BoxGeometry(220, 9, 16), std({ color: palette.pit, roughness: 0.6 }));
    building.position.set(0, 4.5, 8);
    building.castShadow = building.receiveShadow = true;
    pits.add(building);
    const glass = new THREE.Mesh(new THREE.BoxGeometry(220, 2.4, 0.2),
        new THREE.MeshStandardMaterial({ color: 0x1e2a33, roughness: 0.05, metalness: 0.9 }));
    glass.position.set(0, 6.6, -0.05);
    pits.add(glass);
    for (let k = 0; k < 18; k++) {
        const door = new THREE.Mesh(new THREE.PlaneGeometry(9, 4.2), std({ color: 0x111316 }));
        door.position.set(-100 + k * 11.8, 2.1, -0.02);
        door.rotation.y = Math.PI;
        pits.add(door);
    }
    const pitLane = new THREE.Mesh(new THREE.PlaneGeometry(240, 12), std({ map: asphalt, color: 0xd8d8d8 }));
    pitLane.rotation.x = -Math.PI / 2;
    pitLane.position.set(0, 0.01, -6);
    pitLane.receiveShadow = true;
    pits.add(pitLane);
    const sideR = c.wr[i0] + B + 2;
    // local -z (the garage fronts) points left, toward the track
    pits.position.copy(startFrame.at(10, -sideR - 12, 0));
    pits.rotation.y = startFrame.yaw;
    world.add(pits);

    // trees
    const keepClear = (x, y) => near(x, y, -1, 0, 70) > 26;
    const clearOfBuildings = (x, y) => {
        const d = startFrame.local(x, y);
        return !(Math.abs(d.along - 15) < 140 && Math.abs(d.left) < 70);
    };
    const trunks = [], crowns = [];
    const tries = 9000;
    for (let t = 0; t < tries && crowns.length < 2600; t++) {
        const x = minX - 400 + Math.random() * (maxX - minX + 800);
        const y = minY - 400 + Math.random() * (maxY - minY + 800);
        if (!keepClear(x, y) || !clearOfBuildings(x, y)) continue;
        // clump: trees prefer to grow near other trees
        const n = 1 + Math.floor(Math.random() * 6);
        for (let k = 0; k < n; k++) {
            const tx = x + (Math.random() - 0.5) * 30, ty = y + (Math.random() - 0.5) * 30;
            if (!keepClear(tx, ty)) continue;
            const h = 8 + Math.random() * 10;
            trunks.push([tx, ty, h]);
            crowns.push([tx, ty, h]);
        }
    }
    const trunkMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.25, 0.4, 1, 6), std({ color: 0x4a3a2c }), trunks.length);
    const crownGeo = new THREE.IcosahedronGeometry(1, 1);
    const crownMesh = new THREE.InstancedMesh(crownGeo, std({ color: 0xffffff, flatShading: true }), crowns.length);
    trunks.forEach(([x, y, h], k) => {
        tmp.position.set(x, h * 0.2, -y);
        tmp.scale.set(1, h * 0.4, 1);
        tmp.rotation.set(0, 0, 0);
        tmp.updateMatrix();
        trunkMesh.setMatrixAt(k, tmp.matrix);
        const r = h * (0.28 + Math.random() * 0.12);
        tmp.position.set(x, h * 0.62, -y);
        tmp.scale.set(r, h * 0.42, r);
        tmp.rotation.set(0, Math.random() * 6, 0);
        tmp.updateMatrix();
        crownMesh.setMatrixAt(k, tmp.matrix);
        crownMesh.setColorAt(k, col.set(palette.trees[Math.floor(Math.random() * palette.trees.length)]).offsetHSL(0, 0, (Math.random() - 0.5) * 0.06));
    });
    trunkMesh.castShadow = crownMesh.castShadow = true;
    crownMesh.receiveShadow = true;
    world.add(trunkMesh, crownMesh);

    // trackside TV cameras, alternating sides every ~220 m
    const cameras = [];
    for (let i = 0, k = 0; i < c.n; i += 110, k++) {
        const side = k % 2 ? 1 : -1;
        const o = side > 0 ? c.wl[i] + 16 : -(c.wr[i] + 16);
        const nx = -c.ty[i], ny = c.tx[i];
        const px = c.x[i] + nx * o, py = c.y[i] + ny * o;
        if (near(px, py, i, 40, 40) < 8) continue;
        cameras.push({ i, s: c.s[i], pos: toWorld(px, py, 5 + Math.random() * 5) });
    }

    return { world, setLights, cameras, startFrame, i0, bounds: { minX, maxX, minY, maxY } };
}

function nearestCenter(c, x, y) {
    let best = 0, bd = Infinity;
    for (let i = 0; i < c.n; i++) {
        const d = (c.x[i] - x) ** 2 + (c.y[i] - y) ** 2;
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

// A local frame at centerline point i: at(along, left, h) -> world position.
function frameAt(c, i) {
    const tx = c.tx[i], ty = c.ty[i], nx = -ty, ny = tx;
    return {
        yaw: Math.atan2(ty, tx),
        at: (along, left, h) => toWorld(c.x[i] + tx * along + nx * left, c.y[i] + ty * along + ny * left, h),
        local: (x, y) => ({ along: (x - c.x[i]) * tx + (y - c.y[i]) * ty, left: (x - c.x[i]) * nx + (y - c.y[i]) * ny }),
    };
}

// The racing line as a centerline-like object (for the rubber band).
export function racingLine(track) {
    const n = track.count;
    return {
        n, x: track.x, y: track.y, tx: track.tangentX, ty: track.tangentY, s: track.s, length: track.length,
    };
}
