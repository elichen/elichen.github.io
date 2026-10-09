// Top-down pictures of the simulation, for checking the model by eye.
//   node tools/snap.mjs --angle 20 --traffic 150 --at 30,120,300 --out /tmp/snap [--params '{...}']
// Brown: twig. Grey: air an ant can stretch over. Orange: bridge ants, brighter with more
// traffic over them. White: walking ants. Cyan: ants holding over air.
import { Sim } from '../sim.js';
import { writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]);
    return acc;
}, []));
const sim = new Sim({ angle: +(args.angle ?? 20), traffic: +(args.traffic ?? 150), seed: +(args.seed ?? 1),
    params: args.params ? JSON.parse(args.params) : {} });
const times = (args.at ?? '60').split(',').map(Number);
const out = args.out ?? '/tmp/snap';
const S = +(args.scale ?? 4);
// crop to the region around the crotch
const zTop = sim.crotch.z + 1.5, zBot = Math.max(sim.z0, sim.crotch.z - +(args.depth ?? 12));

function png(w, h, rgb) {
    const raw = Buffer.alloc((w * 3 + 1) * h);
    for (let y = 0; y < h; y++) { raw[y * (w * 3 + 1)] = 0; rgb.copy(raw, y * (w * 3 + 1) + 1, y * w * 3, (y + 1) * w * 3); }
    const crcT = []; for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; crcT[n] = c >>> 0; }
    const crc = (b) => { let c = 0xffffffff; for (const x of b) c = crcT[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
    const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]); const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([len, td, c]); };
    const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
    return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

function draw() {
    const h = sim.p.cell, W = sim.nx * S;
    const j0 = Math.floor((zBot - sim.z0) / h), j1 = Math.min(sim.nz, Math.ceil((zTop - sim.z0) / h));
    const H = (j1 - j0) * S, img = Buffer.alloc(W * H * 3);
    const put = (px, py, r, g, b) => { if (px < 0 || py < 0 || px >= W || py >= H) return; const o = (py * W + px) * 3; img[o] = r; img[o + 1] = g; img[o + 2] = b; };
    const toPx = (x, z) => [Math.round((x - sim.x0) / h * S), Math.round((zTop - z) / h * S)];
    for (let j = j0; j < j1; j++) for (let i = 0; i < sim.nx; i++) {
        const c = j * sim.nx + i, own = sim.owner[c];
        let col = [20, 22, 26];
        if (sim.staticType[c] === 2) col = [92, 70, 50];
        else if (own >= 0) { const a = sim.bridge[own], k = Math.min(1, a.rate / 1.5); col = [140 + 115 * k, 70 + 90 * k, 30]; }
        else if (sim.type[c] === 1) col = [45, 48, 54];
        const [px, py] = toPx(sim.x0 + i * h, sim.z0 + (j + 1) * h);
        for (let dy = 0; dy < S; dy++) for (let dx = 0; dx < S; dx++) put(px + dx, py + dy, ...col);
    }
    const line = (x0, z0, x1, z1, col) => {
        const [a, b] = toPx(x0, z0), [c, d] = toPx(x1, z1), n = Math.max(Math.abs(c - a), Math.abs(d - b), 1);
        for (let k = 0; k <= n; k++) put(Math.round(a + (c - a) * k / n), Math.round(b + (d - b) * k / n), ...col);
    };
    for (const a of sim.bridge) {
        const P = sim.px;
        line(P[a.b * 3], P[a.b * 3 + 2], P[a.f * 3], P[a.f * 3 + 2], [60, 25, 10]);
    }
    for (const w of sim.walkers) {
        const col = w.state === 1 ? [80, 220, 255] : w.carry ? [255, 255, 255] : [200, 200, 200];
        line(w.x - Math.cos(w.hd) * w.size / 2, w.z - Math.sin(w.hd) * w.size / 2, w.x + Math.cos(w.hd) * w.size / 2, w.z + Math.sin(w.hd) * w.size / 2, col);
    }
    // centre line, where Reid et al. measured the bridge
    line(0, sim.innerCorner, 0, zBot, [90, 90, 160]);
    return png(W, H, img);
}

for (const t of times) {
    while (sim.t < t) sim.step();
    const m = sim.measure();
    const f = `${out}-${String(t).padStart(5, '0')}.png`;
    writeFileSync(f, draw());
    console.log(f, `d ${m.d.toFixed(2)} width ${m.width.toFixed(2)} ants ${m.ants} saved ${m.saved.toFixed(2)}`);
}
