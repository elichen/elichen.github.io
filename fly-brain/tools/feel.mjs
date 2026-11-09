// What the body's own sensory neurons do in the model (BANC, engine.wasm, with adaptation): drive them as the page does, read the motor neurons.
// Then the input stops, and the last 0.5 s of the next second shows if the model goes back to rest.
//   node tools/feel.mjs [seeds]
import fs from 'node:fs'; import zlib from 'node:zlib';
const seeds = +(process.argv[2] || 3), T = 1, n = JSON.parse(fs.readFileSync('model/neurons.json')), L = n.life, F = n.feel, bytes = zlib.gunzipSync(fs.readFileSync('model/banc.gz'));
const wasm = await WebAssembly.compile(fs.readFileSync('engine.wasm'));
const legs = (...kinds) => Object.values(F.legs).flatMap(l => kinds.flatMap(k => l[k])), both = s => [...s.left, ...s.right];
const allLegs = legs('claw', 'hook', 'hairplate', 'campaniform', 'touch'), looms = s => [...Object.values(n.looms).flatMap(v => v[s]), ...L.LPLC2[s]];
const cases = {
    'nothing': [],
    'claw 50': [[legs('claw'), 50]], 'hook 50': [[legs('hook'), 50]], 'hair plates 50': [[legs('hairplate'), 50]], 'campaniform 50': [[legs('campaniform'), 50]], 'foot touch 30': [[legs('touch'), 30]],
    'all leg sensors 50': [[allLegs, 50]], 'P9': [[L.P9, 150]], 'P9 + leg sensors': [[L.P9, 150], [allLegs, 50]], 'P9 + sugar': [[L.P9, 150], [L.sugar, 150]], 'P9 + sugar + leg sensors': [[L.P9, 150], [L.sugar, 150], [allLegs, 50]],
    'halteres 220': [[both(F.haltere), 220]], 'left haltere 220': [[F.haltere.left, 220]], 'wing base 220': [[both(F.wing), 220]], 'halteres + wing base 220': [[[...both(F.haltere), ...both(F.wing)], 220]],
    'left loom': [[looms('left'), 100]], 'left loom + halteres + wing base': [[looms('left'), 100], [[...both(F.haltere), ...both(F.wing)], 220]],
};
const one = (g, c) => g.reduce((x, i) => x + c[i], 0) / Math.max(1, g.length) / T;
for (const [k, inp] of Object.entries(cases)) {
    const r = {}, add = (key, v) => { r[key] = (r[key] || 0) + v / seeds; };
    for (let s = 1; s <= seeds; s++) {
        const w = (await WebAssembly.instantiate(wasm)).exports, p = w.alloc(bytes.length); new Uint8Array(w.memory.buffer, p, bytes.length).set(bytes); w.init(p, bytes.length, s * 1000); w.set_adaptation(0.05, 200);
        for (const [c, hz] of inp) for (const i of c) w.set_rate(i, hz);
        const c = new Float64Array(n.count); let all = 0, after = 0;
        for (let i = 0; i < T * 100; i++) { const m = w.run(100), a = new Uint32Array(w.memory.buffer, w.spikes(), 2 * m); all += m; for (let j = 1; j < a.length; j += 2) c[a[j]]++; }
        for (const [cc] of inp) for (const i of cc) w.set_rate(i, 0);
        for (let i = 0; i < 100; i++) { const m = w.run(100); if (i >= 50) after += m; }
        add('brain', all / T); add('after', after / 0.5); add('oDN1', one(L.oDN1, c)); add('MN9', one(L.MN9, c)); add('DNp03L', one(n.named.DNp03.left, c)); add('DNp03R', one(n.named.DNp03.right, c));
        const leg = ['front_leg', 'middle_leg', 'hind_leg'].flatMap(q => both(F.motor[q]));
        add('legMN', leg.filter(i => c[i] >= 5 * T).length); add('neckMN', both(F.motor.neck).filter(i => c[i] >= 5 * T).length);
        for (const t of ['b1', 'b2', 'b3', 'i1']) for (const sd of ['left', 'right']) add(t + sd[0].toUpperCase(), one(n.wing[t][sd], c));
    }
    const f = x => r[x].toFixed(0);
    console.log(k.padEnd(34), `spikes/s ${f('brain')}, after ${f('after')} | oDN1 ${f('oDN1')} MN9 ${f('MN9')} DNp03 ${f('DNp03L')}/${f('DNp03R')} | leg MNs at 5 Hz or more ${f('legMN')} of 391, neck ${f('neckMN')} of 49 | b1 ${f('b1L')}/${f('b1R')} b2 ${f('b2L')}/${f('b2R')} b3 ${f('b3L')}/${f('b3R')} i1 ${f('i1L')}/${f('i1R')} Hz (left/right)`);
}
