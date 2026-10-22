// Prints the steady-cornering table for sim.js (CORNERING): for each speed,
// the steering input that gives the most steady lateral g with the speed held
// by the pedal (traction-limited), before the car slides (slip > 0.2 rad).
// Re-run after any change to the plant and paste the output into sim.js.
//
//   node tools/corner-table.mjs
import { integrate, tractionLimit } from '../sim.js';

function corner(kmh, s) {
    const st = new Float64Array([0, 0, 0, kmh / 3.6, 0, 0, 0, 0.3, 0]);
    let g = 0;
    for (let k = 0; k < 250; k++) {
        let pedal = Math.max(-1, Math.min(1, (kmh / 3.6 - st[3]) * 0.5));
        if (pedal > 0) pedal = Math.min(pedal, tractionLimit(st));
        integrate(st, s, pedal);
        if (Math.abs(Math.atan2(st[4], Math.max(st[3], 2))) > 0.2 || Math.abs(st[5]) > 3.2) return null;
        if (k > 200) g = Math.abs(st[3] * st[5]) / 9.81;
    }
    return g;
}

const rows = [];
for (let kmh = 20; kmh <= 340; kmh += 10) {
    let best = { s: 0.02, g: 0 };
    for (let s = 0.005; s <= 1.0001; s += 0.005) {
        const g = corner(kmh, s);
        if (g === null) break;
        if (g > best.g) best = { s, g };
    }
    rows.push(`[${kmh}, ${best.s.toFixed(3)}, ${best.g.toFixed(2)}]`);
}
console.log(`// [km/h, steering input, lateral g]: tools/corner-table.mjs\nexport const CORNERING = [\n    ${rows.join(',\n    ')},\n];`);
