// node parity.js <export dir> : compare the JS engine to torch on golden.json
const fs = require("fs");
const { HandModel } = require("../../engine.js");
const dir = process.argv[2] || "../export";
const meta = JSON.parse(fs.readFileSync(`${dir}/hand.json`));
const buf = fs.readFileSync(`${dir}/hand.bin`);
const model = new HandModel(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const g = JSON.parse(fs.readFileSync(`${dir}/golden.json`));
const s = model.newState(g.text);
let y10;
const t0 = Date.now();
g.x.forEach(([dx, dy, pen], t) => { model.step(s, dx, dy, pen); if (t === 10) y10 = s.y.slice(); });
const ms = (Date.now() - t0) / g.x.length;
const err = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
console.log("max |dy| step10", err(Array.from(y10), g.y_10).toExponential(2));
console.log("max |dy| last  ", err(Array.from(s.y), g.y_last).toExponential(2));
console.log("max |dphi| last", err(Array.from(s.phi), g.phi_last).toExponential(2));
console.log("max |dkappa|   ", err(Array.from(s.kappa), g.kappa).toExponential(2));
// speed
const t1 = Date.now(); for (let i = 0; i < 400; i++) model.step(s, 0.5, 0.1, 0);
console.log(`ms/step ${((Date.now() - t1) / 400).toFixed(3)}`);
