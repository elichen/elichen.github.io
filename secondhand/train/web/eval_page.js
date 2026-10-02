// node web/eval_page.js evalset.json out.json [part nparts]: eval_style.py's protocol run
// through the page's own worker.js (chunking and all). Score with eval_page.py.
const fs = require("fs"), path = require("path"), vm = require("vm");
const ROOT = path.join(__dirname, "../..");
const Ink = require(ROOT + "/ink.js");
const [evalset, out, part = 0, nparts = 1] = process.argv.slice(2);
const workerFile = path.join(ROOT, "worker.js");
const E = JSON.parse(fs.readFileSync(evalset));
const ctx = {
  console, Float32Array, Int32Array, Uint16Array, Map, Math, Promise, Array, Object, String, Number, MessageChannel,
  fetch: async (u) => { const b = fs.readFileSync(path.join(ROOT, u.split("?")[0])); return { json: async () => JSON.parse(b), arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.length) }; },
};
let msgs = [], done = null;
ctx.postMessage = (m) => { msgs.push(m); if ((m.type === "done" || m.type === "error") && done) done(); };
ctx.self = ctx;
ctx.importScripts = (u) => vm.runInContext(fs.readFileSync(path.join(ROOT, u.split("?")[0]), "utf8"), ctx);
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(workerFile, "utf8"), ctx);
const write = (data) => new Promise((r) => { msgs = []; done = r; ctx.onmessage({ data }); });
(async () => {
  const ws = Object.keys(E.writers), res = {};
  let id = 1;
  for (let i = 0; i < ws.length; i++) {
    if (i % Number(nparts) !== Number(part)) continue;
    for (const [kind, pw, seed0] of [["self", ws[i], 1], ["other", ws[(i + 5) % ws.length], 2]]) {
      const line = E.writers[pw].prime, gens = [];
      for (let j = 0; j < 3; j++) {
        await write({ type: "write", id: id++, prime: Ink.toMoves(line, 0.2), primeText: "The quick brown fox", lines: E.sentences, bias: 1.0, seed: seed0 * 1000 + j });
        const g = E.sentences.map(() => []);
        for (const m of msgs) if (m.type === "points") for (let k = 0; k < m.pts.length; k += 4) g[m.line].push([m.pts[k], m.pts[k + 1], m.pts[k + 2]]);
        gens.push(g);
      }
      res[`${ws[i]}:${kind}`] = { prime_w: pw, gens };
    }
    console.error("writer", ws[i]);
  }
  fs.writeFileSync(out, JSON.stringify(res)); process.exit(0);
})();
