const test = require("node:test");
const assert = require("node:assert/strict");
const { TopOpt, buildKE } = require("./solver.js");

function makeCantilever({ nelx = 60, nely = 20, volfrac = 0.4, rmin = 2.0 } = {}) {
  const opt = new TopOpt({ nelx, nely, volfrac, rmin });
  opt.clearBCs();
  for (let iy = 0; iy <= nely; iy++) opt.fixNode(0, iy); // clamp left edge
  opt.addLoad(nelx, Math.floor(nely / 2), 0, 1); // transverse tip load
  opt.resetDesign();
  return opt;
}

test("element stiffness matrix is symmetric with zero-force rigid translations", () => {
  const KE = buildKE(0.3);
  for (let r = 0; r < 8; r++) {
    assert.ok(KE[r * 8 + r] > 0, `positive diagonal at ${r}`);
    for (let c = 0; c < 8; c++) {
      assert.ok(Math.abs(KE[r * 8 + c] - KE[c * 8 + r]) < 1e-12, `symmetry ${r},${c}`);
    }
  }
  // rigid x-translation: u = [1,0,1,0,1,0,1,0] must produce zero force
  for (let r = 0; r < 8; r++) {
    let fx = 0, fy = 0;
    for (let c = 0; c < 4; c++) {
      fx += KE[r * 8 + 2 * c];
      fy += KE[r * 8 + 2 * c + 1];
    }
    assert.ok(Math.abs(fx) < 1e-12 && Math.abs(fy) < 1e-12, `rigid mode row ${r}`);
  }
});

test("full-density cantilever tip deflection matches beam theory within 15%", () => {
  const nelx = 60, nely = 20;
  const opt = makeCantilever({ nelx, nely });
  opt.x.fill(1);
  opt.filterForward(opt.x, opt.xPhys);
  opt.solve(1e-8, 20000);

  const tip = opt.nodeId(nelx, nely / 2);
  const deflection = Math.abs(opt.U[2 * tip + 1]);
  // Timoshenko: PL^3/(3EI) + PL/(k G A) with E=1, nu=0.3, I=h^3/12
  const L = nelx, h = nely, I = (h * h * h) / 12;
  const G = 1 / (2 * 1.3);
  const expected = (L * L * L) / (3 * I) + L / ((5 / 6) * G * h);
  const ratio = deflection / expected;
  assert.ok(ratio > 0.85 && ratio < 1.15, `deflection ${deflection} vs theory ${expected}`);
});

test("optimization reduces compliance and honors the volume constraint", () => {
  const opt = makeCantilever();
  const first = opt.step();
  let last = first;
  for (let i = 0; i < 29; i++) last = opt.step();

  assert.ok(last.compliance < first.compliance * 0.8,
    `compliance ${first.compliance} -> ${last.compliance}`);
  assert.ok(Math.abs(last.volume - 0.4) < 0.01, `volume ${last.volume}`);

  // densities should polarize toward 0/1 rather than staying gray
  let crisp = 0;
  for (const v of opt.xPhys) if (v < 0.2 || v > 0.8) crisp++;
  assert.ok(crisp / opt.nel > 0.6, `only ${crisp / opt.nel} of elements are crisp`);
});

test("a symmetric bridge problem produces a near-symmetric design", () => {
  const nelx = 60, nely = 20;
  const opt = new TopOpt({ nelx, nely, volfrac: 0.4, rmin: 2.0 });
  opt.clearBCs();
  opt.fixNode(0, nely);
  opt.fixNode(1, nely);
  opt.fixNode(nelx, nely);
  opt.fixNode(nelx - 1, nely);
  opt.addLoad(nelx / 2, nely, 0, 1);
  opt.resetDesign();
  for (let i = 0; i < 20; i++) opt.step();

  let sumDiff = 0;
  for (let ex = 0; ex < nelx; ex++) {
    for (let ey = 0; ey < nely; ey++) {
      const a = opt.xPhys[ex * nely + ey];
      const b = opt.xPhys[(nelx - 1 - ex) * nely + ey];
      sumDiff += Math.abs(a - b);
    }
  }
  assert.ok(sumDiff / opt.nel < 0.02, `mean asymmetry ${sumDiff / opt.nel}`);
});

test("passive void regions stay empty", () => {
  const opt = makeCantilever();
  const passive = new Uint8Array(opt.nel);
  const cx = 30, cy = 10, r = 5;
  for (let ex = 0; ex < opt.nelx; ex++) {
    for (let ey = 0; ey < opt.nely; ey++) {
      if ((ex - cx) ** 2 + (ey - cy) ** 2 <= r * r) passive[ex * opt.nely + ey] = 1;
    }
  }
  opt.setPassive(passive);
  opt.resetDesign();
  for (let i = 0; i < 10; i++) opt.step();
  for (let e = 0; e < opt.nel; e++) {
    if (passive[e] === 1) assert.ok(opt.xPhys[e] === 0, `void element ${e} has density`);
  }
});

test("boundary condition validation catches under-constrained setups", () => {
  const opt = new TopOpt({ nelx: 20, nely: 10 });
  opt.clearBCs();
  assert.equal(opt.hasValidBCs(), false); // nothing
  opt.addLoad(20, 5, 0, 1);
  assert.equal(opt.hasValidBCs(), false); // load, no support
  opt.fixNode(0, 0);
  assert.equal(opt.hasValidBCs(), false); // one pin cannot stop rotation... (2 dofs < 3)
  opt.fixNode(0, 10);
  assert.equal(opt.hasValidBCs(), true);
});
