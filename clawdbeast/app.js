(function () {
  'use strict';
  const CB = window.Clawdbeast;
  const $ = (id) => document.getElementById(id);
  const IN = 25.4;
  const TAU = Math.PI * 2;

  const state = { scale: 1.7, material: '5052-090', hardware: 'M3', slabs: 3, units: 'in', playing: true, rpm: 18, paths: true, xray: false };
  try {
    const saved = JSON.parse(localStorage.getItem('clawdbeast') || 'null');
    if (saved) for (const k of ['scale', 'material', 'hardware', 'slabs', 'units', 'rpm', 'paths', 'xray']) if (k in saved) state[k] = saved[k];
  } catch (e) { /* storage unavailable */ }
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) state.playing = false;
  function remember() {
    try { localStorage.setItem('clawdbeast', JSON.stringify(state)); } catch (e) { /* ignore */ }
  }

  let D = null;
  let partsById = {};
  let cutByKind = {};
  let sceneItems = [];
  let footLoops = [];

  // ------------------------------------------------------------------ names
  const LEG_NAME = { R: 'rear', F: 'front' };
  const KIND_NAME = { j: 'j', k: 'k', c: 'c', f: 'f', bde: 'hip', ghi: 'foot' };
  function partName(id) {
    if (id === 'armI' || id === 'armO') return 'crank arm';
    const leg = id.slice(-1), kind = id.slice(0, -1);
    return `${KIND_NAME[kind]} ${LEG_NAME[leg]}`;
  }
  function jointName(J) {
    const leg = J.slice(-1), letter = J[0];
    const what = { C: 'j + hip', D: 'hip + f', E: 'k + c + foot', F: 'f + foot', B: 'hip rod' }[letter];
    return `${letter} ${LEG_NAME[leg]}`.trim() + ` <span class="muted">(${what})</span>`;
  }

  // ------------------------------------------------------------------ design
  function rebuild() {
    D = CB.design({ scale: state.scale, material: state.material, hardware: state.hardware, slabs: state.slabs });
    partsById = Object.fromEntries(D.parts.map((P) => [P.id, P]));
    cutByKind = Object.fromEntries(D.cut.map((c) => [c.id, c]));
    buildScene();
    renderPanel();
    renderParts();
    renderAssembly();
    remember();
  }

  // ------------------------------------------------------------------ panel
  function fmtMM(v) { return Math.round(v).toLocaleString(); }
  function renderPanel() {
    const s = D.size, n = state.slabs;
    $('scaleOut').textContent = `${state.scale.toFixed(2)} mm/unit`;
    $('scale').value = state.scale;
    for (const [id, key] of [['hw', 'hardware'], ['legs', 'slabs'], ['units', 'units']]) {
      for (const b of $(id).querySelectorAll('button')) b.setAttribute('aria-pressed', String(String(state[key]) === b.dataset.v));
    }
    const mat = D.cfg.mat;
    const nParts = D.cut.reduce((a, p) => a + p.qty, 0);
    $('spec').innerHTML = [
      `${4 * n} legs`,
      `${fmtMM(s.length)} mm long`,
      `${mat.gauge} ${mat.name.replace(' aluminum', ' Al')}`,
      `${nParts} laser-cut parts`,
    ].map((t) => `<span class="chip">${t}</span>`).join('');

    const statsHTML = [
      ['Length', fmtMM(s.length), 'mm', `${(s.length / IN).toFixed(1)} in`],
      ['Width', isFinite(s.width) ? fmtMM(s.width) : '—', 'mm', isFinite(s.width) ? `${(s.width / IN).toFixed(1)} in` : ''],
      ['Height', fmtMM(s.height), 'mm', `${(s.height / IN).toFixed(1)} in`],
      ['Crank throw', (CB.HOLY.m * state.scale).toFixed(1), 'mm', `${((CB.HOLY.m * state.scale) / IN).toFixed(2)} in`],
      ['Parts to cut', nParts, '', `${D.cut.length} files`],
      ['Feet down', D.stability ? `≥ ${D.stability.minFeet}` : '—', '', 'at every moment'],
    ].map(([k, v, unit, sub]) => `<div class="stat"><div class="label">${k}</div><div class="v">${v}${unit ? ` <span class="u">${unit}</span>` : ''}</div><div class="u">${sub}</div></div>`).join('');
    $('stats').innerHTML = statsHTML;

    const items = [];
    for (const i of D.issues.filter((x) => x.level === 'fail')) items.push(['fail', i.text]);
    for (const i of D.issues.filter((x) => x.level === 'warn')) items.push(['warn', i.text]);
    const has = (re) => D.issues.some((i) => re.test(i.text));
    if (D.layout) items.push(['ok', `No collisions over a full crank turn with ${D.layout.N} layers per phase.`]);
    if (!has(/sweeps within|Tie rod/)) items.push(['ok', 'Hip rods and tie rods clear every moving part.']);
    if (!has(/Pivot holes/)) items.push(['ok', `Every hole is at least the ${D.cfg.t.toFixed(2)} mm material thickness.`]);
    if (!has(/between hole and edge|laser minimum/)) items.push(['ok', `Hole-to-edge web is at least 2× thickness (${D.cfg.edgeRule.toFixed(2)} mm).`]);
    if (!has(/minimum part size|maximum/)) items.push(['ok', `Every part is inside ${mat.minPartIn[0]}" × ${mat.minPartIn[1]}" to 30" × 44".`]);
    if (D.stability && D.stability.stableFrac >= 0.999) items.push(['ok', 'Balanced on its feet through the whole stride.']);
    $('checks').innerHTML = items.map(([lvl, t]) => `<li class="${lvl}"><span class="dot" aria-hidden="true"></span><span>${t}</span></li>`).join('');
    const blocked = D.issues.some((i) => i.level === 'fail');
    $('zip').disabled = blocked || !window.JSZip;
    $('zip').title = blocked ? 'Fix the failing checks first' : '';
  }

  // ------------------------------------------------------------------ parts
  function partSVG(part) {
    const b = part.bbox, pad = Math.max(b.w, b.h) * 0.06;
    const d = part.contours.map(CB.toSVGPath).join(' ');
    const fill = part.id === 'plate' ? 'var(--clawd)' : 'var(--metal)';
    const stroke = part.id === 'plate' ? 'var(--clawd-edge)' : 'var(--metal-edge)';
    return `<svg viewBox="${b.x0 - pad} ${-(b.y1 + pad)} ${b.w + 2 * pad} ${b.h + 2 * pad}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <g transform="scale(1,-1)"><path d="${d}" fill="${fill}" fill-rule="evenodd" stroke="${stroke}" stroke-width="${Math.max(b.w, b.h) / 180}"/></g></svg>`;
  }

  function renderParts() {
    const local = !inViewer;
    const u = state.units;
    const dim = (mm) => (u === 'in' ? `${(mm / IN).toFixed(2)}"` : `${mm.toFixed(1)}`);
    $('parts').innerHTML = D.cut.map((p) => `
      <article class="part">
        <div class="row"><h3>${p.name}</h3><span class="qty">×${p.qty}</span></div>
        ${partSVG(p)}
        <div>
          <div class="dims">${dim(p.bbox.w)} × ${dim(p.bbox.h)}${u === 'mm' ? ' mm' : ''}</div>
          <div class="dims">${p.centers}</div>
          <div class="file">${p.file}.dxf</div>
          ${p.coat && D.cfg.mat.coat ? '<div class="tag">2 of 4 in Safety Orange</div>' : ''}
        </div>
        ${local ? `<button type="button" data-dxf="${p.id}">Download DXF</button>` : ''}
      </article>`).join('');
    const n = D.cut.reduce((a, p) => a + p.qty, 0);
    const mat = D.cfg.mat;
    $('cutSummary').textContent = `${n} parts from ${D.cut.length} files, all ${mat.gauge} ${mat.name}. The ZIP holds every DXF plus the cut list, hardware list and assembly notes.`;
    $('orderNote').innerHTML = mat.coat
      ? `<b>Ordering:</b> upload each DXF as its own part, confirm ${u === 'in' ? 'inches' : 'millimetres'} as the units, and set the quantities shown. Add the Clawd plate twice: 2 with Safety Orange powder coat for the outside, 2 bare for the inside. Powder coat narrows holes by up to 0.25 mm, which the plate holes already allow for. Deburring is worth adding on the links.`
      : `<b>Ordering:</b> upload each DXF as its own part, confirm ${u === 'in' ? 'inches' : 'millimetres'} as the units, and set the quantities shown. Acrylic can't be powder coated, so Clawd stays clear or whatever colour you pick.`;
  }

  // ------------------------------------------------------------------ assembly
  function renderAssembly() {
    const A = D.assembly, S = D.stack;
    if (!A) {
      for (const id of ['layerMap', 'pinTable', 'armTable', 'bom', 'shaftTable', 'steps']) $(id).innerHTML = '';
      $('sideBar').innerHTML = '';
      $('dial').innerHTML = '';
      return;
    }
    const t = D.cfg.t, pitch = D.cfg.pitch;

    // Side cross-section bar.
    const W = 760, H = 96, total = S.sideWidth, k = (W - 20) / total;
    const x = (z) => 10 + z * k;
    let svg = `<rect x="${x(0)}" y="30" width="${t * k}" height="40" fill="var(--clawd)"/>`;
    svg += `<rect x="${x(S.outerZ)}" y="30" width="${t * k}" height="40" fill="var(--clawd)"/>`;
    svg += `<text x="${x(0)}" y="20" font-size="11" fill="var(--muted)">inner plate</text>`;
    svg += `<text x="${x(S.outerZ + t)}" y="20" font-size="11" fill="var(--muted)" text-anchor="end">outer plate</text>`;
    S.slabZ.forEach((z0, i) => {
      for (let L = 0; L < S.N; L++) {
        const arm = L === A.aI || L === A.aO;
        svg += `<rect x="${x(z0 + L * pitch)}" y="36" width="${t * k}" height="28" fill="${arm ? 'var(--ink)' : 'var(--metal)'}"/>`;
      }
      const mid = x(z0 + S.slabT / 2);
      svg += `<text x="${mid}" y="20" font-size="11" fill="var(--ink)" text-anchor="middle">phase ${i + 1}</text>`;
      svg += `<text x="${mid}" y="86" font-size="10.5" fill="var(--muted)" text-anchor="middle">${S.slabT.toFixed(1)}</text>`;
    });
    const gapLbl = (z0, z1) => `<text x="${(x(z0) + x(z1)) / 2}" y="86" font-size="10.5" fill="var(--muted)" text-anchor="middle">${(z1 - z0).toFixed(1)}</text>`;
    svg += gapLbl(t, S.slabZ[0]);
    svg += `<line x1="${x(0)}" x2="${x(S.sideWidth)}" y1="50" y2="50" stroke="var(--line-strong)" stroke-dasharray="2 3"/>`;
    for (let i = 0; i < S.slabZ.length - 1; i++) svg += gapLbl(S.slabZ[i] + S.slabT, S.slabZ[i + 1]);
    svg += gapLbl(S.slabZ[S.slabZ.length - 1] + S.slabT, S.outerZ);
    $('sideLabel').textContent = `One side, to scale: ${total.toFixed(1)} mm from inner plate to outer plate (${D.params.bodyGap} mm body gap between the sides)`;
    $('sideBar').setAttribute('viewBox', `0 0 ${W} ${H}`);
    $('sideBar').setAttribute('width', W);
    $('sideBar').style.maxWidth = '100%';
    $('sideBar').style.minWidth = '520px';
    $('sideBar').innerHTML = svg;

    // Layer map.
    $('layerMap').innerHTML = A.layers.map((ids, L) => `
      <div class="layer"><span class="ln">Layer ${L + 1}</span><div class="stackseq">${ids.map((id) => `<span class="pc ${id.startsWith('arm') ? 'arm' : ''}">${partName(id)}</span>`).join('')}</div></div>`).join('');

    // Pin stacks.
    const seqHTML = (items) => items.map((it) => {
      if (it.part === 'plate') return '<span class="pc plate">plate</span>';
      if (it.part) return `<span class="pc">${partName(it.part)}${it.slab !== undefined ? ` ·${it.slab + 1}` : ''}</span>`;
      if (it.washer) return '<span class="pc gap">washer</span>';
      if (it.spacer) return `<span class="pc gap">spacer ${it.spacer.toFixed(1)}</span>`;
      return `<span class="pc gap">bare ${it.bare.toFixed(1)}</span>`;
    }).join('');
    let rows = A.pinStacks.map((p) => `<tr><td>${jointName(p.joint)}</td><td><div class="stackseq">${seqHTML(p.items)}</div></td><td class="num">${state.hardware}×${p.bolt}</td></tr>`).join('');
    rows += A.hipStacks.map((p) => `<tr><td>${jointName(p.joint)}</td><td><div class="stackseq">${seqHTML(p.items)}</div></td><td class="num">rod</td></tr>`).join('');
    $('pinTable').innerHTML = `<thead><tr><th>Pivot</th><th>Stack</th><th class="num">Screw</th></tr></thead><tbody>${rows}</tbody>`;

    // Crank dial.
    let dial = `<circle r="112" fill="none" stroke="var(--line)" />`;
    for (let a = 0; a < 360; a += 30) {
      const r0 = a % 90 === 0 ? 102 : 107, c = Math.cos((a * Math.PI) / 180), s = -Math.sin((a * Math.PI) / 180);
      dial += `<line x1="${r0 * c}" y1="${r0 * s}" x2="${112 * c}" y2="${112 * s}" stroke="var(--line-strong)"/>`;
    }
    dial += `<text x="120" y="4" font-size="10" fill="var(--muted)">front</text>`;
    const arms = [];
    for (const side of ['L', 'R']) D.phases[side].forEach((ph, i) => {
      const a = ((ph % 360) + 360) % 360, key = Math.min(a, ((180 - a) % 360 + 360) % 360);
      arms.push({ side, i, ph: a, key, flip: key !== a });
    });
    for (const arm of arms) {
      const c = Math.cos((arm.ph * Math.PI) / 180), s = -Math.sin((arm.ph * Math.PI) / 180);
      const col = arm.side === 'L' ? 'var(--accent)' : 'var(--steel)';
      dial += `<line x1="0" y1="0" x2="${86 * c}" y2="${86 * s}" stroke="${col}" stroke-width="5" stroke-linecap="round" ${arm.side === 'R' ? 'stroke-dasharray="7 5"' : ''}/>`;
      dial += `<text x="${96 * c}" y="${96 * s + 4}" font-size="11" text-anchor="middle" fill="${col}">${arm.side}${arm.i + 1}</text>`;
    }
    dial += `<circle r="14" fill="var(--surface-2)" stroke="var(--ink)" stroke-width="1.5"/>`;
    dial += `<line x1="-9" x2="9" y1="-10" y2="-10" stroke="var(--ink)" stroke-width="2"/>`;
    const rot = D.dir > 0 ? 'counter-clockwise' : 'clockwise';
    dial += `<text x="0" y="128" font-size="10" text-anchor="middle" fill="var(--muted)">turns ${rot} to walk forward</text>`;
    $('dial').innerHTML = dial;
    $('armTable').innerHTML = `<thead><tr><th>Arm</th><th class="num">Angle</th><th>File</th><th>Face</th></tr></thead><tbody>${arms.map((a) => `<tr><td>${a.side === 'L' ? 'Left' : 'Right'} phase ${a.i + 1} <span class="muted">(×2)</span></td><td class="num">${Math.round(a.ph)}°</td><td class="mono">crank-arm-${String(a.key).padStart(3, '0')}</td><td>${a.flip ? 'flipped' : 'as cut'}</td></tr>`).join('')}</tbody>`;

    // Hardware and shaft.
    $('bom').innerHTML = `<thead><tr><th class="num">Qty</th><th>Item</th></tr></thead><tbody>${A.items.map((i) => `<tr><td class="num">${i.qty}</td><td>${i.what}</td></tr>`).join('')}</tbody>`;
    $('shaftTable').innerHTML = `<thead><tr><th>Piece</th><th class="num">Length</th><th class="num">Qty</th></tr></thead><tbody>${A.cuts.map((c) => `<tr><td>${c.what}</td><td class="num">${c.len.toFixed(1)} mm</td><td class="num">${c.qty}</td></tr>`).join('')}</tbody>`;

    $('steps').innerHTML = buildSteps().map((s) => `<li>${s}</li>`).join('');
  }

  function buildSteps() {
    const hw = state.hardware, n = state.slabs;
    return [
      'Order the parts (ZIP → <span class="mono">dxf/</span>) and the hardware list. While you wait, cut the D-shaft into the pieces listed and deburr the ends.',
      'Test-fit a crank arm on a scrap of shaft. The D-holes leave 0.1 mm of air; if a laser-cut hole comes out tight, dress the flat with a file rather than the round.',
      `Build the crankshaft one side at a time, inner plate outward. Main axle into the first inner crank arm, crank pin into its partner, link stub to the next phase, and so on. Every arm points where the phasing dial says with the shaft flat up. Lock each D-shaft piece with retaining compound and let it cure before loading it.`,
      `Assemble the ${4 * n} legs off the machine. Each bolted pivot follows its stack in the table: ${hw} screw from the head side, a washer against every moving face, nylon-insert nut snugged and then backed off until the joint swings under its own weight.`,
      'Thread the hip rods through the inner plate, then slide each leg\'s hip triangle and link c on in phase order, with spacers where the stack calls for them and bare rod where link k sweeps past. Drop links j and k onto their crank pins as you reach each phase.',
      'Close the side with the outer plate, then join the two sides with the main axle and the two tie rods. Turn the crank a full revolution by hand before tightening the tie-rod nuts: nothing should tick or bind.',
      'Fit the hand crank on the end stub with a shaft collar. Put the feet on a rug, turn the handle, and Clawd walks.',
    ];
  }

  // ------------------------------------------------------------------ export
  const inViewer = !!(window.claude && typeof window.claude.use === 'function');
  let downloads = null;
  if (inViewer) {
    window.claude.use('downloads').then((d) => { downloads = d; if (!d) { $('zip').hidden = true; } }).catch(() => { $('zip').hidden = true; });
  }

  let toastTimer = 0;
  function toast(msg) {
    const el = $('toast');
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 2600);
  }

  async function saveFile(filename, data) {
    if (inViewer) {
      if (!downloads) { toast('Downloads aren’t available in this view.'); return; }
      try {
        await downloads.save({ filename, data });
        toast(`Saved ${filename}`);
      } catch (e) {
        const code = e && e.code;
        if (code === 'declined') toast('Download cancelled.');
        else if (code === 'rate_limited') toast('A save prompt is already open.');
        else toast(`Couldn’t save the file (${code || 'error'}).`);
      }
      return;
    }
    const blob = data instanceof Blob ? data : new Blob([data], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function csv(rows) {
    return rows.map((r) => r.map((v) => { const s = String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; }).join(',')).join('\n') + '\n';
  }

  function assemblyText() {
    const A = D.assembly, P = D.params, mat = D.cfg.mat;
    const L = [];
    L.push(`CLAWDBEAST — ${4 * P.slabs} legs, scale ${P.scale} mm per Jansen unit`);
    L.push(`Material: ${mat.gauge} ${mat.name}. Pivots: ${P.hardware}. Overall ${Math.round(D.size.length)} x ${Math.round(D.size.width)} x ${Math.round(D.size.height)} mm (L x W x H).`);
    L.push('');
    L.push('LAYERS IN ONE PHASE (inner to outer)');
    A.layers.forEach((ids, i) => L.push(`  ${i + 1}: ${ids.map(partName).join(', ')}`));
    L.push('');
    L.push('SIDE STACK (mm from the inner plate\'s inner face)');
    L.push(`  inner plate 0.0 - ${D.cfg.t.toFixed(2)}`);
    D.stack.slabZ.forEach((z, i) => L.push(`  phase ${i + 1}: ${z.toFixed(2)} - ${(z + D.stack.slabT).toFixed(2)}`));
    L.push(`  outer plate ${D.stack.outerZ.toFixed(2)} - ${(D.stack.outerZ + D.cfg.t).toFixed(2)}`);
    L.push('');
    L.push('PIVOT STACKS (inner to outer)');
    const seq = (items) => items.map((it) => it.part === 'plate' ? 'plate' : it.part ? partName(it.part) + (it.slab !== undefined ? ` (phase ${it.slab + 1})` : '') : it.washer ? 'washer' : it.spacer ? `spacer ${it.spacer.toFixed(1)}` : `bare rod ${it.bare.toFixed(1)}`).join(' | ');
    for (const p of A.pinStacks) L.push(`  ${p.joint}: ${P.hardware}x${p.bolt}, head ${p.headLow ? 'inner' : 'outer'} side: ${seq(p.items)}`);
    for (const p of A.hipStacks) L.push(`  ${p.joint} hip rod: ${seq(p.items)}`);
    L.push('');
    L.push('CRANK PHASING (seen from the left, shaft flat up, 0 deg = pointing forward, angles counter-clockwise)');
    for (const side of ['L', 'R']) D.phases[side].forEach((ph, i) => {
      const a = ((ph % 360) + 360) % 360, key = Math.min(a, ((180 - a) % 360 + 360) % 360);
      L.push(`  ${side === 'L' ? 'Left ' : 'Right'} phase ${i + 1}: ${Math.round(a)} deg, 2x clawdbeast-crank-arm-${String(key).padStart(3, '0')} ${key !== a ? '(flipped)' : '(as cut)'}`);
    });
    L.push('');
    L.push('D-SHAFT CUT LIST');
    for (const c of A.cuts) L.push(`  ${c.qty} x ${c.len.toFixed(1)} mm  ${c.what}`);
    L.push('');
    L.push('BUILD');
    buildSteps().forEach((s, i) => L.push(`  ${i + 1}. ${s.replace(/<[^>]+>/g, '')}`));
    return L.join('\n') + '\n';
  }

  async function downloadZip() {
    if (!window.JSZip) return;
    const zip = new window.JSZip();
    const u = state.units, mat = D.cfg.mat;
    for (const p of D.cut) zip.file(`dxf/${p.file}.dxf`, CB.toDXF(p, u));
    const rows = [['file', 'part', 'quantity', 'material', 'thickness_in', 'finish', 'width_in', 'height_in']];
    for (const p of D.cut) {
      const base = [`${p.file}.dxf`, p.name];
      const tail = [mat.name, mat.gauge.replace('"', ''), '', (p.bbox.w / IN).toFixed(3), (p.bbox.h / IN).toFixed(3)];
      if (p.coat && mat.coat) {
        rows.push([...base, 2, tail[0], tail[1], 'Powder coat, Safety Orange', tail[3], tail[4]]);
        rows.push([...base, 2, tail[0], tail[1], 'None', tail[3], tail[4]]);
      } else rows.push([...base, p.qty, tail[0], tail[1], 'None', tail[3], tail[4]]);
    }
    zip.file('cut-list.csv', csv(rows));
    zip.file('hardware.csv', csv([['quantity', 'item'], ...D.assembly.items.map((i) => [i.qty, i.what])]));
    zip.file('ASSEMBLY.txt', assemblyText());
    zip.file('design.json', JSON.stringify(D.params, null, 2) + '\n');
    const blob = await zip.generateAsync({ type: 'blob' });
    await saveFile(`clawdbeast-${4 * state.slabs}-legs-${u}.zip`, blob);
  }

  $('zip').addEventListener('click', downloadZip);
  $('parts').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-dxf]');
    if (!b) return;
    const p = D.cut.find((c) => c.id === b.dataset.dxf);
    saveFile(`${p.file}.dxf`, CB.toDXF(p, state.units));
  });

  // ------------------------------------------------------------------ controls
  let pending = 0;
  $('scale').addEventListener('input', (e) => {
    state.scale = +e.target.value;
    $('scaleOut').textContent = `${state.scale.toFixed(2)} mm/unit`;
    clearTimeout(pending);
    pending = setTimeout(rebuild, 80);
  });
  const mats = CB.MATERIALS;
  $('material').innerHTML = Object.keys(mats).map((k) => `<option value="${k}">${mats[k].gauge} ${mats[k].name}</option>`).join('');
  $('material').value = state.material;
  $('material').addEventListener('change', (e) => { state.material = e.target.value; rebuild(); });
  for (const [id, key, cast] of [['hw', 'hardware', String], ['legs', 'slabs', Number], ['units', 'units', String]]) {
    $(id).addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      state[key] = cast(b.dataset.v);
      if (key === 'units') { renderPanel(); renderParts(); remember(); } else rebuild();
    });
  }
  $('play').addEventListener('click', () => { state.playing = !state.playing; syncPlay(); remember(); });
  function syncPlay() {
    $('play').textContent = state.playing ? 'Pause' : 'Walk';
    $('play').setAttribute('aria-pressed', String(state.playing));
  }
  $('rpm').value = state.rpm;
  $('rpmOut').textContent = `${state.rpm} rpm`;
  $('rpm').addEventListener('input', (e) => { state.rpm = +e.target.value; $('rpmOut').textContent = `${state.rpm} rpm`; remember(); });
  $('paths').checked = state.paths;
  $('xray').checked = state.xray;
  $('paths').addEventListener('change', (e) => { state.paths = e.target.checked; remember(); draw(); });
  $('xray').addEventListener('change', (e) => { state.xray = e.target.checked; remember(); draw(); });

  // ------------------------------------------------------------------ stage
  const canvas = $('stage');
  const ctx = canvas.getContext('2d');
  let colors = {};
  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    for (const k of ['stage', 'stage-line', 'ink', 'muted', 'line', 'line-strong', 'clawd', 'clawd-edge', 'metal', 'metal-edge', 'steel', 'accent', 'surface', 'mono']) colors[k] = cs.getPropertyValue('--' + k).trim();
  }
  readColors();
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readColors(); draw(); });
  new MutationObserver(() => { readColors(); draw(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // Blend a colour toward the stage background to push it back in depth.
  function parseColor(c) {
    if (c.startsWith('#')) { const h = c.slice(1); const v = h.length === 3 ? h.split('').map((x) => x + x).join('') : h; return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16)); }
    const m = c.match(/[\d.]+/g); return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
  }
  function haze(c, amount) {
    const a = parseColor(c), b = parseColor(colors.stage);
    return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * amount)).join(',')})`;
  }

  // Back-to-front drawing order: far side outer plate first, near side
  // outer plate last. Each entry is a plate or one layer of one phase.
  function buildScene() {
    sceneItems = [];
    footLoops = [];
    if (!D.layout) return;
    const n = state.slabs, N = D.layout.N, S = D.stack, t = D.cfg.t, half = D.params.bodyGap / 2;
    const zmax = half + S.sideWidth;
    const depth = (zView) => 0.58 * (zmax - zView) / (2 * zmax);
    const zL = (k, L) => half + S.slabZ[k] + L * D.cfg.pitch + t / 2;
    sceneItems.push({ plate: true, side: 'R', haze: depth(-(half + S.outerZ)) });
    for (let k = n - 1; k >= 0; k--) {
      for (let L = N - 1; L >= 0; L--) sceneItems.push({ side: 'R', k, L, haze: depth(-zL(k, L)) });
      sceneItems.push({ joints: true, side: 'R', k, haze: depth(-zL(k, 0)) });
    }
    sceneItems.push({ plate: true, side: 'R', haze: depth(-half) });
    sceneItems.push({ plate: true, side: 'L', haze: depth(half), inner: true });
    for (let k = 0; k < n; k++) {
      for (let L = 0; L < N; L++) sceneItems.push({ side: 'L', k, L, haze: depth(zL(k, L)) });
      sceneItems.push({ joints: true, side: 'L', k, haze: depth(zL(k, N - 1)) });
    }
    sceneItems.push({ plate: true, side: 'L', haze: 0, outer: true });

    // Foot paths in the body frame, bottom of the foot pad.
    for (const front of [false, true]) {
      const pts = [];
      for (let i = 0; i <= 120; i++) { const G = CB.legPose((i / 120) * TAU, state.scale, front).G; pts.push([G[0], G[1] - D.cfg.R]); }
      footLoops.push(pts);
    }
    prevFeet = null;
  }

  let theta = 0, bodyX = 0, camX = 0, prevFeet = null, bodyY = 0, feetDown = 0;
  function poseAll(th) {
    const out = { L: [], R: [] };
    for (const side of ['L', 'R']) D.phases[side].forEach((ph) => out[side].push(CB.slabPose(th + (ph * Math.PI) / 180, state.scale)));
    return out;
  }
  function feetOf(poses) {
    const f = [];
    const R = D.cfg.R;
    for (const side of ['L', 'R']) poses[side].forEach((p) => { f.push([p.GR[0], p.GR[1] - R]); f.push([p.GF[0], p.GF[1] - R]); });
    return f;
  }
  function advance(dt) {
    theta += D.dir * (state.rpm / 60) * TAU * dt;
    const feet = feetOf(poseAll(theta));
    let lo = 0;
    for (let i = 1; i < feet.length; i++) if (feet[i][1] < feet[lo][1]) lo = i;
    if (prevFeet && prevFeet.length === feet.length) bodyX -= feet[lo][0] - prevFeet[lo][0];
    prevFeet = feet;
    bodyY = -feet[lo][1];
    feetDown = feet.filter((f) => f[1] < feet[lo][1] + 0.5 * state.scale).length;
  }

  function contourPath(c, T) {
    // T maps local [x,y] to screen; mirror flips arc direction.
    const pts = c.map((v) => T.p([v.x, v.y]));
    for (let i = 0; i < c.length; i++) {
      const P = pts[i], Q = pts[(i + 1) % c.length], b = c[i].b * (T.flip ? -1 : 1);
      if (i === 0) ctx.moveTo(P[0], P[1]);
      if (Math.abs(b) < 1e-12) ctx.lineTo(Q[0], Q[1]);
      else {
        const a = CB.arcCenter({ x: P[0], y: P[1] }, { x: Q[0], y: Q[1] }, b);
        const a0 = Math.atan2(P[1] - a.y, P[0] - a.x);
        ctx.arc(a.x, a.y, a.r, a0, a0 + a.th, a.th < 0);
      }
    }
    ctx.closePath();
  }
  function circlePath(cx, cy, r) { ctx.moveTo(cx + r, cy); ctx.arc(cx, cy, r, 0, TAU); }

  let view = { k: 1, ox: 0, oy: 0 };
  // World (mm, y up) to screen (px, y down). The screen is y-down, so a
  // counter-clockwise arc in the world is clockwise on screen: the bulge
  // sign flips once for the y flip, handled by T.flip parity below.
  function toScreen(x, y) { return [view.ox + (x + bodyX - camX) * view.k, view.oy - (y + bodyY) * view.k]; }

  function partTransform(P, pose) {
    const cut = cutByKind[P.kind];
    const world = P.circles.map((c) => pose[c.k]);
    const local = cut.anchors;
    const w0 = world[0], w1 = world[1];
    const rot = Math.atan2(w1[1] - w0[1], w1[0] - w0[0]);
    let mirror = false;
    if (local.length === 3) {
      const cl = local[1][0] * local[2][1] - local[1][1] * local[2][0];
      const cw = (w1[0] - w0[0]) * (world[2][1] - w0[1]) - (w1[1] - w0[1]) * (world[2][0] - w0[0]);
      mirror = cl * cw < 0;
    }
    const c = Math.cos(rot), s = Math.sin(rot);
    return {
      flip: !mirror, // y-flip to screen, plus an optional mirror
      p([x, y]) {
        const yy = mirror ? -y : y;
        return toScreen(w0[0] + x * c - yy * s, w0[1] + x * s + yy * c);
      },
    };
  }

  function drawPart(P, pose, hz) {
    ctx.beginPath();
    if (P.kind === 'arm') {
      const circles = P.circles.map((c) => { const [x, y] = pose[c.k]; return { x, y, r: c.r }; });
      const hull = CB.hullOfCircles(circles);
      contourPath(hull, { flip: true, p: ([x, y]) => toScreen(x, y) });
      for (const c of circles) { const [sx, sy] = toScreen(c.x, c.y); circlePath(sx, sy, 3.1 * view.k); }
      ctx.fillStyle = haze(colors.steel, hz);
      ctx.strokeStyle = haze(colors['metal-edge'], hz);
    } else {
      const cut = cutByKind[P.kind];
      const T = partTransform(P, pose);
      for (const c of cut.contours) contourPath(c, T);
      ctx.fillStyle = haze(colors.metal, hz);
      ctx.strokeStyle = haze(colors['metal-edge'], hz);
    }
    ctx.lineWidth = Math.max(0.6, 0.35 * view.k);
    ctx.fill('evenodd');
    ctx.stroke();
  }

  function drawPlate(item) {
    const plate = D.cut.find((c) => c.id === 'plate');
    const T = { flip: true, p: ([x, y]) => toScreen(x, y) };
    ctx.save();
    if (state.xray) ctx.globalAlpha = item.outer ? 0.16 : 0.1;
    ctx.beginPath();
    for (const c of plate.contours) contourPath(c, T);
    ctx.fillStyle = haze(colors.clawd, item.haze);
    ctx.fill('evenodd');
    ctx.strokeStyle = haze(colors['clawd-edge'], item.haze);
    ctx.lineWidth = Math.max(0.8, 0.5 * view.k);
    ctx.stroke();
    ctx.restore();
    if (item.outer) {
      // Rod ends and the axle showing through the outer plate.
      ctx.fillStyle = colors.ink;
      for (const [x, y] of [...D.plate.pivots, ...D.plate.ties, [0, 0]]) {
        const [sx, sy] = toScreen(x, y);
        ctx.beginPath(); circlePath(sx, sy, (x === 0 && y === 0 ? 4.2 : 2.6) * view.k); ctx.fill();
      }
    }
  }

  function drawJoints(item, poses) {
    const pose = poses[item.side][item.k];
    ctx.fillStyle = haze(colors.ink, Math.min(0.85, item.haze + 0.15));
    for (const J of ['CR', 'DR', 'ER', 'FR', 'CF', 'DF', 'EF', 'FF']) {
      const [sx, sy] = toScreen(pose[J][0], pose[J][1]);
      ctx.beginPath(); circlePath(sx, sy, 2.7 * view.k); ctx.fill();
    }
    const [ax, ay] = toScreen(pose.A[0], pose.A[1]);
    ctx.beginPath(); circlePath(ax, ay, 3.4 * view.k); ctx.fill();
  }

  function drawGround(W, H) {
    const gy = view.oy;
    ctx.strokeStyle = colors['stage-line'];
    ctx.lineWidth = 1;
    // Faint 50 mm grid anchored to the ground so it scrolls as Clawd walks.
    const step = 50 * view.k;
    const x0 = view.ox - (camX % 50) * view.k;
    ctx.beginPath();
    for (let x = x0 - step * Math.ceil(x0 / step); x < W; x += step) { ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, gy); }
    for (let y = gy - step; y > 0; y -= step) { ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(W, Math.round(y) + 0.5); }
    ctx.stroke();
    // Ground and a millimetre rule.
    ctx.strokeStyle = colors['line-strong'];
    ctx.beginPath(); ctx.moveTo(0, gy + 0.5); ctx.lineTo(W, gy + 0.5); ctx.stroke();
    ctx.strokeStyle = colors.muted;
    ctx.fillStyle = colors.muted;
    ctx.font = `500 10px ${colors.mono || 'monospace'}`;
    ctx.textAlign = 'center';
    const left = camX - view.ox / view.k, right = camX + (W - view.ox) / view.k;
    ctx.beginPath();
    for (let mm = Math.floor(left / 10) * 10; mm <= right; mm += 10) {
      const x = Math.round(view.ox + (mm - camX) * view.k) + 0.5;
      const len = mm % 100 === 0 ? 9 : mm % 50 === 0 ? 6 : 3;
      if (view.k * 10 < 3 && mm % 50 !== 0) continue;
      ctx.moveTo(x, gy + 1); ctx.lineTo(x, gy + 1 + len);
    }
    ctx.stroke();
    for (let mm = Math.floor(left / 100) * 100; mm <= right; mm += 100) {
      const x = view.ox + (mm - camX) * view.k;
      if (x > W - 36) continue; // leave room for the unit label
      ctx.fillText(String(mm), x, gy + 22);
    }
    ctx.textAlign = 'right';
    ctx.fillText('mm', W - 8, gy + 22);
  }

  let cssW = 0, cssH = 0, dpr = 1;
  function resize() {
    const r = canvas.getBoundingClientRect();
    dpr = Math.min(2, window.devicePixelRatio || 1);
    cssW = r.width; cssH = r.height;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    draw();
  }
  new ResizeObserver(resize).observe(canvas);

  function draw() {
    if (!D || !cssW) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = colors.stage;
    ctx.fillRect(0, 0, cssW, cssH);
    const s = D.size;
    view.k = Math.min(cssW / (s.length * 1.45), (cssH - 44) / (s.height * 1.22));
    view.ox = cssW / 2;
    view.oy = cssH - 34;
    drawGround(cssW, cssH);
    if (!D.layout) {
      ctx.fillStyle = colors.muted; ctx.textAlign = 'center'; ctx.font = '500 14px sans-serif';
      ctx.fillText('No working layout at this size. Make it bigger or pick thinner material.', cssW / 2, cssH / 2);
      return;
    }
    const poses = poseAll(theta);
    for (const item of sceneItems) {
      if (item.plate) drawPlate(item);
      else if (item.joints) drawJoints(item, poses);
      else {
        const pose = poses[item.side][item.k];
        for (const id of D.assembly.layers[item.L]) drawPart(partsById[id], pose, item.haze);
      }
    }
    if (state.paths) {
      ctx.save();
      ctx.strokeStyle = colors.accent;
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.4;
      for (const loop of footLoops) {
        ctx.beginPath();
        loop.forEach(([x, y], i) => { const [sx, sy] = toScreen(x, y); if (i) ctx.lineTo(sx, sy); else ctx.moveTo(sx, sy); });
        ctx.stroke();
      }
      ctx.restore();
    }
    const deg = ((((theta * 180) / Math.PI) % 360) + 360) % 360;
    $('hud').innerHTML = `crank <b>${deg.toFixed(0).padStart(3, ' ')}°</b> · <b>${feetDown}</b> feet down · walked <b>${Math.abs(bodyX / 1000).toFixed(2)}</b> m`;
  }

  let last = 0;
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000 || 0);
    last = now;
    if (state.playing && D && D.layout) {
      advance(dt);
      camX += (bodyX - camX) * Math.min(1, dt * 4);
    }
    draw();
    requestAnimationFrame(frame);
  }

  syncPlay();
  rebuild();
  if (D.layout) { advance(0); camX = bodyX; }
  requestAnimationFrame(frame);
})();
