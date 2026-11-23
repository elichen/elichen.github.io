// Zero-dependency headless Chrome runner (uses Node 24 built-in http + WebSocket CDP)
// with Cache-Control: no-store and WebGPU Metal enabled.
//
// Usage:
//   node tools/chrome_test.mjs verify [goals=20] [backend=] [mode=hyb-cem]
//   node tools/chrome_test.mjs page   [seconds=12]

import http from 'node:http';
import { readFileSync, writeFileSync, statSync, existsSync } from 'node:fs';
import { extname, join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.xml': 'application/xml',
  '.bin': 'application/octet-stream',
  '.stl': 'application/octet-stream',
  '.png': 'image/png',
  '.wasm': 'application/wasm',
};

function startServer(dir) {
  return new Promise(res => {
    const srv = http.createServer((req, resp) => {
      const urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
      const p = join(dir, urlPath === '/' ? 'index.html' : urlPath);
      if (!existsSync(p) || statSync(p).isDirectory()) {
        resp.writeHead(404);
        resp.end('Not found');
        return;
      }
      resp.writeHead(200, {
        'Content-Type': MIME[extname(p)] || 'application/octet-stream',
        'Cache-Control': 'no-store, no-cache, must-revalidate',
      });
      resp.end(readFileSync(p));
    });
    srv.listen(0, '127.0.0.1', () => res({ srv, port: srv.address().port }));
  });
}

async function launchChrome(url, width = 1280, height = 900) {
  const debugPort = 9222 + Math.floor(Math.random() * 500);
  const proc = spawn(CHROME, [
    '--headless=new',
    `--remote-debugging-port=${debugPort}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--enable-unsafe-webgpu',
    '--use-angle=metal',
    `--window-size=${width},${height}`,
    '--user-data-dir=/tmp/chrome-wm-test-' + debugPort,
    url,
  ], { stdio: ['ignore', 'pipe', 'pipe'] });

  // Wait for DevTools HTTP endpoint
  let target = null;
  for (let i = 0; i < 60; i++) {
    try {
      const list = await fetch(`http://127.0.0.1:${debugPort}/json/list`).then(r => r.json());
      target = list.find(t => t.type === 'page' && t.webSocketDebuggerUrl);
      if (target) break;
    } catch {}
    await new Promise(r => setTimeout(r, 200));
  }
  if (!target) throw new Error('Chrome DevTools failed to start');

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  let seq = 1;
  const pending = new Map();
  const errors = [];
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { res, rej } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) rej(new Error(msg.error.message));
      else res(msg.result);
    } else if (msg.method === 'Runtime.consoleAPICalled') {
      const txt = (msg.params.args || []).map(a => a.value ?? a.description ?? '').join(' ');
      if (msg.params.type === 'error') { errors.push(txt); console.error('[CHROME ERR]', txt); }
      else console.log('[CHROME]', txt);
    } else if (msg.method === 'Runtime.exceptionThrown') {
      const desc = msg.params.exceptionDetails?.exception?.description || msg.params.exceptionDetails?.text || 'exception';
      errors.push(desc);
      console.error('[CHROME EXC]', desc);
    }
  };
  const send = (method, params = {}) => new Promise((res, rej) => {
    const id = seq++;
    pending.set(id, { res, rej });
    ws.send(JSON.stringify({ id, method, params }));
  });
  await send('Runtime.enable');
  await send('Page.enable');
  return { proc, ws, send, errors, close: () => { try { ws.close(); } catch {} try { proc.kill('SIGKILL'); } catch {} } };
}

const cmd = process.argv[2] || 'verify';
const { srv, port } = await startServer(root);

try {
  if (cmd === 'verify') {
    const goals = process.argv[3] || '10';
    const backend = process.argv[4] || '';
    const mode = process.argv[5] || 'hyb-cem';
    const url = `http://127.0.0.1:${port}/tools/verify.html?goals=${goals}&backend=${backend}&mode=${mode}`;
    const c = await launchChrome(url, 1280, 960);
    const t0 = Date.now();
    while (Date.now() - t0 < 240_000) {
      const r = await c.send('Runtime.evaluate', { expression: 'Boolean(window.__VERIFY_DONE__)', returnByValue: true });
      if (r.result?.value) break;
      await new Promise(r => setTimeout(r, 500));
    }
    const shot = await c.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync('/tmp/wm_verify.png', Buffer.from(shot.data, 'base64'));
    console.log('Saved screenshot -> /tmp/wm_verify.png');
    c.close();
    if (c.errors.length) process.exitCode = 1;
  } else if (cmd === 'page') {
    const secs = Number(process.argv[3] || '14');
    const url = `http://127.0.0.1:${port}/index.html`;
    const c = await launchChrome(url, 1280, 960);
    // Wait until status is hidden + run live for `secs`
    const t0 = Date.now();
    while (Date.now() - t0 < 60_000) {
      const r = await c.send('Runtime.evaluate', { expression: 'Boolean(window.demo && document.getElementById("status").hidden)', returnByValue: true });
      if (r.result?.value) break;
      await new Promise(r => setTimeout(r, 300));
    }
    await new Promise(r => setTimeout(r, secs * 1000));
    const stats = await c.send('Runtime.evaluate', {
      expression: `JSON.stringify({
        placed: document.getElementById('placed').textContent,
        dist: document.getElementById('dist').textContent,
        unc: document.getElementById('unc').textContent,
        ms: document.getElementById('replan-ms').textContent,
        backend: document.getElementById('backend-label').textContent,
        phase: document.getElementById('eye-phase').textContent,
      })`,
      returnByValue: true,
    });
    console.log('Live page stats:', stats.result?.value);
    const desk = await c.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync('/tmp/wm_desktop.png', Buffer.from(desk.data, 'base64'));
    await c.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
    await new Promise(r => setTimeout(r, 500));
    const mob = await c.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync('/tmp/wm_mobile.png', Buffer.from(mob.data, 'base64'));
    console.log('Saved /tmp/wm_desktop.png and /tmp/wm_mobile.png, errors:', c.errors.length);
    c.close();
    if (c.errors.length) process.exitCode = 1;
  } else if (cmd === 'arena') {
    const url = `http://127.0.0.1:${port}/index.html`;
    const c = await launchChrome(url, 1280, 1180);
    const t0 = Date.now();
    while (Date.now() - t0 < 90_000) {
      const r = await c.send('Runtime.evaluate', { expression: 'Boolean(window.demo && document.getElementById("status").hidden)', returnByValue: true });
      if (r.result?.value) break;
      await new Promise(r => setTimeout(r, 300));
    }
    const evalJs = async (expr) => (await c.send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.value;

    const runCmd = async (prompt, targetSteps = 78, maxMs = 14000) => {
      await evalJs(`window.demo.applyPrompt(${JSON.stringify(prompt)}, { fromUser: true })`);
      const c0 = JSON.parse(await evalJs('JSON.stringify(window.demo.env.cubePos())'));
      let maxZ = 0, maxV = 0, minPan = 99, maxPan = -99, maxEz = 0, minEz = 99, lastStep = 0;
      const tStart = Date.now();
      while (Date.now() - tStart < maxMs) {
        const st = JSON.parse(await evalJs(`JSON.stringify({
          c: window.demo.env.cubePos(),
          e: window.demo.env.effectorPos(),
          pan: window.demo.env.data.qpos[0],
          step: window.demo.getSkillStep(),
          v: Math.hypot(
            window.demo.env.data.qvel[window.demo.env.cubeDofAdr],
            window.demo.env.data.qvel[window.demo.env.cubeDofAdr + 1],
            window.demo.env.data.qvel[window.demo.env.cubeDofAdr + 2]
          ),
        })`));
        maxZ = Math.max(maxZ, st.c[2]);
        maxV = Math.max(maxV, st.v);
        minPan = Math.min(minPan, st.pan);
        maxPan = Math.max(maxPan, st.pan);
        maxEz = Math.max(maxEz, st.e[2]);
        minEz = Math.min(minEz, st.e[2]);
        lastStep = st.step;
        if (st.step >= targetSteps) break;
        await new Promise(r => setTimeout(r, 120));
      }
      const fin = JSON.parse(await evalJs(`JSON.stringify({
        c: window.demo.env.cubePos(),
        hud: document.getElementById('active-cmd').textContent,
        skill: document.getElementById('matched-skill').textContent,
        ortho: document.getElementById('trace-ortho').textContent,
        badge: document.getElementById('rot-status-badge').textContent,
        stepMs: window.demo.getLastStepMs(),
      })`));
      const fly = Math.hypot(fin.c[0] - c0[0], fin.c[1] - c0[1]);
      return { ...fin, steps: lastStep, wallMs: Date.now() - tStart, fly, maxZ, maxV, spinDeg: (maxPan - minPan) * 180 / Math.PI, maxEz, minEz };
    };

    // Verify ZERO auto-mode on load: arm stays idle (skillStep === 0) until user clicks or submits!
    await new Promise(r => setTimeout(r, 800));
    const idleStep = await evalJs('window.demo.getSkillStep()');
    console.log('0. Idle check (no auto-mode): skillStep =', idleStep);

    // Trigger first command via actual sample chip click!
    const rThrow = await (async () => {
      await evalJs(`document.querySelector('#prompt-chips [data-prompt="throw cube"]').click()`);
      const c0 = JSON.parse(await evalJs('JSON.stringify(window.demo.env.cubePos())'));
      let maxZ = 0, maxV = 0, minPan = 99, maxPan = -99, maxEz = 0, minEz = 99, lastStep = 0;
      const ezHist = [], exHist = [], eyHist = [], gHist = [];
      const tStart = Date.now();
      while (Date.now() - tStart < 14000) {
        const st = JSON.parse(await evalJs(`JSON.stringify({
          c: window.demo.env.cubePos(), e: window.demo.env.effectorPos(), g: window.demo.env.gripClosed(),
          pan: window.demo.env.data.qpos[0], step: window.demo.getSkillStep(),
          v: Math.hypot(window.demo.env.data.qvel[window.demo.env.cubeDofAdr], window.demo.env.data.qvel[window.demo.env.cubeDofAdr+1], window.demo.env.data.qvel[window.demo.env.cubeDofAdr+2]),
        })`));
        maxZ = Math.max(maxZ, st.c[2]); maxV = Math.max(maxV, st.v);
        minPan = Math.min(minPan, st.pan); maxPan = Math.max(maxPan, st.pan);
        maxEz = Math.max(maxEz, st.e[2]); minEz = Math.min(minEz, st.e[2]);
        if (st.step >= 18) { exHist.push(st.e[0]); eyHist.push(st.e[1]); ezHist.push(st.e[2]); gHist.push(st.g); }
        lastStep = st.step;
        if (st.step >= 78) break;
        await new Promise(r => setTimeout(r, 100));
      }
      const fin = JSON.parse(await evalJs(`JSON.stringify({
        c: window.demo.env.cubePos(), hud: document.getElementById('active-cmd').textContent,
        skill: document.getElementById('matched-skill').textContent, ortho: document.getElementById('trace-ortho').textContent,
        badge: document.getElementById('rot-status-badge').textContent, stepMs: window.demo.getLastStepMs(),
      })`));
      const fly = Math.hypot(fin.c[0] - c0[0], fin.c[1] - c0[1]);
      return { ...fin, steps: lastStep, wallMs: Date.now() - tStart, fly, maxZ, maxV, spinDeg: (maxPan - minPan) * 180 / Math.PI };
    })();
    console.log('1. "throw cube" (chip click):', rThrow);
    const desk = await c.send('Page.captureScreenshot', {
      format: 'png',
      clip: { x: 0, y: 0, width: 1280, height: 1460, scale: 1 },
      captureBeyondViewport: true,
    });
    writeFileSync('/tmp/wm_arena_desktop.png', Buffer.from(desk.data, 'base64'));

    const rSpin = await runCmd('rotate around your base', 78);
    console.log('2. "rotate around your base":', rSpin);

    const rWave = await runCmd('say hi', 78);
    console.log('3. "say hi":', rWave);

    await evalJs(`window.demo.resetEpisode()`);
    const rPlace = await runCmd('put the cube in the front left corner', 78);
    console.log('4. "put the cube in the front left corner":', rPlace);

    // Test Shuffled control switch
    await evalJs(`document.querySelector('#n-toggle [data-n="shuffled"]').click()`);
    await new Promise(r => setTimeout(r, 400));
    const rShuf = JSON.parse(await evalJs(`JSON.stringify({
      badge: document.getElementById('rot-status-badge').textContent,
      hud: document.getElementById('active-cmd').textContent,
    })`));
    console.log('5. Shuffled control:', rShuf);

    await evalJs(`document.querySelector('#n-toggle [data-n="10000"]').click()`);
    await new Promise(r => setTimeout(r, 400));
    await c.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
    await new Promise(r => setTimeout(r, 400));
    const mob = await c.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync('/tmp/wm_arena_mobile.png', Buffer.from(mob.data, 'base64'));
    console.log('Saved /tmp/wm_arena_desktop.png and /tmp/wm_arena_mobile.png, errors:', c.errors.length);
    c.close();
    const ok = c.errors.length === 0
      && (rThrow.fly > 0.25 || rThrow.maxV > 0.9 || rThrow.maxZ > 0.30)
      && rSpin.spinDeg > 150
      && rWave.maxEz > 0.42;
    console.log('CHROME_ARENA:', ok ? 'PASS' : 'CHECK');
    if (!ok) process.exitCode = 1;
  } else if (cmd === 'comp') {
    const url = `http://127.0.0.1:${port}/index.html`;
    const c = await launchChrome(url, 1280, 1180);
    const t0 = Date.now();
    while (Date.now() - t0 < 90_000) {
      const r = await c.send('Runtime.evaluate', { expression: 'Boolean(window.demo && document.getElementById("status").hidden)', returnByValue: true });
      if (r.result?.value) break;
      await new Promise(r => setTimeout(r, 300));
    }
    const evalJs = async (expr) => (await c.send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.value;

    const runCmd = async (prompt, targetSteps = 78, maxMs = 14000) => {
      await evalJs(`window.demo.applyPrompt(${JSON.stringify(prompt)}, { fromUser: true })`);
      const c0 = JSON.parse(await evalJs('JSON.stringify(window.demo.env.cubePos())'));
      let maxZ = 0, maxV = 0, minPan = 99, maxPan = -99, maxEz = 0, minEz = 99, lastStep = 0;
      const ezHist = [], exHist = [], eyHist = [], gHist = [];
      const tStart = Date.now();
      while (Date.now() - tStart < maxMs) {
        const st = JSON.parse(await evalJs(`JSON.stringify({
          c: window.demo.env.cubePos(),
          e: window.demo.env.effectorPos(),
          g: window.demo.env.gripClosed(),
          pan: window.demo.env.data.qpos[0],
          step: window.demo.getSkillStep(),
          v: Math.hypot(
            window.demo.env.data.qvel[window.demo.env.cubeDofAdr],
            window.demo.env.data.qvel[window.demo.env.cubeDofAdr + 1],
            window.demo.env.data.qvel[window.demo.env.cubeDofAdr + 2]
          ),
        })`));
        maxZ = Math.max(maxZ, st.c[2]);
        maxV = Math.max(maxV, st.v);
        minPan = Math.min(minPan, st.pan);
        maxPan = Math.max(maxPan, st.pan);
        maxEz = Math.max(maxEz, st.e[2]);
        minEz = Math.min(minEz, st.e[2]);
        if (st.step >= 18) {
          exHist.push(st.e[0]); eyHist.push(st.e[1]); ezHist.push(st.e[2]); gHist.push(st.g);
        }
        lastStep = st.step;
        if (st.step >= targetSteps) break;
        await new Promise(r => setTimeout(r, 100));
      }
      const fin = JSON.parse(await evalJs(`JSON.stringify({
        c: window.demo.env.cubePos(),
        hud: document.getElementById('active-cmd').textContent,
        skill: document.getElementById('matched-skill').textContent,
        ortho: document.getElementById('trace-ortho').textContent,
        badge: document.getElementById('rot-status-badge').textContent,
        stepMs: window.demo.getLastStepMs(),
      })`));
      const span = (a) => a.length ? (Math.max(...a) - Math.min(...a)) : 0;
      const mean = (a) => a.length ? (a.reduce((s, v) => s + v, 0) / a.length) : 0;
      const fly = Math.hypot(fin.c[0] - c0[0], fin.c[1] - c0[1]);
      return {
        ...fin, steps: lastStep, wallMs: Date.now() - tStart, fly, maxZ, maxV,
        spinDeg: (maxPan - minPan) * 180 / Math.PI,
        maxEz, minEz, zSpan: span(ezHist), xSpan: span(exHist), ySpan: span(eyHist),
        meanGrip: mean(gHist), meanEz: mean(ezHist),
      };
    };

    // Verify ZERO auto-mode on load: arm stays idle (skillStep === 0) until user clicks or submits!
    await new Promise(r => setTimeout(r, 800));
    const idleStep = await evalJs('window.demo.getSkillStep()');
    console.log('0. Idle check (no auto-mode): skillStep =', idleStep);

    // Trigger first command via actual sample chip click!
    const rThrow = await (async () => {
      await evalJs(`document.querySelector('#prompt-chips [data-prompt="throw cube"]').click()`);
      const c0 = JSON.parse(await evalJs('JSON.stringify(window.demo.env.cubePos())'));
      let maxZ = 0, maxV = 0, minPan = 99, maxPan = -99, maxEz = 0, minEz = 99, lastStep = 0;
      const ezHist = [], exHist = [], eyHist = [], gHist = [];
      const tStart = Date.now();
      while (Date.now() - tStart < 14000) {
        const st = JSON.parse(await evalJs(`JSON.stringify({
          c: window.demo.env.cubePos(), e: window.demo.env.effectorPos(), g: window.demo.env.gripClosed(),
          pan: window.demo.env.data.qpos[0], step: window.demo.getSkillStep(),
          v: Math.hypot(window.demo.env.data.qvel[window.demo.env.cubeDofAdr], window.demo.env.data.qvel[window.demo.env.cubeDofAdr+1], window.demo.env.data.qvel[window.demo.env.cubeDofAdr+2]),
        })`));
        maxZ = Math.max(maxZ, st.c[2]); maxV = Math.max(maxV, st.v);
        minPan = Math.min(minPan, st.pan); maxPan = Math.max(maxPan, st.pan);
        maxEz = Math.max(maxEz, st.e[2]); minEz = Math.min(minEz, st.e[2]);
        if (st.step >= 18) { exHist.push(st.e[0]); eyHist.push(st.e[1]); ezHist.push(st.e[2]); gHist.push(st.g); }
        lastStep = st.step;
        if (st.step >= 78) break;
        await new Promise(r => setTimeout(r, 100));
      }
      const fin = JSON.parse(await evalJs(`JSON.stringify({
        c: window.demo.env.cubePos(), hud: document.getElementById('active-cmd').textContent,
        skill: document.getElementById('matched-skill').textContent, ortho: document.getElementById('trace-ortho').textContent,
        badge: document.getElementById('rot-status-badge').textContent, stepMs: window.demo.getLastStepMs(),
      })`));
      const fly = Math.hypot(fin.c[0] - c0[0], fin.c[1] - c0[1]);
      return { ...fin, steps: lastStep, wallMs: Date.now() - tStart, fly, maxZ, maxV, spinDeg: (maxPan - minPan) * 180 / Math.PI };
    })();
    console.log('1. "throw cube" (chip click):', rThrow);
    const desk = await c.send('Page.captureScreenshot', {
      format: 'png', clip: { x: 0, y: 0, width: 1280, height: 1020, scale: 1 }, captureBeyondViewport: true,
    });
    writeFileSync('/tmp/wm_comp_desktop.png', Buffer.from(desk.data, 'base64'));

    await evalJs(`document.getElementById('inspector').open = true`);
    await new Promise(r => setTimeout(r, 120));
    const rFist = await runCmd('pump your hand like fists', 78);
    console.log('2. "pump your hand like fists":', rFist);
    const fistShot = await c.send('Page.captureScreenshot', {
      format: 'png', clip: { x: 0, y: 0, width: 1280, height: 1260, scale: 1 }, captureBeyondViewport: true,
    });
    writeFileSync('/tmp/wm_comp_fist.png', Buffer.from(fistShot.data, 'base64'));
    await evalJs(`document.getElementById('inspector').open = false`);

    const rSpin = await runCmd('rotate around your base', 78);
    console.log('3. "rotate around your base":', rSpin);

    const rCirc = await runCmd('slowly circle down low with your claw closed', 78);
    console.log('4. "slowly circle down low with your claw closed":', rCirc);

    await evalJs(`window.demo.resetEpisode()`);
    const rPlace = await runCmd('put the cube in the front left corner', 78);
    console.log('5. "put the cube in the front left corner":', rPlace);

    await evalJs(`document.querySelector('#n-toggle [data-n="shuffled"]').click()`);
    await new Promise(r => setTimeout(r, 400));
    const rShuf = JSON.parse(await evalJs(`JSON.stringify({
      badge: document.getElementById('rot-status-badge').textContent,
      hud: document.getElementById('active-cmd').textContent,
    })`));
    console.log('6. Shuffled control:', rShuf);

    await evalJs(`document.querySelector('#n-toggle [data-n="10000"]').click()`);
    await new Promise(r => setTimeout(r, 400));
    await c.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
    await new Promise(r => setTimeout(r, 400));
    const mob = await c.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync('/tmp/wm_comp_mobile.png', Buffer.from(mob.data, 'base64'));
    console.log('Saved /tmp/wm_comp_{desktop,fist,mobile}.png, errors:', c.errors.length);
    c.close();
    const ok = c.errors.length === 0
      && (rThrow.fly > 0.25 || rThrow.maxV > 0.9 || rThrow.maxZ > 0.30)
      && rFist.zSpan > 0.14 && rFist.maxEz > 0.42
      && rSpin.spinDeg > 150;
    console.log('CHROME_COMP:', ok ? 'PASS' : 'CHECK');
    if (!ok) process.exitCode = 1;
  } else if (cmd === 'text') {
    const url = `http://127.0.0.1:${port}/index.html`;
    const c = await launchChrome(url, 1280, 1180);
    const t0 = Date.now();
    while (Date.now() - t0 < 60_000) {
      const r = await c.send('Runtime.evaluate', { expression: 'Boolean(window.demo && document.getElementById("status").hidden)', returnByValue: true });
      if (r.result?.value) break;
      await new Promise(r => setTimeout(r, 300));
    }
    const evalJs = async (expr) => (await c.send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.value;

    // 1. Wait for opening prompt ("front left") to reach or run 12s
    const waitPlaced = async (targetCount, maxMs = 15000) => {
      const s = Date.now();
      while (Date.now() - s < maxMs) {
        const p = Number(await evalJs('document.getElementById("placed").textContent'));
        if (p >= targetCount) return true;
        await new Promise(r => setTimeout(r, 300));
      }
      return false;
    };
    const ok1 = await waitPlaced(1, 16000);
    const s1 = JSON.parse(await evalJs(`JSON.stringify({
      prompt: document.getElementById('prompt-input').value,
      hud: document.getElementById('active-cmd').textContent,
      placed: document.getElementById('placed').textContent,
      dist: document.getElementById('dist').textContent,
      badge: document.getElementById('rot-status-badge').textContent,
      ortho: document.getElementById('trace-ortho').textContent,
      goal: window.demo.env.goal,
      cube: window.demo.env.cubePos(),
    })`));
    console.log('Step 1 (opening "front left"):', ok1 ? 'PLACED' : 'IN-PROGRESS', s1);

    // Capture crisp full-section desktop screenshot (header + command deck + 3D view + latent-space inspector)
    const desk = await c.send('Page.captureScreenshot', {
      format: 'png',
      clip: { x: 0, y: 0, width: 1280, height: 1450, scale: 1 },
      captureBeyondViewport: true,
    });
    writeFileSync('/tmp/wm_text_desktop.png', Buffer.from(desk.data, 'base64'));

    // 2. Submit a natural spatial command via the form
    await evalJs(`(() => {
      document.getElementById('prompt-input').value = 'put the cube 15 cm to the right, near the front edge';
      document.getElementById('prompt-form').dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    })()`);
    const prevPlaced = Number(s1.placed);
    const ok2 = await waitPlaced(prevPlaced + 1, 18000);
    const s2 = JSON.parse(await evalJs(`JSON.stringify({
      hud: document.getElementById('active-cmd').textContent,
      placed: document.getElementById('placed').textContent,
      dist: document.getElementById('dist').textContent,
      goal: window.demo.env.goal,
      cube: window.demo.env.cubePos(),
    })`));
    console.log('Step 2 ("15 cm right, front edge"):', ok2 ? 'PLACED' : 'IN-PROGRESS', s2);

    // 3. Test switching to Masked caption (bad) and Shuffled controls
    await evalJs(`document.querySelector('#enc-toggle [data-kind="bad"]').click()`);
    await new Promise(r => setTimeout(r, 400));
    const sBad = JSON.parse(await evalJs(`JSON.stringify({
      badge: document.getElementById('rot-status-badge').textContent,
      hud: document.getElementById('active-cmd').textContent,
      goal: window.demo.env.goal,
    })`));
    console.log('Step 3a (Masked caption control):', sBad);

    await evalJs(`document.querySelector('#enc-toggle [data-kind="good"]').click(); document.querySelector('#n-toggle [data-n="shuffled"]').click()`);
    await new Promise(r => setTimeout(r, 400));
    const sShuf = JSON.parse(await evalJs(`JSON.stringify({
      badge: document.getElementById('rot-status-badge').textContent,
      hud: document.getElementById('active-cmd').textContent,
      goal: window.demo.env.goal,
    })`));
    console.log('Step 3b (Shuffled control):', sShuf);

    // Restore good + 10k and capture mobile screenshot
    await evalJs(`document.querySelector('#n-toggle [data-n="10000"]').click()`);
    await new Promise(r => setTimeout(r, 500));
    await c.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
    await new Promise(r => setTimeout(r, 500));
    const mob = await c.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync('/tmp/wm_text_mobile.png', Buffer.from(mob.data, 'base64'));
    console.log('Saved /tmp/wm_text_desktop.png and /tmp/wm_text_mobile.png, errors:', c.errors.length);
    c.close();
    if (c.errors.length || !ok1 || !ok2) process.exitCode = 1;
  }
} finally {
  srv.close();
}
