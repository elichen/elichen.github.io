// Zero-dependency headless Chrome visual & runtime test for /rubiks-hand/.
// Uses Node 24 built-in http + WebSocket CDP with Metal WebGL.
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
    '.obj': 'text/plain',
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

async function launchChrome(url, width = 1440, height = 900) {
    const debugPort = 9322 + Math.floor(Math.random() * 400);
    const proc = spawn(CHROME, [
        '--headless=new',
        `--remote-debugging-port=${debugPort}`,
        '--no-first-run',
        '--no-default-browser-check',
        '--use-angle=metal',
        `--window-size=${width},${height}`,
        '--user-data-dir=/tmp/chrome-rubik-test-' + debugPort,
        url,
    ], { stdio: ['ignore', 'pipe', 'pipe'] });

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

const { srv, port } = await startServer(root);
try {
    const c = await launchChrome(`http://127.0.0.1:${port}/index.html`, 1440, 900);
    const evalJS = async (expression) => {
        const r = await c.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
        return r.result?.value;
    };

    // Wait up to 30s for MuJoCo WASM + Three.js + Policies to initialize
    const t0 = Date.now();
    let ready = false;
    while (Date.now() - t0 < 30_000) {
        ready = await evalJS('Boolean(window.rubikDemo)');
        if (ready) break;
        await new Promise(r => setTimeout(r, 250));
    }
    if (!ready) throw new Error('Timed out waiting for window.rubikDemo');
    console.log(`Initialized in ${Date.now() - t0} ms`);

    // Trigger a crisp 5-move scramble, step forward 2 seconds, capture studio screenshot
    await evalJS('window.rubikDemo.startScramble(5); for (let i=0;i<60;i++) window.rubikDemo.advance(0.05);');
    await new Promise(r => setTimeout(r, 400));
    const st1 = await evalJS('({ phase: rubikDemo.ctrl.phase, done: rubikDemo.ctrl.completedMoves.length, todo: rubikDemo.ctrl.plan.length, facelets: rubikDemo.env.facelets() })');
    console.log('State after 3.0s sim:', st1);

    const shot1 = await c.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync('/tmp/rubik_studio.png', Buffer.from(shot1.data, 'base64'));
    console.log('Saved -> /tmp/rubik_studio.png');

    // Switch to Close-up camera & step forward another 8s sim time
    await evalJS('rubikDemo.view.setPreset("closeup"); for (let i=0;i<160;i++) rubikDemo.advance(0.05); for (let i=0;i<40;i++) rubikDemo.view.draw(0.05, rubikDemo.ctrl);');
    await new Promise(r => setTimeout(r, 300));
    const st2 = await evalJS('({ phase: rubikDemo.ctrl.phase, done: rubikDemo.ctrl.completedMoves.length, todo: rubikDemo.ctrl.plan.length, solved: rubikDemo.env.isSolved() })');
    console.log('State after 11.0s sim:', st2);

    const shot2 = await c.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync('/tmp/rubik_closeup.png', Buffer.from(shot2.data, 'base64'));
    console.log('Saved -> /tmp/rubik_closeup.png');

    console.log('Console errors:', c.errors.length);
    c.close();
    if (c.errors.length > 0) process.exitCode = 1;
} finally {
    srv.close();
}
