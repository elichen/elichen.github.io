// GPT-2 attention visualizer — main thread: tokenization (transformers.js),
// worker orchestration, and all canvas rendering.
import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/+esm';

const MAX_TOKENS = 48; // cap so the forward stays a few seconds and labels stay legible

const $ = (id) => document.getElementById(id);
const loader = $('loader'), barFill = $('bar-fill'), loaderStatus = $('loader-status');
const app = $('app');

const state = {
    tokenizer: null,
    worker: null,
    cfg: null,
    tokens: [],     // display strings
    ids: [],
    T: 0,
    attn: null,     // Float32Array [L*H*T*T]
    logits: null,
    selLayer: 0,
    selHead: 0,
    queryTok: 0,    // index of token whose outgoing attention we inspect
    reqId: 0,
};

// ---------- tooltip ----------
const tooltip = document.createElement('div');
tooltip.id = 'tooltip';
document.body.appendChild(tooltip);
function showTip(x, y, text) {
    tooltip.textContent = text;
    tooltip.style.display = 'block';
    tooltip.style.left = (x + 14) + 'px';
    tooltip.style.top = (y + 14) + 'px';
}
function hideTip() { tooltip.style.display = 'none'; }

// ---------- color ----------
function attnColor(w) {
    const v = Math.pow(Math.max(0, Math.min(1, w)), 0.65);
    const r = Math.round(20 * v);
    const g = Math.round(45 + 205 * v);
    const b = Math.round(35 * v);
    return `rgb(${r},${g},${b})`;
}

// ---------- worker ----------
function initWorker() {
    const worker = new Worker('./worker.js', { type: 'module' });
    state.worker = worker;
    worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'progress') {
            const pct = m.total ? (m.loaded / m.total) : 0;
            barFill.style.width = (pct * 100).toFixed(1) + '%';
            loaderStatus.textContent =
                `${(m.loaded / 1048576).toFixed(0)} / ${(m.total / 1048576).toFixed(0)} MB`;
        } else if (m.type === 'ready') {
            state.cfg = m.config;
            onModelReady();
        } else if (m.type === 'result') {
            if (m.reqId === state.reqId) onResult(m);
        } else if (m.type === 'run-error') {
            if (m.reqId === state.reqId) {
                $('run').disabled = false;
                $('run-stats').textContent = 'forward failed: ' + m.message;
            }
        } else if (m.type === 'error') {
            loaderStatus.textContent = 'error: ' + m.message;
        }
    };
    worker.postMessage({ type: 'init' });
}

async function onModelReady() {
    loaderStatus.textContent = 'loading tokenizer…';
    try {
        state.tokenizer = await AutoTokenizer.from_pretrained('Xenova/gpt2');
    } catch (err) {
        loaderStatus.textContent = 'tokenizer failed to load: ' + err;
        return;
    }
    loader.hidden = true;
    app.hidden = false;
    bindUI();
    run();
}

// ---------- tokenize + run ----------
function tokenize(text) {
    const enc = state.tokenizer.encode(text);
    let ids = Array.from(enc).map(Number);
    let truncated = false;
    if (ids.length > MAX_TOKENS) { ids = ids.slice(0, MAX_TOKENS); truncated = true; }
    if (ids.length === 0) ids = [state.tokenizer.encode('.')[0]];
    const tokens = ids.map((id) => state.tokenizer.decode([id]));
    return { ids, tokens, truncated };
}

function run() {
    if ($('run').disabled) return; // a forward is already in flight
    const text = $('text').value;
    const { ids, tokens, truncated } = tokenize(text);
    state.ids = ids;
    state.tokens = tokens;
    state.T = ids.length;
    state.queryTok = ids.length - 1;
    renderTokens();
    $('run-stats').textContent = truncated ? `truncated to ${MAX_TOKENS} tokens` : '';
    $('run').disabled = true;
    state.reqId++;
    state.worker.postMessage({ type: 'run', ids, reqId: state.reqId });
}

function onResult(m) {
    state.attn = m.attn;
    state.logits = m.logits;
    state.T = m.T;
    $('run').disabled = false;
    $('run-stats').textContent = `${m.T} tokens · forward ${m.ms.toFixed(0)} ms`;
    autoSelectHead(state.queryTok);
    drawGrid();
    drawDetail();
    drawArc();
    drawPredictions();
}

// ---------- auto-select the head that most sharply focuses a query token ----------
// For query token q, find the head whose attention puts the most weight on a
// single earlier word, skipping the first-token attention sink (k=0) and the
// immediately-preceding token (k=q-1). Those two are present in almost every
// head — the sink as a no-op dump and the neighbor as the ubiquitous
// previous-token head — so excluding them surfaces longer-range structure like
// coreference and induction instead. Tokens too near the start have no interior
// key, so we relax the range in two fallback steps.
function autoSelectHead(q) {
    if (!state.attn || q <= 0) return;
    const L = state.cfg.n_layer, H = state.cfg.n_head, T = state.T;
    const scan = (lo, hi) => {
        let best = null;
        for (let l = 0; l < L; l++) {
            for (let h = 0; h < H; h++) {
                const base = (l * H + h) * T * T + q * T;
                for (let k = lo; k <= hi; k++) {
                    const w = state.attn[base + k];
                    if (!best || w > best.w) best = { l, h, w };
                }
            }
        }
        return best;
    };
    const best = scan(1, q - 2) || scan(1, q - 1) || scan(0, q - 1);
    if (!best) return;
    state.selLayer = best.l;
    state.selHead = best.h;
    $('sel-layer').textContent = best.l;
    $('sel-head').textContent = best.h;
}

// ---------- token chips ----------
function renderTokens() {
    const wrap = $('tokens');
    wrap.innerHTML = '';
    state.tokens.forEach((tok, i) => {
        const el = document.createElement('div');
        el.className = 'tok' + (i === state.queryTok ? ' sel' : '');
        el.textContent = tok.replace(/\n/g, '⏎');
        el.title = `token ${i}`;
        el.onclick = () => {
            state.queryTok = i;
            autoSelectHead(i);  // jump to the head that focuses on this token most
            renderTokens();     // also refreshes the query label
            drawGrid();         // move the selected-head highlight box
            drawDetail();
            drawArc();
        };
        wrap.appendChild(el);
    });
    updateQueryLabel();
}
function updateQueryLabel() {
    const t = state.tokens[state.queryTok] ?? '';
    $('query-label').textContent = `[${state.queryTok}] "${t.replace(/\n/g, '⏎')}"`;
    $('arc-query').textContent = `"${t.replace(/\n/g, '⏎')}"`;
}

// ---------- head grid ----------
const gridCanvas = $('grid');
let gridGeom = null;
function drawGrid() {
    if (!state.attn) return;
    const L = state.cfg.n_layer, H = state.cfg.n_head, T = state.T;
    const dpr = window.devicePixelRatio || 1;
    const cssW = gridCanvas.clientWidth || 520;
    const padL = 22, padT = 16, gap = 3;
    const cell = (cssW - padL) / H;
    const cssH = padT + cell * L;
    gridCanvas.style.height = cssH + 'px';
    gridCanvas.width = Math.round(cssW * dpr);
    gridCanvas.height = Math.round(cssH * dpr);
    const ctx = gridCanvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.imageSmoothingEnabled = false;

    ctx.font = '9px "Source Code Pro", monospace';
    ctx.fillStyle = '#4d8b4d';
    ctx.textAlign = 'center';
    for (let h = 0; h < H; h++) ctx.fillText(h, padL + h * cell + cell / 2, 11);
    ctx.textAlign = 'right';
    for (let l = 0; l < L; l++) ctx.fillText(l, padL - 5, padT + l * cell + cell / 2 + 3);

    const inner = cell - gap;
    const sub = inner / T;
    for (let l = 0; l < L; l++) {
        for (let h = 0; h < H; h++) {
            const x0 = padL + h * cell, y0 = padT + l * cell;
            const base = (l * H + h) * T * T;
            for (let q = 0; q < T; q++) {
                for (let k = 0; k <= q; k++) {
                    ctx.fillStyle = attnColor(state.attn[base + q * T + k]);
                    ctx.fillRect(x0 + k * sub, y0 + q * sub, Math.ceil(sub), Math.ceil(sub));
                }
            }
            if (l === state.selLayer && h === state.selHead) {
                ctx.strokeStyle = '#90ee90';
                ctx.lineWidth = 1.5;
                ctx.strokeRect(x0 + 0.5, y0 + 0.5, inner, inner);
            }
        }
    }
    gridGeom = { padL, padT, cell, gap, inner, sub, L, H, T };
}

gridCanvas.addEventListener('mousemove', (e) => {
    if (!gridGeom) return;
    const r = gridCanvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const g = gridGeom;
    const h = Math.floor((x - g.padL) / g.cell);
    const l = Math.floor((y - g.padT) / g.cell);
    if (h < 0 || h >= g.H || l < 0 || l >= g.L) { hideTip(); return; }
    const cx = (x - g.padL) - h * g.cell, cy = (y - g.padT) - l * g.cell;
    const k = Math.floor(cx / g.sub), q = Math.floor(cy / g.sub);
    if (q < 0 || q >= g.T || k < 0 || k > q) {
        showTip(e.clientX, e.clientY, `layer ${l} · head ${h}`);
        return;
    }
    const w = state.attn[((l * g.H + h) * g.T + q) * g.T + k];
    showTip(e.clientX, e.clientY,
        `L${l} H${h}\n"${trim(state.tokens[q])}" → "${trim(state.tokens[k])}"\n${(w * 100).toFixed(1)}%`);
});
gridCanvas.addEventListener('mouseleave', hideTip);
gridCanvas.addEventListener('click', (e) => {
    if (!gridGeom) return;
    const r = gridCanvas.getBoundingClientRect();
    const g = gridGeom;
    const h = Math.floor((e.clientX - r.left - g.padL) / g.cell);
    const l = Math.floor((e.clientY - r.top - g.padT) / g.cell);
    if (h < 0 || h >= g.H || l < 0 || l >= g.L) return;
    state.selLayer = l; state.selHead = h;
    $('sel-layer').textContent = l;
    $('sel-head').textContent = h;
    drawGrid();
    drawDetail();
    drawArc();
});

function trim(s) { return (s ?? '').replace(/\n/g, '⏎'); }

// ---------- detail heatmap ----------
const detailCanvas = $('detail');
let detailGeom = null;
function drawDetail() {
    if (!state.attn) return;
    const T = state.T;
    const dpr = window.devicePixelRatio || 1;
    const wrapW = (detailCanvas.parentElement && detailCanvas.parentElement.clientWidth) || 480;

    // Choose a cell size that fills the available width but stays bounded: small
    // inputs don't balloon, long ones stay legible. Labels are dropped when cells
    // get too small to carry text, so a 40-token sentence still renders cleanly.
    const labelL = 84;
    let cell = Math.floor((wrapW - labelL - 8) / T);
    cell = Math.max(6, Math.min(30, cell));
    const gridSize = cell * T;
    const fs = Math.max(7, Math.min(12, cell * 0.66));
    const showLabels = cell >= 8;
    const maxChars = Math.max(3, Math.floor((labelL - 8) / (fs * 0.6)));
    const proj = Math.ceil(maxChars * fs * 0.6 * 0.72); // -45° projection of a label
    const labelT = showLabels ? proj + 10 : 12;
    const rightPad = showLabels ? Math.max(8, proj) : 8;

    const cssW = labelL + gridSize + rightPad;
    const cssH = labelT + gridSize + 4;
    detailCanvas.style.width = cssW + 'px';
    detailCanvas.style.height = cssH + 'px';
    detailCanvas.width = Math.round(cssW * dpr);
    detailCanvas.height = Math.round(cssH * dpr);
    const ctx = detailCanvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const base = (state.selLayer * state.cfg.n_head + state.selHead) * T * T;
    const x0 = labelL, y0 = labelT;
    for (let q = 0; q < T; q++) {
        for (let k = 0; k <= q; k++) {
            ctx.fillStyle = attnColor(state.attn[base + q * T + k]);
            ctx.fillRect(x0 + k * cell, y0 + q * cell, Math.ceil(cell), Math.ceil(cell));
        }
    }
    // highlight selected query row
    ctx.strokeStyle = 'rgba(144,238,144,0.85)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0 + 0.5, y0 + state.queryTok * cell + 0.5, gridSize - 1, cell);

    ctx.font = `${fs}px "Source Code Pro", monospace`;
    ctx.textBaseline = 'middle';
    // query labels (left, right-aligned) — always show the selected row's label
    ctx.textAlign = 'right';
    for (let q = 0; q < T; q++) {
        if (!showLabels && q !== state.queryTok) continue;
        ctx.fillStyle = q === state.queryTok ? '#90ee90' : '#7fbf7f';
        ctx.fillText(clip(trim(state.tokens[q]), maxChars), labelL - 6, y0 + q * cell + cell / 2);
    }
    // key labels (top, rotated up-left so the last column never clips off the edge)
    if (showLabels) {
        ctx.textAlign = 'left';
        for (let k = 0; k < T; k++) {
            ctx.fillStyle = k === state.queryTok ? '#90ee90' : '#7fbf7f';
            ctx.save();
            ctx.translate(x0 + k * cell + cell / 2, labelT - 6);
            ctx.rotate(-Math.PI / 4);
            ctx.fillText(clip(trim(state.tokens[k]), maxChars), 0, 0);
            ctx.restore();
        }
    }

    // detail stats: entropy of the selected query row
    let ent = 0;
    const rowBase = base + state.queryTok * T;
    for (let k = 0; k <= state.queryTok; k++) {
        const p = state.attn[rowBase + k];
        if (p > 1e-9) ent -= p * Math.log2(p);
    }
    $('detail-stats').textContent = `row entropy ${ent.toFixed(2)} bits`;
    detailGeom = { y0, cell, T };
}

function clip(s, n) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }

detailCanvas.addEventListener('click', (e) => {
    if (!detailGeom) return;
    const r = detailCanvas.getBoundingClientRect();
    const g = detailGeom;
    const q = Math.floor((e.clientY - r.top - g.y0) / g.cell);
    if (q >= 0 && q < g.T) {
        state.queryTok = q;
        renderTokens(); // also refreshes the query label
        drawDetail();   // clicking a row keeps the current head, just moves the query
        drawArc();
    }
});

// ---------- attention arcs ----------
const arcCanvas = $('arc');
function drawArc() {
    if (!state.attn) return;
    const T = state.T;
    const dpr = window.devicePixelRatio || 1;
    const cssW = (arcCanvas.parentElement && arcCanvas.parentElement.clientWidth) || 900;

    // Labels are rotated below the axis so they never run together, even at 40+
    // tokens; the font and label length shrink as the sequence grows.
    const fs = Math.max(8, Math.min(12, 760 / T));
    const labelChars = T > 26 ? 6 : 10;
    const labelRoom = Math.ceil(labelChars * fs * 0.6 * 0.72) + 14;
    const arcH = 92;
    const yTok = arcH + 10;
    const cssH = yTok + labelRoom;
    arcCanvas.style.height = cssH + 'px';
    arcCanvas.width = Math.round(cssW * dpr);
    arcCanvas.height = Math.round(cssH * dpr);
    const ctx = arcCanvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    // rotated labels drop to the lower-left, so reserve room on the left edge
    const marginL = Math.max(26, Math.ceil(labelChars * fs * 0.6 * 0.72));
    const marginR = 26;
    const span = cssW - marginL - marginR;
    const xOf = (i) => marginL + (T <= 1 ? span / 2 : (i / (T - 1)) * span);

    const base = (state.selLayer * state.cfg.n_head + state.selHead) * T * T;
    const q = state.queryTok;

    // arcs from query token to each key
    for (let k = 0; k <= q; k++) {
        const w = state.attn[base + q * T + k];
        if (w < 0.01) continue;
        const x1 = xOf(q), x2 = xOf(k);
        const lift = Math.min(arcH - 6, 16 + Math.abs(x1 - x2) * 0.5);
        ctx.beginPath();
        ctx.moveTo(x1, yTok);
        ctx.quadraticCurveTo((x1 + x2) / 2, yTok - lift, x2, yTok);
        ctx.strokeStyle = `rgba(144,238,144,${Math.min(1, w * 0.9 + 0.06)})`;
        ctx.lineWidth = Math.max(0.5, w * 7);
        ctx.stroke();
    }

    // token ticks + rotated labels
    ctx.font = `${fs}px "Source Code Pro", monospace`;
    ctx.textBaseline = 'middle';
    for (let i = 0; i < T; i++) {
        const x = xOf(i);
        ctx.fillStyle = i === q ? '#90ee90' : '#2f5f2f';
        ctx.fillRect(x - 1.5, yTok - 2, 3, 4);
        ctx.save();
        ctx.translate(x, yTok + 8);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = 'right';
        ctx.fillStyle = i === q ? '#90ee90' : (i <= q ? '#7fbf7f' : '#345834');
        ctx.fillText(clip(trim(state.tokens[i]), labelChars), 0, 0);
        ctx.restore();
    }
}

// ---------- predictions ----------
function drawPredictions() {
    const logits = state.logits;
    // softmax over top candidates (numerically stable)
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    // find top-6 by partial scan
    const k = 6;
    const idx = [];
    const used = new Set();
    for (let n = 0; n < k; n++) {
        let best = -1, bv = -Infinity;
        for (let i = 0; i < logits.length; i++) {
            if (used.has(i)) continue;
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        }
        used.add(best); idx.push(best);
    }
    // softmax denominator over full vocab
    let denom = 0;
    for (let i = 0; i < logits.length; i++) denom += Math.exp(logits[i] - max);

    const list = $('pred-list');
    list.innerHTML = '';
    for (const id of idx) {
        const p = Math.exp(logits[id] - max) / denom;
        const row = document.createElement('div');
        row.className = 'pred-row';
        const tok = state.tokenizer.decode([id]).replace(/\n/g, '⏎');
        row.innerHTML =
            `<span class="pred-tok">"${escapeHtml(tok)}"</span>` +
            `<span class="pred-bar-bg"><span class="pred-bar" style="width:${(p * 100).toFixed(1)}%"></span></span>` +
            `<span class="pred-pct">${(p * 100).toFixed(1)}%</span>`;
        list.appendChild(row);
    }
}
function escapeHtml(s) {
    return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// ---------- UI binding ----------
function bindUI() {
    $('run').onclick = run;
    $('text').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
    $('sel-layer').textContent = state.selLayer;
    $('sel-head').textContent = state.selHead;
    let resizeRAF = 0;
    window.addEventListener('resize', () => {
        cancelAnimationFrame(resizeRAF);
        resizeRAF = requestAnimationFrame(() => {
            if (!state.attn) return;
            drawGrid(); drawDetail(); drawArc();
        });
    });
}

initWorker();
