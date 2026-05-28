// GPT-2 attention visualizer — main thread: tokenization (transformers.js),
// worker orchestration, and all canvas rendering.
import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/+esm';

const MAX_TOKENS = 24; // keep the 12x12 head grid legible and the forward fast

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
    drawGrid();
    drawDetail();
    drawArc();
    drawPredictions();
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
            renderTokens(); // also refreshes the query label
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
function drawDetail() {
    if (!state.attn) return;
    const T = state.T;
    const dpr = window.devicePixelRatio || 1;
    const cssW = detailCanvas.clientWidth || 520;
    const labelL = 96, labelT = 96;
    const gridSize = cssW - labelL - 6;
    const cell = gridSize / T;
    const cssH = labelT + gridSize + 4;
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
    ctx.strokeStyle = 'rgba(144,238,144,0.8)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0 + 0.5, y0 + state.queryTok * cell + 0.5, gridSize - 1, cell);

    const fs = Math.max(8, Math.min(12, cell * 0.62));
    ctx.font = `${fs}px "Source Code Pro", monospace`;
    // query labels (left, right-aligned)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let q = 0; q < T; q++) {
        ctx.fillStyle = q === state.queryTok ? '#90ee90' : '#7fbf7f';
        ctx.fillText(clip(trim(state.tokens[q]), 13), labelL - 6, y0 + q * cell + cell / 2);
    }
    // key labels (top, rotated)
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#7fbf7f';
    for (let k = 0; k < T; k++) {
        ctx.save();
        ctx.translate(x0 + k * cell + cell / 2, labelT - 6);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(clip(trim(state.tokens[k]), 13), 0, 0);
        ctx.restore();
    }

    // detail stats: entropy of the selected query row
    let ent = 0;
    const rowBase = base + state.queryTok * T;
    for (let k = 0; k <= state.queryTok; k++) {
        const p = state.attn[rowBase + k];
        if (p > 1e-9) ent -= p * Math.log2(p);
    }
    $('detail-stats').textContent = `row entropy ${ent.toFixed(2)} bits`;
}

function clip(s, n) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }

detailCanvas.addEventListener('click', (e) => {
    const T = state.T;
    const r = detailCanvas.getBoundingClientRect();
    const cssW = detailCanvas.clientWidth;
    const labelL = 96, labelT = 96;
    const cell = (cssW - labelL - 6) / T;
    const q = Math.floor((e.clientY - r.top - labelT) / cell);
    if (q >= 0 && q < T) {
        state.queryTok = q;
        renderTokens(); // also refreshes the query label
        drawDetail();
        drawArc();
    }
});

// ---------- attention arcs ----------
const arcCanvas = $('arc');
function drawArc() {
    if (!state.attn) return;
    const T = state.T;
    const dpr = window.devicePixelRatio || 1;
    const cssW = arcCanvas.clientWidth || 520;
    const cssH = 130;
    arcCanvas.style.height = cssH + 'px';
    arcCanvas.width = Math.round(cssW * dpr);
    arcCanvas.height = Math.round(cssH * dpr);
    const ctx = arcCanvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const margin = 12;
    const yTok = cssH - 22;
    const span = cssW - margin * 2;
    const xOf = (i) => margin + (T <= 1 ? span / 2 : (i / (T - 1)) * span);

    const base = (state.selLayer * state.cfg.n_head + state.selHead) * T * T;
    const q = state.queryTok;

    // arcs from query token to each key
    for (let k = 0; k <= q; k++) {
        const w = state.attn[base + q * T + k];
        if (w < 0.01) continue;
        const x1 = xOf(q), x2 = xOf(k);
        const lift = Math.min(70, 18 + Math.abs(x1 - x2) * 0.45);
        ctx.beginPath();
        ctx.moveTo(x1, yTok);
        ctx.quadraticCurveTo((x1 + x2) / 2, yTok - lift, x2, yTok);
        ctx.strokeStyle = `rgba(144,238,144,${Math.min(1, w * 0.9 + 0.06)})`;
        ctx.lineWidth = Math.max(0.5, w * 7);
        ctx.stroke();
    }

    // token row
    const fs = Math.max(8, Math.min(12, span / T * 0.7));
    ctx.font = `${fs}px "Source Code Pro", monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i < T; i++) {
        ctx.fillStyle = i === q ? '#90ee90' : (i <= q ? '#7fbf7f' : '#345834');
        ctx.fillText(clip(trim(state.tokens[i]), 8), xOf(i), yTok + 6);
        ctx.fillStyle = i === q ? '#90ee90' : '#2f5f2f';
        ctx.fillRect(xOf(i) - 1.5, yTok - 2, 3, 4);
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
