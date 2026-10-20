// The page: passage -> concepts (worker), pins, steered vs unsteered continuations, then where the pins fire in them.
// The passage box is a textarea over a backdrop that draws the highlights; a click picks the word under the caret.
const $ = id => document.getElementById(id);
const PASSAGE = 'The humpback whale surfaced a few meters from our boat, blew a cloud of mist into the cold morning air, and slid back under the grey water. Somewhere below, its calf was singing.';
const COLORS = 5;                                          // --p0 .. --p4 in styles.css

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const state = {
    ready: false, reading: null, pinned: [], color: new Map(), token: 1, concept: null, seed: 1, job: 0,   // job: writes + stops sent
    prompt: '', rows: null, writing: false, scored: null,
};
let examples = null, names = null, shows = [];

fetch('data/examples.json').then(r => r.json()).then(e => { examples = e; render(); });
fetch('data/names.json').then(r => r.json()).then(n => { names = n; render(); });

function esc(s) { return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]); }
// a token marked with a span; its leading space stays outside so the mark hugs the word
function mark(t, cls, style) {
    const [, ws, rest] = t.match(/^(\s*)([\s\S]*)$/);
    return rest ? `${esc(ws)}<span class="${cls}" style="${style}">${esc(rest)}</span>` : esc(t);
}
const label = j => names?.[j] || `Concept #${j}`;
const pct = v => `${Math.round(100 * v)}%`;
// pinned concepts keep their color while pinned; a new pin takes the first free one
const colorOf = j => state.color.has(j) ? `var(--p${state.color.get(j)})` : null;
const cvar = j => colorOf(j) ? ` style="--c:${colorOf(j)}"` : '';

function pin(j) {
    if (state.pinned.includes(j)) {
        state.pinned = state.pinned.filter(x => x !== j);
        state.color.delete(j);
    } else {
        const used = new Set(state.color.values());
        let k = 0;
        while (used.has(k) && k < COLORS) k++;
        state.pinned = [...state.pinned, j];
        state.color.set(j, k % COLORS);
    }
    render();
    write();
}

worker.onmessage = e => {
    const m = e.data;
    if (m.type === 'progress') {
        $('status-text').textContent = `Loading Gemma 3 270M and its concept dictionary: ${(m.loaded / 1e6).toFixed(0)} of ${(m.total / 1e6).toFixed(0)} MB…`;
        $('progress').style.width = `${(100 * m.loaded / m.total).toFixed(1)}%`;
    } else if (m.type === 'ready') {
        state.ready = true;
        shows = m.shows;                                   // does steering with it show up in the text
        $('status').hidden = true;
        $('again').disabled = false;
        read();
    } else if (m.type === 'read') {
        if (m.text !== $('text').value) return;           // typed since; a newer read is coming
        // the first reading opens on the first word with a concept that can be pinned
        const first = !state.reading && Math.max(1, m.perToken.findIndex((row, n) => n > 0 && row.some(([j]) => shows[j])));
        state.reading = m;
        if (state.token !== null) state.token = m.tokens.length > 1 ? Math.min(first || state.token, m.tokens.length - 1) : null;
        state.concept = null;
        render();
    } else if (m.type === 'text') {
        if (m.job !== state.job) return;
        state.rows = m.rows;
        renderOutputs();
    } else if (m.type === 'done') {
        if (m.job !== state.job) return;
        state.writing = false;
        renderOutputs();
    } else if (m.type === 'scored') {
        if (m.job !== state.job) return;
        state.scored = m;
        renderOutputs();
    } else if (m.type === 'error') {
        $('status').hidden = false;
        $('status').classList.add('error');
        $('status-text').textContent = m.message;
    }
};
worker.postMessage({ type: 'load' });

function read() {
    if (state.ready && $('text').value.trim()) worker.postMessage({ type: 'read', text: $('text').value });
}

function write() {
    if (!state.ready) return;
    state.prompt = $('prompt').value;
    state.rows = null;
    state.scored = null;
    state.writing = state.pinned.length > 0;
    state.job++;                                           // messages from earlier writes are dropped
    if (!state.pinned.length) worker.postMessage({ type: 'stop' });
    else worker.postMessage({ type: 'write', prompt: state.prompt, concepts: state.pinned, scale: +$('strength').value, seed: state.seed, tokens: +$('tokens').value });
    renderOutputs();
}

// character range of each token in the passage (token 0 is <bos>)
function offsets(tokens) {
    const out = [];
    let o = 0;
    for (const t of tokens) { out.push([o, o + t.length]); o += t.length; }
    out[0] = [0, 0];
    return out;
}

// activation of concept j at every token, as a share of its best (null if the worker didn't send them)
function actsOf(j) {
    const r = state.reading;
    const ci = r.spread.findIndex(([k]) => k === j);
    return ci >= 0 ? r.acts[ci] : r.perToken.map(row => row.find(([k]) => k === j)?.[1] || 0);
}

function render() {
    renderPassage();
    renderList();
    renderDetail();
    renderPinned();
}

function renderPassage() {
    const r = state.reading;
    const text = $('text').value;
    if (!r || r.text !== text) { $('passage').textContent = text + '\n'; return; }
    const acts = state.concept === null ? null : actsOf(state.concept);
    $('passage').innerHTML = r.tokens.map((t, n) => {
        if (n === 0) return '';
        const a = acts ? acts[n] : 0;
        const cls = (a > 0 ? ' lit' : '') + (n === state.token ? ' sel' : '');
        if (!cls) return esc(t);
        return mark(t, cls.trim(), a > 0 ? `--a:${Math.min(1, a).toFixed(2)}${colorOf(state.concept) ? `;--c:${colorOf(state.concept)}` : ''}` : '');
    }).join('') + '\n';
}

function renderList() {
    const r = state.reading;
    const byWord = state.token !== null;
    $('view-word').setAttribute('aria-pressed', byWord);
    $('view-all').setAttribute('aria-pressed', !byWord);
    $('stepper').hidden = !byWord;
    if (!r) {
        $('word').textContent = '';
        $('concept-list').innerHTML = `<li class="empty">${state.ready ? 'Reading…' : 'The concepts appear here once the model has loaded.'}</li>`;
        return;
    }
    if (byWord) {
        $('word').innerHTML = `<b>“${esc(r.tokens[state.token].trim() || '·')}”</b> ${state.token} of ${r.tokens.length - 1}`;
        $('prev').disabled = state.token <= 1;
        $('next').disabled = state.token >= r.tokens.length - 1;
    }
    const items = byWord ? r.perToken[state.token] : r.spread.slice(0, 16);
    if (!items.length) { $('concept-list').innerHTML = '<li class="empty">No concepts fire here.</li>'; return; }
    const maxv = Math.max(...items.map(([, v]) => v), 1e-6);
    $('concept-list').innerHTML = items.map(([j, v]) => {
        const on = state.pinned.includes(j);
        const lead = shows[j]
            ? `<button class="pin${on ? ' on' : ''}" type="button" data-pin="${j}" aria-pressed="${on}" aria-label="${on ? 'Unpin' : 'Pin'} ${esc(label(j))}" title="${on ? 'Unpin' : 'Pin to write with it'}"></button>`
            : '<span class="no-pin" title="The model recognizes this concept, but steering with it doesn’t make it write about it">–</span>';
        return `<li class="concept${state.concept === j ? ' sel' : ''}${shows[j] ? '' : ' faint'}" data-j="${j}"${cvar(j)}>
            ${lead}
            <span class="what" title="${esc(label(j))}">${esc(label(j))}</span>
            <span class="amount"><span class="bar"><span style="width:${(100 * v / maxv).toFixed(0)}%"></span></span>${pct(v)}</span>
        </li>`;
    }).join('');
}

function renderDetail() {
    const j = state.concept, r = state.reading;
    if (j === null || !r) { $('detail').hidden = true; return; }
    const acts = actsOf(j);
    const fires = acts.slice(1).filter(a => a > 0).length;
    const on = state.pinned.includes(j);
    const ex = (examples?.[j] || []).slice(0, 3);
    $('detail').hidden = false;
    $('detail').setAttribute('style', colorOf(j) ? `--c:${colorOf(j)}` : '');
    $('detail').innerHTML = `
        <div class="detail-head">
            <div>
                <h4>${esc(label(j))}</h4>
                <p class="meta">Fires on ${fires} of ${r.tokens.length - 1} tokens here${shows[j] ? '' : '. The model recognizes it, but steering with it doesn’t make the model write about it'}.</p>
            </div>
            ${shows[j] ? `<button class="pin-big${on ? ' on' : ''}" type="button" data-pin="${j}">${on ? 'Pinned' : 'Pin'}</button>` : ''}
        </div>
        ${ex.length ? `<h5>Where it fires hardest elsewhere</h5>
        <ul class="examples">${ex.map(([a, t, b, v]) => `<li><span class="ctx">…${esc(a.slice(-30).replace(/\s+/g, ' '))}<mark>${esc(t.replace(/\n/g, ' '))}</mark>${esc(b.replace(/\s+/g, ' '))}…</span><span class="pct">${pct(v)}</span></li>`).join('')}</ul>` : ''}`;
}

function renderPinned() {
    $('pinned').innerHTML = state.pinned.length
        ? state.pinned.map(j => `<span class="chip"${cvar(j)}><span class="name" title="${esc(label(j))}">${esc(label(j))}</span><button type="button" data-unpin="${j}" aria-label="Unpin ${esc(label(j))}">×</button></span>`).join('')
        : 'Pin concepts with + above, and the model writes with them turned up.';
}

// a continuation: plain while it streams, then colored by the pinned concept that fires hardest on each token
function continuation(b) {
    const s = state.scored?.rows[b];
    if (!s) return esc(state.rows?.[b] || '');
    return s.pieces.map((p, n) => {
        let k = -1, a = 0;
        s.acts.forEach((row, i) => { if (row[n] > a) { a = row[n]; k = i; } });
        return k < 0 ? esc(p) : mark(p, 'lit', `--c:${colorOf(state.scored.concepts[k]) || 'var(--lit)'};--a:${Math.min(1, a).toFixed(2)}`);
    }).join('');
}

function renderOutputs() {
    for (const b of [0, 1]) {
        const el = $(`out${b}`);
        el.classList.toggle('writing', state.writing);
        el.classList.toggle('scored', !!state.scored);
        if (!state.pinned.length) {
            el.innerHTML = b === 0 ? `<span class="wait">${state.ready ? 'Pin a concept to write with it.' : 'Waiting for the model to load.'}</span>` : '';
            $(`pres${b}`).innerHTML = '';
            continue;
        }
        el.innerHTML = `<span class="prompt">${esc(state.prompt)}</span>${continuation(b)}`;
        const s = state.scored;
        $(`pres${b}`).innerHTML = s?.rows[b] ? s.concepts.map((j, i) => {
            const fired = s.rows[b].acts[i].filter(a => a > 0).length;
            return `<li${cvar(j)}><span class="dot"></span><span class="name">${esc(label(j))}</span><span>fires on <b>${pct(fired / Math.max(1, s.rows[b].pieces.length))}</b> of the new tokens</span></li>`;
        }).join('') : '';
    }
}

function fit() {
    const t = $('text');
    t.style.height = 'auto';
    t.style.height = `${t.scrollHeight + 2}px`;
}

let readTimer;
$('text').addEventListener('input', () => {
    fit();
    renderPassage();
    clearTimeout(readTimer);
    readTimer = setTimeout(read, 500);
});
$('text').addEventListener('scroll', () => { $('passage').scrollTop = $('text').scrollTop; });
$('text').addEventListener('click', () => {
    const r = state.reading;
    const t = $('text');
    if (!r || r.text !== t.value || t.selectionStart !== t.selectionEnd) return;
    const c = t.selectionStart;
    const n = offsets(r.tokens).findIndex(([a, b], i) => i > 0 && c >= a && c < b);
    if (n > 0) { state.token = n; state.concept = null; }
    render();
});
$('view-word').addEventListener('click', () => { if (state.token === null) { state.token = 1; render(); } });
$('view-all').addEventListener('click', () => { state.token = null; render(); });
function step(d) {
    const r = state.reading;
    if (!r || state.token === null) return;
    state.token = Math.min(r.tokens.length - 1, Math.max(1, state.token + d));
    state.concept = null;
    render();
}
$('prev').addEventListener('click', () => step(-1));
$('next').addEventListener('click', () => step(1));
$('concept-list').addEventListener('click', e => {
    const p = e.target.closest('[data-pin]');
    if (p) { pin(+p.dataset.pin); return; }
    const li = e.target.closest('.concept');
    if (li) { const j = +li.dataset.j; state.concept = state.concept === j ? null : j; render(); }
});
$('detail').addEventListener('click', e => {
    const p = e.target.closest('[data-pin]');
    if (p) pin(+p.dataset.pin);
});
$('pinned').addEventListener('click', e => {
    const b = e.target.closest('[data-unpin]');
    if (b) pin(+b.dataset.unpin);
});

$('text').value = PASSAGE;
fit();
render();
renderOutputs();
window.addEventListener('resize', fit);
$('again').addEventListener('click', () => { state.seed++; write(); });
document.addEventListener('keydown', e => {
    if (e.target.matches('textarea, input, select')) return;
    if (e.key === 'ArrowLeft') step(-1);
    else if (e.key === 'ArrowRight') step(1);
});
$('prompt').addEventListener('change', write);
$('prompt').addEventListener('keydown', e => { if (e.key === 'Enter') e.target.blur(); });
$('strength').addEventListener('input', () => { $('strength-out').textContent = `${(+$('strength').value).toFixed(2).replace(/\.?0+$/, '')}×`; });
$('strength').addEventListener('change', write);
$('tokens').addEventListener('change', write);
window.steeringApp = { state, worker };
