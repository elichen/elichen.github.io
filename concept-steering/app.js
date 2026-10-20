// The page: passage -> concepts (worker), pins, steered vs unsteered continuations.
// The passage box is a textarea over a backdrop that draws the highlights; a click picks the word under the caret.
const $ = id => document.getElementById(id);
const PASSAGE = 'The humpback whale surfaced a few meters from our boat, blew a cloud of mist into the cold morning air, and slid back under the grey water. Somewhere below, its calf was singing.';

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const state = { ready: false, reading: null, readText: null, pinned: [], token: null, concept: null, seed: 1, job: 0 };
let examples = null, names = null;

fetch('data/examples.json').then(r => r.json()).then(e => { examples = e; render(); });
fetch('data/names.json').then(r => r.json()).then(n => { names = n; render(); });

function esc(s) { return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]); }

// a concept is shown by its Neuronpedia label; the passage where it fired hardest is its tooltip (or its label
// when it has none)
function example(j) {
    const ex = examples?.[j]?.[0];
    return ex ? `…${ex[0].slice(-48)}[${ex[1]}]${ex[2]}…`.replace(/\n/g, ' ') : '';
}
const label = j => names?.[j] || example(j) || `#${j}`;

worker.onmessage = e => {
    const m = e.data;
    if (m.type === 'progress') $('status').textContent = `Loading Gemma 3 270M and its concept dictionary: ${(m.loaded / 1e6).toFixed(0)} of ${(m.total / 1e6).toFixed(0)} MB…`;
    else if (m.type === 'ready') {
        state.ready = true;
        $('status').textContent = '';
        $('again').disabled = false;
        read();
    } else if (m.type === 'read') {
        if (m.text !== $('text').value) return;           // typed since; a newer read is coming
        state.reading = m;
        state.token = state.concept = null;
        state.pinned = m.spread.slice(0, 3).map(([j]) => j);
        render();
        write();
    } else if (m.type === 'text') {
        if (m.job < state.job) return;
        state.job = m.job;
        $('out0').innerHTML = `<span class="prompt">${esc(state.prompt)}</span>${esc(m.rows[0])}`;
        $('out1').innerHTML = `<span class="prompt">${esc(state.prompt)}</span>${esc(m.rows[1])}`;
    } else if (m.type === 'error') {
        $('status').textContent = m.message;
    }
};
worker.postMessage({ type: 'load' });

function read() {
    if (state.ready && $('text').value.trim()) worker.postMessage({ type: 'read', text: $('text').value });
}

function write() {
    if (!state.ready) return;
    state.prompt = $('prompt').value;
    $('out0').innerHTML = $('out1').innerHTML = '';
    worker.postMessage({ type: 'write', prompt: state.prompt, concepts: state.pinned, scale: +$('strength').value, seed: state.seed });
}

// character range of each token in the passage (token 0 is <bos>)
function offsets(tokens) {
    const out = [];
    let o = 0;
    for (const t of tokens) { out.push([o, o + t.length]); o += t.length; }
    out[0] = [0, 0];
    return out;
}

function render() {
    const r = state.reading;
    const text = $('text').value;
    if (!r || r.text !== text) { $('passage').textContent = text + '\n'; return; }
    const ci = state.concept === null ? -1 : r.spread.findIndex(([j]) => j === state.concept);
    $('passage').innerHTML = r.tokens.map((t, n) => {
        if (n === 0) return '';
        let a = 0;
        if (state.concept !== null) a = ci >= 0 ? r.acts[ci][n] : (r.perToken[n].find(([j]) => j === state.concept)?.[1] || 0);
        const cls = (a > 0 ? ' lit' : '') + (n === state.token ? ' sel' : '');
        return cls ? `<span class="${cls.trim()}"${a > 0 ? ` style="--a:${Math.min(1, a).toFixed(2)}"` : ''}>${esc(t)}</span>` : esc(t);
    }).join('') + '\n';
    const items = state.token === null ? r.spread.slice(0, 16) : r.perToken[state.token];
    $('list-title').textContent = state.token === null ? 'Concepts in this passage' : `Concepts at “${r.tokens[state.token].trim()}”`;
    $('list-back').hidden = state.token === null;
    const maxv = Math.max(...items.map(([, v]) => v), 1e-6);
    $('concept-list').innerHTML = items.map(([j, v]) => `
        <li class="concept${state.concept === j ? ' sel' : ''}" data-j="${j}" title="${esc(example(j))}">
            <button class="pin${state.pinned.includes(j) ? ' on' : ''}" type="button" data-pin="${j}" aria-label="${state.pinned.includes(j) ? 'Unpin' : 'Pin'}">${state.pinned.includes(j) ? '●' : '○'}</button>
            <span class="bar"><span style="width:${(100 * v / maxv).toFixed(0)}%"></span></span>
            <span class="what">${esc(label(j))}</span>
        </li>`).join('');
    $('pinned').innerHTML = state.pinned.length
        ? state.pinned.map(j => `<span class="chip"><span class="name">${esc(label(j))}</span><button type="button" data-unpin="${j}" aria-label="Unpin">×</button></span>`).join('')
        : '<span class="muted">Nothing pinned: both columns are the plain model.</span>';
}

function fit() {
    const t = $('text');
    t.style.height = 'auto';
    t.style.height = `${t.scrollHeight + 2}px`;
}

let readTimer;
$('text').addEventListener('input', () => {
    fit();
    render();
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
    state.token = n > 0 && n !== state.token ? n : null;
    render();
});
$('list-back').addEventListener('click', () => { state.token = null; render(); });
$('concept-list').addEventListener('click', e => {
    const pin = e.target.closest('[data-pin]');
    if (pin) {
        const j = +pin.dataset.pin;
        state.pinned = state.pinned.includes(j) ? state.pinned.filter(x => x !== j) : [...state.pinned, j];
        render();
        write();
        return;
    }
    const li = e.target.closest('.concept');
    if (li) { const j = +li.dataset.j; state.concept = state.concept === j ? null : j; render(); }
});
$('pinned').addEventListener('click', e => {
    const b = e.target.closest('[data-unpin]');
    if (b) { state.pinned = state.pinned.filter(x => x !== +b.dataset.unpin); render(); write(); }
});

$('text').value = PASSAGE;
fit();
render();
window.addEventListener('resize', fit);
$('again').addEventListener('click', () => { state.seed++; write(); });
$('prompt').addEventListener('change', write);
$('prompt').addEventListener('keydown', e => { if (e.key === 'Enter') e.target.blur(); });
$('strength').addEventListener('input', () => { $('strength-out').textContent = `${(+$('strength').value).toFixed(2).replace(/0$/, '')}×`; });
$('strength').addEventListener('change', write);
window.steeringApp = { state, worker };
