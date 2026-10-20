// The page: passage -> concepts (worker), pins, steered vs unsteered continuations.
const $ = id => document.getElementById(id);
const PRESETS = {
    Whales: 'The humpback whale surfaced a few meters from our boat, blew a cloud of mist into the cold morning air, and slid back under the grey water. Somewhere below, its calf was singing.',
    Grief: 'It has been a year since my father died. I still reach for the phone on Sunday evenings to call him, and for a second I forget that nobody will pick up.',
    Recipe: 'Melt the butter in a heavy pan over medium heat, add the chopped onions and a pinch of salt, and stir until they turn soft and golden, about ten minutes. Then add the garlic.',
    Space: "When the rocket's second stage ignited, the capsule was already above most of the atmosphere, and the astronauts could see the curve of the Earth against the black sky.",
    Lease: 'The Tenant shall not sublet the Premises or assign this Agreement without the prior written consent of the Landlord, which consent shall not be unreasonably withheld.',
};
const OPENINGS = ['I went outside and', 'Yesterday I', 'My grandmother always said that', 'The city was quiet when', 'The best part of my job is'];

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const state = { ready: false, reading: null, pinned: [], token: null, concept: null, seed: 1, job: 0 };
let stats = null, examples = null, names = null;

fetch('data/stats.bin').then(r => r.arrayBuffer()).then(b => { stats = new Float32Array(b); });
fetch('data/examples.json').then(r => r.json()).then(e => { examples = e; render(); });
fetch('data/names.json').then(r => r.json()).then(n => { names = n; render(); });

const F = 16384;
const lift = j => (stats ? stats[3 * F + j] : 0);

function esc(s) { return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]); }
function show(s) { return esc(s).replace(/\n/g, '⏎'); }

// a concept is shown by its Neuronpedia label and the passage where it fired hardest
function snippet(j) {
    const ex = examples?.[j]?.[0];
    if (!ex) return '';
    const [l, t, r] = ex;
    return `<span class="ex">…${show(l.slice(-48))}<mark>${show(t)}</mark>${show(r)}…</span>`;
}
function label(j) {
    const n = names?.[j];
    return n ? `<span class="name">${esc(n)}</span>` : snippet(j) || `<span class="muted">#${j}</span>`;
}

worker.onmessage = e => {
    const m = e.data;
    if (m.type === 'progress') $('status').textContent = `Loading Gemma 3 270M and its concept dictionary: ${(m.loaded / 1e6).toFixed(0)} of ${(m.total / 1e6).toFixed(0)} MB…`;
    else if (m.type === 'ready') {
        state.ready = true;
        $('status').textContent = '';
        $('read').disabled = $('write').disabled = $('again').disabled = false;
        read();
    } else if (m.type === 'read') {
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
    if (!state.ready) return;
    worker.postMessage({ type: 'read', text: $('text').value });
}

function write() {
    if (!state.ready) return;
    state.prompt = $('prompt').value;
    $('out0').innerHTML = $('out1').innerHTML = '';
    worker.postMessage({ type: 'write', prompt: state.prompt, concepts: state.pinned, scale: +$('strength').value, seed: state.seed });
}

function render() {
    const r = state.reading;
    if (!r) return;
    // passage: tokens shaded by the selected concept (or marked when clicked)
    const ci = state.concept === null ? -1 : r.spread.findIndex(([j]) => j === state.concept);
    $('passage').innerHTML = r.tokens.map((t, n) => {
        if (n === 0) return '';
        let a = 0;
        if (state.concept !== null) {
            a = ci >= 0 ? r.acts[ci][n] : (r.perToken[n].find(([j]) => j === state.concept)?.[1] || 0);
        }
        const style = a > 0 ? ` style="--a:${Math.min(1, a).toFixed(2)}"` : '';
        return `<span class="tok${a > 0 ? ' lit' : ''}${n === state.token ? ' sel' : ''}" data-n="${n}"${style}>${show(t)}</span>`;
    }).join('') + (r.truncated ? '<span class="muted"> (first 512 tokens)</span>' : '');
    // list: passage-wide, or one token's concepts
    const items = state.token === null ? r.spread.slice(0, 16) : r.perToken[state.token];
    $('list-title').textContent = state.token === null ? 'Concepts in this passage' : `Concepts at “${r.tokens[state.token].trim()}”`;
    $('list-back').hidden = state.token === null;
    const maxv = Math.max(...items.map(([, v]) => v), 1e-6);
    $('concept-list').innerHTML = items.map(([j, v]) => `
        <li class="concept${state.concept === j ? ' sel' : ''}" data-j="${j}">
            <button class="pin${state.pinned.includes(j) ? ' on' : ''}" type="button" data-pin="${j}" aria-label="Pin">${state.pinned.includes(j) ? '●' : '○'}</button>
            <span class="bar"><span style="width:${(100 * v / maxv).toFixed(0)}%"></span></span>
            <span class="what">${label(j)}${names?.[j] ? snippet(j) : ''}</span>
            <span class="tag${lift(j) >= 0.03 ? ' strong' : ''}" title="How visibly this concept shows up when it steers">${lift(j) >= 0.03 ? 'steers' : 'subtle'}</span>
        </li>`).join('');
    $('pinned').innerHTML = state.pinned.length
        ? state.pinned.map(j => `<span class="chip">${label(j)}<button type="button" data-unpin="${j}" aria-label="Unpin">×</button></span>`).join('')
        : '<span class="muted">No concepts pinned: both columns are the plain model.</span>';
}

$('passage').addEventListener('click', e => {
    const t = e.target.closest('.tok');
    if (!t) return;
    const n = +t.dataset.n;
    state.token = state.token === n ? null : n;
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

$('presets').innerHTML = Object.keys(PRESETS).map(k => `<button type="button" data-preset="${k}">${k}</button>`).join('');
$('presets').addEventListener('click', e => {
    const b = e.target.closest('[data-preset]');
    if (b) { $('text').value = PRESETS[b.dataset.preset]; read(); }
});
$('openings').innerHTML = OPENINGS.map(o => `<button type="button" class="link" data-open="${esc(o)}">${esc(o)}…</button>`).join('');
$('openings').addEventListener('click', e => {
    const b = e.target.closest('[data-open]');
    if (b) { $('prompt').value = b.dataset.open; write(); }
});
$('text').value = PRESETS.Whales;
$('read').addEventListener('click', read);
$('write').addEventListener('click', write);
$('again').addEventListener('click', () => { state.seed++; write(); });
$('prompt').addEventListener('keydown', e => { if (e.key === 'Enter') write(); });
$('strength').addEventListener('input', () => { $('strength-out').textContent = `${(+$('strength').value).toFixed(2).replace(/0$/, '')}×`; });
$('strength').addEventListener('change', write);
window.steeringApp = { state, worker };
