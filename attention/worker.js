// Owns the GPT-2 weights and runs the forward pass off the main thread so the
// UI never janks while attention is computed.
import { loadModel, forward } from './gpt2.js';

const MODEL_DIR = './model/';
const CACHE_NAME = 'gpt2-attn-v1';

let model = null;

async function fetchPart(cache, url) {
    // Serve from Cache Storage if we've already downloaded this part once.
    let res = await cache.match(url);
    if (!res) {
        res = await fetch(url);
        if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
        cache.put(url, res.clone()).catch(() => {}); // best-effort persist
    }
    return res;
}

// The weights are split into sub-100MB parts (GitHub rejects files >100MB).
// Stream them in order into one preallocated buffer of the known total size, so
// we never hold a second full copy and report a single combined progress bar.
async function loadWeights(parts, total, onProgress) {
    const cache = await caches.open(CACHE_NAME);
    const out = new Uint8Array(total);
    let loaded = 0;
    for (const name of parts) {
        const res = await fetchPart(cache, MODEL_DIR + name);
        const reader = res.body.getReader();
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            out.set(value, loaded);
            loaded += value.length;
            onProgress(loaded, total);
        }
    }
    if (loaded !== total) throw new Error(`weight size mismatch: ${loaded} != ${total}`);
    return out.buffer;
}

async function init() {
    const manifest = await (await fetch(MODEL_DIR + 'manifest.json')).json();
    const buffer = await loadWeights(manifest.parts, manifest.total_bytes, (loaded, total) => {
        self.postMessage({ type: 'progress', loaded, total });
    });
    model = loadModel(buffer, manifest);
    self.postMessage({ type: 'ready', config: model.cfg });
}

self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === 'init') {
        try { await init(); }
        catch (err) { self.postMessage({ type: 'error', message: String(err) }); }
    } else if (msg.type === 'run') {
        if (!model) return;
        try {
            const t0 = performance.now();
            const { attn, logits, T } = forward(model, msg.ids);
            const ms = performance.now() - t0;
            self.postMessage(
                { type: 'result', reqId: msg.reqId, attn, logits, T, ms },
                [attn.buffer, logits.buffer]
            );
        } catch (err) {
            self.postMessage({ type: 'run-error', reqId: msg.reqId, message: String(err) });
        }
    }
};
