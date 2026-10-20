// Gemma 3's tokenizer (tokenizer.json): added tokens are matched first on the raw text; everything between them
// has spaces replaced by "▁" and is BPE-merged as one word, lowest merge rank first (leftmost on ties);
// characters missing from the vocabulary fall back to their UTF-8 bytes (<0xXX> tokens).

export class Tokenizer {
  constructor(vocab, mergesBuffer) {
    this.vocab = vocab.vocab;
    this.bos = vocab.bos;
    this.eos = vocab.eos;
    this.ids = new Map(this.vocab.map((s, i) => [s, i]));
    const m = new Int32Array(mergesBuffer);
    this.merges = new Map();
    for (let r = 0; r < m.length / 2; r++) {
      const a = m[2 * r], b = m[2 * r + 1];
      this.merges.set(a * 262144 + b, [r, this.ids.get(this.vocab[a] + this.vocab[b])]);
    }
    this.added = new Map(vocab.added.map(t => [t.content, t.id]));
    const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const alts = [...this.added.keys()].sort((a, b) => b.length - a.length).map(esc);
    this.addedRe = new RegExp(alts.join('|'), 'g');
    this.special = new Set([0, 1, 2, 3]);
    this.bytes = new Map();
    for (let b = 0; b < 256; b++) this.bytes.set(this.ids.get(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`), b);
  }

  encode(text, bos = true) {
    const out = bos ? [this.bos] : [];
    let last = 0;
    for (const m of text.matchAll(this.addedRe)) {
      if (m.index > last) this.bpe(text.slice(last, m.index), out);
      out.push(this.added.get(m[0]));
      last = m.index + m[0].length;
    }
    if (last < text.length) this.bpe(text.slice(last), out);
    return out;
  }

  bpe(text, out) {
    const sym = [];
    for (const ch of text.replaceAll(' ', '▁')) {
      const id = this.ids.get(ch);
      if (id !== undefined) sym.push(id);
      else for (const b of new TextEncoder().encode(ch)) sym.push(this.ids.get(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`));
    }
    const n = sym.length;
    const prev = Int32Array.from({ length: n }, (_, i) => i - 1);
    const next = Int32Array.from({ length: n }, (_, i) => (i + 1 < n ? i + 1 : -1));
    const alive = new Uint8Array(n).fill(1);
    const heap = new Heap();
    const push = i => {
      const j = next[i];
      if (j < 0) return;
      const m = this.merges.get(sym[i] * 262144 + sym[j]);
      if (m) heap.push(m[0], i, sym[i], sym[j], m[1]);
    };
    for (let i = 0; i < n - 1; i++) push(i);
    while (heap.size) {
      const [, i, a, b, id] = heap.pop();
      const j = next[i];
      if (!alive[i] || j < 0 || sym[i] !== a || sym[j] !== b) continue;    // stale entry
      sym[i] = id;
      alive[j] = 0;
      next[i] = next[j];
      if (next[j] >= 0) prev[next[j]] = i;
      if (prev[i] >= 0) push(prev[i]);
      push(i);
    }
    for (let i = 0; i >= 0 && i < n; i = next[i]) out.push(sym[i]);
  }

  decode(ids) {
    const bytes = [];
    const enc = new TextEncoder();
    for (const id of ids) {
      if (this.special.has(id)) continue;
      const b = this.bytes.get(id);
      if (b !== undefined) bytes.push(b);
      else bytes.push(...enc.encode(this.vocab[id].replaceAll('▁', ' ')));
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }
}

// min-heap on (rank, position)
class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  less(x, y) { return x[0] < y[0] || (x[0] === y[0] && x[1] < y[1]); }
  push(...e) {
    const a = this.a;
    a.push(e);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(a[i], a[p])) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop() {
    const a = this.a, top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && this.less(a[l], a[m])) m = l;
        if (r < a.length && this.less(a[r], a[m])) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]];
        i = m;
      }
    }
    return top;
  }
}
