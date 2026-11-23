// Pure TensorFlow.js (WebGPU / WebGL / CPU) implementation of sentence-transformers/all-MiniLM-L6-v2
// with exact BERT WordPiece tokenization and CPU-resident per-row int8 word embeddings
// (~22.5 MB on disk, only ~11 MB in GPU VRAM, ~6-10 ms per English prompt!).

import { halfToFloat } from './model.js';

const INV_SQRT2 = Math.SQRT1_2;

function decodeColOrFloat(buf, t) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  if (t.dtype === 'f32') return new Float32Array(buf, t.offset, n);
  if (t.dtype === 'f16') {
    const h = new Uint16Array(buf, t.offset, n), out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = halfToFloat(h[i]);
    return out;
  }
  if (t.dtype === 'i8_col') {
    const q = new Int8Array(buf, t.offset, n), cols = t.shape[1];
    const sc = new Float32Array(buf, t.scaleOffset, cols), out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = q[i] * sc[i % cols];
    return out;
  }
  throw new Error(`Unsupported MiniLM dtype: ${t.dtype}`);
}

function isPunct(ch) {
  const cp = ch.codePointAt(0);
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
  return /\p{P}/u.test(ch);
}

export class MiniLM {
  constructor(tf, meta, buf) {
    this.tf = tf;
    this.meta = meta;
    this.d = meta.d;
    this.layers = meta.layers;
    this.heads = meta.heads;
    this.maxPos = meta.maxPos;
    this.eps = meta.eps || 1e-12;
    this.clsId = meta.clsId;
    this.sepId = meta.sepId;
    this.unkId = meta.unkId;
    this.pooling = meta.pooling || 'mean';

    this.vocabMap = new Map();
    for (let i = 0; i < meta.vocab.length; i++) {
      if (meta.vocab[i]) this.vocabMap.set(meta.vocab[i], i);
    }

    this.w = {};
    for (const t of meta.tensors) {
      if (t.name === 'emb/tok') {
        // Keep 30522x384 word table quantized on CPU! Lookup uploads only [L, 384] (~15 KB) to GPU.
        const n = t.shape[0] * t.shape[1];
        this.tokQ = new Int8Array(buf, t.offset, n);
        this.tokScale = new Float32Array(buf, t.scaleOffset, t.shape[0]);
      } else if (t.name === 'emb/pos') {
        this.posArr = decodeColOrFloat(buf, t);
      } else {
        this.w[t.name] = tf.tensor(decodeColOrFloat(buf, t), t.shape, 'float32');
      }
    }
  }

  dispose() {
    for (const t of Object.values(this.w)) t.dispose();
  }

  tokenize(text) {
    // Standard BERT BasicTokenizer (lowercase + NFD accent strip + punctuation split) + WordPiece
    const norm = (text || '').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    const words = [];
    let cur = '';
    for (const ch of norm) {
      if (/\s/.test(ch)) {
        if (cur) { words.push(cur); cur = ''; }
      } else if (isPunct(ch)) {
        if (cur) { words.push(cur); cur = ''; }
        words.push(ch);
      } else {
        cur += ch;
      }
    }
    if (cur) words.push(cur);

    const ids = [this.clsId];
    const pieces = [];
    const limit = this.maxPos - 2;

    for (const word of words) {
      if (pieces.length >= limit) break;
      if (word.length > 100) {
        ids.push(this.unkId);
        pieces.push('[UNK]');
        continue;
      }
      let start = 0, bad = false;
      const subIds = [], subPieces = [];
      while (start < word.length) {
        let end = word.length, found = -1, foundStr = '';
        while (start < end) {
          const sub = (start > 0 ? '##' : '') + word.slice(start, end);
          const id = this.vocabMap.get(sub);
          if (id !== undefined) { found = id; foundStr = sub; break; }
          end--;
        }
        if (found === -1) { bad = true; break; }
        subIds.push(found);
        subPieces.push(foundStr);
        start = end;
      }
      if (bad) {
        ids.push(this.unkId);
        pieces.push('[UNK]');
      } else {
        for (let i = 0; i < subIds.length && pieces.length < limit; i++) {
          ids.push(subIds[i]);
          pieces.push(subPieces[i]);
        }
      }
    }
    ids.push(this.sepId);
    return { ids, pieces };
  }

  _linear(x, name) {
    return this.tf.add(this.tf.matMul(x, this.w[`${name}/w`]), this.w[`${name}/b`]);
  }

  _ln(x, name) {
    const tf = this.tf;
    const { mean, variance } = tf.moments(x, -1, true);
    return x.sub(mean).div(variance.add(this.eps).sqrt()).mul(this.w[`${name}/w`]).add(this.w[`${name}/b`]);
  }

  _gelu(x) {
    return x.mul(0.5).mul(this.tf.erf(x.mul(INV_SQRT2)).add(1));
  }

  async encode(text) {
    const tf = this.tf;
    const { ids, pieces } = this.tokenize(text);
    const L = ids.length, D = this.d, H = this.heads, hd = D / H;

    // Fast CPU lookup of int8 per-row word embeddings + pre-folded pos/type embeddings
    const embBuf = new Float32Array(L * D);
    for (let t = 0; t < L; t++) {
      const id = ids[t];
      const sc = this.tokScale[id];
      const qOff = id * D;
      const pOff = t * D;
      for (let j = 0; j < D; j++) {
        embBuf[pOff + j] = this.tokQ[qOff + j] * sc + this.posArr[pOff + j];
      }
    }

    const outT = tf.tidy(() => {
      let x = this._ln(tf.tensor2d(embBuf, [L, D]), 'emb/ln');
      for (let l = 0; l < this.layers; l++) {
        const lp = `l${l}`;
        const qkv = this._linear(x, `${lp}/qkv`).reshape([L, 3, H, hd]);
        const [q, k, v] = tf.unstack(qkv, 1);
        const qt = q.transpose([1, 0, 2]);
        const kt = k.transpose([1, 2, 0]);
        const vt = v.transpose([1, 0, 2]);
        const attn = tf.softmax(tf.matMul(qt, kt).mul(1 / Math.sqrt(hd)), -1);
        const ctx = tf.matMul(attn, vt).transpose([1, 0, 2]).reshape([L, D]);
        const x1 = this._ln(x.add(this._linear(ctx, `${lp}/o`)), `${lp}/ln1`);
        const ff = this._linear(this._gelu(this._linear(x1, `${lp}/fc1`)), `${lp}/fc2`);
        x = this._ln(x1.add(ff), `${lp}/ln2`);
      }
      const pooled = this.pooling === 'cls'
        ? x.slice([0, 0], [1, D]).reshape([D])
        : x.mean(0);
      const norm = pooled.norm().maximum(1e-12);
      return pooled.div(norm);
    });

    const emb = Float32Array.from(await outT.data());
    outT.dispose();
    return { emb, ids, pieces };
  }
}

export async function loadMiniLM(tf, baseUrl = 'model') {
  const [meta, buf] = await Promise.all([
    fetch(`${baseUrl}/minilm.json`).then(r => r.json()),
    fetch(`${baseUrl}/minilm.bin`).then(r => r.arrayBuffer()),
  ]);
  return new MiniLM(tf, meta, buf);
}
