// GPT-2 small forward pass in pure JS, run on int8-quantized weights.
//
// The point of running the model ourselves (rather than through transformers.js)
// is that the standard ONNX export only emits logits and the key/value cache —
// it never exposes the attention matrices. Computing Q and K directly from the
// hidden states with the real weights gives the genuine, exact attention maps,
// which is the whole subject of the visualization.

const SQRT2_OVER_PI = Math.sqrt(2 / Math.PI);

// Parse the single-file weight blob described by manifest.json into typed-array
// views. int8 matrices keep a per-column (or per-row) f32 scale; we dequantize
// on the fly during matmul so peak memory stays near the ~122MB download size
// instead of the ~500MB an fp32 expansion would cost.
export function loadModel(buffer, manifest) {
    const bytes = new Uint8Array(buffer);
    const tensors = {};
    for (const t of manifest.tensors) {
        const entry = { shape: t.shape, kind: t.kind };
        if (t.kind === 'f32') {
            entry.data = new Float32Array(buffer, t.off, t.len / 4);
        } else {
            // i8c (per output column) or i8r (per row)
            entry.scale = new Float32Array(buffer, t.scale_off, t.scale_len / 4);
            entry.q = new Int8Array(buffer, t.off, t.len);
        }
        tensors[t.name] = entry;
    }
    return { cfg: manifest.config, t: tensors, bytes };
}

function layerNorm(x, T, d, g, b, eps) {
    const out = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
        const base = t * d;
        let mean = 0;
        for (let i = 0; i < d; i++) mean += x[base + i];
        mean /= d;
        let varSum = 0;
        for (let i = 0; i < d; i++) {
            const dv = x[base + i] - mean;
            varSum += dv * dv;
        }
        const inv = 1 / Math.sqrt(varSum / d + eps);
        for (let i = 0; i < d; i++) out[base + i] = (x[base + i] - mean) * inv * g[i] + b[i];
    }
    return out;
}

// y[T,outDim] = x[T,inDim] @ W + b, where W is int8 [inDim,outDim] (row-major)
// with a per-output-column scale. We accumulate the integer-weighted sum in f32
// and apply the per-column scale once at the end, then add the bias.
function matmul(x, T, inDim, weight, bias) {
    const { q, scale } = weight;
    const outDim = scale.length;
    const y = new Float32Array(T * outDim);
    for (let t = 0; t < T; t++) {
        const xBase = t * inDim;
        const yBase = t * outDim;
        for (let i = 0; i < inDim; i++) {
            const xi = x[xBase + i];
            if (xi === 0) continue;
            const qBase = i * outDim;
            for (let o = 0; o < outDim; o++) {
                y[yBase + o] += xi * q[qBase + o];
            }
        }
        for (let o = 0; o < outDim; o++) {
            y[yBase + o] = y[yBase + o] * scale[o] + (bias ? bias[o] : 0);
        }
    }
    return y;
}

function gelu(v) {
    return 0.5 * v * (1 + Math.tanh(SQRT2_OVER_PI * (v + 0.044715 * v * v * v)));
}

// Run the model on token ids. Returns { attn, logits }.
//   attn: Float32Array of length n_layer*n_head*T*T, layer-major then head, row=query, col=key.
//   logits: Float32Array[n_vocab] for the final position (next-token distribution).
export function forward(model, ids) {
    const { cfg, t } = model;
    const { n_layer, n_head, n_embd: d, n_vocab, eps } = cfg;
    const T = ids.length;
    if (T === 0) throw new Error('forward: empty token sequence');
    const hd = d / n_head;
    const scaleAttn = 1 / Math.sqrt(hd);

    // token + position embeddings (dequantize the needed wte rows)
    const wte = t.wte, wpe = t.wpe.data;
    let x = new Float32Array(T * d);
    for (let p = 0; p < T; p++) {
        const id = ids[p];
        const rowScale = wte.scale[id];
        const qBase = id * d;
        const wpeBase = p * d;
        const xBase = p * d;
        for (let i = 0; i < d; i++) {
            x[xBase + i] = wte.q[qBase + i] * rowScale + wpe[wpeBase + i];
        }
    }

    const attn = new Float32Array(n_layer * n_head * T * T);

    for (let L = 0; L < n_layer; L++) {
        const p = `h.${L}.`;
        const h = layerNorm(x, T, d, t[p + 'ln_1.w'].data, t[p + 'ln_1.b'].data, eps);
        const qkv = matmul(h, T, d, t[p + 'attn.cattn.w'], t[p + 'attn.cattn.b'].data); // [T, 3d]

        // attention per head; also accumulate context vectors
        const ctx = new Float32Array(T * d);
        const scores = new Float32Array(T); // reused per (head, query)
        for (let hi = 0; hi < n_head; hi++) {
            const qOff = hi * hd;          // queries live in [0,d)
            const kOff = d + hi * hd;      // keys in [d,2d)
            const vOff = 2 * d + hi * hd;  // values in [2d,3d)
            const layerHeadBase = (L * n_head + hi) * T * T;
            for (let qi = 0; qi < T; qi++) {
                const qRow = qi * 3 * d + qOff;
                let maxScore = -Infinity;
                for (let ki = 0; ki <= qi; ki++) {
                    const kRow = ki * 3 * d + kOff;
                    let dot = 0;
                    for (let c = 0; c < hd; c++) dot += qkv[qRow + c] * qkv[kRow + c];
                    dot *= scaleAttn;
                    scores[ki] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                let sum = 0;
                for (let ki = 0; ki <= qi; ki++) {
                    const e = Math.exp(scores[ki] - maxScore);
                    scores[ki] = e;
                    sum += e;
                }
                const inv = 1 / sum;
                const outRow = layerHeadBase + qi * T;
                const ctxRow = qi * d + hi * hd;
                for (let ki = 0; ki <= qi; ki++) {
                    const a = scores[ki] * inv;
                    attn[outRow + ki] = a;
                    const vRow = ki * 3 * d + vOff;
                    for (let c = 0; c < hd; c++) ctx[ctxRow + c] += a * qkv[vRow + c];
                }
            }
        }

        // output projection + residual
        const attnOut = matmul(ctx, T, d, t[p + 'attn.cproj.w'], t[p + 'attn.cproj.b'].data);
        for (let i = 0; i < T * d; i++) x[i] += attnOut[i];

        // MLP + residual
        const h2 = layerNorm(x, T, d, t[p + 'ln_2.w'].data, t[p + 'ln_2.b'].data, eps);
        const ff = matmul(h2, T, d, t[p + 'mlp.cfc.w'], t[p + 'mlp.cfc.b'].data); // [T,4d]
        for (let i = 0; i < ff.length; i++) ff[i] = gelu(ff[i]);
        const ffOut = matmul(ff, T, 4 * d, t[p + 'mlp.cproj.w'], t[p + 'mlp.cproj.b'].data);
        for (let i = 0; i < T * d; i++) x[i] += ffOut[i];
    }

    // final layer norm + tied-embedding logits for the last position only
    const xf = layerNorm(x, T, d, t['ln_f.w'].data, t['ln_f.b'].data, eps);
    const last = (T - 1) * d;
    const logits = new Float32Array(n_vocab);
    const wte2 = t.wte;
    for (let v = 0; v < n_vocab; v++) {
        const rowScale = wte2.scale[v];
        const qBase = v * d;
        let acc = 0;
        for (let i = 0; i < d; i++) acc += xf[last + i] * wte2.q[qBase + i];
        logits[v] = acc * rowScale;
    }

    return { attn, logits, T };
}
