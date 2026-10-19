// A TopK sparse autoencoder over one robust ResNet-50 layer.
//
// Activations x (one 2048-vector per position) are divided by `scale`, then
//   z = ReLU(TopK(W_enc (x - b_dec) + b_enc)),   x ~ scale * (W_dec z + b_dec).
// Only the encoder's matrix product runs on the GPU. The code is sparse (k concepts per position), so the
// top-k and everything after it run on the CPU.
// layer4.2 feeds the average pool and the classifier directly, so a concept's contribution to a class
// logit is exact: scale * mean over positions of z_j * (W_dec[j] . fc[:, class]).

// Weights are int8 with one scale per concept: encoder.bin is W_enc [channels, size] and decoder.bin is
// W_dec [size, channels], both row-major; sae.json holds the scales, biases and settings.

async function fetchOk(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load ${url}: HTTP ${response.status}`);
    }
    return response;
}

class ConceptDictionary {
    static async load(baseUrl) {
        const [meta, encoder, decoder] = await Promise.all([
            fetchOk(`${baseUrl}sae.json`).then(r => r.json()),
            fetchOk(`${baseUrl}encoder.bin`).then(r => r.arrayBuffer()),
            fetchOk(`${baseUrl}decoder.bin`).then(r => r.arrayBuffer())
        ]);
        return new ConceptDictionary(meta, new Int8Array(encoder), new Int8Array(decoder));
    }

    constructor(meta, encoder, decoder) {
        this.layer = meta.layer;
        this.size = meta.latents;
        this.channels = meta.channels;
        this.k = meta.k;
        this.scale = meta.scale;
        this.ids = meta.ids;  // the latents' numbers in the full SAE
        const m = this.size, d = this.channels;
        const W_enc = new Float32Array(d * m);
        for (let i = 0; i < d * m; i++) W_enc[i] = encoder[i] * meta.encoderScale[i % m];
        this.W_enc = tf.tensor2d(W_enc, [d, m]);
        this.b_enc = tf.tensor1d(meta.b_enc);
        this.b_dec = tf.tensor1d(meta.b_dec);
        // Row j is concept j's direction, times decoderScale[j]. Kept as int8 (12 MB, not 48 as floats).
        this.decoder = decoder;
        this.decoderScale = Float32Array.from(meta.decoderScale);
    }

    // acts: [1, H, W, C] layer output. Resolves to the sparse code: for each position, the k largest
    // concepts' indices and activations (activations are 0 where the pre-activation was negative).
    async encode(acts) {
        const pre = tf.tidy(() => {
            const x = acts.reshape([-1, this.channels]).div(this.scale).sub(this.b_dec);
            return tf.matMul(x, this.W_enc).add(this.b_enc);
        });
        const data = await pre.data();
        pre.dispose();
        const m = this.size, k = this.k, positions = data.length / m;
        const idx = new Int32Array(positions * k), val = new Float32Array(positions * k);
        for (let p = 0; p < positions; p++) {
            // Keep the row's k largest in descending order; most entries fail the first comparison.
            const vs = val.subarray(p * k, p * k + k).fill(-Infinity), is = idx.subarray(p * k, p * k + k);
            for (let j = p * m, end = j + m; j < end; j++) {
                const v = data[j];
                if (v <= vs[k - 1]) continue;
                let q = k - 1;
                for (; q > 0 && vs[q - 1] < v; q--) {
                    vs[q] = vs[q - 1];
                    is[q] = is[q - 1];
                }
                vs[q] = v;
                is[q] = j - p * m;
            }
            for (let q = 0; q < k; q++) vs[q] = Math.max(vs[q], 0);
        }
        return { positions, idx, val };
    }

    // Mean activation of every concept over positions: Float32Array [size].
    means(code) {
        const out = new Float32Array(this.size);
        for (let q = 0; q < code.idx.length; q++) out[code.idx[q]] += code.val[q];
        return out.map(v => v / code.positions);
    }

    // One concept's activation at every position: Float32Array [positions].
    map(code, concept) {
        const out = new Float32Array(code.positions);
        for (let q = 0; q < code.idx.length; q++) {
            if (code.idx[q] === concept) out[Math.floor(q / this.k)] = code.val[q];
        }
        return out;
    }

    // W_dec[j] . fc[:, c], scaled back to the layer's units: what one unit of concept j adds to class c's logit.
    // fc: Float32Array [channels * classes] (the classifier's [C, classes] kernel).
    weight(concept, fc, classIndex, classes = 1000) {
        const row = concept * this.channels;
        let s = 0;
        for (let i = 0; i < this.channels; i++) s += this.decoder[row + i] * fc[i * classes + classIndex];
        return s * this.decoderScale[concept] * this.scale;
    }

    // What one unit of concept j adds to every class's logit: Float32Array [classes].
    classWeights(concept, fc, classes = 1000) {
        const out = new Float32Array(classes), row = concept * this.channels;
        const scale = this.decoderScale[concept] * this.scale;
        for (let i = 0; i < this.channels; i++) {
            const w = this.decoder[row + i] * scale;
            for (let c = 0, f = i * classes; c < classes; c++) out[c] += w * fc[f + c];
        }
        return out;
    }

    // The n concepts whose directions are closest to concept j's (cosine): [[concept, cosine], ...].
    // Cosines don't depend on the rows' scales, so this works on the int8 rows directly.
    neighbours(concept, n) {
        const d = this.channels, W = this.decoder;
        if (!this.norms) {
            this.norms = new Float32Array(this.size);
            for (let j = 0; j < this.size; j++) {
                let s = 0;
                for (let i = j * d; i < (j + 1) * d; i++) s += W[i] * W[i];
                this.norms[j] = Math.sqrt(s);
            }
        }
        const scores = [], row = concept * d;
        for (let j = 0; j < this.size; j++) {
            if (j === concept) continue;
            let s = 0;
            for (let i = 0; i < d; i++) s += W[row + i] * W[j * d + i];
            scores.push([j, s / (this.norms[j] * this.norms[concept])]);
        }
        return scores.sort((a, b) => b[1] - a[1]).slice(0, n);
    }

    // How much each active concept adds to a class logit, or with `versus` to the gap between two classes'
    // logits (all the softmax cares about): Map concept -> contribution.
    contributions(code, fc, classIndex, versus = null) {
        const means = this.means(code), out = new Map();
        means.forEach((v, j) => {
            if (v > 0) out.set(j, v * (this.weight(j, fc, classIndex) - (versus === null ? 0 : this.weight(j, fc, versus))));
        });
        return out;
    }
}
