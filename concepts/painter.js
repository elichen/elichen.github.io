// Painting a concept into a photo: gradient steps that raise the concept's activation under the brush.
//
// The change is drawn in Lucid's image parameterization (as in lucid/script.js): a 1/f-scaled Fourier
// spectrum in a decorrelated color space, here without the sigmoid, added to the photo inside the brushed
// mask. On this robust network raw pixels reach about the same activation for the same change; the 1/f
// scaling makes the change smooth and the decorrelated colors keep it natural instead of rainbow-speckled.

const PAINT_SIZE = 224;
// At ~20 steps a second a drag shows the concept within about 2 s. Bigger steps paint faster but harshly,
// crushing blacks and blowing out whites; 0.008 took about 4 s.
const PAINT_LEARNING_RATE = 0.02;

// Square root of ImageNet's color covariance, from lucid.optvis.param.color.
const PAINT_COLOR_SQRT = [
    [0.26, 0.09, 0.02],
    [0.27, 0.00, -0.05],
    [0.27, -0.09, 0.03]
];

// DFT matrices for a real 2D inverse FFT done as matrix products (tfjs has no gradient for its FFT ops).
function paintFourierBasis(size) {
    const freqWidth = Math.floor(size / 2) + 1;
    const table = (rows, cols, f) => {
        const out = new Float32Array(rows * cols);
        for (let k = 0; k < rows; k++) {
            for (let n = 0; n < cols; n++) out[k * cols + n] = f(2 * Math.PI * k * n / size);
        }
        return out;
    };
    // Only non-negative x frequencies are stored, so every column but DC and Nyquist stands in for its mirror.
    const columnWeights = new Float32Array(freqWidth).map((_, k) => (k === 0 || 2 * k === size ? 1 : 2) / size);
    // Lucid's 1/f scaling, with its divide-by-4.
    const spectrumScale = new Float32Array(size * freqWidth);
    for (let y = 0; y < size; y++) {
        const fy = (y <= (size - 1) / 2 ? y : y - size) / size;
        for (let x = 0; x < freqWidth; x++) {
            spectrumScale[y * freqWidth + x] = 1 / Math.max(Math.hypot(x / size, fy), 1 / size) / 4;
        }
    }
    const C = PAINT_COLOR_SQRT;
    const maxNorm = Math.max(...[0, 1, 2].map(j => Math.hypot(C[0][j], C[1][j], C[2][j])));
    return {
        size,
        freqWidth,
        cosY: tf.tidy(() => tf.tensor2d(table(size, size, Math.cos), [size, size]).expandDims(0).tile([3, 1, 1])),
        sinY: tf.tidy(() => tf.tensor2d(table(size, size, Math.sin), [size, size]).expandDims(0).tile([3, 1, 1])),
        cosX: tf.tensor2d(table(freqWidth, size, Math.cos), [freqWidth, size]),
        sinX: tf.tensor2d(table(freqWidth, size, Math.sin), [freqWidth, size]),
        columnWeights: tf.tensor1d(columnWeights),
        spectrumScale: tf.tensor2d(spectrumScale, [size, freqWidth]),
        // 1x1 conv filter: rgb = decorrelated @ C^T, C normalized by its largest column norm as in Lucid.
        colorFilter: tf.tensor4d([0, 1, 2].flatMap(i => [0, 1, 2].map(o => C[o][i] / maxNorm)), [1, 1, 3, 3])
    };
}

class Painter {
    constructor(resnet, sae) {
        this.resnet = resnet;
        this.sae = sae;
        this.basis = paintFourierBasis(PAINT_SIZE);
        this.stroke = null;
    }

    // Starts a stroke on image x0 ([1, 224, 224, 3] in [0, 1]) that paints one concept.
    begin(x0, concept, learningRate = PAINT_LEARNING_RATE) {
        const { size, freqWidth } = this.basis;
        this.stroke = {
            x0: x0.clone(),
            real: tf.variable(tf.zeros([3, size, freqWidth])),
            imag: tf.variable(tf.zeros([3, size, freqWidth])),
            optimizer: tf.train.adam(learningRate),
            column: this.sae.W_enc.slice([0, concept], [-1, 1]),
            bias: this.sae.b_enc.slice([concept], [1]),
            maskData: new Float32Array(size * size),
            mask: null,
            steps: 0
        };
    }

    // A soft round dab of the brush centered at (x, y), in image pixels.
    dab(x, y, radius) {
        const s = this.stroke, size = PAINT_SIZE;
        for (let py = Math.max(0, Math.floor(y - radius - 4)); py < Math.min(size, y + radius + 4); py++) {
            for (let px = Math.max(0, Math.floor(x - radius - 4)); px < Math.min(size, x + radius + 4); px++) {
                const a = Math.min(1, Math.max(0, (radius + 4 - Math.hypot(px - x, py - y)) / 8));
                const i = py * size + px;
                if (a > s.maskData[i]) s.maskData[i] = a;
            }
        }
        if (s.mask) s.mask.dispose();
        s.mask = null;
    }

    maskTensor() {
        const s = this.stroke;
        if (!s.mask) s.mask = tf.tensor4d(s.maskData, [1, PAINT_SIZE, PAINT_SIZE, 1]);
        return s.mask;
    }

    // The change, [1, 224, 224, 3]: inverse real FFT of the scaled spectrum, then recorrelated colors.
    delta() {
        const { basis } = this, { real: realVar, imag: imagVar } = this.stroke;
        const { size, freqWidth } = basis;
        const real = realVar.mul(basis.spectrumScale);
        const imag = imagVar.mul(basis.spectrumScale);
        const colReal = tf.matMul(basis.cosY, real).sub(tf.matMul(basis.sinY, imag)).mul(basis.columnWeights);
        const colImag = tf.matMul(basis.cosY, imag).add(tf.matMul(basis.sinY, real)).mul(basis.columnWeights);
        const decorrelated = tf.matMul(colReal.reshape([3 * size, freqWidth]), basis.cosX)
            .sub(tf.matMul(colImag.reshape([3 * size, freqWidth]), basis.sinX))
            .reshape([3, size, size]);
        return tf.conv2d(decorrelated.transpose([1, 2, 0]).expandDims(0), basis.colorFilter, 1, 'valid');
    }

    // The photo with the stroke so far: [1, 224, 224, 3].
    image() {
        const s = this.stroke, mask = this.maskTensor();
        return tf.tidy(() => s.x0.add(this.delta().mul(mask)).clipByValue(0, 1));
    }

    // One Adam step on the mean pre-activation of the concept over the positions under the brush.
    step() {
        const s = this.stroke, sae = this.sae, mask = this.maskTensor();
        const weights = tf.tidy(() => tf.avgPool(mask, 32, 32, 'valid').reshape([-1, 1]));
        const total = weights.sum();
        const loss = s.optimizer.minimize(() => tf.tidy(() => {
            const x = s.x0.add(this.delta().mul(mask)).clipByValue(0, 1);
            const acts = this.resnet.execute(x, sae.layer).reshape([-1, sae.channels]);
            const pre = tf.matMul(acts.div(sae.scale).sub(sae.b_dec), s.column).add(s.bias);
            return pre.mul(weights).sum().div(total).neg();
        }), true, [s.real, s.imag]);
        tf.dispose([weights, total]);
        s.steps++;
        return loss;
    }

    // Ends the stroke and returns the painted photo.
    end() {
        const out = this.image(), s = this.stroke;
        tf.dispose([s.x0, s.real, s.imag, s.column, s.bias, s.mask]);
        s.optimizer.dispose();
        this.stroke = null;
        return out;
    }
}
