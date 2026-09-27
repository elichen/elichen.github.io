// Adversarially robust ResNet-50 (L2, eps = 3) in plain TensorFlow.js ops.
//
// Weights come from Salman et al., "Do Adversarially Robust ImageNet Models Transfer Better?"
// (https://huggingface.co/madrylab/robust-imagenet-models, MIT license), exported with batch norm
// folded into each convolution's bias and stored as float16.
//
// Mirrors tf.GraphModel.execute(): execute(images, outputs) takes [N, H, W, 3] images in [0, 1]
// and returns the named output, or an array for an array of names. Names are 'conv1', 'layerL.B'
// (a bottleneck block's output) and 'layerL.B/pre_relu' (the residual sum before its ReLU); with
// no name it returns the 1000 class logits.

const RESNET50_STAGES = [
    { layer: 1, blocks: 3, stride: 1 },
    { layer: 2, blocks: 4, stride: 2 },
    { layer: 3, blocks: 6, stride: 2 },
    { layer: 4, blocks: 3, stride: 2 }
];

class RobustResNet50 {
    static async load(baseUrl) {
        const response = await fetch(`${baseUrl}model.json`);
        if (!response.ok) {
            throw new Error(`Failed to load ${baseUrl}model.json: HTTP ${response.status}`);
        }
        const manifest = await response.json();
        const weights = await tf.io.loadWeights(manifest.weightsManifest, baseUrl);
        return new RobustResNet50(manifest, weights);
    }

    constructor(manifest, weights) {
        this.weights = weights;
        this.inputSize = manifest.inputSize;
        this.mean = tf.tensor1d(manifest.preprocessing.mean);
        this.std = tf.tensor1d(manifest.preprocessing.std);
    }

    // Channel counts of every named output, for building layer menus.
    static layerChannels() {
        const channels = { conv1: 64 };
        RESNET50_STAGES.forEach(({ layer, blocks }) => {
            for (let block = 0; block < blocks; block++) {
                channels[`layer${layer}.${block}`] = 256 * 2 ** (layer - 1);
            }
        });
        return channels;
    }

    // PyTorch pads symmetrically; TF's 'same' pads stride-2 convs only at the bottom/right, so pad explicitly.
    conv(x, name, stride, relu) {
        const filter = this.weights[`${name}/kernel`];
        const pad = Math.floor(filter.shape[0] / 2);
        const padded = pad > 0 ? tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]]) : x;
        return tf.fused.conv2d({
            x: padded,
            filter,
            strides: stride,
            pad: 'valid',
            bias: this.weights[`${name}/bias`],
            activation: relu ? 'relu' : 'linear'
        });
    }

    bottleneck(x, name, stride, downsample) {
        let y = this.conv(x, `${name}.conv1`, 1, true);
        y = this.conv(y, `${name}.conv2`, stride, true);
        y = this.conv(y, `${name}.conv3`, 1, false);
        const shortcut = downsample ? this.conv(x, `${name}.downsample`, stride, false) : x;
        return y.add(shortcut);
    }

    execute(images, outputs = 'logits') {
        const names = Array.isArray(outputs) ? outputs : [outputs];
        const found = {};
        const record = (name, tensor) => {
            if (names.includes(name)) {
                found[name] = tensor;
            }
            return names.every(n => n in found);
        };

        const results = tf.tidy(() => {
            let x = images.sub(this.mean).div(this.std);
            x = this.conv(x, 'conv1', 2, true);
            let done = record('conv1', x);

            // The input is post-ReLU, so zero padding acts like PyTorch's -inf padding for max pooling.
            x = tf.maxPool(tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]]), 3, 2, 'valid');

            for (const { layer, blocks, stride } of RESNET50_STAGES) {
                for (let block = 0; block < blocks && !done; block++) {
                    const name = `layer${layer}.${block}`;
                    const preRelu = this.bottleneck(x, name, block === 0 ? stride : 1, block === 0);
                    record(`${name}/pre_relu`, preRelu);
                    x = tf.relu(preRelu);
                    done = record(name, x);
                }
            }

            if (!done) {
                const pooled = tf.mean(x, [1, 2]);
                record('logits', tf.matMul(pooled, this.weights['fc/kernel']).add(this.weights['fc/bias']));
            }

            const missing = names.filter(n => !(n in found));
            if (missing.length > 0) {
                throw new Error(`Unknown ResNet-50 outputs: ${missing.join(', ')}`);
            }
            return names.map(n => found[n]);
        });

        return Array.isArray(outputs) ? results : results[0];
    }
}
