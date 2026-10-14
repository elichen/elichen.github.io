// Parity check on tfjs's CPU backend in Node (no GPU): the browser code in lucid/robust-resnet.js and
// concepts/sae.js must match PyTorch's references in concepts/tools/test/test.json.
//   node concepts/tools/parity.js <repo> [n_images]   (npm install in concepts/tools first)
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const tf = require('@tensorflow/tfjs');
const { PNG } = require('pngjs');

const repo = process.argv[2];
const n = Number(process.argv[3] || 2);
global.tf = tf;
for (const file of ['lucid/robust-resnet.js', 'concepts/sae.js']) {
    vm.runInThisContext(fs.readFileSync(path.join(repo, file), 'utf8') +
        '\nglobalThis.RobustResNet50 = typeof RobustResNet50 !== "undefined" ? RobustResNet50 : globalThis.RobustResNet50;' +
        '\nglobalThis.ConceptDictionary = typeof ConceptDictionary !== "undefined" ? ConceptDictionary : globalThis.ConceptDictionary;');
}

function loadWeights(dir) {
    const manifest = JSON.parse(fs.readFileSync(path.join(dir, 'model.json')));
    const group = manifest.weightsManifest[0];
    const buf = Buffer.concat(group.paths.map(p => fs.readFileSync(path.join(dir, p))));
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    return [manifest, tf.io.decodeWeights(ab, group.weights)];
}

const overlap = (a, b) => a.filter(x => b.includes(x)).length;
// Indices and values of the n largest entries of [index, value] pairs.
const topn = (pairs, n) => [...pairs].sort((a, b) => b[1] - a[1]).slice(0, n);
const maxDiff = (a, b) => Math.max(...b.map((v, i) => Math.abs(a[i] - v)));

(async () => {
    await tf.setBackend('cpu');
    const resnet = new RobustResNet50(...loadWeights(path.join(repo, 'lucid/robust-resnet50')));
    const dir = path.join(repo, 'concepts/model');
    const sae = new ConceptDictionary(JSON.parse(fs.readFileSync(path.join(dir, 'sae.json'))),
        new Int8Array(fs.readFileSync(path.join(dir, 'encoder.bin'))), new Int8Array(fs.readFileSync(path.join(dir, 'decoder.bin'))));
    const fc = resnet.weights['fc/kernel'].dataSync();
    const cases = JSON.parse(fs.readFileSync(path.join(repo, 'concepts/tools/test/test.json')));
    for (const c of cases.slice(0, n)) {
        const png = PNG.sync.read(fs.readFileSync(path.join(repo, 'concepts/tools/test', c.image)));
        const rgb = new Float32Array(224 * 224 * 3);
        for (let i = 0; i < 224 * 224; i++) {
            for (let ch = 0; ch < 3; ch++) rgb[i * 3 + ch] = png.data[i * 4 + ch] / 255;
        }
        const x = tf.tensor4d(rgb, [1, 224, 224, 3]);
        const t0 = Date.now();
        const [acts, logits] = resnet.execute(x, [sae.layer, 'logits']);
        const code = await sae.encode(acts);
        const top = topn(sae.means(code).entries(), 10);
        const contrib = sae.contributions(code, fc, c.class);
        const topC = topn(contrib.entries(), 10);
        const top5 = topn(logits.dataSync().entries(), 5).map(e => e[0]);
        const total = [...contrib.values()].reduce((a, b) => a + b, 0);
        console.log(`${c.image}: top5 ${top5} vs ${c.top5} | concepts overlap ${overlap(top.map(e => e[0]), c.concepts)}/10,` +
            ` means diff ${maxDiff(top.map(e => e[1]), c.concept_means).toFixed(3)} (top ${c.concept_means[0].toFixed(2)}) |` +
            ` contrib overlap ${overlap(topC.map(e => e[0]), c.contrib_concepts)}/10,` +
            ` diff ${maxDiff(topC.map(e => e[1]), c.contrib_values).toFixed(3)} (top ${c.contrib_values[0].toFixed(2)}),` +
            ` total ${total.toFixed(2)} vs ${c.contrib_total.toFixed(2)} | ${Date.now() - t0} ms`);
    }
})();
