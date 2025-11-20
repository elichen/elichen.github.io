const TFJS_MODEL_URL = 'web_model/snake_tfjs/model.json';

async function loadGraphPolicy() {
    await tf.ready();
    const model = await tf.loadGraphModel(TFJS_MODEL_URL);
    return model;
}
