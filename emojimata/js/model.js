class CAModel {
    constructor() {
        this.D = 96;
        this.scale = 4;
        this.state = null;
        this.model = null;
    }

    async loadModel() {
        const modelPath = './8000.weights.h5.json';
        const response = await fetch(modelPath);
        const modelJSON = await response.json();
        const consts = parseConsts(modelJSON);
        
        this.model = await tf.loadGraphModel(modelPath);
        Object.assign(this.model.weights, consts);
        
        this.initState();
    }

    initState() {
        const seed = new Array(16).fill(0).map((x, i) => i < 3 ? 0 : 1);
        const seedTensor = tf.tensor(seed, [1, 1, 1, 16]);
        
        const initState = tf.tidy(() => {
            const D2 = this.D / 2;
            return seedTensor.pad([[0, 0], [D2-1, D2], [D2-1, D2], [0, 0]]);
        });
        
        this.state = tf.variable(initState);
    }

    damage(x, y, r) {
        const [_, h, w, ch] = this.state.shape;
        tf.tidy(() => {
            const rx = tf.range(0, w).sub(x).div(r).square().expandDims(0);
            const ry = tf.range(0, h).sub(y).div(r).square().expandDims(1);
            const mask = rx.add(ry).greater(1.0).expandDims(2);
            this.state.assign(this.state.mul(mask));
        });
    }

    plantSeed(x, y) {
        const [_, h, w, ch] = this.state.shape;
        const seed = new Array(16).fill(0).map((x, i) => i < 3 ? 0 : 1);
        const seedTensor = tf.tensor(seed, [1, 1, 1, 16]);
        
        const x2 = w - x - seedTensor.shape[2];
        const y2 = h - y - seedTensor.shape[1];
        
        if (x < 0 || x2 < 0 || y2 < 0 || y2 < 0) return;
        
        tf.tidy(() => {
            const a = seedTensor.pad([[0, 0], [y, y2], [x, x2], [0, 0]]);
            this.state.assign(this.state.add(a));
        });
    }

    step() {
        tf.tidy(() => {
            this.state.assign(this.model.execute(
                {
                    x: this.state,
                    fire_rate: tf.tensor(0.5),
                    angle: tf.tensor(0.0),
                    step_size: tf.tensor(1.0)
                },
                ['Identity']
            ));
        });
    }
} 