class CAModel {
    constructor() {
        this.D = 96;
        this.scale = 4;
        this.state = null;
        this.model = null;
        this.tileSize = 96;  // Original model size
        this.padding = 16;   // Padding for tile boundaries
    }

    async loadModel() {
        const modelPath = './8000.weights.h5.json';
        console.log('Loading model from:', modelPath);
        const response = await fetch(modelPath);
        const modelJSON = await response.json();
        const consts = parseConsts(modelJSON);
        
        this.model = await tf.loadGraphModel(modelPath);
        console.log('Model loaded');
        Object.assign(this.model.weights, consts);
        
        this.initState();
    }

    initState() {
        // Start with a completely blank state (all 0.0s)
        const initState = tf.tidy(() => {
            const base = tf.zeros([1, this.D, this.D * 2, 16]);
            
            // Verify initial state
            const data = base.dataSync();
            console.log('Initial state:', {
                shape: base.shape,
                min: 0,
                max: 0,
                channels: 16,
                totalPixels: this.D * this.D * 2
            });
            
            return base;
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
        console.log('Planting seed at:', x, y);
        console.log('Current state shape:', [h, w, ch]);
        
        // Use the exact same seed values as initialization
        const seed = new Array(16).fill(0).map((x, i) => i < 3 ? 0.0 : 1.0);
        
        tf.tidy(() => {
            // Create a single seed at the specified position
            const newState = this.state.clone();  // Clone current state
            for (let i = 0; i < 16; i++) {
                newState.bufferSync().set(seed[i], 0, y, x, i);
            }
            this.state.assign(newState);
        });
    }

    step() {
        tf.tidy(() => {
            const [_, height, width, channels] = this.state.shape;
            
            // Process left tile - no extra padding
            const leftTile = this.state.slice(
                [0, 0, 0, 0],
                [1, height, this.tileSize + this.padding, channels]
            ).pad([[0, 0], [1, 1], [1, 1], [0, 0]], 0.0);
            
            const processedLeft = this.model.execute(
                {
                    x: leftTile,
                    fire_rate: tf.scalar(0.5),
                    angle: tf.scalar(0.0),
                    step_size: tf.scalar(1.0)
                },
                ['Identity']
            );

            // Process right tile - no extra padding
            const rightTile = this.state.slice(
                [0, 0, this.tileSize - this.padding, 0],
                [1, height, this.tileSize + this.padding, channels]
            ).pad([[0, 0], [1, 1], [1, 1], [0, 0]], 0.0);
            
            const processedRight = this.model.execute(
                {
                    x: rightTile,
                    fire_rate: tf.scalar(0.5),
                    angle: tf.scalar(0.0),
                    step_size: tf.scalar(1.0)
                },
                ['Identity']
            );

            // Remove padding before combining
            const leftValid = processedLeft.slice(
                [0, 1, 1, 0],
                [1, height, this.tileSize, channels]
            );
            const rightValid = processedRight.slice(
                [0, 1, this.padding + 1, 0],
                [1, height, this.tileSize, channels]
            );

            const combined = tf.concat([leftValid, rightValid], 2);
            this.state.assign(combined);
        });
    }
} 