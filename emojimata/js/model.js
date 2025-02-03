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
        console.log('Model loaded, weights:', Object.keys(this.model.weights));
        Object.assign(this.model.weights, consts);
        console.log('Constants assigned:', Object.keys(consts));
        
        // Verify model inputs and outputs
        console.log('Model inputs:', this.model.inputs);
        console.log('Model outputs:', this.model.outputs);
        
        this.initState();
    }

    initState() {
        // Initialize state for 2 tiles side by side with only one seed
        const seed = new Array(16).fill(0).map((x, i) => i < 3 ? 0.0 : 1.0);
        console.log('Initial seed values:', JSON.stringify(seed, null, 2));
        
        // Start with a clean state of all 1.0s (non-active cells)
        const initState = tf.tidy(() => {
            // Create a base tensor of all 1.0s
            const base = tf.ones([1, this.D, this.D * 2, 16]);
            
            // Plant a single seed in the center of the left tile
            const centerX = Math.floor(this.D / 2);
            const centerY = Math.floor(this.D / 2);
            
            // Clone and modify the base tensor
            const withSeed = base.clone();
            const buffer = withSeed.bufferSync();
            for (let i = 0; i < 16; i++) {
                buffer.set(seed[i], 0, centerY, centerX, i);
            }
            
            // Verify seed placement
            console.log('State at seed:', 
                JSON.stringify(withSeed.slice([0,centerY,centerX,0], [1,1,1,-1]).arraySync(), null, 2)
            );
            
            return withSeed;
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
            console.log('Current state shape:', [height, width, channels]);

            // Process left tile with proper padding
            const leftTile = this.state.slice(
                [0, 0, 0, 0],
                [1, height, this.tileSize + this.padding, channels]
            );
            
            // Process right tile with proper padding
            const rightTile = this.state.slice(
                [0, 0, this.tileSize - this.padding, 0],
                [1, height, this.tileSize + this.padding, channels]
            );
            
            // Process tiles independently
            const processedLeft = this.model.execute(
                {
                    x: leftTile,
                    fire_rate: tf.scalar(0.5),
                    angle: tf.scalar(0.0),
                    step_size: tf.scalar(1.0)
                },
                ['Identity']
            );

            const processedRight = this.model.execute(
                {
                    x: rightTile,
                    fire_rate: tf.scalar(0.5),
                    angle: tf.scalar(0.0),
                    step_size: tf.scalar(1.0)
                },
                ['Identity']
            );

            // Take only the valid regions (no overlap)
            const leftValid = processedLeft.slice(
                [0, 0, 0, 0],
                [1, height, this.tileSize, channels]
            );
            const rightValid = processedRight.slice(
                [0, 0, this.padding, 0],
                [1, height, this.tileSize, channels]
            );

            // Combine
            const combined = tf.concat([leftValid, rightValid], 2);
            this.state.assign(combined);
        });
    }
} 