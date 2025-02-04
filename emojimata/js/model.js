class CAModel {
    constructor() {
        this.tileSize = 96;  // Original model size
        this.padding = 16;   // Padding for tile boundaries
        this.scale = 4;
        this.state = null;
        this.model = null;
        this.numTilesX = 0;  // Will be calculated based on window width
        this.numTilesY = 2;  // Keep 2 rows
    }

    calculateTiles() {
        // Calculate how many tiles we need to cover the window width
        const windowWidth = window.innerWidth;
        this.numTilesX = Math.ceil(windowWidth / (this.tileSize * this.scale));
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
        this.calculateTiles();
        // Start with a completely blank state (all 0.0s)
        const initState = tf.tidy(() => {
            const width = this.tileSize * this.numTilesX;
            const height = this.tileSize * this.numTilesY;
            const base = tf.zeros([1, height, width, 16]);
            
            console.log('Initial state:', {
                shape: base.shape,
                tilesX: this.numTilesX,
                tilesY: this.numTilesY,
                width,
                height,
                channels: 16
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
        
        // Use the exact same seed values as initialization
        const seed = new Array(16).fill(0).map((x, i) => i < 3 ? 0.0 : 1.0);
        
        tf.tidy(() => {
            const newState = this.state.clone();
            for (let i = 0; i < 16; i++) {
                newState.bufferSync().set(seed[i], 0, y, x, i);
            }
            this.state.assign(newState);
        });
    }

    step() {
        tf.tidy(() => {
            const [_, height, width, channels] = this.state.shape;
            const processedTiles = [];
            
            // Process each tile
            for (let y = 0; y < this.numTilesY; y++) {
                const rowTiles = [];
                for (let x = 0; x < this.numTilesX; x++) {
                    // Calculate start positions with padding
                    const startY = Math.max(0, y * this.tileSize - this.padding);
                    const startX = Math.max(0, x * this.tileSize - this.padding);
                    
                    // Calculate slice dimensions including padding in both directions
                    const sliceHeight = Math.min(this.tileSize + 2 * this.padding, height - startY);
                    const sliceWidth = Math.min(this.tileSize + 2 * this.padding, width - startX);
                    
                    // Extract tile with padding
                    const tile = this.state.slice(
                        [0, startY, startX, 0],
                        [1, sliceHeight, sliceWidth, channels]
                    ).pad([[0, 0], [1, 1], [1, 1], [0, 0]], 0.0);
                    
                    // Process the tile
                    const processed = this.model.execute(
                        {
                            x: tile,
                            fire_rate: tf.scalar(0.5),
                            angle: tf.scalar(0.0),
                            step_size: tf.scalar(1.0)
                        },
                        ['Identity']
                    );
                    
                    // Calculate the offset for extracting the valid region
                    const validStartY = startY === 0 ? 1 : this.padding + 1;
                    const validStartX = startX === 0 ? 1 : this.padding + 1;
                    
                    // Extract the valid region (remove padding)
                    const valid = processed.slice(
                        [0, validStartY, validStartX, 0],
                        [1, this.tileSize, this.tileSize, channels]
                    );
                    
                    rowTiles.push(valid);
                }
                // Combine tiles in this row
                const row = tf.concat(rowTiles, 2);
                processedTiles.push(row);
            }
            
            // Combine all rows
            const combined = tf.concat(processedTiles, 1);
            this.state.assign(combined);
        });
    }
} 