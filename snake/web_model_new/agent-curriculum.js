/**
 * Snake RL Agent - Curriculum-trained PPO
 * 8-channel input, 3 relative actions (left/straight/right)
 */

class SnakeCurriculumAgent {
    constructor(gridSize = 20) {
        this.gridSize = gridSize;
        this.nChannels = 8;
        this.nActions = 3;
        this.model = null;
        this.currentDirection = 0; // 0=up, 1=right, 2=down, 3=left
    }

    async load(weightsUrl = 'web_model_new/weights.json') {
        const response = await fetch(weightsUrl);
        this.weights = await response.json();
        await this.buildModel();
        console.log('Snake curriculum agent loaded');
    }

    testModel() {
        // Test full model with simple input (only channel 3 = 1 for direction up)
        const n = this.gridSize;
        const testObs = new Float32Array(8 * n * n);
        for (let i = 0; i < n * n; i++) {
            testObs[3 * n * n + i] = 1.0;
        }

        const obsNCHW = tf.tensor(testObs, [1, 8, n, n]);
        const obsNHWC = obsNCHW.transpose([0, 2, 3, 1]);

        const output = this.model.predict(obsNHWC);
        const outputData = output.dataSync();
        console.log('Model test output:', Array.from(outputData).map(v => v.toFixed(4)));
        console.log('Python reference:  [-0.1345, -0.8707, -0.5112]');

        // Check if close enough
        const pyRef = [-0.1345, -0.8707, -0.5112];
        const maxDiff = Math.max(...outputData.map((v, i) => Math.abs(v - pyRef[i])));
        console.log('Max difference from Python:', maxDiff.toFixed(6));
        if (maxDiff < 0.01) {
            console.log('✓ Model output matches Python!');
        } else {
            console.warn('✗ Model output differs from Python');
        }

        obsNCHW.dispose();
        obsNHWC.dispose();
        output.dispose();
    }

    async buildModel() {
        // Build the CNN architecture matching Python training
        // Using channelsLast (NHWC) format which is TF.js native
        // Input will be transposed from NCHW to NHWC before feeding

        this.model = tf.sequential();

        // Conv layers - using channelsLast (default)
        this.model.add(tf.layers.conv2d({
            inputShape: [this.gridSize, this.gridSize, this.nChannels],
            filters: 64,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu'
        }));

        this.model.add(tf.layers.conv2d({
            filters: 128,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu'
        }));

        this.model.add(tf.layers.conv2d({
            filters: 256,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu'
        }));

        // Global average pooling
        this.model.add(tf.layers.globalAveragePooling2d({dataFormat: 'channelsLast'}));

        // Feature extractor linear
        this.model.add(tf.layers.dense({units: 256, activation: 'relu'}));

        // Policy MLP - uses tanh activation (not relu!)
        this.model.add(tf.layers.dense({units: 256, activation: 'tanh'}));
        this.model.add(tf.layers.dense({units: 256, activation: 'tanh'}));

        // Action output
        this.model.add(tf.layers.dense({units: this.nActions}));

        // Load weights
        await this.loadWeights();
    }

    async loadWeights() {
        const w = this.weights;
        const layers = this.model.layers;

        // Map PyTorch weight names to layer indices
        // Layer 0: Conv2D, Layer 1: Conv2D, Layer 2: Conv2D
        // Layer 3: GlobalAveragePooling2D (no weights)
        // Layer 4: Dense (relu), Layer 5: Dense (tanh), Layer 6: Dense (tanh), Layer 7: Dense (none)
        const weightMap = [
            ['pi_features_extractor.cnn.0.weight', 'pi_features_extractor.cnn.0.bias', 0],
            ['pi_features_extractor.cnn.2.weight', 'pi_features_extractor.cnn.2.bias', 1],
            ['pi_features_extractor.cnn.4.weight', 'pi_features_extractor.cnn.4.bias', 2],
            ['pi_features_extractor.linear.0.weight', 'pi_features_extractor.linear.0.bias', 4],
            ['mlp_extractor.policy_net.0.weight', 'mlp_extractor.policy_net.0.bias', 5],
            ['mlp_extractor.policy_net.2.weight', 'mlp_extractor.policy_net.2.bias', 6],
            ['action_net.weight', 'action_net.bias', 7],
        ];

        for (const [wName, bName, layerIdx] of weightMap) {
            if (w[wName] && w[bName]) {
                const layer = layers[layerIdx];
                let kernel = tf.tensor(w[wName]);
                const bias = tf.tensor(w[bName]);

                // Conv layers need transpose: PyTorch [out, in, h, w] -> TF [h, w, in, out]
                if (layerIdx <= 2) {
                    kernel = kernel.transpose([2, 3, 1, 0]);
                } else {
                    // Dense layers: PyTorch [out, in] -> TF [in, out]
                    kernel = kernel.transpose();
                }

                layer.setWeights([kernel, bias]);
            } else {
                console.error(`Missing weights: ${wName} or ${bName}`);
            }
        }

        // Test model output matches Python
        this.testModel();
    }

    buildObservation(game) {
        // Build 8-channel observation: head, body, food, dir_up, dir_right, dir_down, dir_left, length
        const n = this.gridSize;
        const obs = new Float32Array(this.nChannels * n * n);

        // Get direction as 0-3 index
        let dir = this.currentDirection;

        // Channel 0: Head
        const head = game.snake[0];
        obs[0 * n * n + head.y * n + head.x] = 1.0;

        // Channel 1: Body (all segments including head)
        for (const segment of game.snake) {
            obs[1 * n * n + segment.y * n + segment.x] = 1.0;
        }

        // Channel 2: Food
        if (game.food) {
            obs[2 * n * n + game.food.y * n + game.food.x] = 1.0;
        }

        // Channels 3-6: Direction one-hot (broadcast across grid)
        const dirChannel = 3 + dir;
        for (let i = 0; i < n * n; i++) {
            obs[dirChannel * n * n + i] = 1.0;
        }

        // Channel 7: Normalized length (broadcast)
        const normalizedLength = game.snake.length / (n * n);
        for (let i = 0; i < n * n; i++) {
            obs[7 * n * n + i] = normalizedLength;
        }

        return obs;
    }

    predictAction(game) {
        // Update current direction from game state
        const d = game.direction;
        if (d.x === 0 && d.y === -1) this.currentDirection = 0; // up
        else if (d.x === 1 && d.y === 0) this.currentDirection = 1; // right
        else if (d.x === 0 && d.y === 1) this.currentDirection = 2; // down
        else if (d.x === -1 && d.y === 0) this.currentDirection = 3; // left

        return tf.tidy(() => {
            const obs = this.buildObservation(game);

            // Create NCHW tensor then transpose to NHWC for TF.js
            const obsNCHW = tf.tensor(obs, [1, this.nChannels, this.gridSize, this.gridSize]);
            const obsTensor = obsNCHW.transpose([0, 2, 3, 1]); // NCHW -> NHWC

            const logits = this.model.predict(obsTensor);
            const logitsArray = logits.dataSync();

            // Get argmax action (0=left, 1=straight, 2=right)
            let maxIdx = 0;
            let maxVal = logitsArray[0];
            for (let i = 1; i < this.nActions; i++) {
                if (logitsArray[i] > maxVal) {
                    maxVal = logitsArray[i];
                    maxIdx = i;
                }
            }

            // Convert relative action to absolute direction
            // 0=turn left, 1=straight, 2=turn right
            const delta = [-1, 0, 1];
            const newDir = (this.currentDirection + delta[maxIdx] + 4) % 4;

            return newDir; // Return absolute direction (0=up, 1=right, 2=down, 3=left)
        });
    }

    reset(game) {
        // Sync with game's actual direction (now random)
        if (game) {
            const d = game.direction;
            if (d.x === 0 && d.y === -1) this.currentDirection = 0; // up
            else if (d.x === 1 && d.y === 0) this.currentDirection = 1; // right
            else if (d.x === 0 && d.y === 1) this.currentDirection = 2; // down
            else if (d.x === -1 && d.y === 0) this.currentDirection = 3; // left
        } else {
            this.currentDirection = 0;
        }
    }
}

// For compatibility with existing code
const agent = new SnakeCurriculumAgent(20);
