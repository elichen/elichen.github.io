/**
 * Snake RL Agent - Egocentric FC Network
 * 5-channel egocentric input (rotated so snake faces up), 3 relative actions
 * Architecture: FC with LayerNorm (2x scale = 4.4M params)
 */

class SnakeEgocentricAgent {
    constructor(gridSize = 20) {
        this.gridSize = gridSize;
        this.obsSize = gridSize + 2;  // With wall padding
        this.nChannels = 5;
        this.nActions = 3;
        this.weights = null;
        this.currentDirection = 0; // 0=up, 1=right, 2=down, 3=left
    }

    async load(weightsUrl = 'web_model_egocentric/weights.json') {
        console.log('Loading egocentric agent weights...');
        const response = await fetch(weightsUrl);
        const data = await response.json();
        this.weights = data.weights;
        this.metadata = data.metadata;
        console.log('Snake egocentric agent loaded');
        console.log('  Board size:', this.metadata.board_size);
        console.log('  Network scale:', this.metadata.network_scale + 'x');
        console.log('  Observation:', this.metadata.n_channels + 'x' + this.metadata.obs_size + 'x' + this.metadata.obs_size);
    }

    // LayerNorm: normalize, then scale and shift
    layerNorm(x, weight, bias) {
        const n = x.length;
        let mean = 0;
        for (let i = 0; i < n; i++) mean += x[i];
        mean /= n;

        let variance = 0;
        for (let i = 0; i < n; i++) variance += (x[i] - mean) * (x[i] - mean);
        variance /= n;

        const std = Math.sqrt(variance + 1e-5);
        const result = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            result[i] = ((x[i] - mean) / std) * weight[i] + bias[i];
        }
        return result;
    }

    // Matrix multiply: weight is [out, in], x is [in]
    linear(x, weight, bias) {
        const outDim = weight.length;
        const inDim = weight[0].length;
        const result = new Float32Array(outDim);
        for (let i = 0; i < outDim; i++) {
            let sum = bias[i];
            for (let j = 0; j < inDim; j++) {
                sum += weight[i][j] * x[j];
            }
            result[i] = sum;
        }
        return result;
    }

    relu(x) {
        const result = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            result[i] = Math.max(0, x[i]);
        }
        return result;
    }

    // Rotate grid so snake faces "up" (direction 0)
    rotateGrid(grid, direction) {
        // grid is [channels, height, width]
        // direction: 0=up (no rotation), 1=right (rot90 once), 2=down (rot90 twice), 3=left (rot90 thrice)
        if (direction === 0) return grid;

        const c = grid.length;
        const h = grid[0].length;
        const w = grid[0][0].length;

        // Rotate by direction * 90 degrees counterclockwise
        const rotated = [];
        for (let ch = 0; ch < c; ch++) {
            const newChannel = [];
            for (let r = 0; r < h; r++) {
                newChannel.push(new Float32Array(w));
            }
            rotated.push(newChannel);
        }

        for (let ch = 0; ch < c; ch++) {
            for (let r = 0; r < h; r++) {
                for (let col = 0; col < w; col++) {
                    let newR, newC;
                    if (direction === 1) {
                        // 90 degrees CCW: (r, c) -> (w-1-c, r)
                        newR = w - 1 - col;
                        newC = r;
                    } else if (direction === 2) {
                        // 180 degrees: (r, c) -> (h-1-r, w-1-c)
                        newR = h - 1 - r;
                        newC = w - 1 - col;
                    } else { // direction === 3
                        // 270 degrees CCW (90 CW): (r, c) -> (c, h-1-r)
                        newR = col;
                        newC = h - 1 - r;
                    }
                    rotated[ch][newR][newC] = grid[ch][r][col];
                }
            }
        }

        return rotated;
    }

    buildObservation(game) {
        // Build 5-channel egocentric observation with wall padding
        // Channels: head, body, food, normalized_length, walls
        const n = this.gridSize;
        const obsN = this.obsSize;  // n + 2 for wall padding

        // Initialize 3D grid [channels][row][col]
        const grid = [];
        for (let ch = 0; ch < this.nChannels; ch++) {
            const channel = [];
            for (let r = 0; r < obsN; r++) {
                channel.push(new Float32Array(obsN));
            }
            grid.push(channel);
        }

        // Channel 4: Walls (border cells)
        for (let r = 0; r < obsN; r++) {
            grid[4][r][0] = 1.0;         // Left wall
            grid[4][r][obsN - 1] = 1.0;  // Right wall
        }
        for (let c = 0; c < obsN; c++) {
            grid[4][0][c] = 1.0;         // Top wall
            grid[4][obsN - 1][c] = 1.0;  // Bottom wall
        }

        // Get current direction (0=up, 1=right, 2=down, 3=left)
        let dir = this.currentDirection;

        // Channel 0: Head (offset by 1 for wall padding)
        const head = game.snake[0];
        grid[0][head.y + 1][head.x + 1] = 1.0;

        // Channel 1: Body (all segments)
        for (const segment of game.snake) {
            grid[1][segment.y + 1][segment.x + 1] = 1.0;
        }

        // Channel 2: Food
        if (game.food) {
            grid[2][game.food.y + 1][game.food.x + 1] = 1.0;
        }

        // Channel 3: Normalized length (broadcast)
        const normalizedLength = game.snake.length / (n * n);
        for (let r = 0; r < obsN; r++) {
            for (let c = 0; c < obsN; c++) {
                grid[3][r][c] = normalizedLength;
            }
        }

        // Rotate grid so snake always faces "up"
        const rotatedGrid = this.rotateGrid(grid, dir);

        // Flatten to 1D array [C, H, W] order
        const flat = new Float32Array(this.nChannels * obsN * obsN);
        let idx = 0;
        for (let ch = 0; ch < this.nChannels; ch++) {
            for (let r = 0; r < obsN; r++) {
                for (let c = 0; c < obsN; c++) {
                    flat[idx++] = rotatedGrid[ch][r][c];
                }
            }
        }

        return flat;
    }

    forward(obs) {
        const w = this.weights;

        // features.1: Linear + LayerNorm + ReLU
        let x = this.linear(obs, w['features.1.weight'], w['features.1.bias']);
        x = this.layerNorm(x, w['features.2.weight'], w['features.2.bias']);
        x = this.relu(x);

        // features.4: Linear + LayerNorm + ReLU
        x = this.linear(x, w['features.4.weight'], w['features.4.bias']);
        x = this.layerNorm(x, w['features.5.weight'], w['features.5.bias']);
        x = this.relu(x);

        // features.7: Linear + LayerNorm + ReLU
        x = this.linear(x, w['features.7.weight'], w['features.7.bias']);
        x = this.layerNorm(x, w['features.8.weight'], w['features.8.bias']);
        x = this.relu(x);

        // features.10: Linear + ReLU (no LayerNorm)
        x = this.linear(x, w['features.10.weight'], w['features.10.bias']);
        x = this.relu(x);

        // policy_head.0: Linear + ReLU
        x = this.linear(x, w['policy_head.0.weight'], w['policy_head.0.bias']);
        x = this.relu(x);

        // policy_head.2: Linear (output logits)
        const logits = this.linear(x, w['policy_head.2.weight'], w['policy_head.2.bias']);

        return logits;
    }

    predictAction(game) {
        // Update current direction from game state
        const d = game.direction;
        if (d.x === 0 && d.y === -1) this.currentDirection = 0; // up
        else if (d.x === 1 && d.y === 0) this.currentDirection = 1; // right
        else if (d.x === 0 && d.y === 1) this.currentDirection = 2; // down
        else if (d.x === -1 && d.y === 0) this.currentDirection = 3; // left

        const obs = this.buildObservation(game);
        const logits = this.forward(obs);

        // Get argmax action (0=left, 1=straight, 2=right)
        let maxIdx = 0;
        let maxVal = logits[0];
        for (let i = 1; i < this.nActions; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }

        // Convert relative action to absolute direction
        // 0=turn left, 1=straight, 2=turn right
        const delta = [-1, 0, 1];
        const newDir = (this.currentDirection + delta[maxIdx] + 4) % 4;

        return newDir; // Return absolute direction (0=up, 1=right, 2=down, 3=left)
    }

    reset(game) {
        // Sync with game's actual direction
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
const agent = new SnakeEgocentricAgent(20);
