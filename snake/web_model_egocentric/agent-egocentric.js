/**
 * Snake RL Agent - Egocentric FC Network (Head-Centered)
 * 5-channel head-centered input (rotated so snake faces up), 3 relative actions
 * Architecture: FC with LayerNorm (2x scale = 4.4M params)
 * Observation: 5 x 39 x 39 centered on snake head
 */

class SnakeEgocentricAgent {
    constructor(gridSize = 20) {
        this.gridSize = gridSize;
        this.nChannels = 5;
        this.nActions = 3;
        this.weights = null;
        this.currentDirection = 0; // 0=up, 1=right, 2=down, 3=left
        // Set after loading metadata
        this.obsSize = null;
    }

    async load(weightsUrl = 'web_model_egocentric/weights.json') {
        console.log('Loading egocentric agent weights...');
        const response = await fetch(weightsUrl);
        const data = await response.json();
        this.metadata = data.metadata;
        this.obsSize = this.metadata.obs_size;

        // Reconstruct weights from compact format (shape + flat data)
        this.weights = {};
        for (const [name, w] of Object.entries(data.weights)) {
            this.weights[name] = this.unflatten(w.data, w.shape);
        }

        // Validate weight dimensions match observation size
        const expectedInput = this.nChannels * this.obsSize * this.obsSize;
        const actualInput = this.weights['features.1.weight'][0].length;
        if (actualInput !== expectedInput) {
            throw new Error(`Weight/obs mismatch: weights expect ${actualInput} inputs but obs is ${expectedInput} (${this.nChannels}x${this.obsSize}x${this.obsSize}). Clear cache and reload.`);
        }

        console.log('Snake egocentric agent loaded');
        console.log('  Board size:', this.metadata.board_size);
        console.log('  Network scale:', this.metadata.network_scale + 'x');
        console.log('  Head-centered:', this.metadata.head_centered);
        console.log('  Observation:', this.nChannels + 'x' + this.obsSize + 'x' + this.obsSize, '=', expectedInput, 'features');
    }

    // Reconstruct nested array from flat data and shape
    unflatten(data, shape) {
        if (shape.length === 1) {
            return data.slice(0, shape[0]);
        }
        const result = [];
        const subSize = shape.slice(1).reduce((a, b) => a * b, 1);
        for (let i = 0; i < shape[0]; i++) {
            result.push(this.unflatten(data.slice(i * subSize, (i + 1) * subSize), shape.slice(1)));
        }
        return result;
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
        if (direction === 0) return grid;

        const c = grid.length;
        const h = grid[0].length;
        const w = grid[0][0].length;

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
                        newR = w - 1 - col;
                        newC = r;
                    } else if (direction === 2) {
                        newR = h - 1 - r;
                        newC = w - 1 - col;
                    } else {
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
        // Head-centered 5-channel observation (39x39 for 20x20 board)
        // Head is always at the center of the grid
        // Channels: head, body, food, normalized_length, walls
        const n = this.gridSize;
        const obsN = this.obsSize;  // 39 for n=20
        const center = Math.floor(obsN / 2);  // 19

        // Initialize 3D grid [channels][row][col]
        const grid = [];
        for (let ch = 0; ch < this.nChannels; ch++) {
            const channel = [];
            for (let r = 0; r < obsN; r++) {
                channel.push(new Float32Array(obsN));
            }
            grid.push(channel);
        }

        const dir = this.currentDirection;

        // Head position in board coordinates (game uses x,y where y=row, x=col)
        const head = game.snake[0];
        const headRow = head.y;
        const headCol = head.x;

        // Channel 0: Head (always at center)
        grid[0][center][center] = 1.0;

        // Channel 1: Body (all segments, offset relative to head)
        for (const segment of game.snake) {
            const obsR = segment.y - headRow + center;
            const obsC = segment.x - headCol + center;
            if (obsR >= 0 && obsR < obsN && obsC >= 0 && obsC < obsN) {
                grid[1][obsR][obsC] = 1.0;
            }
        }

        // Channel 2: Food (offset relative to head)
        if (game.food) {
            const foodR = game.food.y - headRow + center;
            const foodC = game.food.x - headCol + center;
            if (foodR >= 0 && foodR < obsN && foodC >= 0 && foodC < obsN) {
                grid[2][foodR][foodC] = 1.0;
            }
        }

        // Channel 3: Normalized length (broadcast)
        const normalizedLength = game.snake.length / (n * n);
        for (let r = 0; r < obsN; r++) {
            for (let c = 0; c < obsN; c++) {
                grid[3][r][c] = normalizedLength;
            }
        }

        // Channel 4: Walls (everything outside the board)
        for (let r = 0; r < obsN; r++) {
            const boardRow = r + headRow - center;
            for (let c = 0; c < obsN; c++) {
                const boardCol = c + headCol - center;
                if (boardRow < 0 || boardRow >= n || boardCol < 0 || boardCol >= n) {
                    grid[4][r][c] = 1.0;
                }
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
        const delta = [-1, 0, 1];
        const newDir = (this.currentDirection + delta[maxIdx] + 4) % 4;

        return newDir;
    }

    reset(game) {
        if (game) {
            const d = game.direction;
            if (d.x === 0 && d.y === -1) this.currentDirection = 0;
            else if (d.x === 1 && d.y === 0) this.currentDirection = 1;
            else if (d.x === 0 && d.y === 1) this.currentDirection = 2;
            else if (d.x === -1 && d.y === 0) this.currentDirection = 3;
        } else {
            this.currentDirection = 0;
        }
    }
}

// For compatibility with existing code
const agent = new SnakeEgocentricAgent(20);
