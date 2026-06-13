/**
 * Snake RL Agent - 5-channel egocentric FC network (head-centered).
 * The deployed model (cont_ctrl_s202r): PPO, 4.4M params, scale 2.
 * Observation: 5 x 39 x 39 centered on the snake head, rotated so it faces "up".
 * Channels: head, body, food, normalized length, walls. (Flood-fill was a
 * training-only auxiliary target; the network never sees it at inference.)
 * Action: single forward pass -> argmax over {left, straight, right}. No search.
 *
 * Works in the browser (load() via fetch) and in node (loadFromBuffer()).
 */

class SnakeAgentV2 {
    constructor(gridSize = 20) {
        this.gridSize = gridSize;
        this.nChannels = 5;
        this.nActions = 3;
        this.weights = null;
        this.currentDirection = 0; // 0=up, 1=right, 2=down, 3=left
        this.obsSize = null;
    }

    async load(weightsUrl = 'web_model_v2/weights.bin') {
        const response = await fetch(weightsUrl);
        const buffer = await response.arrayBuffer();
        this.loadFromBuffer(buffer);
    }

    loadFromBuffer(buffer) {
        const view = new DataView(buffer);
        let offset = 0;

        const metaLen = view.getUint32(offset, true); offset += 4;
        const metaJson = new TextDecoder().decode(new Uint8Array(buffer, offset, metaLen));
        this.metadata = JSON.parse(metaJson);
        offset += metaLen;
        this.obsSize = this.metadata.obs_size;

        const numWeights = view.getUint16(offset, true); offset += 2;
        this.weights = {};
        for (let w = 0; w < numWeights; w++) {
            const nameLen = view.getUint16(offset, true); offset += 2;
            const name = new TextDecoder().decode(new Uint8Array(buffer, offset, nameLen));
            offset += nameLen;

            const ndims = view.getUint8(offset); offset += 1;
            const shape = [];
            let totalSize = 1;
            for (let d = 0; d < ndims; d++) {
                shape.push(view.getUint32(offset, true)); offset += 4;
                totalSize *= shape[shape.length - 1];
            }

            const byteLen = totalSize * 4;
            const aligned = new Float32Array(totalSize);
            new Uint8Array(aligned.buffer).set(new Uint8Array(buffer, offset, byteLen));
            offset += byteLen;

            if (shape.length === 2) {
                const rows = [];
                for (let i = 0; i < shape[0]; i++) {
                    rows.push(aligned.subarray(i * shape[1], (i + 1) * shape[1]));
                }
                this.weights[name] = rows;
            } else {
                this.weights[name] = aligned;
            }
        }

        const expectedInput = this.nChannels * this.obsSize * this.obsSize;
        const actualInput = this.weights['features.1.weight'][0].length;
        if (actualInput !== expectedInput) {
            throw new Error(`Weight/obs mismatch: weights expect ${actualInput} inputs but obs is ${expectedInput}.`);
        }
    }

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
        for (let i = 0; i < n; i++) result[i] = ((x[i] - mean) / std) * weight[i] + bias[i];
        return result;
    }

    linear(x, weight, bias) {
        const outDim = weight.length;
        const inDim = weight[0].length;
        const result = new Float32Array(outDim);
        for (let i = 0; i < outDim; i++) {
            let sum = bias[i];
            const row = weight[i];
            for (let j = 0; j < inDim; j++) sum += row[j] * x[j];
            result[i] = sum;
        }
        return result;
    }

    relu(x) {
        const result = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) result[i] = Math.max(0, x[i]);
        return result;
    }

    rotateGrid(grid, direction) {
        if (direction === 0) return grid;
        const c = grid.length, h = grid[0].length, w = grid[0][0].length;
        const rotated = [];
        for (let ch = 0; ch < c; ch++) {
            const nc = [];
            for (let r = 0; r < h; r++) nc.push(new Float32Array(w));
            rotated.push(nc);
        }
        for (let ch = 0; ch < c; ch++) {
            for (let r = 0; r < h; r++) {
                for (let col = 0; col < w; col++) {
                    let newR, newC;
                    if (direction === 1) { newR = w - 1 - col; newC = r; }
                    else if (direction === 2) { newR = h - 1 - r; newC = w - 1 - col; }
                    else { newR = col; newC = h - 1 - r; }
                    rotated[ch][newR][newC] = grid[ch][r][col];
                }
            }
        }
        return rotated;
    }

    buildObservation(game) {
        const n = this.gridSize;
        const obsN = this.obsSize;
        const center = Math.floor(obsN / 2);

        const grid = [];
        for (let ch = 0; ch < this.nChannels; ch++) {
            const channel = [];
            for (let r = 0; r < obsN; r++) channel.push(new Float32Array(obsN));
            grid.push(channel);
        }

        const dir = this.currentDirection;
        const head = game.snake[0];
        const headRow = head.y, headCol = head.x;

        grid[0][center][center] = 1.0; // head at center

        for (const seg of game.snake) {
            const r = seg.y - headRow + center, c = seg.x - headCol + center;
            if (r >= 0 && r < obsN && c >= 0 && c < obsN) grid[1][r][c] = 1.0;
        }

        if (game.food) {
            const r = game.food.y - headRow + center, c = game.food.x - headCol + center;
            if (r >= 0 && r < obsN && c >= 0 && c < obsN) grid[2][r][c] = 1.0;
        }

        const normLen = game.snake.length / (n * n);
        for (let r = 0; r < obsN; r++)
            for (let c = 0; c < obsN; c++) grid[3][r][c] = normLen;

        for (let r = 0; r < obsN; r++) {
            const boardRow = r + headRow - center;
            for (let c = 0; c < obsN; c++) {
                const boardCol = c + headCol - center;
                if (boardRow < 0 || boardRow >= n || boardCol < 0 || boardCol >= n)
                    grid[4][r][c] = 1.0;
            }
        }

        const rg = this.rotateGrid(grid, dir);
        const flat = new Float32Array(this.nChannels * obsN * obsN);
        let idx = 0;
        for (let ch = 0; ch < this.nChannels; ch++)
            for (let r = 0; r < obsN; r++)
                for (let c = 0; c < obsN; c++) flat[idx++] = rg[ch][r][c];
        return flat;
    }

    forward(obs) {
        const w = this.weights;
        let x = this.linear(obs, w['features.1.weight'], w['features.1.bias']);
        x = this.relu(this.layerNorm(x, w['features.2.weight'], w['features.2.bias']));
        x = this.linear(x, w['features.4.weight'], w['features.4.bias']);
        x = this.relu(this.layerNorm(x, w['features.5.weight'], w['features.5.bias']));
        x = this.linear(x, w['features.7.weight'], w['features.7.bias']);
        x = this.relu(this.layerNorm(x, w['features.8.weight'], w['features.8.bias']));
        x = this.relu(this.linear(x, w['features.10.weight'], w['features.10.bias']));
        x = this.relu(this.linear(x, w['policy_head.0.weight'], w['policy_head.0.bias']));
        return this.linear(x, w['policy_head.2.weight'], w['policy_head.2.bias']);
    }

    setDirectionFromGame(game) {
        const d = game.direction;
        if (d.x === 0 && d.y === -1) this.currentDirection = 0;
        else if (d.x === 1 && d.y === 0) this.currentDirection = 1;
        else if (d.x === 0 && d.y === 1) this.currentDirection = 2;
        else if (d.x === -1 && d.y === 0) this.currentDirection = 3;
    }

    predictAction(game) {
        this.setDirectionFromGame(game);
        const logits = this.forward(this.buildObservation(game));
        let maxIdx = 0, maxVal = logits[0];
        for (let i = 1; i < this.nActions; i++) {
            if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
        }
        const delta = [-1, 0, 1];
        return (this.currentDirection + delta[maxIdx] + 4) % 4;
    }

    reset(game) {
        if (game) this.setDirectionFromGame(game);
        else this.currentDirection = 0;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = SnakeAgentV2;
}
const agent = (typeof window !== 'undefined') ? new SnakeAgentV2(20) : null;
