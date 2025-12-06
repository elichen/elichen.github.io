"""
Direct export to TensorFlow.js using tensorflowjs_converter alternative.
Saves model in a format that can be loaded with tf.loadLayersModel.
"""

import json
import os
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO


def export_model_weights(model_path: str, output_dir: str, board_size: int = 20):
    """Export model weights in a compact format for JS reconstruction."""

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Extract weights from the policy
    state_dict = model.policy.state_dict()

    # Only keep policy network weights (not value network duplicates)
    policy_keys = [
        'pi_features_extractor.cnn.0.weight', 'pi_features_extractor.cnn.0.bias',
        'pi_features_extractor.cnn.2.weight', 'pi_features_extractor.cnn.2.bias',
        'pi_features_extractor.cnn.4.weight', 'pi_features_extractor.cnn.4.bias',
        'pi_features_extractor.linear.0.weight', 'pi_features_extractor.linear.0.bias',
        'mlp_extractor.policy_net.0.weight', 'mlp_extractor.policy_net.0.bias',
        'mlp_extractor.policy_net.2.weight', 'mlp_extractor.policy_net.2.bias',
        'action_net.weight', 'action_net.bias',
    ]

    # Convert to numpy and organize
    weights = {}
    for name in policy_keys:
        if name in state_dict:
            arr = state_dict[name].detach().cpu().numpy()
            weights[name] = arr.tolist()
            print(f"  {name}: {arr.shape}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = os.path.join(output_dir, "weights.json")
    with open(weights_path, "w") as f:
        json.dump(weights, f)

    print(f"\nWeights saved to: {weights_path}")
    print(f"File size: {os.path.getsize(weights_path) / 1024 / 1024:.2f} MB")

    return weights


def create_tfjs_agent(output_dir: str, board_size: int = 20):
    """Create the JavaScript agent file that loads weights and runs inference."""

    js_code = '''/**
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

    async buildModel() {
        // Build the CNN architecture matching Python training
        // Architecture: Conv(8->64) -> Conv(64->128) -> Conv(128->256) -> GlobalAvgPool -> Linear(256->256) -> Linear(256->3)

        this.model = tf.sequential();

        // Conv layers
        this.model.add(tf.layers.conv2d({
            inputShape: [this.nChannels, this.gridSize, this.gridSize],
            filters: 64,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu',
            dataFormat: 'channelsFirst'
        }));

        this.model.add(tf.layers.conv2d({
            filters: 128,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu',
            dataFormat: 'channelsFirst'
        }));

        this.model.add(tf.layers.conv2d({
            filters: 256,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu',
            dataFormat: 'channelsFirst'
        }));

        // Global average pooling
        this.model.add(tf.layers.globalAveragePooling2d({dataFormat: 'channelsFirst'}));

        // Feature extractor linear
        this.model.add(tf.layers.dense({units: 256, activation: 'relu'}));

        // Policy MLP
        this.model.add(tf.layers.dense({units: 256, activation: 'relu'}));
        this.model.add(tf.layers.dense({units: 256, activation: 'relu'}));

        // Action output
        this.model.add(tf.layers.dense({units: this.nActions}));

        // Load weights
        await this.loadWeights();
    }

    async loadWeights() {
        const w = this.weights;
        const layers = this.model.layers;

        // Map PyTorch weight names to layer indices
        // features_extractor.cnn.0 = conv1 (layer 0)
        // features_extractor.cnn.3 = conv2 (layer 1)
        // features_extractor.cnn.6 = conv3 (layer 2)
        // features_extractor.linear.0 = dense after pool (layer 4)
        // mlp_extractor.policy_net.0 = policy dense 1 (layer 5)
        // mlp_extractor.policy_net.2 = policy dense 2 (layer 6)
        // action_net = output (layer 7)

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
            }
        }
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
            const obsTensor = tf.tensor(obs, [1, this.nChannels, this.gridSize, this.gridSize]);

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

    reset() {
        this.currentDirection = 0; // Reset to up
    }
}

// For compatibility with existing code
const agent = new SnakeCurriculumAgent(20);
'''

    agent_path = os.path.join(output_dir, "agent-curriculum.js")
    with open(agent_path, "w") as f:
        f.write(js_code)

    print(f"Agent JS saved to: {agent_path}")


def create_script_file(output_dir: str):
    """Create the main script file."""

    js_code = '''// Main script for curriculum-trained Snake AI

let game;
let totalFoodEaten = 0;
let episodeCount = 1;
let gameLoopId = null;
const GAME_SPEED = 50; // ms per frame

async function init() {
    document.getElementById('modelStatus').textContent = 'Loading model...';

    game = new SnakeGame('gameCanvas', 20);
    game.maxMovesWithoutFood = 200;
    game.draw();

    try {
        await agent.load('web_model_new/weights.json');
        agent.reset();
        document.getElementById('modelStatus').textContent = 'Model loaded! Starting game...';
        setTimeout(startGame, 500);
    } catch (error) {
        console.error('Failed to load model:', error);
        document.getElementById('modelStatus').textContent = 'Failed to load model: ' + error.message;
    }
}

function startGame() {
    document.getElementById('modelStatus').textContent = '';
    gameLoop();
}

function gameLoop() {
    if (game.gameOver) {
        // Reset and start new episode
        episodeCount++;
        document.getElementById('episode').textContent = episodeCount;
        game.reset();
        agent.reset();
    }

    // Get action from agent
    const action = agent.predictAction(game);

    // Take step
    const result = game.step(action);

    // Update stats
    document.getElementById('score').textContent = game.score;
    if (result.reward > 0) {
        totalFoodEaten++;
        document.getElementById('foodEaten').textContent = totalFoodEaten;
    }

    // Schedule next frame
    gameLoopId = setTimeout(gameLoop, GAME_SPEED);
}

// Start when page loads
window.addEventListener('load', init);
'''

    script_path = os.path.join(output_dir, "script-curriculum.js")
    with open(script_path, "w") as f:
        f.write(js_code)

    print(f"Script JS saved to: {script_path}")


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/snake_curriculum_20251205_121938/final_model.zip"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "web_model_new"

    export_model_weights(model_path, output_dir)
    create_tfjs_agent(output_dir)
    create_script_file(output_dir)

    print(f"\nExport complete! Files in {output_dir}/")
    print("To test, update index.html to use:")
    print('  <script src="web_model_new/agent-curriculum.js"></script>')
    print('  <script src="web_model_new/script-curriculum.js"></script>')
