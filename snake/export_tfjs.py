"""
Export trained Snake RL model to TensorFlow.js format for web inference.
Converts PyTorch model from SB3 to ONNX, then to TensorFlow.js.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from snake_env import SnakeEnv
from snake_features import SnakeFeatureExtractor


class PolicyNetwork(nn.Module):
    """
    Standalone policy network for export.
    Extracts just the policy (actor) network from PPO.
    """

    def __init__(self, model: PPO):
        super().__init__()

        # Get the feature extractor
        self.features_extractor = model.policy.features_extractor

        # Get the policy network (mlp_extractor + action_net)
        self.mlp_extractor = model.policy.mlp_extractor
        self.action_net = model.policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning action logits.

        Args:
            obs: Observation tensor (B, C, H, W)

        Returns:
            Action logits (B, n_actions)
        """
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        action_logits = self.action_net(latent_pi)
        return action_logits


def export_to_onnx(
    model_path: str,
    output_path: str,
    board_size: int = 20,
    opset_version: int = 11,
) -> str:
    """
    Export SB3 PPO model to ONNX format.

    Args:
        model_path: Path to trained .zip model
        output_path: Output directory for ONNX model
        board_size: Board size the model was trained on
        opset_version: ONNX opset version

    Returns:
        Path to saved ONNX model
    """
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Create standalone policy network
    policy_net = PolicyNetwork(model)
    policy_net.eval()

    # Create dummy input
    n_channels = 8
    dummy_input = torch.zeros(1, n_channels, board_size, board_size)

    # Test forward pass
    with torch.no_grad():
        test_output = policy_net(dummy_input)
        print(f"Test output shape: {test_output.shape}")
        print(f"Test output (logits): {test_output}")

    # Export to ONNX
    Path(output_path).mkdir(parents=True, exist_ok=True)
    onnx_path = os.path.join(output_path, "snake_policy.onnx")

    print(f"Exporting to ONNX: {onnx_path}")

    torch.onnx.export(
        policy_net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action_logits"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
    )

    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path


def export_to_tfjs(
    onnx_path: str,
    output_path: str,
) -> str:
    """
    Convert ONNX model to TensorFlow.js format.

    Args:
        onnx_path: Path to ONNX model
        output_path: Output directory for TF.js model

    Returns:
        Path to TF.js model directory
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError:
        print("Please install required packages:")
        print("  pip install onnx onnx-tf tensorflow tensorflowjs")
        raise

    print(f"Loading ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Convert to TensorFlow
    print("Converting to TensorFlow...")
    tf_rep = prepare(onnx_model)

    tf_model_path = os.path.join(output_path, "tf_model")
    tf_rep.export_graph(tf_model_path)
    print(f"TensorFlow model saved to: {tf_model_path}")

    # Convert to TensorFlow.js
    print("Converting to TensorFlow.js...")
    tfjs_path = os.path.join(output_path, "tfjs_model")

    try:
        import tensorflowjs as tfjs
        tfjs.converters.convert_tf_saved_model(
            tf_model_path,
            tfjs_path,
        )
        print(f"TensorFlow.js model saved to: {tfjs_path}")
    except Exception as e:
        print(f"TF.js conversion failed: {e}")
        print("Try manual conversion with: tensorflowjs_converter --input_format=tf_saved_model")
        print(f"  {tf_model_path} {tfjs_path}")

    return tfjs_path


def export_weights_json(
    model_path: str,
    output_path: str,
    board_size: int = 20,
) -> str:
    """
    Export model weights as JSON for simple JS implementation.
    This is a simpler alternative that doesn't require ONNX/TF.js.

    Args:
        model_path: Path to trained .zip model
        output_path: Output directory
        board_size: Board size

    Returns:
        Path to weights JSON file
    """
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    policy_net = PolicyNetwork(model)
    policy_net.eval()

    # Extract weights
    weights = {}
    for name, param in policy_net.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()

    # Model metadata
    metadata = {
        "board_size": board_size,
        "n_channels": 8,
        "n_actions": 3,
        "architecture": "SnakeFeatureExtractor + MLP",
    }

    output = {
        "metadata": metadata,
        "weights": weights,
    }

    Path(output_path).mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(output_path, "snake_weights.json")

    print(f"Saving weights to: {json_path}")
    with open(json_path, "w") as f:
        json.dump(output, f)

    # Also save a compact version
    compact_path = os.path.join(output_path, "snake_weights_compact.json")
    with open(compact_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    print(f"Compact weights saved to: {compact_path}")
    print(f"File size: {os.path.getsize(compact_path) / 1024:.1f} KB")

    return json_path


def create_js_inference_template(
    output_path: str,
    board_size: int = 20,
) -> str:
    """
    Create a JavaScript template for running inference.

    Args:
        output_path: Output directory
        board_size: Board size

    Returns:
        Path to JS file
    """
    js_template = f'''/**
 * Snake RL Agent - Browser Inference
 * Loads trained model and runs inference for Snake game.
 *
 * Usage:
 *   const agent = new SnakeRLAgent();
 *   await agent.loadModel('tfjs_model/model.json');
 *   const action = agent.predict(observation);
 */

class SnakeRLAgent {{
    constructor(boardSize = {board_size}) {{
        this.boardSize = boardSize;
        this.nChannels = 8;
        this.nActions = 3;
        this.model = null;
    }}

    /**
     * Load TensorFlow.js model.
     * @param {{string}} modelPath - Path to model.json
     */
    async loadModel(modelPath) {{
        this.model = await tf.loadGraphModel(modelPath);
        console.log('Snake RL model loaded');
    }}

    /**
     * Build observation tensor from game state.
     * @param {{Object}} gameState - Current game state
     * @returns {{tf.Tensor}} Observation tensor
     */
    buildObservation(gameState) {{
        const {{ head, body, food, direction, length }} = gameState;

        // Create 8-channel observation
        const obs = new Float32Array(this.nChannels * this.boardSize * this.boardSize);

        // Helper to set channel value
        const setCell = (channel, row, col, value) => {{
            const idx = channel * this.boardSize * this.boardSize + row * this.boardSize + col;
            obs[idx] = value;
        }};

        // Channel 0: Head position
        setCell(0, head.row, head.col, 1.0);

        // Channel 1: Body positions
        for (const segment of body) {{
            setCell(1, segment.row, segment.col, 1.0);
        }}

        // Channel 2: Food position
        setCell(2, food.row, food.col, 1.0);

        // Channels 3-6: Direction one-hot (broadcast)
        const dirChannel = 3 + direction;
        for (let r = 0; r < this.boardSize; r++) {{
            for (let c = 0; c < this.boardSize; c++) {{
                setCell(dirChannel, r, c, 1.0);
            }}
        }}

        // Channel 7: Normalized length (broadcast)
        const normalizedLength = length / (this.boardSize * this.boardSize);
        for (let r = 0; r < this.boardSize; r++) {{
            for (let c = 0; c < this.boardSize; c++) {{
                setCell(7, r, c, normalizedLength);
            }}
        }}

        return tf.tensor4d(obs, [1, this.nChannels, this.boardSize, this.boardSize]);
    }}

    /**
     * Predict action from game state.
     * @param {{Object}} gameState - Current game state
     * @param {{boolean}} deterministic - Use argmax (true) or sample (false)
     * @returns {{number}} Action: 0=left, 1=straight, 2=right
     */
    predict(gameState, deterministic = true) {{
        if (!this.model) {{
            throw new Error('Model not loaded. Call loadModel() first.');
        }}

        const obsTensor = this.buildObservation(gameState);
        const logits = this.model.predict(obsTensor);
        const logitsArray = logits.dataSync();

        // Clean up tensors
        obsTensor.dispose();
        logits.dispose();

        if (deterministic) {{
            // Argmax
            let maxIdx = 0;
            let maxVal = logitsArray[0];
            for (let i = 1; i < this.nActions; i++) {{
                if (logitsArray[i] > maxVal) {{
                    maxVal = logitsArray[i];
                    maxIdx = i;
                }}
            }}
            return maxIdx;
        }} else {{
            // Sample from softmax distribution
            const expLogits = logitsArray.map(x => Math.exp(x));
            const sumExp = expLogits.reduce((a, b) => a + b, 0);
            const probs = expLogits.map(x => x / sumExp);

            const r = Math.random();
            let cumsum = 0;
            for (let i = 0; i < this.nActions; i++) {{
                cumsum += probs[i];
                if (r < cumsum) {{
                    return i;
                }}
            }}
            return this.nActions - 1;
        }}
    }}

    /**
     * Convert relative action to absolute direction.
     * @param {{number}} currentDir - Current direction (0=up, 1=right, 2=down, 3=left)
     * @param {{number}} action - Relative action (0=left, 1=straight, 2=right)
     * @returns {{number}} New absolute direction
     */
    static actionToDirection(currentDir, action) {{
        const delta = [-1, 0, 1];  // left, straight, right
        return (currentDir + delta[action] + 4) % 4;
    }}
}}

// Direction vectors
const DIRECTIONS = {{
    0: {{ dr: -1, dc: 0 }},  // up
    1: {{ dr: 0, dc: 1 }},   // right
    2: {{ dr: 1, dc: 0 }},   // down
    3: {{ dr: 0, dc: -1 }},  // left
}};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = {{ SnakeRLAgent, DIRECTIONS }};
}}
'''

    Path(output_path).mkdir(parents=True, exist_ok=True)
    js_path = os.path.join(output_path, "snake_rl_agent.js")

    with open(js_path, "w") as f:
        f.write(js_template)

    print(f"JavaScript inference template saved to: {js_path}")
    return js_path


def main():
    parser = argparse.ArgumentParser(description="Export Snake RL model for web inference")

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="web_model",
        help="Output directory for exported model",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=20,
        help="Board size the model was trained on",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["onnx", "tfjs", "json", "all"],
        help="Export format",
    )
    parser.add_argument(
        "--skip-tfjs",
        action="store_true",
        help="Skip TF.js conversion (requires onnx-tf)",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Always create JS inference template
    create_js_inference_template(output_dir, args.board_size)

    if args.format in ["onnx", "tfjs", "all"]:
        onnx_path = export_to_onnx(
            args.model_path,
            output_dir,
            args.board_size,
        )

        if args.format in ["tfjs", "all"] and not args.skip_tfjs:
            try:
                export_to_tfjs(onnx_path, output_dir)
            except Exception as e:
                print(f"TF.js export failed: {e}")
                print("You can try manual conversion or use JSON weights instead.")

    if args.format in ["json", "all"]:
        export_weights_json(
            args.model_path,
            output_dir,
            args.board_size,
        )

    print(f"\nExport complete! Files saved to: {output_dir}")
    print("\nTo use in browser:")
    print("1. Include TensorFlow.js: <script src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs'></script>")
    print("2. Include snake_rl_agent.js")
    print("3. Load model: await agent.loadModel('tfjs_model/model.json')")


if __name__ == "__main__":
    main()
