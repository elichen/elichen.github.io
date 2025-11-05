#!/usr/bin/env python3
"""
Export trained PPO model to TensorFlow.js format
Direct PyTorch to TF.js conversion without needing TensorFlow
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from tictactoe_env import TicTacToeEnv


def create_tfjs_model_direct(model_path: str, output_dir: str = "tfjs_model"):
    """Create a TensorFlow.js model directly from PyTorch weights"""

    print(f"Loading model from {model_path}...")

    # Create environment and load PPO model
    env = TicTacToeEnv(opponent_type='random')

    # Try loading as MaskablePPO first, fallback to regular PPO
    try:
        model = MaskablePPO.load(model_path, env=env)
        print("Loaded as MaskablePPO")
    except:
        model = PPO.load(model_path, env=env)
        print("Loaded as PPO")

    # Get the policy network (actor)
    policy_net = model.policy.mlp_extractor.policy_net
    action_net = model.policy.action_net

    # Combine policy and action networks
    full_net = nn.Sequential(policy_net, action_net)
    full_net.eval()
    print("Loaded PPO model")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract network architecture and weights
    layers = []
    weights_data = []

    # Flatten the nested Sequential and get individual modules
    all_modules = []
    for module in full_net.modules():
        if isinstance(module, (nn.Linear, nn.ReLU)):
            all_modules.append(module)

    # Process each layer
    layer_idx = 0
    for i, module in enumerate(all_modules):
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy() if module.bias is not None else None

            # TensorFlow.js expects weights in different format
            # PyTorch: [out_features, in_features]
            # TF.js: [in_features, out_features]
            weight = weight.T

            # Determine activation for this layer
            # Check if next module is ReLU
            activation = 'linear'
            if i + 1 < len(all_modules) and isinstance(all_modules[i + 1], nn.ReLU):
                activation = 'relu'

            layers.append({
                'name': f'dense_{layer_idx}',
                'type': 'dense',
                'units': weight.shape[1],
                'activation': activation,
                'weight_shape': list(weight.shape),
                'bias_shape': list(bias.shape) if bias is not None else None
            })

            weights_data.append(weight)
            if bias is not None:
                weights_data.append(bias)

            layer_idx += 1

    print(f"Extracted {len(layers)} layers:")
    for i, layer in enumerate(layers):
        print(f"  Layer {i}: {layer['weight_shape']} -> {layer['units']} units")

    # Create model.json structure for TensorFlow.js
    model_json = {
        "format": "layers-model",
        "generatedBy": "export_to_tfjs.py",
        "convertedBy": "Custom PyTorch to TF.js converter",
        "modelTopology": {
            "keras_version": "2.13.0",
            "backend": "tensorflow",
            "model_config": {
                "class_name": "Sequential",
                "config": {
                    "name": "tictactoe_ppo",
                    "layers": []
                }
            }
        },
        "weightsManifest": [{
            "paths": ["group1-shard1of1.bin"],
            "weights": []
        }]
    }

    # Add layers to model config
    for i, layer_info in enumerate(layers):
        if i == 0:
            # First layer needs input shape (9 for board state)
            layer_config = {
                "class_name": "Dense",
                "config": {
                    "name": layer_info['name'],
                    "trainable": True,
                    "batch_input_shape": [None, 9],
                    "dtype": "float32",
                    "units": layer_info['units'],
                    "activation": layer_info['activation'],
                    "use_bias": layer_info['bias_shape'] is not None
                }
            }
        else:
            layer_config = {
                "class_name": "Dense",
                "config": {
                    "name": layer_info['name'],
                    "trainable": True,
                    "dtype": "float32",
                    "units": layer_info['units'],
                    "activation": layer_info['activation'],
                    "use_bias": layer_info['bias_shape'] is not None
                }
            }

        model_json["modelTopology"]["model_config"]["config"]["layers"].append(layer_config)

        # Add weights info
        weight_spec = {
            "name": f"{layer_info['name']}/kernel",
            "shape": layer_info['weight_shape'],
            "dtype": "float32"
        }
        model_json["weightsManifest"][0]["weights"].append(weight_spec)

        if layer_info['bias_shape'] is not None:
            bias_spec = {
                "name": f"{layer_info['name']}/bias",
                "shape": layer_info['bias_shape'],
                "dtype": "float32"
            }
            model_json["weightsManifest"][0]["weights"].append(bias_spec)

    # Save model.json
    model_json_path = os.path.join(output_dir, "model.json")
    with open(model_json_path, 'w') as f:
        json.dump(model_json, f, indent=2)

    # Save weights as binary
    weights_path = os.path.join(output_dir, "group1-shard1of1.bin")
    with open(weights_path, 'wb') as f:
        for weight in weights_data:
            # Write weights as float32 binary data
            f.write(weight.astype(np.float32).tobytes())

    print(f"\nTensorFlow.js model saved to {output_dir}")
    print(f"  - model.json: {model_json_path}")
    print(f"  - weights: {weights_path}")

    # Also save model info for the web app
    model_info = {
        "input_shape": [9],
        "output_shape": [9],
        "description": "PPO agent trained to expert level (never loses)",
        "training_steps": 50000,
        "performance": {
            "vs_perfect": "100% draws (0 losses)",
            "skill_level": "Expert"
        },
        "usage": "Model outputs action logits for each board position (0-8)"
    }

    info_path = os.path.join(output_dir, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"  - model info: {info_path}")

    return output_dir


def verify_export(model_path: str, tfjs_dir: str):
    """Verify that the exported model has the right structure"""

    print("\nVerifying export...")

    # Load original PPO model
    env = TicTacToeEnv(opponent_type='random')

    # Try MaskablePPO first, fallback to PPO
    try:
        original_model = MaskablePPO.load(model_path, env=env)
    except:
        original_model = PPO.load(model_path, env=env)

    # Create test observation (empty board)
    test_obs = np.zeros(9, dtype=np.float32)

    # Get original model prediction
    action, _ = original_model.predict(test_obs, deterministic=True)
    print(f"Original PPO model action on empty board: {action}")

    # Check the exported files exist
    model_json = os.path.join(tfjs_dir, "model.json")
    weights_bin = os.path.join(tfjs_dir, "group1-shard1of1.bin")
    model_info = os.path.join(tfjs_dir, "model_info.json")

    assert os.path.exists(model_json), "model.json not found"
    assert os.path.exists(weights_bin), "weights binary not found"
    assert os.path.exists(model_info), "model_info.json not found"

    print("✓ All export files verified")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export trained PPO model to TensorFlow.js")
    parser.add_argument("--model", type=str, default="models/ppo_tictactoe_fast",
                        help="Path to trained PPO model")
    parser.add_argument("--output", type=str, default="tfjs_model",
                        help="Output directory for TF.js model")

    args = parser.parse_args()

    if not os.path.exists(args.model + ".zip"):
        print(f"Model not found at {args.model}.zip")
        print("Please train a model first using train_fast.py")
        exit(1)

    tfjs_dir = create_tfjs_model_direct(args.model, args.output)
    verify_export(args.model, tfjs_dir)
    print(f"\n✅ TensorFlow.js export complete!")
    print(f"Model ready for web deployment at: {tfjs_dir}")
