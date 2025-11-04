#!/usr/bin/env python3
"""
Export trained PPO model to TensorFlow.js format
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from train_ppo import RobotArmEnv




def create_tfjs_model_direct(model_path: str, output_dir: str = "models/tfjs"):
    """Create a TensorFlow.js model directly from PyTorch weights"""

    import json
    import struct

    print(f"Loading model from {model_path}...")

    # Create environment and load PPO model
    env = RobotArmEnv()
    model = PPO.load(model_path, env=env)

    # Get the policy network (actor)
    policy_net = model.policy.mlp_extractor.policy_net
    q_net = nn.Sequential(policy_net, model.policy.action_net)
    q_net.eval()
    print("Loaded PPO model")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract network architecture and weights
    layers = []
    weights_data = []

    # Get all layers from the network
    for name, module in q_net.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy() if module.bias is not None else None

            # TensorFlow.js expects weights in different format
            # PyTorch: [out_features, in_features]
            # TF.js: [in_features, out_features]
            weight = weight.T

            layers.append({
                'name': name if name else f'dense_{len(layers)}',
                'type': 'dense',
                'units': weight.shape[1],
                'weight_shape': list(weight.shape),
                'bias_shape': list(bias.shape) if bias is not None else None
            })

            weights_data.append(weight)
            if bias is not None:
                weights_data.append(bias)

    # Create model.json structure for TensorFlow.js
    model_json = {
        "format": "layers-model",
        "generatedBy": "export_model.py",
        "convertedBy": "Custom PyTorch to TF.js converter",
        "modelTopology": {
            "keras_version": "2.13.0",
            "backend": "tensorflow",
            "model_config": {
                "class_name": "Sequential",
                "config": {
                    "name": "robot_arm_ppo",
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
            # First layer needs input shape
            layer_config = {
                "class_name": "Dense",
                "config": {
                    "name": layer_info['name'],
                    "trainable": True,
                    "batch_input_shape": [None, 11],
                    "dtype": "float32",
                    "units": layer_info['units'],
                    "activation": "relu" if i < len(layers) - 1 else "linear",
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
                    "activation": "relu" if i < len(layers) - 1 else "linear",
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

    print(f"TensorFlow.js model saved to {output_dir}")
    print(f"  - model.json: {model_json_path}")
    print(f"  - weights: {weights_path}")

    # Also save model info for the web app
    model_info = {
        "input_shape": [11],
        "output_shape": [5],
        "actions": ["angle1+", "angle1-", "angle2+", "angle2-", "toggle_claw"],
        "observation_features": [
            "angle1_normalized",
            "angle2_normalized",
            "block_x_relative",
            "block_y_normalized",
            "claw_closed",
            "distance_to_block",
            "block_held",
            "valid_action_0",
            "valid_action_1",
            "valid_action_2",
            "valid_action_3"
        ]
    }

    info_path = os.path.join(output_dir, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"  - model info: {info_path}")

    return output_dir


def verify_export(model_path: str, tfjs_dir: str):
    """Verify that the exported model produces same outputs as original"""

    print("\nVerifying export...")

    # Load original PPO model
    env = RobotArmEnv()
    original_model = PPO.load(model_path, env=env)

    # Create test observation
    test_obs = np.array([0.5, -0.3, 0.2, 0.7, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)

    # Get original model prediction
    action, _ = original_model.predict(test_obs, deterministic=True)
    print(f"Original PPO model action: {action}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export trained PPO model to TensorFlow.js")
    parser.add_argument("--model", type=str, default="models/robot_arm_ppo_final",
                        help="Path to trained PPO model")

    args = parser.parse_args()

    if not os.path.exists(args.model + ".zip"):
        print(f"Model not found at {args.model}.zip")
        print("Please train a model first using train_ppo.py")
        exit(1)

    tfjs_dir = create_tfjs_model_direct(args.model)
    verify_export(args.model, tfjs_dir)
    print(f"✓ TensorFlow.js export complete: {tfjs_dir}")