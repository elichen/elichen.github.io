# -*- coding: utf-8 -*-
"""
Export trained SAC model to TensorFlow.js format
Adapted from PPO export scripts for continuous action SAC
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stick_env import StickBalancingEnv


def create_tfjs_model_from_sac(model_path: str, vec_norm_path: str, output_dir: str = "../models/tfjs"):
    """Create a TensorFlow.js model from SAC actor network"""

    print(f"Loading SAC model from {model_path}...")

    # Load the SAC model
    model = SAC.load(model_path)

    # Load normalization parameters if available
    norm_mean = None
    norm_std = None
    if vec_norm_path and os.path.exists(vec_norm_path):
        print("Loading normalization statistics...")
        env = DummyVecEnv([lambda: StickBalancingEnv()])
        env = VecNormalize.load(vec_norm_path, env)
        norm_mean = env.obs_rms.mean
        norm_var = env.obs_rms.var
        norm_std = np.sqrt(norm_var + 1e-8)
        print(f"Normalization mean: {norm_mean}")
        print(f"Normalization std: {norm_std}")

    # Get the actor network from SAC (we only need the actor for inference)
    actor = model.actor

    # Extract the mean network (SAC actor outputs mean and log_std)
    # SAC's actor has: features_extractor -> latent_pi -> mu (mean)
    features_extractor = actor.features_extractor
    latent_pi_net = actor.latent_pi
    mu_net = actor.mu  # This outputs the mean action

    print("Extracted SAC actor network")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract network architecture and weights
    layers = []
    weights_data = []

    # Process all linear layers in order
    all_modules = []

    # Add features extractor layers
    for module in features_extractor.modules():
        if isinstance(module, nn.Linear):
            all_modules.append(module)

    # Add latent_pi layers
    for module in latent_pi_net.modules():
        if isinstance(module, nn.Linear):
            all_modules.append(module)

    # Add mu layer (final output)
    all_modules.append(mu_net)

    # Extract weights from all layers
    layer_count = 0
    for module in all_modules:
        weight = module.weight.detach().cpu().numpy()
        bias = module.bias.detach().cpu().numpy() if module.bias is not None else None

        # TensorFlow.js expects weights in different format
        # PyTorch: [out_features, in_features]
        # TF.js: [in_features, out_features]
        weight = weight.T

        layer_name = f'dense_{layer_count}'
        layers.append({
            'name': layer_name,
            'type': 'dense',
            'units': weight.shape[1],
            'weight_shape': list(weight.shape),
            'bias_shape': list(bias.shape) if bias is not None else None
        })

        weights_data.append(weight)
        if bias is not None:
            weights_data.append(bias)

        layer_count += 1

    # Create model.json structure for TensorFlow.js
    model_json = {
        "format": "layers-model",
        "generatedBy": "export_sac_to_tfjs.py",
        "convertedBy": "Custom SAC to TF.js converter",
        "modelTopology": {
            "keras_version": "2.13.0",
            "backend": "tensorflow",
            "model_config": {
                "class_name": "Sequential",
                "config": {
                    "name": "stick_balancing_sac",
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
            # First layer needs input shape (4 for stick balancing)
            layer_config = {
                "class_name": "Dense",
                "config": {
                    "name": layer_info['name'],
                    "trainable": True,
                    "batch_input_shape": [None, 4],  # 4 observation dimensions
                    "dtype": "float32",
                    "units": layer_info['units'],
                    "activation": "relu" if i < len(layers) - 1 else "linear",
                    "use_bias": layer_info['bias_shape'] is not None
                }
            }
        else:
            # Last layer should have tanh activation for bounded continuous actions
            activation = "tanh" if i == len(layers) - 1 else "relu"
            layer_config = {
                "class_name": "Dense",
                "config": {
                    "name": layer_info['name'],
                    "trainable": True,
                    "dtype": "float32",
                    "units": layer_info['units'],
                    "activation": activation,
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

    # Save normalization parameters
    if norm_mean is not None:
        norm_params = {
            'mean': norm_mean.tolist(),
            'std': norm_std.tolist()
        }
        norm_json_path = os.path.join(output_dir, 'normalization.json')
        with open(norm_json_path, 'w') as f:
            json.dump(norm_params, f, indent=2)
        print(f"Normalization parameters saved to: {norm_json_path}")

    print(f"\nTensorFlow.js model saved to {output_dir}")
    print(f"  - model.json: {model_json_path}")
    print(f"  - weights: {weights_path}")

    # Print layer info
    print("\nModel architecture:")
    for i, layer_info in enumerate(layers):
        activation = "tanh" if i == len(layers) - 1 else "relu"
        print(f"  Layer {i}: {layer_info['weight_shape']} -> {layer_info['units']} units ({activation})")

    print("\n✅ Export complete! The model outputs continuous actions in range [-1, 1]")
    print("   This matches the expected format for the web app's AI controller.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export SAC model to TensorFlow.js")
    parser.add_argument("--model", type=str, default="models_sac_final/sac_stick_final.zip",
                       help="Path to trained SAC model")
    parser.add_argument("--vec-norm", type=str, default="models_sac_final/vec_normalize.pkl",
                       help="Path to normalization statistics")
    parser.add_argument("--output", type=str, default="../models/tfjs",
                       help="Output directory for TensorFlow.js model")

    args = parser.parse_args()

    create_tfjs_model_from_sac(args.model, args.vec_norm, args.output)


if __name__ == "__main__":
    main()