"""
Convert Stable Baselines3 PPO model to TensorFlow.js format
Extracts the actor network from PPO and converts weights to JSON
"""

import os
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from racer_env import RacerEnv


def extract_ppo_architecture(model):
    """Extract network architecture from PPO model"""

    # Get the policy network (actor)
    policy = model.policy

    # Extract features extractor architecture
    features_extractor = policy.features_extractor
    mlp_extractor = policy.mlp_extractor
    action_net = policy.action_net

    # Build architecture info
    architecture = {
        'input_size': policy.observation_space.shape[0],
        'output_size': policy.action_space.shape[0],
        'features_extractor_layers': [],
        'policy_layers': [],
        'action_layer': None,
        'log_std_init': None
    }

    # Extract features extractor layers
    if hasattr(features_extractor, 'net'):
        for layer in features_extractor.net:
            if isinstance(layer, torch.nn.Linear):
                architecture['features_extractor_layers'].append({
                    'type': 'linear',
                    'in_features': layer.in_features,
                    'out_features': layer.out_features
                })
            elif isinstance(layer, torch.nn.ReLU):
                architecture['features_extractor_layers'].append({
                    'type': 'activation',
                    'activation': 'relu'
                })

    # Extract policy MLP layers
    if hasattr(mlp_extractor, 'policy_net'):
        for layer in mlp_extractor.policy_net:
            if isinstance(layer, torch.nn.Linear):
                architecture['policy_layers'].append({
                    'type': 'linear',
                    'in_features': layer.in_features,
                    'out_features': layer.out_features
                })
            elif isinstance(layer, torch.nn.ReLU):
                architecture['policy_layers'].append({
                    'type': 'activation',
                    'activation': 'relu'
                })

    # Extract action layer
    if action_net is not None:
        architecture['action_layer'] = {
            'in_features': action_net.in_features,
            'out_features': action_net.out_features
        }

    # Get log_std for continuous actions
    if hasattr(policy, 'log_std_init'):
        architecture['log_std_init'] = policy.log_std_init

    return architecture


def extract_ppo_weights(model):
    """Extract weights from PPO model for TensorFlow.js"""

    weights = {}
    policy = model.policy

    # Features extractor weights
    if hasattr(policy.features_extractor, 'net'):
        layer_idx = 0
        for layer in policy.features_extractor.net:
            if isinstance(layer, torch.nn.Linear):
                # TensorFlow.js expects transposed weights compared to PyTorch
                weights[f'features_extractor_{layer_idx}_weight'] = (
                    layer.weight.cpu().detach().numpy().T.tolist()
                )
                weights[f'features_extractor_{layer_idx}_bias'] = (
                    layer.bias.cpu().detach().numpy().tolist()
                )
                layer_idx += 1

    # Policy MLP weights
    if hasattr(policy.mlp_extractor, 'policy_net'):
        layer_idx = 0
        for layer in policy.mlp_extractor.policy_net:
            if isinstance(layer, torch.nn.Linear):
                weights[f'policy_{layer_idx}_weight'] = (
                    layer.weight.cpu().detach().numpy().T.tolist()
                )
                weights[f'policy_{layer_idx}_bias'] = (
                    layer.bias.cpu().detach().numpy().tolist()
                )
                layer_idx += 1

    # Action layer weights (mean actions)
    if policy.action_net is not None:
        weights['action_weight'] = (
            policy.action_net.weight.cpu().detach().numpy().T.tolist()
        )
        weights['action_bias'] = (
            policy.action_net.bias.cpu().detach().numpy().tolist()
        )

    # Log standard deviation for continuous actions
    if hasattr(policy, 'log_std'):
        weights['log_std'] = policy.log_std.cpu().detach().numpy().tolist()

    return weights


def validate_conversion(model, weights_json, env, n_tests=10):
    """Validate that the converted weights produce similar outputs"""

    print("\n🔍 Validating conversion...")

    # Test with random observations
    for i in range(n_tests):
        obs, _ = env.reset()

        # Get original model prediction (deterministic)
        with torch.no_grad():
            action_original, _ = model.predict(obs, deterministic=True)

        # Manual forward pass with extracted weights
        x = obs

        # Features extractor
        layer_idx = 0
        for layer_info in weights_json['architecture']['features_extractor_layers']:
            if layer_info['type'] == 'linear':
                weight = np.array(weights_json['weights'][f'features_extractor_{layer_idx}_weight']).T
                bias = np.array(weights_json['weights'][f'features_extractor_{layer_idx}_bias'])
                x = np.dot(weight, x) + bias
                layer_idx += 1
            elif layer_info['type'] == 'activation' and layer_info['activation'] == 'relu':
                x = np.maximum(0, x)

        # Policy MLP
        layer_idx = 0
        for layer_info in weights_json['architecture']['policy_layers']:
            if layer_info['type'] == 'linear':
                weight = np.array(weights_json['weights'][f'policy_{layer_idx}_weight']).T
                bias = np.array(weights_json['weights'][f'policy_{layer_idx}_bias'])
                x = np.dot(weight, x) + bias
                layer_idx += 1
            elif layer_info['type'] == 'activation' and layer_info['activation'] == 'relu':
                x = np.maximum(0, x)

        # Action layer
        if weights_json['architecture']['action_layer'] is not None:
            weight = np.array(weights_json['weights']['action_weight']).T
            bias = np.array(weights_json['weights']['action_bias'])
            action_manual = np.dot(weight, x) + bias
        else:
            action_manual = x

        # Apply tanh to bound actions to [-1, 1]
        action_manual = np.tanh(action_manual)

        # Compare
        error = np.mean(np.abs(action_original - action_manual))
        print(f"Test {i+1}: Error = {error:.6f}")

        if error > 0.01:  # Threshold for acceptable error
            print(f"⚠️  Warning: Large error detected!")
            print(f"  Original: {action_original}")
            print(f"  Manual: {action_manual}")

    print("✅ Validation complete!")


def convert_ppo_to_tfjs(model_path, output_path='models/ppo_weights.json'):
    """
    Convert a trained PPO model to TensorFlow.js format

    Args:
        model_path: Path to the saved PPO model (.zip file)
        output_path: Path to save the JSON weights file
    """

    print(f"\n🔄 Converting PPO model to TensorFlow.js format...")
    print(f"Input: {model_path}")
    print(f"Output: {output_path}")

    # Load the model
    print("\n📦 Loading PPO model...")
    model = PPO.load(model_path)

    # Create environment for validation
    env = RacerEnv()

    # Extract architecture and weights
    print("🏗️  Extracting architecture...")
    architecture = extract_ppo_architecture(model)

    print("⚖️  Extracting weights...")
    weights = extract_ppo_weights(model)

    # Create JSON structure
    weights_json = {
        'architecture': architecture,
        'weights': weights,
        'metadata': {
            'model_type': 'PPO',
            'action_space': 'continuous',
            'observation_size': architecture['input_size'],
            'action_size': architecture['output_size'],
            'converted_from': model_path
        }
    }

    # Validate conversion
    validate_conversion(model, weights_json, env)

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(weights_json, f, indent=2)

    print(f"\n✅ Model converted and saved to: {output_path}")

    # Print summary
    print("\n📊 Model Summary:")
    print(f"  Input size: {architecture['input_size']}")
    print(f"  Output size: {architecture['output_size']}")
    print(f"  Features extractor layers: {len(architecture['features_extractor_layers'])}")
    print(f"  Policy layers: {len(architecture['policy_layers'])}")
    print(f"  Total parameters: {sum(np.array(w).size for w in weights.values())}")

    env.close()
    return weights_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PPO model to TensorFlow.js")
    parser.add_argument("model_path", help="Path to the PPO model (.zip file)")
    parser.add_argument("--output", default="models/ppo_weights.json",
                       help="Output path for JSON weights (default: models/ppo_weights.json)")

    args = parser.parse_args()

    # Convert the model
    convert_ppo_to_tfjs(args.model_path, args.output)