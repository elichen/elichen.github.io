"""
Convert trained PyTorch DQN model to TensorFlow.js format.

This script extracts the Q-network weights from a trained SB3 DQN model
and converts them to a JSON format that can be loaded in TensorFlow.js.
"""
import sys
sys.path.insert(0, "/Users/elichen/code/stable-baselines3")

import json
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from mtncar_env import make_env


def extract_network_architecture(model):
    """
    Extract the network architecture from the DQN model.

    Returns:
        dict: Network architecture description
    """
    q_net = model.q_net

    # Get the q_net structure
    # SB3 DQN uses a QNetwork with features_extractor and q_net
    arch = {
        "input_size": 2,  # Mountain car: [position, velocity]
        "output_size": 3,  # 3 actions: left, none, right
        "hidden_layers": [],
        "activation": "relu"
    }

    # Extract hidden layer sizes from the network
    # The q_net has a sequential structure
    for name, module in q_net.q_net.named_modules():
        if isinstance(module, torch.nn.Linear):
            arch["hidden_layers"].append(module.out_features)

    # Remove the last layer (output layer) from hidden layers
    if arch["hidden_layers"]:
        arch["hidden_layers"] = arch["hidden_layers"][:-1]

    return arch


def convert_weights_to_json(model, output_path="models/sb3_weights.json"):
    """
    Convert PyTorch model weights to JSON format for TensorFlow.js.

    Args:
        model: Trained SB3 DQN model
        output_path: Path to save the JSON weights
    """
    q_net = model.q_net

    # Extract weights and biases from the Q-network
    weights_dict = {
        "architecture": extract_network_architecture(model),
        "weights": []
    }

    # Get all layers from the q_net
    # Structure: features_extractor -> q_net (Sequential with Linear layers)
    layer_idx = 0

    for name, module in q_net.q_net.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get weights and biases
            weight = module.weight.data.cpu().numpy()  # Shape: (out_features, in_features)
            bias = module.bias.data.cpu().numpy()      # Shape: (out_features,)

            # TensorFlow.js expects weights in transposed format (in_features, out_features)
            weight_transposed = weight.T

            layer_data = {
                "layer_index": layer_idx,
                "weight": weight_transposed.tolist(),
                "bias": bias.tolist(),
                "input_size": weight_transposed.shape[0],
                "output_size": weight_transposed.shape[1]
            }

            weights_dict["weights"].append(layer_data)
            layer_idx += 1

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)

    print(f"Weights saved to: {output_path}")
    print(f"Architecture: {weights_dict['architecture']}")
    print(f"Number of layers: {len(weights_dict['weights'])}")

    return weights_dict


def test_conversion(model, weights_dict, n_tests=10):
    """
    Test that the converted weights match the original model's output.

    Args:
        model: Original PyTorch model
        weights_dict: Converted weights dictionary
        n_tests: Number of random test cases
    """
    print("\nTesting conversion accuracy...")
    print("=" * 60)

    # Create a test environment
    env = make_env()

    # Test on random states
    max_error = 0

    for i in range(n_tests):
        # Generate random state
        state = env.observation_space.sample()

        # Get PyTorch Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values_torch = model.q_net(state_tensor).squeeze().numpy()

        # Manually compute Q-values using converted weights
        # This simulates what TF.js will do
        x = state
        for layer in weights_dict["weights"]:
            weight = np.array(layer["weight"])
            bias = np.array(layer["bias"])

            # Linear transformation
            x = np.dot(x, weight) + bias

            # Apply ReLU activation (except for last layer)
            if layer["layer_index"] < len(weights_dict["weights"]) - 1:
                x = np.maximum(0, x)

        q_values_manual = x

        # Compare
        error = np.abs(q_values_torch - q_values_manual).max()
        max_error = max(max_error, error)

        print(f"Test {i+1}:")
        print(f"  State: [{state[0]:.4f}, {state[1]:.4f}]")
        print(f"  PyTorch Q-values: {q_values_torch}")
        print(f"  Manual Q-values:  {q_values_manual}")
        print(f"  Max error: {error:.6f}")

    print("=" * 60)
    print(f"Maximum error across all tests: {max_error:.6f}")

    if max_error < 1e-5:
        print("✓ Conversion successful! Errors are within tolerance.")
    else:
        print("✗ Warning: Conversion may have issues. Check the implementation.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert SB3 DQN to TensorFlow.js")
    parser.add_argument("--model-path", type=str,
                       default="models/best_model/best_model.zip",
                       help="Path to trained SB3 model")
    parser.add_argument("--output", type=str,
                       default="models/sb3_weights.json",
                       help="Output path for JSON weights")
    parser.add_argument("--test", action="store_true",
                       help="Test conversion accuracy")

    args = parser.parse_args()

    print("Loading model...")
    env = DummyVecEnv([make_env])
    model = DQN.load(args.model_path, env=env)

    print("Converting weights...")
    weights_dict = convert_weights_to_json(model, args.output)

    if args.test:
        test_conversion(model, weights_dict)

    print("\nConversion complete!")
    print(f"To use in TensorFlow.js, load the weights from: {args.output}")


if __name__ == "__main__":
    main()
