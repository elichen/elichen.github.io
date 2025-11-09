#!/usr/bin/env python3
"""
Export trained PPO model to TensorFlow.js format for browser deployment.
This converts the stable-baselines3 PPO model to a format that can be loaded in the browser.
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from typing import Dict, Any

from stable_baselines3 import PPO
from snake_gym_env import SnakeGymEnv


class PPOToTFJSExporter:
    """Export PPO model from stable-baselines3 to TensorFlow.js format."""

    def __init__(self, model_path: str, grid_size: int = 20):
        """
        Initialize exporter with trained model.

        Args:
            model_path: Path to saved PPO model
            grid_size: Grid size the model was trained on
        """
        print(f"📂 Loading PPO model from {model_path}")
        self.model = PPO.load(model_path)
        self.grid_size = grid_size

        # Get network architecture info
        self.policy = self.model.policy
        self._extract_architecture()

    def _extract_architecture(self):
        """Extract the neural network architecture from PPO policy."""
        # PPO uses an actor-critic architecture
        # We mainly need the actor (policy) network for deployment

        # Get the policy network
        self.actor = self.policy.action_net
        self.value = self.policy.value_net
        self.features_extractor = self.policy.features_extractor

        # Get layer dimensions
        if hasattr(self.features_extractor, 'policy_net'):
            self.hidden_layers = []
            for layer in self.features_extractor.policy_net:
                if hasattr(layer, 'out_features'):
                    self.hidden_layers.append(layer.out_features)
        else:
            # Default architecture
            self.hidden_layers = [64, 64]  # Standard PPO architecture

        print(f"   Architecture: Input(24) -> {' -> '.join(map(str, self.hidden_layers))} -> Output(4)")

    def create_tensorflow_model(self) -> tf.keras.Model:
        """
        Create equivalent TensorFlow/Keras model.

        Returns:
            TensorFlow model with same architecture
        """
        print("🏗️ Creating TensorFlow model...")

        # Build the model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(24,)),  # Our state size
        ])

        # Add hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(tf.keras.layers.Dense(
                units,
                activation='tanh',  # PPO typically uses tanh
                name=f'hidden_{i}'
            ))

        # Output layer (4 actions)
        model.add(tf.keras.layers.Dense(
            4,
            activation='linear',  # No activation for policy logits
            name='output'
        ))

        model.compile(optimizer='adam', loss='mse')

        print(f"   Model created with {model.count_params()} parameters")
        model.summary()

        return model

    def transfer_weights(self, tf_model: tf.keras.Model):
        """
        Transfer weights from PyTorch PPO to TensorFlow model.

        Args:
            tf_model: TensorFlow model to transfer weights to
        """
        print("🔄 Transferring weights from PPO to TensorFlow...")

        try:
            # Get PyTorch state dict
            state_dict = self.model.policy.state_dict()

            # Map PyTorch weights to TensorFlow layers
            # This mapping depends on the specific architecture
            layer_idx = 0

            # Transfer mlp_extractor.policy_net layers (2 layers: 24→64→64)
            mlp_layers = [
                'mlp_extractor.policy_net.0',  # Layer 0: 24→64
                'mlp_extractor.policy_net.2'   # Layer 2: 64→64 (layer 1 is Tanh)
            ]

            for mlp_layer in mlp_layers:
                weight_key = f'{mlp_layer}.weight'
                bias_key = f'{mlp_layer}.bias'

                if weight_key in state_dict and bias_key in state_dict:
                    weights = state_dict[weight_key].cpu().numpy().T  # Transpose for TF
                    bias = state_dict[bias_key].cpu().numpy()

                    tf_model.layers[layer_idx].set_weights([weights, bias])
                    print(f"   ✓ Transferred layer {layer_idx}: {weights.shape}")
                    layer_idx += 1

            # Transfer action network weights (final layer: 64→4)
            if 'action_net.weight' in state_dict and 'action_net.bias' in state_dict:
                action_weights = state_dict['action_net.weight'].cpu().numpy().T
                action_bias = state_dict['action_net.bias'].cpu().numpy()
                tf_model.layers[-1].set_weights([action_weights, action_bias])
                print(f"   ✓ Transferred action layer: {action_weights.shape}")

        except Exception as e:
            print(f"⚠️ Weight transfer may be incomplete: {e}")
            print("   Using simplified weight extraction method...")

            # Simplified method: create dummy input and trace through network
            self._simplified_weight_transfer(tf_model)

    def _simplified_weight_transfer(self, tf_model: tf.keras.Model):
        """
        Simplified weight transfer using model prediction matching.
        """
        print("   Using prediction matching for weight calibration...")

        # Generate test inputs
        env = SnakeGymEnv(grid_size=self.grid_size)
        test_states = []

        for _ in range(100):
            obs, _ = env.reset()
            test_states.append(obs)
            for _ in range(10):
                action = env.action_space.sample()
                obs, _, done, _, _ = env.step(action)
                test_states.append(obs)
                if done:
                    break

        test_states = np.array(test_states[:1000])

        # Get PPO predictions
        ppo_actions = []
        for state in test_states:
            action, _ = self.model.predict(state, deterministic=True)
            ppo_actions.append(action)

        ppo_actions = np.array(ppo_actions)

        # Create one-hot encoded targets
        targets = np.zeros((len(ppo_actions), 4))
        for i, action in enumerate(ppo_actions):
            targets[i, action] = 1

        # Fine-tune TensorFlow model to match PPO predictions
        print("   Fine-tuning TensorFlow model to match PPO behavior...")
        tf_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        tf_model.fit(
            test_states,
            targets,
            epochs=50,
            batch_size=32,
            verbose=0
        )

        # Validate accuracy
        predictions = tf_model.predict(test_states, verbose=0)
        predicted_actions = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_actions == ppo_actions)
        print(f"   ✓ Prediction matching accuracy: {accuracy*100:.1f}%")

    def export_to_tfjs(self, tf_model: tf.keras.Model, output_dir: str):
        """
        Export TensorFlow model to TensorFlow.js format.

        Args:
            tf_model: TensorFlow model to export
            output_dir: Directory to save the exported model
        """
        print(f"📦 Exporting to TensorFlow.js format...")

        os.makedirs(output_dir, exist_ok=True)

        # Export the model
        tfjs.converters.save_keras_model(tf_model, output_dir)

        print(f"   ✓ Model exported to {output_dir}")

        # Create metadata file
        metadata = {
            'grid_size': self.grid_size,
            'input_size': 24,
            'output_size': 4,
            'hidden_layers': self.hidden_layers,
            'architecture': 'mlp',
            'activation': 'tanh',
            'description': 'PPO Snake agent converted to TensorFlow.js'
        }

        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"   ✓ Metadata saved to {output_dir}/metadata.json")

    def create_usage_example(self, output_dir: str):
        """Create example JavaScript code for using the model."""

        example_code = '''// Load and use the Snake PPO model in TensorFlow.js

// Load the model
async function loadSnakeModel() {
    const model = await tf.loadLayersModel('./tfjs_model/model.json');
    return model;
}

// Predict action from game state
function predictAction(model, state) {
    // State should be a 24-element array with the features:
    // - Direction one-hot (4)
    // - Food direction (2)
    // - Danger detection (4)
    // - Distance to walls (4)
    // - Snake length (1)
    // - Grid fill ratio (1)
    // - Food distance (1)
    // - Body pattern (3)
    // - Connectivity features (4)

    const input = tf.tensor2d([state]);
    const prediction = model.predict(input);
    const action = prediction.argMax(-1).dataSync()[0];

    // Clean up tensors
    input.dispose();
    prediction.dispose();

    return action;  // 0: Up, 1: Right, 2: Down, 3: Left
}

// Example usage
async function playSnake() {
    const model = await loadSnakeModel();

    // Game loop
    while (!gameOver) {
        const state = getGameState();  // Your function to get current state
        const action = predictAction(model, state);
        executeAction(action);  // Your function to execute the action
    }
}

// Feature extraction example
function getGameState(snake, food, gridSize) {
    const state = [];

    // 1. Direction one-hot (4 features)
    const directionMap = {
        'up': [1, 0, 0, 0],
        'right': [0, 1, 0, 0],
        'down': [0, 0, 1, 0],
        'left': [0, 0, 0, 1]
    };
    state.push(...directionMap[snake.direction]);

    // 2. Food direction (2 features)
    const head = snake.body[0];
    state.push((food.x - head.x) / gridSize);
    state.push((food.y - head.y) / gridSize);

    // 3. Danger detection (4 features)
    const dangers = [
        isDanger(head.x, head.y - 1),  // Up
        isDanger(head.x + 1, head.y),  // Right
        isDanger(head.x, head.y + 1),  // Down
        isDanger(head.x - 1, head.y)   // Left
    ];
    state.push(...dangers.map(d => d ? 1 : 0));

    // 4. Distance to walls (4 features)
    state.push(head.y / gridSize);  // Top
        state.push((gridSize - 1 - head.x) / gridSize);  // Right
        state.push((gridSize - 1 - head.y) / gridSize);  // Bottom
        state.push(head.x / gridSize);  // Left

    // 5. Snake length (1 feature)
    state.push(snake.body.length / (gridSize * gridSize));

    // 6. Grid fill ratio (1 feature)
    state.push((snake.body.length - 1) / (gridSize * gridSize));

    // 7. Food distance (1 feature)
    const foodDist = Math.abs(food.x - head.x) + Math.abs(food.y - head.y);
    state.push(foodDist / (2 * gridSize));

    // 8. Body pattern (3 features) - simplified
    state.push(0, 0, 0);  // Placeholder for body patterns

    // 9. Connectivity features (4 features) - simplified
    state.push(0.5, 0.5, 0.5, 0.5);  // Placeholder for connectivity

    return state;
}
'''

        with open(os.path.join(output_dir, 'example_usage.js'), 'w') as f:
            f.write(example_code)

        print(f"   ✓ Usage example saved to {output_dir}/example_usage.js")

    def export(self, output_dir: str = "tfjs_model"):
        """
        Complete export pipeline.

        Args:
            output_dir: Directory to save the exported model
        """
        print(f"\n{'='*60}")
        print("🚀 Starting PPO to TensorFlow.js Export")
        print(f"{'='*60}\n")

        # Create TensorFlow model
        tf_model = self.create_tensorflow_model()

        # Transfer weights
        self.transfer_weights(tf_model)

        # Export to TensorFlow.js
        self.export_to_tfjs(tf_model, output_dir)

        # Create usage example
        self.create_usage_example(output_dir)

        print(f"\n{'='*60}")
        print("✅ Export Complete!")
        print(f"{'='*60}")
        print(f"\nTo use the model in your browser:")
        print(f"1. Copy the '{output_dir}' directory to your web server")
        print(f"2. Load the model using TensorFlow.js:")
        print(f"   const model = await tf.loadLayersModel('./{output_dir}/model.json');")
        print(f"3. See '{output_dir}/example_usage.js' for implementation details")

        return True


def main():
    """Main entry point for export script."""
    parser = argparse.ArgumentParser(description="Export PPO model to TensorFlow.js")

    parser.add_argument('model_path', type=str,
                      help='Path to trained PPO model')
    parser.add_argument('--grid-size', type=int, default=20,
                      help='Grid size the model was trained on (default: 20)')
    parser.add_argument('--output-dir', type=str, default='tfjs_model',
                      help='Output directory for TensorFlow.js model (default: tfjs_model)')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found at {args.model_path}")
        sys.exit(1)

    # Check dependencies
    try:
        import tensorflowjs
    except ImportError:
        print("❌ TensorFlow.js not installed. Run: pip install tensorflowjs")
        sys.exit(1)

    # Export the model
    exporter = PPOToTFJSExporter(args.model_path, args.grid_size)
    success = exporter.export(args.output_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()