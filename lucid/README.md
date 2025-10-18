# Lucid Feature Visualization

A browser-based implementation of neural network feature visualization, inspired by the [Lucid library](https://github.com/tensorflow/lucid) and [Distill's Feature Visualization](https://distill.pub/2017/feature-visualization/) article.

## Overview

This tool visualizes what individual neurons in InceptionV3's late layers "see" by optimizing an input image to maximally activate specific neurons. Unlike DeepDream which enhances all features, this focuses on single neuron activations to understand what patterns each neuron responds to.

## Features

### Core Visualization
- **Single Neuron Maximization**: Visualize individual neurons/channels in InceptionV3 layers
- **Fourier Basis Initialization**: Generates cleaner, more structured patterns than random noise
- **Progressive Resolution**: Starts at 128x128, scales up to 512x512 for better quality

### Regularization Techniques
All techniques from the original Lucid library:
- **Transformation Robustness**: Random jitter, rotation, and scaling
- **Frequency Penalization**: Reduces high-frequency artifacts
- **Total Variation**: Encourages spatial smoothness
- **L2 Decay**: Prevents extreme pixel values

### Layers Available
- **Mixed_6a** (768 channels): Early patterns and textures
- **Mixed_6b** (768 channels): Mid-level features and parts
- **Mixed_6c** (768 channels): Complex recurring patterns
- **Mixed_6d** (768 channels): Higher-level object parts
- **Mixed_6e** (1280 channels): Late layer complex representations

## Usage

1. **Select Layer**: Choose which InceptionV3 layer to visualize
2. **Choose Channel**: Select a specific neuron/channel index (0 to max channels)
3. **Adjust Settings** (optional):
   - **Steps**: Number of optimization iterations (default: 500)
   - **Learning Rate**: Step size for gradient ascent (default: 0.05)
   - **Regularization Weights**: Fine-tune different penalties
4. **Click "Visualize Neuron"** to start the optimization
5. **Download** the resulting visualization

## Technical Details

### Fourier Initialization
Instead of starting from random noise, we initialize images using a Fourier basis:
- Generates frequency components with 1/f^1.5 power spectrum
- Random phase for each frequency
- Creates more natural, structured patterns

### Optimization Process
1. Initialize image with Fourier basis at 128x128
2. For each resolution stage (128, 256, 512):
   - Apply random transformations for robustness
   - Compute gradient of neuron activation
   - Apply regularization penalties
   - Update image using gradient ascent
   - Clip values to valid range [-1, 1]
3. Display final 512x512 visualization

### Implementation Stack
- **TensorFlow.js**: Neural network operations and model execution
- **InceptionV3**: Pretrained model from TensorFlow Hub
- **WebGL Backend**: GPU acceleration in the browser
- **No Build Process**: Pure JavaScript, runs directly in browser

## Interesting Neurons to Try

### Mixed_6e (Late Layer)
- Channel 0-100: Often shows animal-like features
- Channel 200-300: Architectural and geometric patterns
- Channel 400-500: Text and symbol-like patterns
- Channel 800-900: Natural textures and landscapes
- Channel 1000-1279: Complex object parts

### Mixed_6c (Mid Layer)
- Channel 0-100: Basic textures and patterns
- Channel 300-400: Repeating geometric structures
- Channel 500-600: Curved and organic shapes

## Tips for Best Results

1. **Start with defaults**: The default settings are tuned for good results
2. **Experiment with channels**: Different channels show vastly different patterns
3. **Try different layers**: Earlier layers show simpler patterns, later layers show more complex features
4. **Adjust regularization**:
   - Increase frequency penalty for smoother results
   - Increase TV weight for less noisy patterns
   - Decrease L2 weight for more vibrant colors

## Differences from DeepDream

| Aspect | Lucid (This Tool) | DeepDream |
|--------|------------------|-----------|
| **Goal** | Understand single neurons | Enhance all features |
| **Initialization** | Fourier basis | Existing image |
| **Optimization** | Maximize one neuron | Maximize all activations |
| **Result** | Clean, isolated patterns | Psychedelic, enhanced image |
| **Use Case** | Scientific visualization | Artistic effect |

## Browser Requirements

- Modern browser with WebGL support
- Recommended: Chrome, Firefox, or Edge (latest versions)
- Requires ~500MB RAM for model loading
- GPU acceleration recommended for faster optimization

## Credits

- Original [Lucid library](https://github.com/tensorflow/lucid) by TensorFlow team
- [Feature Visualization](https://distill.pub/2017/feature-visualization/) article on Distill.pub
- InceptionV3 model from [TensorFlow Hub](https://tfhub.dev/)
- Built with [TensorFlow.js](https://www.tensorflow.org/js)

## License

This implementation is for educational purposes. The visualization technique and approach are based on research published by the TensorFlow/Lucid team.