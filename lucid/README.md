# Lucid Feature Visualization

A browser-based implementation of neural network feature visualization, inspired by the [Lucid library](https://github.com/tensorflow/lucid) and [Distill's Feature Visualization](https://distill.pub/2017/feature-visualization/) article.

## Overview

This tool visualizes what units in InceptionV3's late layers "see" by optimizing an input image to maximally activate either a center neuron or an entire channel. Unlike DeepDream which enhances all features, this focuses on targeted internal activations to show what patterns each unit responds to.

## Features

### Core Visualization
- **Switchable Objective Modes**: Choose between center-neuron and full-channel maximization
- **Fourier Parameterization**: Optimize a learned Fourier basis instead of raw pixels
- **Progressive Resolution**: Optimize at 128, 192, 256, and 299 px, then display at 512 px

### Regularization Techniques
- **Transformation Robustness**: Constant padding, jitter crops, and random scaling
- **Frequency Bias**: Low frequencies are favored directly in the Fourier parameterization
- **Total Variation**: Encourages spatial smoothness
- **L2 Decay**: Prevents extreme pixel values

### Layers Available
- **Mixed_6a** (768 channels): Early patterns and textures
- **Mixed_6b** (768 channels): Mid-level features and parts
- **Mixed_6c** (768 channels): Complex recurring patterns
- **Mixed_6d** (768 channels): Higher-level object parts
- **Mixed_6e** (768 channels): Late layer abstract features

## Usage

1. **Select Layer**: Choose which InceptionV3 layer to visualize
2. **Choose Channel**: Select a specific channel index
3. **Choose Objective**:
   - **Center Neuron**: More localized and closer to classic Lucid neuron renders
   - **Full Channel**: Faster and often cleaner in-browser
4. **Adjust Settings** (optional):
   - **Steps**: Number of optimization iterations (default: 128)
   - **Learning Rate**: Adam learning rate (default: 0.05)
   - **Regularization Weights**: Fine-tune different penalties
5. **Click the visualize button** to start the optimization
6. **Download** the resulting visualization

## Technical Details

### Fourier Parameterization
Instead of optimizing pixels directly, the app learns Fourier coefficients:
- Uses a decayed frequency spectrum so low frequencies dominate early
- Keeps the image in the model's expected `[0, 1]` range
- Applies Lucid-style color decorrelation before the final sigmoid

### Optimization Process
1. Initialize a Fourier basis at the model's native resolution
2. Optimize over progressive stages (128, 192, 256, 299):
   - Resize the rendered image to the current stage
   - Apply padded jitter and random scaling
   - Maximize either the selected channel's center neuron or the full channel map
   - Apply L2 and total-variation penalties on the rendered image
3. Render the final result at 512x512 for display and download

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
- Channel 600-767: More abstract object parts and motifs

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
| **Goal** | Understand targeted units/channels | Enhance all features |
| **Initialization** | Fourier basis | Existing image |
| **Optimization** | Maximize one neuron or one channel | Maximize all activations |
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
