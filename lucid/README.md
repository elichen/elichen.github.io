# Lucid Feature Visualization

A browser-based implementation of neural network feature visualization, inspired by the [Lucid library](https://github.com/tensorflow/lucid) and [Distill's Feature Visualization](https://distill.pub/2017/feature-visualization/) article.

## Overview

This tool visualizes what units in InceptionV3 and an adversarially robust ResNet-50 "see" by optimizing an input image to maximally activate either a center neuron, an entire channel, or a final ImageNet class output. Unlike DeepDream which enhances all features, this focuses on targeted internal activations to show what patterns each unit responds to.

## Features

### Core Visualization
- **Switchable Objective Modes**: Choose between center-neuron, full-channel, and final-class maximization
- **Fourier Parameterization**: Optimize a learned Fourier basis instead of raw pixels
- **Native Resolution**: Optimize at InceptionV3's 299 px input, then display at 512 px
- **ImageNet Labels**: Class mode exposes the final classifier output with human-readable labels

### Regularization Techniques
- **Transformation Robustness**: Lucid's constant padding, jitter crops, and random scaling (rotation is omitted because tfjs can't backpropagate through it)
- **Frequency Bias**: Low frequencies are favored directly in the Fourier parameterization
- **Total Variation**: Encourages spatial smoothness
- **L2 Decay**: Pulls pixels toward mid-gray to rein in saturation

### Models
- **InceptionV3** (TF Hub): the standard model. Neuron and channel renders show detailed textures, but class renders mostly fool the network with patterns people can't recognize.
- **Robust ResNet-50** (L2, ε=3) from [Salman et al.](https://arxiv.org/abs/2007.08489), weights from [madrylab/robust-imagenet-models](https://huggingface.co/madrylab/robust-imagenet-models) (MIT). Adversarial training means no imperceptible pattern can move its outputs much, so its class renders show recognizable objects (apples, pandas, toilet paper). The 51 MB of float16 weights in `robust-resnet50/` load on first use; `robust-resnet.js` implements the network in TensorFlow.js ops, with batch norm folded into the convolutions.

### Layers Available (InceptionV3)
- **Mixed_6a** (768 channels): Early patterns and textures
- **Mixed_6b** (768 channels): Mid-level features and parts
- **Mixed_6c** (768 channels): Complex recurring patterns
- **Mixed_6d** (768 channels): Higher-level object parts
- **Mixed_6e** (768 channels): Late layer abstract features

## Usage

The app opens on the robust ResNet-50 in ImageNet class mode, targeting class 1000 (toilet tissue). InceptionV3 loads from TensorFlow Hub the first time you select it.

1. **Select Model and Layer**: Choose the robust ResNet-50 or InceptionV3, and which layer to visualize
2. **Choose Channel**: Select a specific channel index
3. **Choose Objective**:
    - **Center Neuron**: More localized and closer to classic Lucid neuron renders
    - **Full Channel**: Faster and often cleaner in-browser
    - **ImageNet Class**: Optimize directly against the final classifier output
4. **Adjust Settings** (optional):
   - **Steps**: Number of optimization iterations (default: 128)
   - **Learning Rate**: Adam learning rate (default: 0.05)
   - **Regularization Weights**: Fine-tune different penalties
5. **Click the visualize button** to start the optimization
6. **Download** the resulting visualization

## Technical Details

### Fourier Parameterization
Instead of optimizing pixels directly, the app learns Fourier coefficients:
- Covers the full spectrum at 299 px, scaling each frequency by 1/f so low frequencies take larger steps, as in Lucid's `param.image(fft=True)`
- Computes the inverse FFT as matrix products, since tfjs can't backpropagate through its FFT ops
- Keeps the image in the model's expected `[0, 1]` range
- Applies Lucid's color decorrelation (the square root of ImageNet's color covariance) before the final sigmoid

### Optimization Process
1. Initialize the spectrum with small noise (a near-uniform gray image) at 299 px
2. For each step, with Adam:
   - Pad, jitter, randomly scale, and jitter the rendered image again
   - Maximize either the selected channel's center neuron, the full channel map, or a final ImageNet class output
   - Apply L2 and total-variation penalties on the rendered image
3. Render the final result at 512x512 for display and download

### Implementation Stack
- **TensorFlow.js**: Neural network operations and model execution
- **InceptionV3**: Pretrained model from TensorFlow Hub
- **Robust ResNet-50**: PyTorch checkpoint exported to float16 TensorFlow.js weights
- **WebGPU Backend**: GPU acceleration in the browser, falling back to WebGL
- **No Build Process**: Pure JavaScript, runs directly in browser

## Interesting Targets to Try

### Mixed_6e (Late Layer)
- Channel 0-100: Often shows animal-like features
- Channel 200-300: Architectural and geometric patterns
- Channel 400-500: Text and symbol-like patterns
- Channel 600-767: More abstract object parts and motifs

### Mixed_6c (Mid Layer)
- Channel 0-100: Basic textures and patterns
- Channel 300-400: Repeating geometric structures
- Channel 500-600: Curved and organic shapes

### Final ImageNet Classes
Class numbers follow the 1001-entry label file, where 0 is "background" (the robust ResNet has no background class, so its range starts at 1). Search by name in class mode, or try:
- Class 282: tabby
- Class 389: giant panda
- Class 949: Granny Smith
- Class 1000: toilet tissue (try it on the robust model)

## Tips for Best Results

1. **Start with defaults**: The default settings are tuned for good results
2. **Experiment with channels**: Different channels show vastly different patterns
3. **Try different layers**: Earlier layers show simpler patterns, later layers show more complex features
4. **Adjust regularization**:
   - Increase TV weight for less noisy patterns
   - Decrease L2 weight for more vibrant colors

## Differences from DeepDream

| Aspect | Lucid (This Tool) | DeepDream |
|--------|------------------|-----------|
| **Goal** | Understand targeted units/channels/classes | Enhance all features |
| **Initialization** | Fourier basis | Existing image |
| **Optimization** | Maximize one neuron, one channel, or one class | Maximize all activations |
| **Result** | Clean, isolated patterns | Psychedelic, enhanced image |
| **Use Case** | Scientific visualization | Artistic effect |

## Browser Requirements

- Modern browser with WebGPU or WebGL support
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
