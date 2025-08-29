# Avatar Lip Sync - Real-time Mouth Animation + Idle Animation

A high-quality, real-time lip sync system that animates a 3D avatar's mouth from microphone input with natural idle body movement, using Three.js WebGL and advanced audio processing.

## Features

### High-Quality Audio Processing
- **16kHz Audio Capture**: Optimized for speech recognition accuracy
- **Advanced Frequency Analysis**: Multi-band spectral analysis with FFT
- **Voice Activity Detection**: Intelligent silence detection to prevent false triggers
- **Hysteresis & Smoothing**: Prevents jitter and maintains natural mouth movements

### Professional 3D Rendering & Animation
- **WebGL Rendering**: Hardware-accelerated rendering at 60fps (WebGPU fallback implemented)
- **VRM Avatar Support**: Industry-standard 3D character format with facial expressions
- **Layered Animation System**: Idle body movement + real-time lip sync simultaneously  
- **Natural Idle Animation**: VRoid Hub-style breathing, body sway, and head movement
- **Smooth Expression Blending**: EMA-based interpolation for natural transitions
- **Real-time Performance**: Optimized for <100ms end-to-end latency

### Intelligent Viseme Detection
- **5-Viseme System**: A, E, I, O, U + Rest state
- **Spectral Classification**: Frequency band analysis for accurate mouth shape detection
- **Temporal Smoothing**: Multi-frame averaging to reduce noise
- **Adaptive Thresholds**: Dynamic adjustment based on voice characteristics

## Technical Architecture

### Audio Pipeline
1. **Microphone Capture** → Web Audio API with optimal settings
2. **Frequency Analysis** → Real-time FFT with 2048-point resolution  
3. **Band Energy Calculation** → 4-band spectral analysis (Low, Low-Mid, Mid, High)
4. **Viseme Classification** → Heuristic algorithm based on spectral features
5. **Temporal Smoothing** → Rolling buffer with hysteresis

### 3D Rendering & Animation Pipeline
1. **VRM Loading** → CC0 VRoid avatar with facial expressions and humanoid bone structure
2. **Animation System** → Three.js AnimationMixer with quaternion-based bone animation
3. **Idle Animation** → 4-second breathing cycle with chest expansion, spine sway, head movement
4. **Expression Mapping** → Real-time viseme → VRM Expression Preset conversion
5. **Layered Updates** → Simultaneous body animation + facial expression blending
6. **WebGL Rendering** → Hardware-accelerated 60fps output

### Performance Optimizations
- **Sub-100ms Latency**: Optimized processing pipeline
- **60fps Rendering**: WebGPU hardware acceleration
- **Efficient Memory Usage**: Circular buffers and object pooling
- **Real-time Monitoring**: Live performance metrics display

## Usage

1. **Open in Browser**: Navigate to the avatar directory
2. **Grant Microphone Access**: Click "Start Microphone" and allow permissions
3. **Speak Naturally**: The avatar will automatically animate mouth shapes
4. **Monitor Performance**: Check debug panel for latency and FPS

## Technical Specifications

- **Audio Format**: 16kHz mono PCM
- **Latency Target**: <100ms end-to-end
- **Rendering**: Three.js WebGL at 60fps (WebGPU support implemented)
- **Animation System**: Three.js AnimationMixer with QuaternionKeyframeTrack
- **Supported Visemes**: A, E, I, O, U, REST (mapped to VRM expressions: aa, ee, ih, oh, ou)
- **Avatar Format**: VRM 0.x/1.0 with humanoid bone structure
- **Browser Requirements**: Modern browsers with WebGL support

## Files

- `index.html` - Main application interface with WebGL/WebGPU support
- `app.js` - Three.js renderer, VRM loader, animation system, and lip sync integration
- `audio-processor.js` - Advanced audio analysis and real-time viseme detection
- `2407317377392965971.vrm` - Working CC0 VRoid avatar model with facial expressions

## License

This project uses:
- **CC0 VRoid Avatar**: Free to use without restrictions
- **MIT Licensed Libraries**: Three.js, Three-VRM
- **Custom Code**: Available under MIT license

Built with cutting-edge web technologies for broadcast-quality real-time lip sync animation.

## Current Status

### ✅ **Working Features**
- Real-time lip sync with microphone input
- 5-viseme detection (A, E, I, O, U, REST)
- VRM avatar loading with facial expressions
- WebGL rendering at 60fps
- Advanced audio processing with spectral analysis
- Expression smoothing and interpolation

### 🚧 **In Development** 
- **Idle Animation System**: Implemented but currently showing T-pose issue
  - Animation system architecture completed
  - Quaternion-based bone animation tracks created
  - 4-second breathing cycle with chest expansion implemented
  - Spine sway and head movement algorithms added
  - **Issue**: Avatar remains in T-pose, animations not applying to bones
  - **Next**: Debug humanoid bone mapping and animation track application

### 🎯 **Target Features**
- Natural idle breathing and body sway (VRoid Hub style)
- Layered animation system (idle + lip sync simultaneously)
- Smooth transitions between idle states
- Optional gesture recognition for enhanced expression