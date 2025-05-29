# GPT TensorFlow.js CLI

Command-line version of the GPT model for testing and training without a browser.

## Installation

```bash
npm install
```

## Usage

### Basic Training
```bash
# Train for 100 epochs (default)
node train_cli.js --train

# Train for specific number of epochs
node train_cli.js --train --epochs 500

# Fresh training (ignore saved model)
node train_cli.js --fresh --train --epochs 100
```

### Generation
```bash
# Generate text using saved model
node train_cli.js --generate

# Train then generate
node train_cli.js --train --epochs 100 --generate

# Interactive mode
node train_cli.js --generate --interactive
```

### NPM Scripts
```bash
npm run train          # Train for 100 epochs
npm run train-long     # Train for 1000 epochs
npm run generate       # Generate text only
npm run interactive    # Interactive generation mode
npm run fresh          # Fresh training from scratch
npm run test           # Quick test (10 epochs + generate)
```

## Options

- `--epochs N`: Number of training epochs (default: 100)
- `--train`: Train the model
- `--generate`: Generate text after training
- `--fresh`: Start fresh training (ignore saved model)
- `--interactive`: Enter interactive generation mode

## Model Details

- Architecture: 2-layer transformer with multi-head attention
- Parameters: ~290K
- Embedding dim: 128
- Attention heads: 4
- Sequence length: 64
- Vocabulary: Character-level (65 unique chars from Shakespeare)

## Saved Model

The model is automatically saved to `./saved_model/` after training and loaded on subsequent runs unless `--fresh` is specified.

## Performance

Training on CPU (Node.js) is slower than GPU (browser). Expect:
- ~1-2 minutes per 100 epochs on modern CPU
- Loss should drop from ~4.5 to ~1.5-2.0 after 300-500 epochs
- Coherent text generation typically requires 1000+ epochs