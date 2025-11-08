# Stick Balancing - Training Code

Python training environment and scripts for the inverted pendulum swingup task.

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_sac_final.py

# Test the trained model
python train_sac_final.py --test

# Export to TensorFlow.js for web
python export_sac_to_tfjs.py
```

## Files

- `stick_env.py` - Gymnasium environment implementing the stick balancing physics
- `train_sac_final.py` - SAC training script (successful 1M step approach)
- `export_sac_to_tfjs.py` - Export trained SAC model to TensorFlow.js format
- `requirements.txt` - Python dependencies

## Training Results

The final SAC model achieved:
- **87.7% balance rate** during training
- **76-82% balance rate** in testing
- Trained in exactly **1,000,000 steps** (~100 minutes)

### Key Techniques

1. **Algorithm**: SAC (Soft Actor-Critic) for better sample efficiency
2. **Reward Shaping**: Combined upright, energy, and height rewards
3. **Observation Normalization**: Stable training with VecNormalize
4. **Network**: [256, 256] deep network for complex control

## Environment

**Observation Space** (4D):
- Cart position: [-2.4, 2.4] meters
- Cart velocity: [-6, 6] m/s
- Stick angle: [-π, π] radians (π = down, 0 = up)
- Angular velocity: unbounded rad/s

**Action Space**:
- Continuous: [-1, 1] mapped to target velocity [-5, 5] m/s

**Reward**:
- Upright: 5.0 × cos(angle)
- Energy: 1.0 × energy_term
- Height: 2.0 × height_term
- Penalty: -0.001 × (v² + ω²)
