#!/bin/bash

# Script to export and deploy trained model
MODEL_NAME="ppo_selfplay_v3"

echo "🚀 Deploying Air Hockey Model: $MODEL_NAME"
echo "=========================================="

# Activate virtual environment
source ../venv/bin/activate

# Export to ONNX
echo "📦 Exporting model to ONNX format..."
python export_to_onnx.py --model "models/${MODEL_NAME}_final.zip"

# Check if export was successful
if [ -f "models/onnx/${MODEL_NAME}_final.zip.onnx" ]; then
    echo "✅ Export successful!"

    # Copy to web app (both model and data files)
    echo "🌐 Deploying to web app..."
    cp "models/onnx/${MODEL_NAME}_final.zip.onnx" "../model/${MODEL_NAME}_final.onnx"
    # Note: data file must keep .zip in name to match internal ONNX reference
    cp "models/onnx/${MODEL_NAME}_final.zip.onnx.data" "../model/${MODEL_NAME}_final.zip.onnx.data"

    echo "✅ Model deployed successfully!"
    echo ""
    echo "🎮 You can now play against the AI at:"
    echo "   file:///Users/elichen/code/elichen.github.io/airhockey/index.html"
else
    echo "❌ Export failed. Please check the model file."
    exit 1
fi