#!/bin/bash
echo "🚀 MyNeuralKingdom - RunPod Installation Script"
echo "================================================"

# Create directories
echo "📁 Creating directories..."
mkdir -p /runpod-volume/models
mkdir -p /runpod-volume/cache
mkdir -p /runpod-volume/outputs
mkdir -p /runpod-volume/data
mkdir -p /runpod-volume/temp

# Set environment variables
echo "🌍 Setting environment variables..."
export TRANSFORMERS_CACHE="/runpod-volume/cache"
export HF_HOME="/runpod-volume/cache"
export TORCH_HOME="/runpod-volume/cache"
export CUDA_VISIBLE_DEVICES="0"

# Install requirements
echo "📦 Installing Python packages..."
pip install --upgrade pip

# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers[torch] transformers accelerate
pip install xformers --index-url https://download.pytorch.org/whl/cu121

# Image processing
pip install mediapipe opencv-python pillow safetensors
pip install insightface onnxruntime-gpu

# Utilities
pip install tqdm requests numpy psutil

# Jupyter and visualization
pip install jupyter matplotlib

echo "✅ Installation completed!"
echo ""
echo "🔍 Checking GPU..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

echo ""
echo "🎉 MyNeuralKingdom environment ready!"
echo "💡 Next steps:"
echo "   1. Upload your code to /workspace/"
echo "   2. Run: python runpod_test.py"
echo "   3. Start Jupyter: jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root"