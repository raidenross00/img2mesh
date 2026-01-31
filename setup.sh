#!/usr/bin/env bash
set -e

echo "=== img2mesh Setup ==="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install TripoSR
if [ ! -d "TripoSR" ]; then
    echo "Cloning TripoSR..."
    git clone https://github.com/VAST-AI-Research/TripoSR.git
fi

echo "Installing TripoSR..."
pip install -e TripoSR/

# Create outputs directory
mkdir -p outputs

echo ""
echo "=== Setup complete ==="
echo "Run with: source venv/bin/activate && python app.py"
