#!/bin/bash
# CalibDrive RunPod Setup Script
# Run this on a RunPod instance to set up the experiment environment.
#
# Recommended RunPod config:
#   - GPU: A100 80GB or H100
#   - Image: runpod/pytorch:2.2.0-py3.10-cuda12.1.0-devel-ubuntu22.04
#   - Disk: 100GB+

set -e

echo "=== CalibDrive RunPod Setup ==="
echo "Setting up experiment environment..."

# Clone repo
cd /workspace
if [ ! -d "Vizuara-VLA-Research" ]; then
    git clone https://github.com/sreedath/Vizuara-VLA-Research.git
fi
cd Vizuara-VLA-Research

# Install dependencies
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets safetensors
pip install numpy scipy scikit-learn pandas pyyaml tqdm wandb einops pillow
pip install netcal
pip install paperbanana

# Install NAVSIM
cd /workspace
if [ ! -d "navsim" ]; then
    git clone https://github.com/autonomousvision/navsim.git
    cd navsim
    pip install -e .
    cd /workspace
fi

# Download NAVSIM data (if not already present)
mkdir -p ~/.cache/navsim
if [ ! -d "$HOME/.cache/navsim/logs" ]; then
    echo "Downloading NAVSIM data..."
    echo "NOTE: You may need to run huggingface-cli login first"
    huggingface-cli download --repo-type dataset autonomousvision/navsim_logs \
        --local-dir ~/.cache/navsim/logs || \
        echo "NAVSIM download failed. Please download manually."
fi

# Pre-download OpenVLA model
python3 -c "
from transformers import AutoProcessor, AutoModelForVision2Seq
print('Downloading OpenVLA-7B...')
try:
    processor = AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
    print('OpenVLA-7B downloaded successfully!')
except Exception as e:
    print(f'OpenVLA download failed: {e}')
    print('Will use alternative model or synthetic data.')
"

echo ""
echo "=== Setup Complete ==="
echo "To run experiments:"
echo "  cd /workspace/Vizuara-VLA-Research"
echo "  PYTHONPATH=. python3 scripts/run_calibdrive_benchmark.py"
echo "  PYTHONPATH=. python3 scripts/run_gpu_experiments.py"
