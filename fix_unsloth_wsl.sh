#!/bin/bash
# Fix Unsloth in WSL by installing CUDA development headers

echo "🔧 Fixing Unsloth CUDA compilation in WSL"
echo "=========================================="

# Check if we're in WSL
if [[ $(uname -r) == *microsoft* ]]; then
    echo "✅ WSL detected"
else
    echo "❌ This script is for WSL only"
    exit 1
fi

# Check CUDA installation
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA toolkit found: $(nvcc --version | grep release)"
else
    echo "⚠️  CUDA toolkit not found, installing..."

    # Add NVIDIA package repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update

    # Install CUDA toolkit
    sudo apt-get install -y cuda-toolkit-12-3
fi

# Install development headers that Unsloth needs
echo "📦 Installing CUDA development headers..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    cuda-nvcc-12-3 \
    cuda-cudart-dev-12-3 \
    cuda-driver-dev-12-3 \
    libcuda1-545 \
    nvidia-cuda-dev

# Set environment variables
echo "🔧 Setting up environment variables..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Make environment variables permanent
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

# Create symlink if needed
if [ ! -f /usr/local/cuda/include/cuda.h ]; then
    echo "🔗 Creating CUDA header symlinks..."
    sudo mkdir -p /usr/local/cuda/include
    sudo ln -sf /usr/include/cuda.h /usr/local/cuda/include/cuda.h
    sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so /usr/local/cuda/lib64/libcuda.so
fi

echo "✅ CUDA headers setup complete!"
echo "Now restart your shell and try installing Unsloth again:"
echo "  pip uninstall unsloth -y"
echo "  pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""