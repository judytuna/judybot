#!/usr/bin/env python3
"""
Setup script for enhanced blog training with Unsloth and optimizations.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}")
    print(f"Running: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU training (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed yet")
        return False

def install_requirements():
    """Install basic requirements."""
    print("📦 Installing basic requirements...")

    # Install core packages
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt",
                      "Installing core requirements"):
        return False

    return True

def install_unsloth():
    """Install Unsloth for faster training."""
    print("🚀 Installing Unsloth for faster training...")

    # Check if we should install CUDA or CPU version
    has_cuda = check_cuda()

    if has_cuda:
        cmd = f'{sys.executable} -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
    else:
        cmd = f'{sys.executable} -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'

    return run_command(cmd, "Installing Unsloth")

def install_flash_attention():
    """Install flash attention for better performance."""
    print("⚡ Installing Flash Attention...")

    # Flash attention requires CUDA
    if not check_cuda():
        print("⚠️  Skipping Flash Attention (requires CUDA)")
        return True

    cmd = f"{sys.executable} -m pip install flash-attn --no-build-isolation"
    success = run_command(cmd, "Installing Flash Attention")

    if not success:
        print("⚠️  Flash Attention installation failed - this is optional")
        print("   Training will work without it, just slightly slower")

    return True  # Don't fail setup if flash-attn fails

def verify_installation():
    """Verify the installation works."""
    print("\n🔍 Verifying installation...")

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")

        import transformers
        print(f"✅ Transformers {transformers.__version__}")

        import datasets
        print(f"✅ Datasets {datasets.__version__}")

        try:
            import unsloth
            print("✅ Unsloth available")
        except ImportError:
            print("⚠️  Unsloth not available")

        try:
            import flash_attn
            print("✅ Flash Attention available")
        except ImportError:
            print("⚠️  Flash Attention not available")

        print("\n🎉 Installation verification complete!")
        return True

    except ImportError as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Enhanced Blog Training Environment")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False

    print(f"✅ Python {sys.version}")

    # Install components
    steps = [
        ("Basic Requirements", install_requirements),
        ("Unsloth (Fast Training)", install_unsloth),
        ("Flash Attention (Optional)", install_flash_attention),
        ("Verification", verify_installation),
    ]

    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            return False

    print("\n" + "="*50)
    print("🎉 Setup complete! You can now run:")
    print("   python train_model_unsloth.py")
    print("\nFor Ollama export, the model will be automatically")
    print("prepared in GGUF format after training.")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)