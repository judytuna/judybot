#!/usr/bin/env python3
"""
Simpler approach to fix Unsloth - install CUDA development tools and retry.
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

def install_cuda_dev_tools():
    """Install CUDA development tools."""
    print("📦 Installing CUDA development tools...")

    commands = [
        ("sudo apt-get update", "Updating package list"),
        ("sudo apt-get install -y build-essential", "Installing build tools"),
        ("sudo apt-get install -y nvidia-cuda-dev", "Installing CUDA dev headers"),
        ("sudo apt-get install -y nvidia-cuda-toolkit", "Installing CUDA toolkit"),
    ]

    for cmd, desc in commands:
        if not run_command(cmd, desc):
            print(f"⚠️  {desc} failed, but continuing...")

    return True

def set_cuda_environment():
    """Set CUDA environment variables."""
    print("🔧 Setting CUDA environment...")

    # Find CUDA installation
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/lib/cuda",
        "/opt/cuda"
    ]

    cuda_home = None
    for path in cuda_paths:
        if os.path.exists(path):
            cuda_home = path
            break

    if not cuda_home:
        print("⚠️  CUDA installation not found in standard locations")
        return False

    print(f"Found CUDA at: {cuda_home}")

    # Set environment variables for current session
    os.environ['CUDA_HOME'] = cuda_home
    os.environ['PATH'] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
    os.environ['LD_LIBRARY_PATH'] = f"{cuda_home}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

    print("✅ Environment variables set for current session")
    return True

def reinstall_unsloth():
    """Reinstall Unsloth with proper CUDA support."""
    print("🚀 Reinstalling Unsloth...")

    # Uninstall first
    run_command(f"{sys.executable} -m pip uninstall unsloth -y", "Uninstalling old Unsloth")

    # Clear pip cache
    run_command(f"{sys.executable} -m pip cache purge", "Clearing pip cache")

    # Reinstall with proper environment
    cmd = f'{sys.executable} -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-cache-dir'
    return run_command(cmd, "Reinstalling Unsloth")

def test_unsloth():
    """Test if Unsloth works now."""
    print("🧪 Testing Unsloth...")

    try:
        # Import in clean environment
        import importlib
        if 'unsloth' in sys.modules:
            importlib.reload(sys.modules['unsloth'])

        from unsloth import FastLanguageModel
        print("✅ Unsloth import successful!")

        # Try to load a small model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="gpt2",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        print("✅ Unsloth model loading successful!")

        del model, tokenizer
        return True

    except Exception as e:
        print(f"❌ Unsloth test failed: {e}")
        return False

def main():
    """Main fix function."""
    print("🚀 Fixing Unsloth for WSL")
    print("=" * 30)

    # Check if we're in WSL
    if 'microsoft' not in os.uname().release.lower():
        print("❌ This script is designed for WSL")
        return False

    steps = [
        ("Install CUDA dev tools", install_cuda_dev_tools),
        ("Set CUDA environment", set_cuda_environment),
        ("Reinstall Unsloth", reinstall_unsloth),
        ("Test Unsloth", test_unsloth),
    ]

    for step_name, step_func in steps:
        print(f"\n{'='*10} {step_name} {'='*10}")
        if not step_func():
            print(f"❌ Failed at: {step_name}")
            if step_name == "Test Unsloth":
                print("⚠️  Unsloth installation may have issues, but training might still work")
                print("   You can proceed with the fallback transformers approach")
            return False

    print("\n" + "="*40)
    print("🎉 Unsloth fix complete!")
    print("Try running: python train_model_unsloth.py")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)