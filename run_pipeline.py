#!/usr/bin/env python3
"""
Complete pipeline to extract blog data, format it, validate it, and train a model.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False

def check_requirements():
    """Check if all required files exist."""
    required_files = [
        "extract_blog_data.py",
        "create_training_data.py",
        "train_model.py",
        "validate_data.py",
        "requirements.txt"
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        return False

    print("✅ All required files present")
    return True

def main():
    print("🚀 BLOG SLM TRAINING PIPELINE")
    print("This will extract your blog data and train a model that sounds like you!")

    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please ensure all files are present.")
        return

    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        print("❌ Failed to install requirements. Please install manually.")
        return

    # Step 1: Extract blog data
    if not run_command(f"{sys.executable} extract_blog_data.py", "Extracting blog content"):
        print("❌ Failed to extract blog data. Check that ../judytuna-jekyll exists.")
        return

    # Step 2: Create training data
    if not run_command(f"{sys.executable} create_training_data.py", "Creating conversational training data"):
        print("❌ Failed to create training data.")
        return

    # Step 3: Validate data
    if not run_command(f"{sys.executable} validate_data.py", "Validating training data quality"):
        print("⚠️  Data validation failed, but continuing...")

    # Step 4: Train model
    print(f"\n{'='*60}")
    print("READY TO TRAIN MODEL")
    print(f"{'='*60}")
    print("Training will start now. This may take several hours.")
    print("You can monitor progress in the terminal output.")

    response = input("\nProceed with training? (y/n): ").lower().strip()
    if response == 'y':
        if not run_command(f"{sys.executable} train_model.py", "Training the model"):
            print("❌ Training failed.")
            return
    else:
        print("Training skipped. You can run train_model.py manually when ready.")

    print(f"\n🎉 PIPELINE COMPLETE!")
    print("Your blog-trained model should be ready in ./blog-model-final/")
    print("\nTo test your model:")
    print("  python -c \"from train_model import BlogModelTrainer; t=BlogModelTrainer(); t.test_model('How are you feeling today?')\"")

if __name__ == "__main__":
    main()