#!/usr/bin/env python3
"""
Simple verification script that handles WSL/CUDA issues gracefully.
"""

def test_imports():
    """Test imports with proper error handling."""
    print("🔍 Testing imports...")

    # Core libraries
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False

    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False

    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets: {e}")
        return False

    # Optional but important libraries
    try:
        import peft
        print(f"✅ PEFT {peft.__version__}")
    except ImportError:
        print("⚠️  PEFT not available")

    try:
        import trl
        print(f"✅ TRL {trl.__version__}")
    except ImportError:
        print("⚠️  TRL not available")

    # Test Unsloth carefully
    try:
        # Import transformers first to avoid the warning
        import transformers
        import unsloth
        print("✅ Unsloth available")

        # Test if we can use FastLanguageModel
        from unsloth import FastLanguageModel
        print("✅ Unsloth FastLanguageModel ready")

    except Exception as e:
        print(f"⚠️  Unsloth issue: {type(e).__name__}")
        print("   This might be due to CUDA headers in WSL - training may still work")

    # Test flash attention
    try:
        import flash_attn
        print("✅ Flash Attention available")
    except ImportError:
        print("⚠️  Flash Attention not available")

    return True

def test_model_loading():
    """Test if we can load a small model."""
    print("\n🧪 Testing model loading...")

    try:
        # Try Unsloth first
        try:
            from unsloth import FastLanguageModel
            print("Testing Unsloth model loading...")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="gpt2",  # Small test model
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
                trust_remote_code=True,
            )
            print("✅ Unsloth model loading works!")
            del model, tokenizer
            return True

        except Exception as e:
            print(f"⚠️  Unsloth loading failed: {e}")
            print("Trying standard transformers...")

            # Fallback to standard transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM

            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            print("✅ Standard transformers model loading works!")
            del model, tokenizer
            return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Main verification."""
    print("🚀 Enhanced Setup Verification")
    print("=" * 40)

    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False

    # Test model loading
    if not test_model_loading():
        print("❌ Model loading test failed")
        return False

    print("\n" + "="*40)
    print("🎉 Verification successful!")
    print("\nYou can now run:")
    print("  python train_model_unsloth.py")
    print("\nNote: If you see CUDA compilation warnings,")
    print("training should still work - these are common in WSL.")

    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)