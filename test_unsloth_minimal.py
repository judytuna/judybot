#!/usr/bin/env python3
"""
Test Unsloth with minimal functionality to see what works.
"""

def test_unsloth_import():
    """Test if basic Unsloth functionality works."""
    try:
        print("🧪 Testing Unsloth import...")
        import unsloth
        print("✅ Unsloth imported successfully")
        return True
    except Exception as e:
        print(f"❌ Unsloth import failed: {e}")
        return False

def test_fastlanguagemodel_import():
    """Test FastLanguageModel import."""
    try:
        print("🧪 Testing FastLanguageModel import...")
        from unsloth import FastLanguageModel
        print("✅ FastLanguageModel imported successfully")
        return True
    except Exception as e:
        print(f"❌ FastLanguageModel import failed: {e}")
        return False

def test_model_creation():
    """Test creating a model without GPT2 (which has the syntax error)."""
    try:
        print("🧪 Testing model creation with Qwen...")
        from unsloth import FastLanguageModel

        # Try with a different model that might not have the compilation issue
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Smaller, different architecture
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        print("✅ Qwen model creation successful!")
        del model, tokenizer
        return True

    except Exception as e:
        print(f"⚠️  Qwen model failed: {e}")
        return False

def test_phi_model():
    """Test with Phi model which is our target."""
    try:
        print("🧪 Testing Phi-3.5 model...")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="microsoft/Phi-3.5-mini-instruct",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        print("✅ Phi-3.5 model creation successful!")
        del model, tokenizer
        return True

    except Exception as e:
        print(f"⚠️  Phi-3.5 model failed: {e}")
        return False

def main():
    """Run minimal tests to see what works."""
    print("🔬 Unsloth Minimal Testing")
    print("=" * 30)

    tests = [
        ("Unsloth Import", test_unsloth_import),
        ("FastLanguageModel Import", test_fastlanguagemodel_import),
        ("Qwen Model Creation", test_model_creation),
        ("Phi-3.5 Model Creation", test_phi_model),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        results[test_name] = test_func()

    print("\n" + "=" * 30)
    print("📊 Test Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")

    # Determine if we can use Unsloth
    if results.get("Phi-3.5 Model Creation", False):
        print("\n🎉 Unsloth works with Phi-3.5! You can use the enhanced training.")
        return True
    elif results.get("Qwen Model Creation", False):
        print("\n✅ Unsloth works with Qwen! Consider using Qwen instead of Phi-3.5.")
        return True
    elif results.get("FastLanguageModel Import", False):
        print("\n⚠️  Unsloth imports work but model loading fails.")
        print("   The training script will fall back to standard transformers.")
        return False
    else:
        print("\n❌ Unsloth has significant issues. Use standard transformers.")
        return False

if __name__ == "__main__":
    main()