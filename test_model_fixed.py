#!/usr/bin/env python3
"""
Test the model with better generation parameters and formatting.
"""

def test_model_properly():
    """Test with proper parameters to avoid garbled output."""
    print("🧪 Testing model with improved parameters")

    try:
        from unsloth import FastLanguageModel

        print("📦 Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="./blog-model-unsloth-final",
            max_seq_length=1024,  # Shorter to avoid issues
            dtype=None,
            load_in_4bit=True,
        )

        # Enable inference mode
        model = FastLanguageModel.for_inference(model)
        print("✅ Model loaded and ready for inference")

        # Test with simpler, shorter prompts first
        simple_prompts = [
            "I think programming",
            "Today I'm working on",
            "My favorite thing about coding is",
            "When I write, I usually",
            "The best way to learn",
        ]

        print(f"\n🎯 Testing with {len(simple_prompts)} simple prompts...")

        for i, prompt in enumerate(simple_prompts, 1):
            print(f"\n--- Test {i}/{len(simple_prompts)} ---")
            print(f"🤖 Prompt: \"{prompt}\"")

            try:
                # Use a very simple format - just the prompt
                inputs = tokenizer.encode(prompt, return_tensors="pt")

                if model.device.type == "cuda":
                    inputs = inputs.cuda()

                # Conservative generation parameters
                outputs = model.generate(
                    inputs,
                    max_new_tokens=30,  # Much shorter
                    temperature=0.3,    # Lower temperature for more coherent output
                    do_sample=True,
                    top_p=0.7,          # More focused sampling
                    repetition_penalty=1.2,  # Stronger repetition penalty
                    no_repeat_ngram_size=2,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )

                # Decode
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract just the new part
                if len(full_response) > len(prompt):
                    response = full_response[len(prompt):].strip()
                else:
                    response = full_response

                print(f"📝 Response: \"{response}\"")

                # Quality check
                if len(response.split()) > 25:
                    print("⚠️  Long response - might be repeating")
                elif len(response) < 5:
                    print("⚠️  Very short response")
                elif any(char in response for char in ['<', '>', '|', '[', ']', '{', '}']):
                    print("⚠️  Contains special characters")
                else:
                    print("✅ Response looks good")

            except Exception as e:
                print(f"❌ Error: {e}")
                continue

        return True

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def test_base_model_comparison():
    """Test the base model to compare outputs."""
    print("\n🔍 Testing base model for comparison...")

    try:
        from unsloth import FastLanguageModel

        print("📦 Loading base Phi-3.5 model...")
        base_model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name="microsoft/Phi-3.5-mini-instruct",
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )

        base_model = FastLanguageModel.for_inference(base_model)

        # Test same prompt
        test_prompt = "I think programming"
        inputs = base_tokenizer.encode(test_prompt, return_tensors="pt")

        if base_model.device.type == "cuda":
            inputs = inputs.cuda()

        outputs = base_model.generate(
            inputs,
            max_new_tokens=30,
            temperature=0.3,
            do_sample=True,
            top_p=0.7,
            repetition_penalty=1.2,
            pad_token_id=base_tokenizer.eos_token_id,
            eos_token_id=base_tokenizer.eos_token_id,
        )

        response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(test_prompt):].strip()

        print(f"🤖 Base model response: \"{response}\"")

        if len(response) > 0 and not any(char in response for char in ['<', '>', '|', '[', ']']):
            print("✅ Base model generates coherently")
            return True
        else:
            print("⚠️  Base model also has issues")
            return False

    except Exception as e:
        print(f"❌ Base model test failed: {e}")
        return False

def main():
    """Main testing function."""
    print("🔧 Model Output Diagnostics")
    print("=" * 30)

    # Test our trained model
    trained_works = test_model_properly()

    # Test base model for comparison
    base_works = test_base_model_comparison()

    print("\n" + "=" * 40)
    print("📊 Diagnosis Results:")

    if trained_works and base_works:
        print("✅ Both models work - training successful!")
    elif not trained_works and base_works:
        print("⚠️  Training may have issues, but base model works")
        print("💡 Possible solutions:")
        print("   1. Retrain with different parameters")
        print("   2. Use lower learning rate")
        print("   3. Check training data format")
    elif trained_works and not base_works:
        print("⚠️  Both models have similar issues - likely inference problem")
        print("💡 This might be a compatibility issue with Unsloth")
    else:
        print("❌ Both models have issues - likely environment problem")

if __name__ == "__main__":
    main()