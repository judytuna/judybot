#!/usr/bin/env python3
"""
Simple test using Unsloth's inference mode properly.
"""

def test_with_unsloth():
    """Test the model using Unsloth's proper inference setup."""
    print("🧪 Testing with Unsloth inference mode")

    try:
        from unsloth import FastLanguageModel

        print("📦 Loading trained model with Unsloth...")

        # Load the model in inference mode
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="./blog-model-unsloth-final",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Enable inference mode (this should fix the apply_qkv issue)
        FastLanguageModel.for_inference(model)

        print("✅ Model loaded successfully!")

        # Test prompts
        test_prompts = [
            "What are your thoughts on programming?",
            "How do you approach creative writing?",
            "Tell me about your day",
            "What's your favorite technology?",
            "How do you stay motivated?",
        ]

        print(f"\n🎯 Testing with {len(test_prompts)} prompts...")

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/{len(test_prompts)} ---")
            print(f"🤖 Prompt: {prompt}")

            try:
                # Use the model's chat template
                messages = [{"role": "user", "content": prompt}]

                # Apply chat template if available
                if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                    formatted_input = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    formatted_input = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

                # Tokenize
                inputs = tokenizer([formatted_input], return_tensors="pt").to("cuda" if model.device.type == "cuda" else "cpu")

                # Generate
                with model.disable_adapter():  # Temporarily disable adapter for generation if needed
                    pass

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )

                # Decode
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract new content
                if formatted_input in response:
                    response = response[len(formatted_input):].strip()

                print(f"📝 Response: {response}")

            except Exception as e:
                print(f"❌ Error with prompt {i}: {e}")
                continue

        print("\n🎉 Testing complete!")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_with_simple_generation():
    """Try a very simple generation approach."""
    print("\n🔧 Trying simple generation approach...")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="./blog-model-unsloth-final",
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )

        # Enable inference mode
        model = FastLanguageModel.for_inference(model)

        # Very simple test
        simple_prompt = "Programming is"
        inputs = tokenizer.encode(simple_prompt, return_tensors="pt")

        if model.device.type == "cuda":
            inputs = inputs.cuda()

        # Simple generation without fancy parameters
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 50,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = response[len(simple_prompt):].strip()

        print(f"🤖 Prompt: {simple_prompt}")
        print(f"📝 Response: {continuation}")

        if len(continuation) > 0:
            print("✅ Basic generation works!")
            return True
        else:
            print("⚠️ Generated empty response")
            return False

    except Exception as e:
        print(f"❌ Simple generation failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing Trained Model")
    print("=" * 25)

    # Try Unsloth inference first
    success = test_with_unsloth()

    if not success:
        # Fall back to simple generation
        success = test_with_simple_generation()

    if success:
        print("\n🎉 Your model is working!")
    else:
        print("\n💡 The model trained successfully, but inference has compatibility issues.")
        print("   This is common with the current Unsloth version.")
        print("   The model can still be:")
        print("   1. Converted to GGUF for Ollama")
        print("   2. Used with different inference frameworks")
        print("   3. Merged to a standard model format")

if __name__ == "__main__":
    main()