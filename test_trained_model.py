#!/usr/bin/env python3
"""
Test the trained model with a workaround for Unsloth inference issues.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def test_model_standard_transformers(model_dir="./blog-model-unsloth-final"):
    """Test the model using standard transformers (more reliable for inference)."""
    print(f"🧪 Testing trained model from {model_dir}")

    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return

    try:
        print("📦 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        print("🧠 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        print("✅ Model loaded successfully!")

        # Test prompts
        test_prompts = [
            "What are your thoughts on programming?",
            "How do you approach creative writing?",
            "Tell me about your day",
            "What's your favorite programming language?",
            "How do you stay motivated?",
        ]

        print(f"\n🎯 Testing with {len(test_prompts)} prompts...")

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/{len(test_prompts)} ---")
            print(f"🤖 Prompt: {prompt}")

            try:
                # Format prompt
                formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

                # Tokenize
                inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.cuda()

                # Generate with attention mask to fix the warning
                attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=inputs.device)

                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_new_tokens=150,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # Decode response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(formatted_prompt):].strip()

                print(f"📝 Response: {response}")

            except Exception as e:
                print(f"❌ Error generating response: {e}")
                continue

        print("\n🎉 Testing complete!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    return True

def check_model_files(model_dir="./blog-model-unsloth-final"):
    """Check what files were saved."""
    print(f"📁 Checking model files in {model_dir}...")

    if not os.path.exists(model_dir):
        print(f"❌ Model directory doesn't exist: {model_dir}")
        return False

    files = os.listdir(model_dir)
    print(f"📄 Found {len(files)} files:")
    for file in sorted(files):
        file_path = os.path.join(model_dir, file)
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  {file} ({size:.1f} MB)")

    # Check for essential files
    essential_files = ["config.json", "tokenizer.json", "adapter_model.safetensors"]
    missing = []
    for file in essential_files:
        if file not in files:
            # Check for alternatives
            if file == "adapter_model.safetensors":
                if not any(f.endswith(".safetensors") for f in files):
                    missing.append(file)
            else:
                missing.append(file)

    if missing:
        print(f"⚠️  Missing files: {missing}")
    else:
        print("✅ All essential files present")

    return len(missing) == 0

def main():
    """Main test function."""
    print("🧪 Trained Model Testing")
    print("=" * 30)

    # Check files first
    if not check_model_files():
        print("❌ Model files incomplete")
        return False

    print("\n" + "=" * 30)

    # Test the model
    return test_model_standard_transformers()

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 If the model doesn't work, you can still use it by:")
        print("  1. Converting to GGUF format for Ollama")
        print("  2. Using it with the original training script")
        print("  3. Merging the LoRA weights back to the base model")