#!/usr/bin/env python3
"""
Test the LoRA-trained model by loading the base model + adapter.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def test_lora_model(
    base_model_name="microsoft/Phi-3.5-mini-instruct",
    adapter_path="./blog-model-unsloth-final"
):
    """Test the LoRA model by loading base + adapter."""
    print(f"🧪 Testing LoRA model")
    print(f"  Base model: {base_model_name}")
    print(f"  Adapter: {adapter_path}")

    if not os.path.exists(adapter_path):
        print(f"❌ Adapter directory not found: {adapter_path}")
        return False

    try:
        print("\n📦 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("🧠 Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        print("🔧 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, adapter_path)

        print("🚀 Merging adapter for inference...")
        model = model.merge_and_unload()  # Merge LoRA weights for faster inference

        print("✅ Model loaded successfully!")

        # Test prompts based on your training data themes
        test_prompts = [
            "What are your thoughts on programming?",
            "How do you approach creative projects?",
            "Tell me about a recent experience",
            "What's your opinion on technology?",
            "How do you stay motivated when working?",
            "What advice would you give about learning?",
        ]

        print(f"\n🎯 Testing with {len(test_prompts)} prompts...")

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/{len(test_prompts)} ---")
            print(f"🤖 Prompt: {prompt}")

            try:
                # Format prompt (using Phi-3.5 format)
                formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

                # Tokenize
                inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.cuda()

                # Create attention mask
                attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=inputs.device)

                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_new_tokens=200,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        early_stopping=True
                    )

                # Decode response
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract just the new part
                if "<|assistant|>" in full_response:
                    response = full_response.split("<|assistant|>")[-1].strip()
                else:
                    response = full_response[len(formatted_prompt):].strip()

                # Clean up any remaining special tokens
                response = response.replace("<|end|>", "").strip()

                print(f"📝 Response: {response}")

                # Check for quality indicators
                if len(response) < 10:
                    print("⚠️  Short response - may need adjustment")
                elif len(response.split()) > 150:
                    print("⚠️  Very long response - consider shorter max_new_tokens")
                else:
                    print("✅ Good response length")

            except Exception as e:
                print(f"❌ Error generating response: {e}")
                continue

        print("\n🎉 Testing complete!")

        # Memory cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def check_adapter_files(adapter_path="./blog-model-unsloth-final"):
    """Check adapter files."""
    print(f"📁 Checking adapter files in {adapter_path}...")

    if not os.path.exists(adapter_path):
        print(f"❌ Adapter directory doesn't exist: {adapter_path}")
        return False

    files = os.listdir(adapter_path)
    print(f"📄 Found {len(files)} files:")
    for file in sorted(files):
        file_path = os.path.join(adapter_path, file)
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  {file} ({size:.1f} MB)")

    # Check for LoRA files
    has_adapter_config = "adapter_config.json" in files
    has_adapter_model = any(f.startswith("adapter_model") for f in files)
    has_tokenizer = any(f.startswith("tokenizer") for f in files)

    print(f"\n📋 Adapter status:")
    print(f"  ✅ Adapter config: {has_adapter_config}")
    print(f"  ✅ Adapter weights: {has_adapter_model}")
    print(f"  ✅ Tokenizer: {has_tokenizer}")

    if has_adapter_config and has_adapter_model:
        print("✅ LoRA adapter looks complete!")
        return True
    else:
        print("❌ LoRA adapter incomplete")
        return False

def main():
    """Main test function."""
    print("🧪 LoRA Model Testing")
    print("=" * 25)

    # Check adapter files
    if not check_adapter_files():
        return False

    print("\n" + "=" * 40)

    # Test the model
    success = test_lora_model()

    if success:
        print("\n🎉 Your blog style model is working!")
        print("💡 The model should now generate text in your writing style.")
    else:
        print("\n❌ Model testing failed")
        print("💡 The training was successful, but there may be compatibility issues.")

    return success

if __name__ == "__main__":
    main()