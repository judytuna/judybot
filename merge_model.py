#!/usr/bin/env python3
"""
Step 1: Merge LoRA adapter with base model to create standalone model.
This is more reliable than trying to export directly from Unsloth.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

def merge_lora_model():
    """Merge LoRA adapter with base model."""
    print("🔧 Merging LoRA adapter with base model...")
    print("=" * 40)

    try:
        print("📦 Loading base model...")
        # Load the EXACT same base model that was used for training
        base_model_name = "unsloth/phi-3.5-mini-instruct-bnb-4bit"
        print(f"🎯 Using: {base_model_name}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        print("🔗 Loading LoRA adapter...")
        # Load your LoRA adapter
        model = PeftModel.from_pretrained(base_model, "./blog-model-unsloth-final/")

        print("🔀 Merging LoRA with base model...")
        # Merge and unload to get a standalone model
        merged_model = model.merge_and_unload()

        print("💾 Saving merged model...")
        # Create output directory
        output_dir = "./blog-model-merged/"
        os.makedirs(output_dir, exist_ok=True)

        # Save the merged model
        merged_model.save_pretrained(output_dir)

        print("📝 Saving tokenizer...")
        # Save tokenizer from your LoRA directory (it has the chat template)
        tokenizer = AutoTokenizer.from_pretrained("./blog-model-unsloth-final/", trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

        print("✅ Merge complete! Files saved to ./blog-model-merged/")
        print("\n📋 Contents:")

        # List what was created
        for item in os.listdir(output_dir):
            file_path = os.path.join(output_dir, item)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  📄 {item} ({size_mb:.1f} MB)")

        return True

    except Exception as e:
        print(f"❌ Error during merge: {e}")
        return False

def test_merged_model():
    """Test the merged model to make sure it works."""
    print("\n🧪 Testing merged model...")

    try:
        # Load the merged model
        model = AutoModelForCausalLM.from_pretrained(
            "./blog-model-merged/",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained("./blog-model-merged/", trust_remote_code=True)

        # Simple test
        test_prompt = "What do you think about programming?"
        print(f"🤖 Testing with: '{test_prompt}'")

        inputs = tokenizer.encode(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(test_prompt):].strip()

        print(f"📝 Response: '{response}'")

        if response and len(response) > 5:
            print("✅ Merged model is working!")
            return True
        else:
            print("⚠️ Response seems short or empty")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 LoRA Model Merger")
    print("=" * 20)
    print("📝 Note: Your GGUF file is LoRA-only (114MB)")
    print("📝 Ollama needs a full merged model, so we'll merge LoRA + base model")
    print()

    # Check if adapter exists
    if not os.path.exists("./blog-model-unsloth-final/"):
        print("❌ LoRA adapter not found at ./blog-model-unsloth-final/")
        return

    # Merge the model
    if merge_lora_model():
        print("\n🎉 Model merge successful!")

        # Test it
        if test_merged_model():
            print("\n✅ Ready for next step!")
            print("Now you can:")
            print("1. Use the merged model directly from ./blog-model-merged/")
            print("2. Convert to GGUF for Ollama with: python export_merged_to_ollama.py")
        else:
            print("\n⚠️ Merge completed but testing had issues")
    else:
        print("\n❌ Merge failed")

if __name__ == "__main__":
    main()