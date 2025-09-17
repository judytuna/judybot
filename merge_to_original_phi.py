#!/usr/bin/env python3
"""
Try merging the quantized LoRA onto the original Microsoft Phi-3.5-mini-instruct.
This might work better than quantized-to-quantized merge.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

def merge_to_original_phi():
    """Merge LoRA adapter with original Microsoft Phi-3.5."""
    print("🔧 Merging LoRA (quantized) with Original Microsoft Phi-3.5")
    print("=" * 60)
    print("⚠️  Experimental: Cross-quantization merge")
    print()

    try:
        print("📦 Loading original Microsoft Phi-3.5-mini-instruct...")
        # Load the ORIGINAL Microsoft model (full precision)
        base_model_name = "microsoft/Phi-3.5-mini-instruct"
        print(f"🎯 Using: {base_model_name}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        print("🔗 Loading LoRA adapter (trained on quantized version)...")
        # Load your LoRA adapter that was trained on the quantized version
        try:
            model = PeftModel.from_pretrained(base_model, "./blog-model-unsloth-final/")
        except Exception as e:
            print(f"❌ LoRA loading failed: {e}")
            print("💡 This might be due to architecture mismatch")
            return False

        print("🔀 Merging LoRA with original model...")
        # Merge and unload to get a standalone model
        merged_model = model.merge_and_unload()

        print("💾 Saving merged model...")
        # Create output directory
        output_dir = "./blog-model-merged-original/"
        os.makedirs(output_dir, exist_ok=True)

        # Save the merged model
        merged_model.save_pretrained(output_dir)

        print("📝 Saving tokenizer...")
        # Use tokenizer from the LoRA directory (has chat template)
        tokenizer = AutoTokenizer.from_pretrained("./blog-model-unsloth-final/", trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

        print("✅ Cross-quantization merge complete!")
        print(f"📁 Files saved to {output_dir}")

        # List what was created
        print("\n📋 Contents:")
        for item in os.listdir(output_dir):
            file_path = os.path.join(output_dir, item)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  📄 {item} ({size_mb:.1f} MB)")

        return True

    except Exception as e:
        print(f"❌ Error during cross-quantization merge: {e}")
        return False

def test_original_merge():
    """Test the cross-quantization merged model."""
    print("\n🧪 Testing Cross-Quantization Merged Model")
    print("=" * 45)

    try:
        # Load the merged model
        model = AutoModelForCausalLM.from_pretrained(
            "./blog-model-merged-original/",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained("./blog-model-merged-original/", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_prompts = [
            "Hello, how are you?",
            "What do you think about programming?",
            "Tell me about yourself"
        ]

        print("🔍 Testing prompts...\n")

        for i, prompt in enumerate(test_prompts, 1):
            print(f"--- Test {i}/{len(test_prompts)} ---")
            print(f"🤖 Prompt: '{prompt}'")

            # Tokenize with attention mask
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            print(f"📝 Response: '{response}'")

            # Quality check
            garbled_indicators = ['ins.', '[', ']', 'atteaksendari', 'silph', 'grillier', 'junqua', 'booksheds']
            if any(weird in response.lower() for weird in garbled_indicators):
                print("❌ Response contains garbled text")
            elif len(response.split()) < 3:
                print("⚠️ Response too short")
            elif len(response) > 150:
                print("⚠️ Response too long")
            else:
                print("✅ Response looks reasonable!")

            print()

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Cross-Quantization LoRA Merge Experiment")
    print("=" * 45)

    # Check if adapter exists
    if not os.path.exists("./blog-model-unsloth-final/"):
        print("❌ LoRA adapter not found at ./blog-model-unsloth-final/")
        return

    # Try the cross-quantization merge
    if merge_to_original_phi():
        print("\n🎉 Cross-quantization merge successful!")

        # Test it
        if test_original_merge():
            print("\n✅ Ready for Ollama export!")
            print("If quality looks good, run:")
            print("  # Update export script to use ./blog-model-merged-original/")
            print("  python export_merged_to_ollama.py")
        else:
            print("\n⚠️ Merge completed but testing had issues")
    else:
        print("\n❌ Cross-quantization merge failed")
        print("💡 The LoRA might be incompatible with the original model")

if __name__ == "__main__":
    main()