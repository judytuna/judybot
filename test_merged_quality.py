#!/usr/bin/env python3
"""
Test the quality of the merged model more thoroughly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_merged_model():
    """Test merged model with multiple prompts."""
    print("🧪 Testing Merged Model Quality")
    print("=" * 35)

    try:
        print("📦 Loading merged model...")
        model = AutoModelForCausalLM.from_pretrained(
            "./blog-model-merged/",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained("./blog-model-merged/", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_prompts = [
            "Hello, how are you?",
            "What do you think about programming?",
            "Tell me about yourself",
            "Programming is",
            "I love coding because"
        ]

        print("🔍 Testing multiple prompts...\n")

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
                    max_new_tokens=40,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            print(f"📝 Response: '{response}'")

            # Quality check
            if any(weird in response.lower() for weird in ['ins.', '[', ']', 'atteaksendari', 'silph']):
                print("❌ Response contains garbled text")
            elif len(response.split()) < 3:
                print("⚠️ Response too short")
            elif len(response) > 100:
                print("⚠️ Response too long")
            else:
                print("✅ Response looks reasonable")

            print()

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_merged_model()