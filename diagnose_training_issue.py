#!/usr/bin/env python3
"""
Diagnose what went wrong with the training to cause garbled outputs.
"""

import json
import torch

def test_base_model():
    """Test if the base model works normally."""
    print("🔍 Testing base Phi-3.5 model (without LoRA)...")

    try:
        from unsloth import FastLanguageModel

        # Load clean base model
        base_model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name="microsoft/Phi-3.5-mini-instruct",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )

        base_model = FastLanguageModel.for_inference(base_model)

        # Test with simple prompts
        test_prompts = [
            "What do you think about sailing?",
            "Today I was thinking about space travel",
            "Programming is"
        ]

        for prompt in test_prompts:
            print(f"\n🤖 Base model test: '{prompt}'")

            inputs = base_tokenizer.encode(prompt + " ", return_tensors="pt")
            attention_mask = torch.ones(inputs.shape, dtype=torch.long)

            device = next(base_model.parameters()).device
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = base_model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=30,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.7,
                    repetition_penalty=1.2,
                    pad_token_id=base_tokenizer.eos_token_id,
                    eos_token_id=base_tokenizer.eos_token_id,
                )

            response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt + " "):].strip()

            print(f"📝 Base response: '{response}'")

            # Check if response is coherent
            if len(response) > 5 and not any(char in response for char in ['$', '^', '[', ']', 'http://']):
                print("✅ Base model response looks coherent")
            else:
                print("❌ Base model also produces garbled output")

        return True

    except Exception as e:
        print(f"❌ Base model test failed: {e}")
        return False

def check_training_data_samples():
    """Check a few training data samples for corruption."""
    print("\n🔍 Checking training data samples...")

    try:
        with open("data/train_data.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"📊 Total training examples: {len(lines)}")

        # Check first few examples
        for i in range(min(3, len(lines))):
            try:
                data = json.loads(lines[i])
                assistant_content = data["messages"][1]["content"]

                print(f"\n--- Training Example {i+1} ---")
                print(f"Length: {len(assistant_content)} chars")
                print(f"Preview: {assistant_content[:200]}...")

                # Check for suspicious patterns
                suspicious_patterns = ['$', '^', 'http://wwwop', 'MessageToAwoeatmeepupfe', '[http http]']
                found_suspicious = [pattern for pattern in suspicious_patterns if pattern in assistant_content]

                if found_suspicious:
                    print(f"⚠️  Found suspicious patterns: {found_suspicious}")
                else:
                    print("✅ Content looks normal")

            except Exception as e:
                print(f"❌ Error parsing example {i+1}: {e}")

        return True

    except Exception as e:
        print(f"❌ Could not check training data: {e}")
        return False

def test_minimal_generation():
    """Test the trained model with minimal parameters."""
    print("\n🔍 Testing trained model with minimal parameters...")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="./blog-model-unsloth-final",
            max_seq_length=256,  # Much shorter
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.for_inference(model)

        # Very simple test
        simple_prompt = "Hello"
        inputs = tokenizer.encode(simple_prompt, return_tensors="pt")
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)

        device = next(model.parameters()).device
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=10,  # Very short
                temperature=0.1,    # Very low temperature
                do_sample=False,    # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(simple_prompt):].strip()

        print(f"🤖 Minimal test response: '{response}'")

        if len(response) > 0 and len(response.split()) <= 15:
            print("✅ Minimal generation works")
            return True
        else:
            print("❌ Even minimal generation fails")
            return False

    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        return False

def main():
    """Main diagnostic function."""
    print("🔬 Training Issue Diagnosis")
    print("=" * 35)

    results = {}

    # Test base model
    results["base_model"] = test_base_model()

    # Check training data
    results["training_data"] = check_training_data_samples()

    # Test minimal generation
    results["minimal_generation"] = test_minimal_generation()

    print("\n" + "=" * 50)
    print("📊 Diagnosis Results:")

    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test.replace('_', ' ').title()}: {status}")

    print("\n💡 Recommendations:")

    if not results.get("base_model", False):
        print("❌ Base model issues - problem with environment/setup")
    elif not results.get("training_data", False):
        print("❌ Training data corruption - need to regenerate training data")
    elif not results.get("minimal_generation", False):
        print("❌ Training failed - need to retrain with different parameters")
    else:
        print("⚠️  Model trained but learned poor patterns")
        print("   Solutions:")
        print("   1. Retrain with lower learning rate")
        print("   2. Clean training data more aggressively")
        print("   3. Use shorter sequences during training")
        print("   4. Try different model (e.g., GPT2-medium worked better)")

if __name__ == "__main__":
    main()