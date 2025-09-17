#!/usr/bin/env python3
"""
Simple working bot with proper memory management and error handling.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

class SimpleBlogBot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load model with proper memory management."""
        print("🔧 Loading model with memory optimizations...")

        try:
            base_model_name = "microsoft/Phi-3.5-mini-instruct"
            adapter_path = "./blog-model-unsloth-final"

            print("📦 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("🧠 Loading base model with CPU offloading...")

            # Create offload directory
            offload_dir = "./offload_cache"
            os.makedirs(offload_dir, exist_ok=True)

            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                offload_folder=offload_dir,  # Fix the offload issue
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            print("🔗 Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

            # Don't merge - keep as LoRA for memory efficiency
            print("✅ Model loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Let's try GPT2-medium instead - it worked well before!")
            return self.load_gpt2_fallback()

    def load_gpt2_fallback(self):
        """Fallback to GPT2-medium which we know works."""
        try:
            print("🔄 Falling back to your working GPT2-medium model...")

            # Check if we have the GPT2 model
            gpt2_path = "./blog-model-final"
            if not os.path.exists(gpt2_path):
                print("❌ GPT2 model not found either")
                return False

            self.tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                gpt2_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            print("✅ GPT2-medium model loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ GPT2 fallback also failed: {e}")
            return False

    def simple_generate(self, prompt, max_tokens=40):
        """Very simple generation to avoid compatibility issues."""
        if not self.model or not self.tokenizer:
            return "Model not loaded"

        try:
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")

            # Move to device
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            # Simple generation without fancy parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # Remove problematic parameters
                )

            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            return response if response else "No response generated"

        except Exception as e:
            return f"Generation failed: {str(e)[:100]}"

    def test_basic(self):
        """Test basic functionality."""
        print("\n🧪 Testing Basic Functionality")
        print("=" * 30)

        test_prompts = [
            "Hello",
            "Programming is",
            "Today I",
        ]

        for prompt in test_prompts:
            print(f"\n🤖 Testing: '{prompt}'")
            response = self.simple_generate(prompt, max_tokens=20)
            print(f"📝 Response: '{response}'")

            # Actual quality check
            if "Error" in response or "failed" in response:
                print("❌ Generation failed")
            elif len(response.split()) > 2 and len(response) < 150:
                print("✅ Response looks reasonable")
            else:
                print("⚠️  Response might have issues")

    def chat(self):
        """Simple chat loop."""
        print("\n💬 Simple Chat")
        print("=" * 15)
        print("Type 'quit' to exit")

        while True:
            user_input = input("\n🧑 You: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                break

            if not user_input:
                continue

            response = self.simple_generate(user_input + " ", max_tokens=30)
            print(f"🤖 Bot: {response}")

def main():
    """Main function."""
    print("🤖 Simple Working Bot")
    print("=" * 20)

    bot = SimpleBlogBot()

    if not bot.model:
        print("❌ Could not load any model")
        return

    while True:
        print("\n" + "="*30)
        print("1. 🧪 Test basic functionality")
        print("2. 💬 Simple chat")
        print("3. ❌ Quit")

        choice = input("\nSelect (1-3): ").strip()

        if choice == '1':
            bot.test_basic()
        elif choice == '2':
            bot.chat()
        elif choice == '3':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()