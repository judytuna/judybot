#!/usr/bin/env python3
"""
Basic bot that works - start with base model, then try to add LoRA if possible.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class BasicBot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_base_model()

    def load_base_model(self):
        """Load just the base model first to ensure something works."""
        print("🔧 Loading base model only (no LoRA)...")

        try:
            # Try smaller model first
            model_name = "gpt2-medium"  # We know this works from earlier tests
            print(f"📦 Loading {model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            print("✅ Base model loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Error loading base model: {e}")
            return False

    def generate_text(self, prompt, max_tokens=40):
        """Simple text generation."""
        if not self.model or not self.tokenizer:
            return "Model not loaded"

        try:
            # Simple tokenization
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")

            # Move to device
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            # Generate with very basic parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            return response if response else "No response"

        except Exception as e:
            return f"Error: {e}"

    def test_functionality(self):
        """Test that the bot works at all."""
        print("\n🧪 Testing Basic Functionality")
        print("=" * 30)

        test_prompts = [
            "Hello, I'm",
            "Programming is",
            "Today I was thinking about",
            "My favorite hobby is",
        ]

        working_count = 0

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/{len(test_prompts)} ---")
            print(f"🤖 Prompt: '{prompt}'")

            response = self.generate_text(prompt, max_tokens=30)
            print(f"📝 Response: '{response}'")

            # Check if it actually works
            if "Error" not in response and len(response.split()) > 2:
                print("✅ Working!")
                working_count += 1
            else:
                print("❌ Not working")

        print(f"\n📊 Results: {working_count}/{len(test_prompts)} tests passed")

        if working_count >= len(test_prompts) // 2:
            print("✅ Basic functionality confirmed!")
            return True
        else:
            print("❌ Basic functionality failing")
            return False

    def simple_chat(self):
        """Very simple chat loop."""
        print("\n💬 Simple Chat Mode")
        print("=" * 20)
        print("Note: This is the BASE model, not your trained one (yet)")
        print("Type 'quit' to exit")

        while True:
            user_input = input("\n🧑 You: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                break

            if not user_input:
                continue

            print("🤖 Thinking...")
            response = self.generate_text(user_input + " ", max_tokens=50)
            print(f"🤖 Bot: {response}")

def main():
    """Main function - get SOMETHING working first."""
    print("🤖 Basic Bot - Getting Something Working!")
    print("=" * 40)

    bot = BasicBot()

    if not bot.model:
        print("❌ Complete failure - no model loaded")
        return

    # Test basic functionality first
    if not bot.test_functionality():
        print("❌ Basic generation not working")
        return

    print("\n🎉 SUCCESS! Basic bot is working!")

    while True:
        print("\n" + "="*30)
        print("What would you like to do?")
        print("1. 🧪 Test again")
        print("2. 💬 Chat with base model")
        print("3. ❌ Quit")

        choice = input("\nSelect (1-3): ").strip()

        if choice == '1':
            bot.test_functionality()
        elif choice == '2':
            bot.simple_chat()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()