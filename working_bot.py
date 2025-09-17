#!/usr/bin/env python3
"""
Working bot using standard transformers + PEFT to avoid Unsloth inference issues.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

class WorkingBlogBot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load the model using standard transformers + PEFT."""
        print("🔧 Loading model with standard transformers + PEFT...")

        try:
            base_model_name = "microsoft/Phi-3.5-mini-instruct"
            adapter_path = "./blog-model-unsloth-final"

            print("📦 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("🧠 Loading base model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

            print("🔗 Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

            print("🔀 Merging adapter weights...")
            self.model = self.model.merge_and_unload()  # Merge for efficiency

            print("✅ Model loaded successfully with standard transformers!")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def generate_response(self, prompt, max_tokens=50, temperature=0.4):
        """Generate response with proper attention masks."""
        if not self.model or not self.tokenizer:
            return "Model not loaded"

        try:
            # Tokenize with attention mask
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            attention_mask = torch.ones(inputs.shape, dtype=torch.long)

            # Move to device
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.8,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()

            # Clean response
            if response:
                # Stop at common markers
                for marker in ['\n\n', '---', '###', '<|']:
                    if marker in response:
                        response = response.split(marker)[0].strip()

            return response if response else "..."

        except Exception as e:
            return f"Error: {e}"

    def chat_mode(self):
        """Simple chat interface."""
        print("\n💬 Working Chat Mode")
        print("=" * 25)
        print("Type 'quit' to exit")

        while True:
            user_input = input("\n🧑 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif not user_input:
                continue

            print("🤖 Thinking...")
            response = self.generate_response(f"{user_input} ", max_tokens=60, temperature=0.4)
            print(f"🤖 BlogBot: {response}")

    def test_quality(self):
        """Test the model quality with various prompts."""
        print("\n🧪 Testing Model Quality")
        print("=" * 30)

        test_prompts = [
            "What do you think about programming?",
            "How do you approach creative projects?",
            "Today I was thinking about",
            "My favorite thing about coding is",
            "When I write, I usually",
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/{len(test_prompts)} ---")
            print(f"🤖 Prompt: {prompt}")

            response = self.generate_response(prompt + " ", max_tokens=50, temperature=0.3)
            print(f"📝 Response: {response}")

            # Quality check
            words = response.split()
            if len(words) > 3 and len(words) < 40 and not any(c in response for c in ['$', '^', '[', ']']):
                print("✅ Response looks good!")
            else:
                print("⚠️  Response may have issues")

def main():
    """Main function."""
    print("🚀 Working Blog Bot (Standard Transformers)")
    print("=" * 45)

    # Check if adapter exists
    if not os.path.exists("./blog-model-unsloth-final"):
        print("❌ LoRA adapter not found at ./blog-model-unsloth-final")
        return

    # Create bot
    bot = WorkingBlogBot()

    if not bot.model:
        print("❌ Failed to load model")
        return

    while True:
        print("\n" + "="*40)
        print("Choose an option:")
        print("1. 🧪 Test model quality")
        print("2. 💬 Chat mode")
        print("3. ❌ Quit")

        choice = input("\nSelect (1-3): ").strip()

        if choice == '1':
            bot.test_quality()
        elif choice == '2':
            bot.chat_mode()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-3.")

if __name__ == "__main__":
    main()