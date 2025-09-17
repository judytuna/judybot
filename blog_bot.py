#!/usr/bin/env python3
"""
Multi-mode blog bot - chatbot, writing assistant, and hybrid modes.
"""

import sys
import os
from typing import Optional

class BlogBot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.mode = "hybrid"
        self.conversation_history = []
        self.load_model()

    def load_model(self):
        """Load the trained model."""
        print("🤖 Loading your blog-trained model...")
        try:
            from unsloth import FastLanguageModel

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name="./blog-model-unsloth-final",
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
            )

            self.model = FastLanguageModel.for_inference(self.model)
            print("✅ Model loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def generate_response(self, prompt: str, max_tokens: int = 50, temperature: float = 0.4) -> str:
        """Generate a response from the model with proper attention mask."""
        try:
            import torch

            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones(inputs.shape, dtype=torch.long)

            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)

            # Generate with attention mask
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

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()

            # Clean up response
            if response:
                # Stop at common end markers
                for marker in ['\n\n', '---', '###', '<|', '##']:
                    if marker in response:
                        response = response.split(marker)[0].strip()

                # Remove incomplete sentences at the end
                sentences = response.split('. ')
                if len(sentences) > 1 and not sentences[-1].endswith('.'):
                    response = '. '.join(sentences[:-1]) + '.'

            return response if response else "..."

        except Exception as e:
            return f"Error: {e}"

    def chatbot_mode(self):
        """Interactive chatbot mode."""
        print("\n💬 Chatbot Mode")
        print("=" * 30)
        print("Chat with your blog personality! Type 'quit' to exit, 'menu' for main menu.")
        print("Tips: Ask questions like 'What do you think about...?' or 'How do you feel about...?'")

        while True:
            user_input = input("\n🧑 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'menu':
                return
            elif not user_input:
                continue

            # Format as a question/conversation
            if not user_input.endswith('?'):
                prompt = f"What do you think about {user_input}? "
            else:
                prompt = f"{user_input} "

            print("🤖 Thinking...")
            response = self.generate_response(prompt, max_tokens=80, temperature=0.5)
            print(f"🤖 BlogBot: {response}")

    def writing_assistant_mode(self):
        """Blog writing assistant mode."""
        print("\n✍️  Writing Assistant Mode")
        print("=" * 35)
        print("Get help with blog posts! Type 'quit' to exit, 'menu' for main menu.")
        print("Examples: 'Write about programming', 'My thoughts on AI', 'Today I learned'")

        while True:
            user_input = input("\n📝 Blog prompt: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'menu':
                return
            elif not user_input:
                continue

            # Format as blog content
            if user_input.lower().startswith(('write about', 'write on')):
                topic = user_input[10:].strip()
                prompt = f"I've been thinking about {topic}. "
            elif user_input.lower().startswith('my thoughts on'):
                topic = user_input[14:].strip()
                prompt = f"My thoughts on {topic}: "
            else:
                prompt = f"{user_input}... "

            print("✍️  Generating content...")
            response = self.generate_response(prompt, max_tokens=120, temperature=0.4)
            print(f"\n📄 Blog content:\n{prompt}{response}")

    def completion_mode(self):
        """Text completion mode."""
        print("\n🔮 Completion Mode")
        print("=" * 25)
        print("Start a sentence and let the model complete it! Type 'quit' to exit, 'menu' for main menu.")
        print("Examples: 'Today I was thinking', 'Programming is', 'The best part about'")

        while True:
            user_input = input("\n🎯 Start sentence: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'menu':
                return
            elif not user_input:
                continue

            print("🔮 Completing...")
            response = self.generate_response(user_input + " ", max_tokens=60, temperature=0.3)

            if response:
                print(f"\n📝 Complete text:\n{user_input} {response}")
            else:
                print("🤷 Couldn't generate a good completion, try a different start!")

    def hybrid_mode(self):
        """Smart hybrid mode that detects intent."""
        print("\n🌟 Hybrid Mode")
        print("=" * 20)
        print("Smart mode that adapts to your input! Type 'quit' to exit, 'menu' for main menu.")
        print("\nTry:")
        print("  💬 Questions: 'What do you think about coding?'")
        print("  ✍️  Blog prompts: 'Write about travel'")
        print("  🔮 Completions: 'Today I learned'")

        while True:
            user_input = input("\n💫 Input: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'menu':
                return
            elif not user_input:
                continue

            # Smart mode detection
            if '?' in user_input or user_input.lower().startswith(('what', 'how', 'why', 'do you')):
                # Question mode
                print("💬 [Chat mode detected]")
                prompt = user_input + " " if user_input.endswith('?') else user_input + "? "
                response = self.generate_response(prompt, max_tokens=80, temperature=0.5)
                print(f"🤖 Response: {response}")

            elif user_input.lower().startswith(('write', 'blog about', 'post about')):
                # Writing mode
                print("✍️  [Writing mode detected]")
                topic = user_input.lower().replace('write about', '').replace('blog about', '').replace('post about', '').strip()
                prompt = f"I've been thinking about {topic}. "
                response = self.generate_response(prompt, max_tokens=120, temperature=0.4)
                print(f"📄 Blog draft:\n{prompt}{response}")

            else:
                # Completion mode
                print("🔮 [Completion mode detected]")
                response = self.generate_response(user_input + " ", max_tokens=60, temperature=0.3)
                print(f"📝 Completed: {user_input} {response}")

    def run(self):
        """Main bot interface."""
        if not self.model:
            print("❌ Model not loaded. Cannot start bot.")
            return

        print("🎉 Welcome to your Personal Blog Bot!")
        print("This bot is trained on YOUR blog data and writes in YOUR style.")

        while True:
            print("\n" + "="*50)
            print("🤖 BLOG BOT - Choose your mode:")
            print("1. 💬 Chatbot (Ask questions, have conversations)")
            print("2. ✍️  Writing Assistant (Generate blog content)")
            print("3. 🔮 Completion (Finish your sentences)")
            print("4. 🌟 Hybrid (Smart mode - adapts to input)")
            print("5. ❌ Quit")

            choice = input("\nSelect mode (1-5): ").strip()

            if choice == '1':
                self.chatbot_mode()
            elif choice == '2':
                self.writing_assistant_mode()
            elif choice == '3':
                self.completion_mode()
            elif choice == '4':
                self.hybrid_mode()
            elif choice == '5':
                print("👋 Thanks for using Blog Bot!")
                break
            else:
                print("Invalid choice. Please select 1-5.")

def main():
    """Start the blog bot."""
    print("🚀 Starting Personal Blog Bot...")

    # Check if model exists
    if not os.path.exists("./blog-model-unsloth-final"):
        print("❌ Trained model not found at ./blog-model-unsloth-final")
        print("Please run the training script first!")
        return

    bot = BlogBot()
    bot.run()

if __name__ == "__main__":
    main()