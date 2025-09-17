#!/usr/bin/env python3
"""
Simple chat interface for the Ollama blog model.
Run this after successfully exporting to Ollama.
"""

import subprocess
import sys

def check_ollama():
    """Check if Ollama is running and blog-model exists."""
    try:
        # Check if Ollama is running
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ollama is not running. Start it with: ollama serve")
            return False

        # Check if any blog model exists
        if "blog-gpt2" in result.stdout:
            print("✅ blog-gpt2 found in Ollama")
            return "blog-gpt2"
        elif "blog-model" in result.stdout:
            print("✅ blog-model found in Ollama")
            return "blog-model"
        else:
            print("❌ No blog model found. Available models:")
            print(result.stdout)
            return False

    except FileNotFoundError:
        print("❌ Ollama not found. Make sure it's installed and in PATH")
        return False

def chat_with_model(model_name):
    """Simple chat interface."""
    print(f"\n💬 Chatting with {model_name}")
    print("=" * 35)
    print("Type 'quit', 'exit', or 'q' to stop")
    print("Type 'clear' to clear screen")
    print("Type 'help' for tips")

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                subprocess.run(['clear'] if sys.platform != 'win32' else ['cls'], shell=True)
                continue
            elif user_input.lower() == 'help':
                print("\n💡 Tips:")
                print("- Ask about programming, creative projects, or writing")
                print("- The model is trained on your blog content")
                print("- Keep questions conversational for best results")
                print("- Try: 'What do you think about...?' or 'How do you approach...?'")
                continue
            elif not user_input:
                continue

            print("🤖 Thinking...")

            # Call Ollama with the user input
            result = subprocess.run(
                ["ollama", "run", model_name, user_input],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                response = result.stdout.strip()
                if response:
                    print(f"🤖 BlogBot: {response}")
                else:
                    print("🤖 BlogBot: (no response)")
            else:
                print(f"❌ Error: {result.stderr}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def test_model(model_name):
    """Test the model with a few prompts."""
    print(f"\n🧪 Testing {model_name}...")

    test_prompts = [
        "What do you think about programming?",
        "How do you approach creative projects?",
        "Tell me about your writing process",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"🤖 Prompt: {prompt}")

        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            response = result.stdout.strip()
            print(f"📝 Response: {response}")
            print("✅ Test passed" if response else "⚠️ Empty response")
        else:
            print(f"❌ Test failed: {result.stderr}")

def main():
    """Main function."""
    print("🚀 Ollama Blog Chat Interface")
    print("=" * 30)

    model_name = check_ollama()
    if not model_name:
        return

    while True:
        print("\n" + "=" * 30)
        print("What would you like to do?")
        print("1. 💬 Chat with your blog model")
        print("2. 🧪 Test the model")
        print("3. 📋 Show available models")
        print("4. ❌ Quit")

        choice = input("\nSelect (1-4): ").strip()

        if choice == '1':
            chat_with_model(model_name)
        elif choice == '2':
            test_model(model_name)
        elif choice == '3':
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            print("\n📋 Available Ollama models:")
            print(result.stdout)
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()