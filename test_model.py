#!/usr/bin/env python3
"""
Test the trained blog model with better generation parameters.
"""

from train_model import BlogModelTrainer

def main():
    print("Loading trained model...")
    trainer = BlogModelTrainer('gpt2-medium')

    # Test prompts
    prompts = [
        "What are your thoughts on writing?",
        "How do you approach creativity?",
        "Tell me about your day",
        "What's your favorite programming language?",
        "How do you stay motivated?"
    ]

    print("Testing model with various prompts:\n")

    for prompt in prompts:
        print(f"🤖 Prompt: {prompt}")
        response = trainer.test_model(prompt)
        print(f"📝 Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()