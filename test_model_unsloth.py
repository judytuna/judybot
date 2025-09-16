#!/usr/bin/env python3
"""
Test the enhanced model trained with Unsloth.
"""

from train_model_unsloth import EnhancedBlogTrainer

def main():
    print("🧪 Testing Enhanced Blog Model")
    print("=" * 40)

    # Determine which model was trained
    trainer = EnhancedBlogTrainer('microsoft/Phi-3.5-mini-instruct')

    # Test prompts designed to showcase different writing styles
    test_prompts = [
        "What are your thoughts on programming?",
        "How do you approach creative writing?",
        "Tell me about a typical day",
        "What's your favorite programming language and why?",
        "How do you stay motivated when working on projects?",
        "What advice would you give to someone starting to code?",
        "Describe your writing process",
        "What's something you learned recently?",
    ]

    print(f"Testing with {len(test_prompts)} prompts...\n")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"🤖 Test {i}/{len(test_prompts)}: {prompt}")
        try:
            response = trainer.test_model(prompt)
            print(f"📝 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 60)

    print("🎉 Testing complete!")

if __name__ == "__main__":
    main()