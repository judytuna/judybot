#!/usr/bin/env python3
"""
Simple command-line chat interface.
"""

def load_model():
    """Load the model once."""
    try:
        from unsloth import FastLanguageModel

        print("🤖 Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="./blog-model-unsloth-final",
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.for_inference(model)
        print("✅ Ready to chat!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def generate(model, tokenizer, prompt, style="chat"):
    """Generate response based on style with proper attention mask."""
    try:
        import torch

        # Different prompt formats for different styles
        if style == "chat":
            formatted_prompt = f"{prompt} "
        elif style == "blog":
            formatted_prompt = f"I've been thinking about {prompt}. "
        else:  # completion
            formatted_prompt = prompt + " "

        # Tokenize with attention mask
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)

        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=50,
            temperature=0.4,
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(formatted_prompt):].strip()

    except Exception as e:
        return f"Error: {e}"

def main():
    """Simple chat loop."""
    model, tokenizer = load_model()
    if not model:
        return

    print("\n💬 Simple Chat Mode")
    print("Commands: /blog, /complete, /chat, /quit")

    mode = "chat"

    while True:
        if mode == "chat":
            prompt = input("\n🧑 Ask: ")
        elif mode == "blog":
            prompt = input("\n📝 Topic: ")
        else:  # complete
            prompt = input("\n🔮 Start: ")

        if prompt.lower() == "/quit":
            break
        elif prompt.lower() == "/chat":
            mode = "chat"
            print("💬 Chat mode")
            continue
        elif prompt.lower() == "/blog":
            mode = "blog"
            print("📝 Blog mode")
            continue
        elif prompt.lower() == "/complete":
            mode = "complete"
            print("🔮 Completion mode")
            continue
        elif not prompt.strip():
            continue

        response = generate(model, tokenizer, prompt, mode)

        if mode == "complete":
            print(f"📝 {prompt} {response}")
        else:
            print(f"🤖 {response}")

if __name__ == "__main__":
    main()