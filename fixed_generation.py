#!/usr/bin/env python3
"""
Fixed generation functions with proper attention masks.
"""

import torch

def generate_with_attention_mask(model, tokenizer, prompt, **kwargs):
    """Generate text with proper attention mask to avoid warnings."""

    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    attention_mask = attention_mask.to(device)

    # Generate with attention mask
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs
        )

    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()

    return response

def safe_generate(model, tokenizer, prompt, max_tokens=50, temperature=0.4, style="chat"):
    """Safe generation with all the fixes."""

    # Format prompt based on style
    if style == "chat":
        formatted_prompt = f"{prompt} "
    elif style == "blog":
        formatted_prompt = f"I've been thinking about {prompt}. "
    elif style == "complete":
        formatted_prompt = f"{prompt} "
    else:
        formatted_prompt = prompt

    try:
        response = generate_with_attention_mask(
            model, tokenizer, formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

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

# Test the fixed generation
def test_fixed_generation():
    """Test the fixed generation function."""
    print("🧪 Testing fixed generation with attention masks...")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="./blog-model-unsloth-final",
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.for_inference(model)

        test_prompts = [
            ("What do you think about programming?", "chat"),
            ("artificial intelligence", "blog"),
            ("Today I learned", "complete"),
        ]

        for prompt, style in test_prompts:
            print(f"\n--- {style.upper()} MODE ---")
            print(f"🤖 Prompt: {prompt}")

            response = safe_generate(model, tokenizer, prompt, style=style)
            print(f"📝 Response: {response}")

        print("\n✅ Fixed generation test complete!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_fixed_generation()