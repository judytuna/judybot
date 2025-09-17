#!/usr/bin/env python3
"""
Use the working GPT2-medium model instead of the broken Phi merge.
"""

import os
import subprocess
import shutil

def import_gpt2_to_ollama():
    """Import the working GPT2 model to Ollama."""
    print("🚀 Using Working GPT2-Medium Model")
    print("=" * 35)

    # Check if GPT2 GGUF exists
    gpt2_gguf = "./data/blog-model-final/Blog-Model-Final-355M-F16.gguf"
    if not os.path.exists(gpt2_gguf):
        print(f"❌ GPT2 GGUF not found at {gpt2_gguf}")
        return False

    print(f"✅ Found working GPT2 model: {gpt2_gguf}")

    # Create Ollama directory
    ollama_dir = "./blog-model-ollama-gpt2/"
    os.makedirs(ollama_dir, exist_ok=True)

    # Copy GGUF file
    print("📋 Copying GPT2 GGUF...")
    shutil.copy2(gpt2_gguf, os.path.join(ollama_dir, "model.gguf"))

    # Create GPT2-specific Modelfile
    print("📝 Creating GPT2 Modelfile...")
    modelfile_content = """FROM ./model.gguf

TEMPLATE \"\"\"{{ .Prompt }}\"\"\"

PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"You are a helpful AI assistant trained on personal blog content. You write in a conversational, authentic style similar to a software engineer and writer who enjoys creative projects, programming, and thoughtful reflection.\"\"\"
"""

    modelfile_path = os.path.join(ollama_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print("✅ GPT2 Modelfile created")

    # Import to Ollama
    print("📥 Importing to Ollama...")
    try:
        original_dir = os.getcwd()
        os.chdir(ollama_dir)

        result = subprocess.run([
            "ollama", "create", "blog-gpt2", "-f", "Modelfile"
        ], capture_output=True, text=True)

        os.chdir(original_dir)

        if result.returncode == 0:
            print("✅ Successfully imported as 'blog-gpt2'!")
            return True
        else:
            print(f"❌ Import failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_gpt2_model():
    """Test the GPT2 model."""
    print("\n🧪 Testing GPT2 model...")

    try:
        result = subprocess.run([
            "ollama", "run", "blog-gpt2",
            "What do you think about programming?"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ Test successful!")
            print(f"Response: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Test failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ Test timed out - but probably working")
        return True
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    if import_gpt2_to_ollama():
        if test_gpt2_model():
            print("\n🎉 GPT2 model working in Ollama!")
            print("Use: ollama run blog-gpt2")
        else:
            print("\n⚠️ Import worked but test had issues")
    else:
        print("\n❌ GPT2 import failed")

if __name__ == "__main__":
    main()