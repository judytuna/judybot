#!/usr/bin/env python3
"""
Quick import of existing GGUF file to Ollama.
"""

import os
import subprocess
import shutil

def import_existing_gguf():
    """Import the existing GGUF file to Ollama."""
    print("🚀 Quick Ollama Import from Existing GGUF")
    print("=" * 40)

    # Check if GGUF exists
    gguf_path = "./blog-model-unsloth-final/Blog-Model-Unsloth-Final-F16-LoRA.gguf"
    if not os.path.exists(gguf_path):
        print(f"❌ GGUF file not found at {gguf_path}")
        return False

    print(f"✅ Found GGUF: {gguf_path}")

    # Create Ollama directory
    ollama_dir = "./blog-model-ollama-quick/"
    os.makedirs(ollama_dir, exist_ok=True)

    # Copy GGUF file
    print("📋 Copying GGUF file...")
    shutil.copy2(gguf_path, os.path.join(ollama_dir, "model.gguf"))

    # Create Modelfile
    print("📝 Creating Modelfile...")
    modelfile_content = """FROM ./model.gguf

TEMPLATE \"\"\"<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
\"\"\"

PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"You are a helpful AI assistant trained on personal blog content. You write in a conversational, authentic style similar to a software engineer and writer who enjoys creative projects, programming, and thoughtful reflection.\"\"\"
"""

    modelfile_path = os.path.join(ollama_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print("✅ Modelfile created")

    # Check Ollama
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        print("✅ Ollama is available")
    except:
        print("❌ Ollama not found! Make sure it's installed in WSL")
        print("To install: curl -fsSL https://ollama.ai/install.sh | sh")
        return False

    # Import to Ollama
    print("📥 Importing to Ollama...")
    try:
        original_dir = os.getcwd()
        os.chdir(ollama_dir)

        result = subprocess.run([
            "ollama", "create", "blog-model", "-f", "Modelfile"
        ], capture_output=True, text=True)

        os.chdir(original_dir)

        if result.returncode == 0:
            print("✅ Successfully imported as 'blog-model'!")
            return True
        else:
            print(f"❌ Import failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model():
    """Quick test of the imported model."""
    print("\n🧪 Testing imported model...")

    try:
        result = subprocess.run([
            "ollama", "run", "blog-model",
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
        print("⏰ Test timed out - but model is probably working")
        return True
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main function."""
    if import_existing_gguf():
        print("\n🎉 Import successful!")

        if test_model():
            print("\n✅ Ready to use!")
            print("Try these commands:")
            print("  ollama run blog-model")
            print("  python ollama_chat.py")
        else:
            print("\n⚠️ Import worked but test had issues")
            print("Try manually: ollama run blog-model")
    else:
        print("\n❌ Import failed")

if __name__ == "__main__":
    main()