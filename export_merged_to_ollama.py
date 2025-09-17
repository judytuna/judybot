#!/usr/bin/env python3
"""
Step 2: Export the merged model to Ollama GGUF format.
Run this after merge_model.py succeeds.
"""

import os
import subprocess
from pathlib import Path

def check_merged_model():
    """Check if merged model exists."""
    merged_path = "./blog-model-merged/"
    if not os.path.exists(merged_path):
        print("❌ Merged model not found at ./blog-model-merged/")
        print("Run merge_model.py first!")
        return False

    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    missing_files = []

    for file in required_files:
        if not os.path.exists(os.path.join(merged_path, file)):
            # Check for .safetensors alternative
            if file == "pytorch_model.bin":
                safetensors_files = [f for f in os.listdir(merged_path) if f.endswith('.safetensors')]
                if not safetensors_files:
                    missing_files.append(file)
            else:
                missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    print("✅ Merged model found and looks complete")
    return True

def convert_to_gguf():
    """Convert merged model to GGUF using llama.cpp."""
    print("🔄 Converting to GGUF format...")

    try:
        # Check if llama.cpp exists
        if not os.path.exists("./llama.cpp"):
            print("📥 Cloning llama.cpp...")
            subprocess.run([
                "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
            ], check=True)

        # Check for conversion script
        convert_script = "./llama.cpp/convert_hf_to_gguf.py"
        if not os.path.exists(convert_script):
            print("❌ Conversion script not found in llama.cpp")
            return False

        # Create output directory
        output_dir = "./blog-model-ollama/"
        os.makedirs(output_dir, exist_ok=True)

        print("🔄 Running conversion...")
        result = subprocess.run([
            "python", convert_script,
            "./blog-model-merged/",
            "--outfile", os.path.join(output_dir, "model.gguf"),
            "--outtype", "q8_0"  # Use valid quantization option
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ GGUF conversion successful!")
            return True
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def create_ollama_modelfile():
    """Create Ollama Modelfile."""
    print("📝 Creating Ollama Modelfile...")

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

    modelfile_path = "./blog-model-ollama/Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"✅ Modelfile created at {modelfile_path}")

def import_to_ollama():
    """Import model into Ollama."""
    print("📥 Importing into Ollama...")

    try:
        # Check if Ollama is available
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        print("✅ Ollama is available")
    except:
        print("❌ Ollama not found! Make sure it's installed and in PATH")
        return False

    try:
        # Change to model directory
        original_dir = os.getcwd()
        os.chdir("./blog-model-ollama/")

        # Create model in Ollama
        result = subprocess.run([
            "ollama", "create", "blog-model", "-f", "Modelfile"
        ], capture_output=True, text=True)

        os.chdir(original_dir)

        if result.returncode == 0:
            print("✅ Model imported into Ollama as 'blog-model'!")
            return True
        else:
            print(f"❌ Import failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_ollama_model():
    """Test the Ollama model."""
    print("\n🧪 Testing Ollama model...")

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
        print("⏰ Test timed out - model might be working but slow")
        return True
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main export function."""
    print("🚀 Export Merged Model to Ollama")
    print("=" * 35)

    # Check prerequisites
    if not check_merged_model():
        return

    # Convert to GGUF
    if not convert_to_gguf():
        print("❌ GGUF conversion failed")
        return

    # Create Modelfile
    create_ollama_modelfile()

    # Import to Ollama
    if not import_to_ollama():
        print("❌ Ollama import failed")
        return

    # Test
    if test_ollama_model():
        print("\n🎉 Export complete and working!")
        print("You can now chat with your model:")
        print("  ollama run blog-model")
        print("Or use the chat interface:")
        print("  python ollama_chat.py")
    else:
        print("\n⚠️ Export completed but testing had issues")
        print("Try manually: ollama run blog-model")

if __name__ == "__main__":
    main()