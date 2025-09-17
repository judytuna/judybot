#!/usr/bin/env python3
"""
Export the trained LoRA model to Ollama format.
Run this in WSL where Ollama is installed.
"""

import os
import subprocess
from pathlib import Path

def export_to_ollama():
    """Export the model to Ollama GGUF format."""
    print("🔄 Exporting trained model to Ollama...")

    try:
        # Check if we have the LoRA adapter
        adapter_path = "./blog-model-unsloth-final"
        if not os.path.exists(adapter_path):
            print(f"❌ LoRA adapter not found at {adapter_path}")
            return False

        # Try using Unsloth's export if available
        try:
            from unsloth import FastLanguageModel
            print("✅ Using Unsloth for GGUF export...")

            # Load the trained model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
            )

            # Enable inference mode
            model = FastLanguageModel.for_inference(model)

            # Export to GGUF
            ollama_dir = "./blog-model-ollama"
            os.makedirs(ollama_dir, exist_ok=True)

            print("🔄 Exporting to GGUF format...")
            model.save_pretrained_gguf(
                ollama_dir,
                tokenizer=tokenizer,
                quantization_method="q4_k_m"
            )

            print("✅ GGUF export successful!")

        except Exception as e:
            print(f"❌ Unsloth export failed: {e}")
            print("🔄 Trying alternative export method...")
            return export_with_llama_cpp()

        # Create Modelfile for Ollama
        create_modelfile(ollama_dir)

        # Import into Ollama
        return import_to_ollama(ollama_dir)

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def export_with_llama_cpp():
    """Alternative export using llama.cpp convert script."""
    print("🔄 Trying llama.cpp conversion...")

    try:
        # Check if llama.cpp exists
        if not os.path.exists("./llama.cpp"):
            print("📥 Cloning llama.cpp...")
            subprocess.run([
                "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
            ], check=True)

        # Build llama.cpp if needed
        if not os.path.exists("./llama.cpp/build"):
            print("🔨 Building llama.cpp...")
            subprocess.run(["make", "-C", "./llama.cpp"], check=True)

        # Convert model
        print("🔄 Converting model...")
        convert_script = "./llama.cpp/convert_hf_to_gguf.py"
        model_path = "./blog-model-unsloth-final"
        output_path = "./blog-model-ollama/model.gguf"

        if os.path.exists(convert_script):
            subprocess.run([
                "python", convert_script,
                model_path,
                "--outfile", output_path,
                "--outtype", "q4_k_m"
            ], check=True)

            print("✅ llama.cpp conversion successful!")
            create_modelfile("./blog-model-ollama")
            return import_to_ollama("./blog-model-ollama")
        else:
            print("❌ llama.cpp convert script not found")
            return False

    except Exception as e:
        print(f"❌ llama.cpp conversion failed: {e}")
        return False

def create_modelfile(ollama_dir):
    """Create the Ollama Modelfile."""
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

    modelfile_path = os.path.join(ollama_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"✅ Modelfile created at {modelfile_path}")

def import_to_ollama(ollama_dir):
    """Import the model into Ollama."""
    print("📥 Importing model into Ollama...")

    try:
        # Change to the model directory
        original_dir = os.getcwd()
        os.chdir(ollama_dir)

        # Create the model in Ollama
        result = subprocess.run([
            "ollama", "create", "blog-model", "-f", "Modelfile"
        ], capture_output=True, text=True)

        os.chdir(original_dir)

        if result.returncode == 0:
            print("✅ Model imported into Ollama successfully!")
            print("\n🎉 Your blog model is ready!")
            print("To use it, run:")
            print("  ollama run blog-model")
            return True
        else:
            print(f"❌ Ollama import failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Ollama import failed: {e}")
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
            print(f"Response: {result.stdout}")
        else:
            print(f"❌ Test failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("⏰ Test timed out - model might be working but slow")
    except Exception as e:
        print(f"❌ Test error: {e}")

def main():
    """Main export function."""
    print("🚀 Exporting Blog Model to Ollama")
    print("=" * 35)
    print("Make sure you're running this in WSL where Ollama is installed!")

    # Check if Ollama is available
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        print("✅ Ollama is available")
    except:
        print("❌ Ollama not found! Install it first:")
        print("  curl -fsSL https://ollama.ai/install.sh | sh")
        return

    # Export the model
    success = export_to_ollama()

    if success:
        print("\n🎉 Export complete!")
        test_ollama_model()
        print("\n💡 You can now chat with your blog model:")
        print("  ollama run blog-model")
    else:
        print("\n❌ Export failed")
        print("💡 Try running the basic bot instead:")
        print("  python basic_bot.py")

if __name__ == "__main__":
    main()