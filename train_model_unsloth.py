#!/usr/bin/env python3
"""
Enhanced fine-tuning with Unsloth for faster training and better memory efficiency.
Supports Phi-3.5, Qwen, and other modern models with optimizations.
"""

import os
import json
import torch
import gc
from pathlib import Path
from typing import Dict, List, Optional

# Import unsloth first to avoid warnings, then fallback to transformers
UNSLOTH_AVAILABLE = False
try:
    # Import unsloth before transformers to avoid optimization warnings
    import unsloth
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth available - using optimized training")
except ImportError:
    print("⚠️  Unsloth not available - falling back to standard transformers")
except Exception as e:
    print(f"⚠️  Unsloth import issue: {e}")
    print("   Falling back to standard transformers")

# Always import transformers for fallback
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

class EnhancedBlogTrainer:
    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct"):
        """
        Initialize enhanced trainer with Unsloth optimizations.

        Recommended models:
        - "microsoft/Phi-3.5-mini-instruct" (3.8B - excellent quality)
        - "Qwen/Qwen2.5-1.5B-Instruct" (1.5B - good balance)
        - "meta-llama/Llama-3.2-1B-Instruct" (1B - fast, requires auth)
        - "google/gemma-2-2b-it" (2B - requires auth)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.use_unsloth = UNSLOTH_AVAILABLE

        # Enhanced training parameters optimized for speed and memory
        self.training_config = {
            "learning_rate": 2e-4,  # Higher LR works well with Unsloth
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2 if UNSLOTH_AVAILABLE else 1,
            "per_device_eval_batch_size": 2 if UNSLOTH_AVAILABLE else 1,
            "gradient_accumulation_steps": 4 if UNSLOTH_AVAILABLE else 8,
            "warmup_steps": 50,
            "weight_decay": 0.01,
            "logging_steps": 5,
            "eval_steps": 100,
            "save_steps": 200,
            "fp16": torch.cuda.is_available(),  # Use FP16 for RTX 2080
            "bf16": False,  # RTX 2080 doesn't support BF16 (needs Ampere+)
            "gradient_checkpointing": True,
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
            "max_grad_norm": 1.0,
            "report_to": None,
            "optim": "adamw_8bit" if UNSLOTH_AVAILABLE else "adamw_torch",
        }

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with Unsloth optimizations if available."""
        print(f"Loading model: {self.model_name}")

        if self.use_unsloth:
            # Use Unsloth for faster loading and training
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=2048,  # Increased context length
                dtype=None,  # Auto-detect best dtype
                load_in_4bit=True,  # 4-bit quantization for memory efficiency
                trust_remote_code=True,
            )

            # Enable LoRA for parameter-efficient fine-tuning
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,  # LoRA rank
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

        else:
            # Fallback to standard transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Check if flash attention is available
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                print("✅ Using Flash Attention 2")
            except ImportError:
                attn_implementation = None
                print("⚠️  Flash Attention not available, using standard attention")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
            )

            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

        print(f"Model loaded. Parameters: {self.model.num_parameters():,}")

        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            torch.cuda.empty_cache()

        gc.collect()

    def format_chat_message(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into training string using the model's chat template."""
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except:
                pass

        # Fallback formatting
        formatted = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                formatted += f"<|user|>\n{content}<|end|>\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}<|end|>\n"

        return formatted

    def prepare_dataset(self, train_file: str, val_file: str) -> tuple:
        """Load and tokenize the training data with optimizations."""
        print("Loading training data...")

        def load_jsonl(filepath):
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data

        train_data = load_jsonl(train_file)
        val_data = load_jsonl(val_file)

        print(f"Loaded {len(train_data)} training examples, {len(val_data)} validation examples")

        def tokenize_function(examples):
            texts = []
            for messages in examples["messages"]:
                formatted_text = self.format_chat_message(messages)
                texts.append(formatted_text)

            # Use longer context length if using Unsloth
            max_length = 1024 if self.use_unsloth else 512

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )

            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )

        return train_dataset, val_dataset

    def train(self, train_file: str = "data/train_data.jsonl", val_file: str = "data/val_data.jsonl"):
        """Train the model with enhanced optimizations."""
        self.load_model_and_tokenizer()
        train_dataset, val_dataset = self.prepare_dataset(train_file, val_file)

        # Output directory
        output_dir = "./blog-model-unsloth" if self.use_unsloth else "./blog-model-enhanced"

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            **self.training_config
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        if self.use_unsloth:
            # Use Unsloth's optimized trainer
            from trl import SFTTrainer

            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                dataset_text_field="input_ids",
                max_seq_length=1024,
                dataset_num_proc=2,
                packing=False,
                args=training_args,
            )
        else:
            # Standard trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )

        print("Starting enhanced training...")
        trainer.train()

        # Save the final model
        final_dir = f"{output_dir}-final"
        trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        print(f"Training complete! Model saved to {final_dir}")

        # Save for Ollama if using Unsloth
        if self.use_unsloth:
            self.save_for_ollama(final_dir)

    def save_for_ollama(self, model_dir: str):
        """Save model in format compatible with Ollama."""
        print("Preparing model for Ollama...")

        try:
            # Save in GGUF format for Ollama
            ollama_dir = f"{model_dir}-ollama"
            os.makedirs(ollama_dir, exist_ok=True)

            if self.use_unsloth:
                # Use Unsloth's GGUF export
                self.model.save_pretrained_gguf(
                    ollama_dir,
                    tokenizer=self.tokenizer,
                    quantization_method="q4_k_m"  # Good balance of size vs quality
                )

                # Create Ollama Modelfile
                modelfile_content = f"""FROM ./{Path(ollama_dir).name}/model.gguf

TEMPLATE \"\"\"<|user|>
{{{{ prompt }}}}<|end|>
<|assistant|>
\"\"\"

PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"You are a helpful AI assistant trained on personal blog content. You write in a conversational, authentic style.\"\"\"
"""

                with open(f"{ollama_dir}/Modelfile", "w") as f:
                    f.write(modelfile_content)

                print(f"✅ Ollama-ready model saved to {ollama_dir}")
                print(f"To use with Ollama:")
                print(f"  1. cd {ollama_dir}")
                print(f"  2. ollama create blog-model -f Modelfile")
                print(f"  3. ollama run blog-model")

        except Exception as e:
            print(f"⚠️  Could not save for Ollama: {e}")

    def test_model(self, prompt: str = "What's on your mind today?"):
        """Test the trained model."""
        model_dir = "./blog-model-unsloth-final" if self.use_unsloth else "./blog-model-enhanced-final"

        if not os.path.exists(model_dir):
            print(f"Model not found at {model_dir}")
            return

        print(f"Loading model from {model_dir}...")

        if self.use_unsloth:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_dir,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)  # Enable inference mode
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            # Try to use flash attention for inference too
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
            except ImportError:
                attn_implementation = None

            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                attn_implementation=attn_implementation,
            )

        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.format_chat_message(messages)

        # Tokenize
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(formatted_prompt):].strip()

        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        return response

if __name__ == "__main__":
    # Model options
    MODEL_OPTIONS = {
        "phi": "microsoft/Phi-3.5-mini-instruct",
        "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
        "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
        "gemma": "google/gemma-2-2b-it",
        "gpt2": "gpt2-medium"  # Fallback option
    }

    # Choose model - Phi-3.5 is excellent if you have enough VRAM
    model_choice = "phi"
    model_name = MODEL_OPTIONS[model_choice]

    print(f"Using model: {model_name}")
    print(f"Unsloth optimization: {'✅ Enabled' if UNSLOTH_AVAILABLE else '❌ Disabled'}")

    trainer = EnhancedBlogTrainer(model_name)
    trainer.train()

    # Test the model
    print("\nTesting enhanced model:")
    trainer.test_model("What are your thoughts on programming?")
    trainer.test_model("How do you stay creative?")
    trainer.test_model("Tell me about your writing process")