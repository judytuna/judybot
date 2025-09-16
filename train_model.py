#!/usr/bin/env python3
"""
Fine-tune a small language model on blog data using modern approaches.
Supports Llama, Qwen, and Gemma models with efficient training.
"""

import os
import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from typing import Dict, List

class BlogModelTrainer:
    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct"):
        """
        Initialize trainer with model selection.

        Recommended models:
        - "microsoft/Phi-3.5-mini-instruct" (3.8B - what you used)
        - "meta-llama/Llama-3.2-1B-Instruct" (1B - faster)
        - "meta-llama/Llama-3.2-3B-Instruct" (3B - balanced)
        - "Qwen/Qwen2.5-1.5B-Instruct" (1.5B - efficient)
        - "google/gemma-2-2b-it" (2B - good quality)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

        # Training parameters optimized for style learning
        self.training_config = {
            "learning_rate": 2e-5,  # Lower LR for style preservation
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "eval_steps": 100,
            "save_steps": 500,
            "fp16": True,  # Memory optimization
            "gradient_checkpointing": True,
            "dataloader_num_workers": 0,  # Avoid Windows multiprocessing issues
            "remove_unused_columns": False
        }

    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        print(f"Model loaded. Parameters: {self.model.num_parameters():,}")

    def format_chat_message(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single training string."""
        # Use the model's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

        # Fallback formatting
        formatted = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"

        return formatted

    def prepare_dataset(self, train_file: str, val_file: str) -> tuple:
        """Load and tokenize the training data."""
        print("Loading training data...")

        # Load JSONL files
        def load_jsonl(filepath):
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data

        train_data = load_jsonl(train_file)
        val_data = load_jsonl(val_file)

        print(f"Loaded {len(train_data)} training examples, {len(val_data)} validation examples")

        # Format and tokenize
        def tokenize_function(examples):
            # Format each conversation
            texts = []
            for messages in examples["messages"]:
                formatted_text = self.format_chat_message(messages)
                texts.append(formatted_text)

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=1024,  # Reasonable context length
                padding=False,
                return_tensors=None
            )

            # For causal language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Convert to datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        # Tokenize
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

    def train(self, train_file: str = "train_data.jsonl", val_file: str = "val_data.jsonl"):
        """Train the model on blog data."""
        # Load model and tokenizer
        self.load_model_and_tokenizer()

        # Prepare datasets
        train_dataset, val_dataset = self.prepare_dataset(train_file, val_file)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./blog-model",
            overwrite_output_dir=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            **self.training_config
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        print("Starting training...")
        trainer.train()

        # Save the final model
        trainer.save_model("./blog-model-final")
        self.tokenizer.save_pretrained("./blog-model-final")

        print("Training complete! Model saved to ./blog-model-final")

    def test_model(self, prompt: str = "Tell me about your day"):
        """Test the trained model with a sample prompt."""
        if self.model is None or self.tokenizer is None:
            print("Loading trained model...")
            self.tokenizer = AutoTokenizer.from_pretrained("./blog-model-final")
            self.model = AutoModelForCausalLM.from_pretrained(
                "./blog-model-final",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.format_chat_message(messages)

        # Tokenize
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the new part
        response = response[len(formatted_prompt):].strip()

        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        return response

if __name__ == "__main__":
    # You can change this to try different models
    MODEL_OPTIONS = {
        "phi": "microsoft/Phi-3.5-mini-instruct",
        "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
        "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
        "gemma": "google/gemma-2-2b-it"
    }

    # Choose your model here
    model_choice = "qwen"  # Change this to try different models
    model_name = MODEL_OPTIONS[model_choice]

    print(f"Using model: {model_name}")

    trainer = BlogModelTrainer(model_name)
    trainer.train()

    # Test the model
    print("\nTesting trained model:")
    trainer.test_model("What's on your mind today?")
    trainer.test_model("Tell me about programming")
    trainer.test_model("How are you feeling?")