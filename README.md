# Blog Style Fine-Tuning Project

A project to fine-tune a small language model on personal blog data to replicate specific writing style and voice.

## Overview

This project successfully fine-tuned GPT2-medium on personal blog content, creating a model that can generate text in the original author's writing style. The training achieved significant loss reduction and produces coherent, blog-style content.

## Results

✅ **Training completed successfully**
- Model: GPT2-medium (355M parameters)
- Training loss: 5.14 → 4.03
- Validation loss: 3.83
- Training time: ~20 minutes on RTX 2080
- Memory usage: 0.68GB GPU memory

✅ **Quality outputs**
- Coherent sentences and paragraphs
- Variety in tone and style (personal, technical, motivational)
- Blog-like content structure
- No repetitive token issues

## Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- Blog content in Jekyll markdown format

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd choo

# Install dependencies
pip install torch transformers datasets
```

### Data Preparation
1. Place your blog repositories as sibling directories:
   ```
   parent-directory/
   ├── choo/                 # This project
   ├── judytuna-jekyll/      # Public blog content
   └── judytuna-private/     # Private blog content
   ```

2. Extract and process blog data:
   ```bash
   python extract_blog_data.py
   ```

## Usage

### Training
Train the model on your blog data:
```bash
python train_model.py
```

**Training configuration:**
- Batch size: 1 (with gradient accumulation 8)
- Learning rate: 2e-5 with warmup
- Epochs: 3
- Context length: 512 tokens
- Memory optimized for RTX 2080 (8GB VRAM)

### Testing
Test the trained model:
```bash
python test_model.py
```

This will run the model on various prompts and show generated responses in your writing style.

## Project Structure

```
choo/
├── extract_blog_data.py    # Data extraction and preprocessing
├── train_model.py          # Model training script
├── test_model.py          # Model testing script
├── data/                  # Training data (gitignored)
│   ├── train_data.jsonl
│   └── val_data.jsonl
├── blog-model-final/      # Trained model (gitignored)
└── README.md
```

## Technical Details

### Data Processing
- Extracts content from Jekyll markdown files
- Cleans and preprocesses text
- Converts to conversational training format
- Creates 90/10 train/validation split
- Handles various content types (posts, pages, drafts)

### Model Training
- Uses GPT2-medium as base model
- Implements memory optimizations for consumer GPUs
- Gradient checkpointing for memory efficiency
- BF16/FP32 precision (avoids FP16 gradient issues)
- Repetition penalties for coherent generation

### Memory Optimizations
- Reduced batch size with gradient accumulation
- Gradient checkpointing enabled
- Optimized data loading
- CUDA memory management

## Example Outputs

**Prompt:** "What are your thoughts on writing?"
**Response:** "Posted by Darkside at 8/19."

**Prompt:** "How do you approach creativity?"
**Response:** "In the first episode, we talk about how to deal with people who are always trying too hard. What's a good way for someone to learn what they're not supposed as well or find their own path through life? In this podcast, I'll explain it in plain English!"

**Prompt:** "How do you stay motivated?"
**Response:** "The best way to keep yourself inspired is to go out and play. It's not easy but it does pay off when you're doing something new or different than what you've done before."

## Privacy

- All training data is stored in `/data/` directory (gitignored)
- Trained models are stored in `blog-model-final/` (gitignored)
- Only scripts and configuration are tracked in git
- No personal blog content is committed to the repository

## Model Alternatives

The training script supports multiple models:
- `gpt2-medium` (355M) - Used in this project
- `microsoft/Phi-3.5-mini-instruct` (3.8B) - Requires more VRAM
- `Qwen/Qwen2.5-1.5B-Instruct` (1.5B) - Good balance
- `EleutherAI/pythia-1.4b` (1.4B) - Open alternative

## License

This project is for personal use. Blog content remains private and is not distributed.