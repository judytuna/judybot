# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repository contains scripts for training a Small Language Model (SLM) on blog data to replicate a specific writing style and voice. The project successfully fine-tuned GPT2-medium on personal blog content.

## Data Sources

The blog content is located in two sister repositories:
- `../judytuna-jekyll` - Main blog content (public Jekyll blog)
- `../judytuna-private` - Private blog content

Blog data is processed and stored in `/data/` directory (gitignored for privacy).

## Project Structure

**Completed components:**
- ✅ `extract_blog_data.py` - Data collection and preprocessing from sister repos
- ✅ `train_model.py` - Model training script with memory optimizations
- ✅ `test_model.py` - Model testing and inference script
- ✅ Training data: `train_data.jsonl`, `val_data.jsonl` (in /data/)
- ✅ Fine-tuned model: `blog-model-final/` (gitignored)

## Training Results

Successfully fine-tuned GPT2-medium (355M parameters) with:
- **Training loss**: 5.14 → 4.03 (significant improvement)
- **Final validation loss**: 3.83
- **Training time**: ~20 minutes on RTX 2080
- **Memory usage**: ~0.68GB GPU memory
- **Model outputs**: Coherent blog-style content matching original writing patterns

## Data Processing

Implemented pipeline:
- ✅ Jekyll markdown parsing and content extraction
- ✅ Text cleaning and preprocessing
- ✅ Conversion to conversational training format
- ✅ Train/validation splits (90/10)
- ✅ Tokenization and formatting for GPT2

## Model Training Configuration

Optimized settings for RTX 2080 (8GB VRAM):
- **Model**: GPT2-medium (355M params)
- **Batch size**: 1 (with gradient accumulation 8)
- **Learning rate**: 2e-5 with warmup
- **Epochs**: 3
- **Context length**: 512 tokens
- **Precision**: BF16/FP32 (avoided FP16 gradient issues)

## Usage

1. Extract blog data: `python extract_blog_data.py`
2. Train model: `python train_model.py`
3. Test model: `python test_model.py`