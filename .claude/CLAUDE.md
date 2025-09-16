# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repository is for training a Small Language Model (SLM) on blog data to replicate a specific writing style and voice.

## Data Sources

The blog content is located in two sister repositories:
- `../judytuna-jekyll` - Main blog content (likely public Jekyll blog)
- `../judytuna-private` - Private blog content

These repositories contain the source material for training the SLM.

## Project Structure

This repository will contain:
- Data collection and preprocessing scripts to extract content from the sister repos
- Model training configuration and scripts
- Evaluation and inference utilities
- Generated model artifacts and checkpoints

## Development Workflow

Initial setup involves:

1. Creating data pipeline to extract and preprocess content from `../judytuna-jekyll` and `../judytuna-private`
2. Configuring the training environment and dependencies
3. Implementing model training scripts
4. Creating evaluation and inference tools

## Data Processing

Blog data processing will require:
- Extracting content from Jekyll markdown files and posts
- Text cleaning and preprocessing
- Tokenization and formatting for training
- Train/validation/test splits
- Style and voice analysis for quality assessment

## Model Training

Consider factors like:
- Model architecture selection (GPT-style, T5, etc.)
- Training hyperparameters
- Compute requirements and optimization
- Checkpointing and model versioning