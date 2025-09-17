#!/usr/bin/env python3
"""
Analyze the training data quality and quantity.
"""

import json
from pathlib import Path

def analyze_dataset(file_path):
    """Analyze a JSONL dataset file."""
    total_words = 0
    total_chars = 0
    total_examples = 0
    response_lengths = []

    print(f"📊 Analyzing {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                response = data['messages'][1]['content']
                words = len(response.split())
                chars = len(response)

                total_words += words
                total_chars += chars
                total_examples += 1
                response_lengths.append(words)

                if total_examples <= 3:
                    print(f"  Example {total_examples}: {words} words, {chars} chars")
                    print(f"    Preview: {response[:100]}...")

    response_lengths.sort()
    median_length = response_lengths[len(response_lengths)//2]
    min_length = min(response_lengths)
    max_length = max(response_lengths)

    print(f"\n📈 Statistics:")
    print(f"  Total examples: {total_examples:,}")
    print(f"  Average response: {total_words/total_examples:.1f} words")
    print(f"  Median response: {median_length} words")
    print(f"  Range: {min_length} - {max_length} words")
    print(f"  Total words: {total_words:,}")
    print(f"  Total chars: {total_chars:,}")

    return {
        'examples': total_examples,
        'avg_words': total_words/total_examples,
        'median_words': median_length,
        'total_words': total_words,
        'min_words': min_length,
        'max_words': max_length
    }

def assess_data_quality(train_stats, val_stats):
    """Assess if the data is sufficient for training."""
    print(f"\n🎯 Data Quality Assessment:")

    # Check quantity
    total_examples = train_stats['examples'] + val_stats['examples']
    print(f"  Total examples: {total_examples:,}")

    if total_examples < 500:
        print("  ❌ LOW: < 500 examples (may underfit)")
    elif total_examples < 1000:
        print("  ⚠️  MODERATE: 500-1000 examples (should work)")
    elif total_examples < 5000:
        print("  ✅ GOOD: 1000-5000 examples (solid training)")
    else:
        print("  🎉 EXCELLENT: > 5000 examples (lots of data)")

    # Check average length
    avg_length = train_stats['avg_words']
    print(f"\n  Average response length: {avg_length:.1f} words")

    if avg_length < 20:
        print("  ❌ SHORT: Responses are quite brief")
    elif avg_length < 50:
        print("  ⚠️  MODERATE: Decent response length")
    elif avg_length < 200:
        print("  ✅ GOOD: Rich, detailed responses")
    else:
        print("  🎉 EXCELLENT: Very detailed responses")

    # Check variety
    word_range = train_stats['max_words'] - train_stats['min_words']
    print(f"\n  Response variety: {train_stats['min_words']}-{train_stats['max_words']} words (range: {word_range})")

    if word_range < 50:
        print("  ❌ LOW VARIETY: Responses are similar length")
    elif word_range < 200:
        print("  ⚠️  MODERATE VARIETY: Some length variation")
    else:
        print("  ✅ GOOD VARIETY: Wide range of response lengths")

    # Overall assessment
    print(f"\n🏆 Overall Assessment:")

    if total_examples >= 1000 and avg_length >= 30:
        print("  ✅ EXCELLENT for blog style fine-tuning!")
        print("  📝 Your data should produce a high-quality model")
        return "excellent"
    elif total_examples >= 500 and avg_length >= 20:
        print("  ✅ GOOD for fine-tuning!")
        print("  📝 Should learn your writing style well")
        return "good"
    elif total_examples >= 200:
        print("  ⚠️  ADEQUATE for basic style transfer")
        print("  📝 May need more epochs or data augmentation")
        return "adequate"
    else:
        print("  ❌ INSUFFICIENT for reliable fine-tuning")
        print("  📝 Consider gathering more training data")
        return "insufficient"

def main():
    """Main analysis function."""
    print("🔍 Blog Training Data Analysis")
    print("=" * 40)

    # Analyze training data
    train_stats = analyze_dataset("data/train_data.jsonl")

    print("\n" + "="*40)

    # Analyze validation data
    val_stats = analyze_dataset("data/val_data.jsonl")

    print("\n" + "="*60)

    # Overall assessment
    quality = assess_data_quality(train_stats, val_stats)

    print(f"\n💡 Recommendations:")
    if quality == "excellent":
        print("  🚀 Ready for training! Consider 2-3 epochs.")
        print("  🎯 Your model should capture your writing style well.")
    elif quality == "good":
        print("  🚀 Ready for training! Consider 3-4 epochs.")
        print("  🎯 May benefit from slightly more training.")
    elif quality == "adequate":
        print("  ⚠️  Consider gathering more data if possible.")
        print("  🎯 Try 4-5 epochs and monitor for overfitting.")
    else:
        print("  ❌ Gather more training data before proceeding.")
        print("  🎯 Aim for at least 500 examples.")

if __name__ == "__main__":
    main()