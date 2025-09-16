#!/usr/bin/env python3
"""
Validate and analyze the prepared training data for quality and potential issues.
"""

import json
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import re

class DataValidator:
    def __init__(self):
        self.stats = {}

    def load_jsonl(self, filepath: str):
        """Load JSONL file."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def analyze_data_quality(self, data):
        """Analyze the quality and characteristics of training data."""
        print("=" * 60)
        print("DATA QUALITY ANALYSIS")
        print("=" * 60)

        total_examples = len(data)
        print(f"Total examples: {total_examples}")

        if total_examples == 0:
            print("No data to analyze!")
            return

        # Analyze message lengths
        user_lengths = []
        assistant_lengths = []

        for example in data:
            messages = example["messages"]
            for msg in messages:
                content_len = len(msg["content"])
                if msg["role"] == "user":
                    user_lengths.append(content_len)
                elif msg["role"] == "assistant":
                    assistant_lengths.append(content_len)

        print(f"\nUSER MESSAGES:")
        print(f"  Count: {len(user_lengths)}")
        print(f"  Avg length: {sum(user_lengths)/len(user_lengths):.1f} chars")
        print(f"  Min length: {min(user_lengths)} chars")
        print(f"  Max length: {max(user_lengths)} chars")

        print(f"\nASSISTANT MESSAGES:")
        print(f"  Count: {len(assistant_lengths)}")
        print(f"  Avg length: {sum(assistant_lengths)/len(assistant_lengths):.1f} chars")
        print(f"  Min length: {min(assistant_lengths)} chars")
        print(f"  Max length: {max(assistant_lengths)} chars")

        # Analyze conversation starters
        user_messages = []
        for example in data:
            messages = example["messages"]
            user_msg = messages[0]["content"] if messages[0]["role"] == "user" else ""
            if user_msg:
                user_messages.append(user_msg.lower())

        # Find common patterns in user messages
        starter_patterns = Counter()
        for msg in user_messages:
            # Extract the first few words
            words = msg.split()[:5]
            if len(words) >= 2:
                pattern = " ".join(words[:3])
                starter_patterns[pattern] += 1

        print(f"\nTOP CONVERSATION STARTERS:")
        for pattern, count in starter_patterns.most_common(10):
            print(f"  '{pattern}': {count} times")

        # Analyze assistant response patterns
        assistant_messages = []
        for example in data:
            messages = example["messages"]
            assistant_msg = messages[1]["content"] if len(messages) > 1 and messages[1]["role"] == "assistant" else ""
            if assistant_msg:
                assistant_messages.append(assistant_msg)

        # Check for style markers from the blog analysis
        style_markers = {
            "exclamation_points": 0,
            "equals_signs": 0,
            "hehe_haha": 0,
            "ellipsis": 0,
            "ALL_CAPS_words": 0
        }

        for msg in assistant_messages:
            style_markers["exclamation_points"] += msg.count("!")
            style_markers["equals_signs"] += msg.count("=)")
            style_markers["hehe_haha"] += len(re.findall(r'\b(hehe|haha|hehehe)\b', msg.lower()))
            style_markers["ellipsis"] += msg.count("...")
            style_markers["ALL_CAPS_words"] += len(re.findall(r'\b[A-Z]{2,}\b', msg))

        print(f"\nSTYLE MARKERS IN RESPONSES:")
        total_responses = len(assistant_messages)
        for marker, count in style_markers.items():
            avg_per_response = count / total_responses if total_responses > 0 else 0
            print(f"  {marker}: {count} total ({avg_per_response:.2f} per response)")

        return {
            "total_examples": total_examples,
            "user_lengths": user_lengths,
            "assistant_lengths": assistant_lengths,
            "style_markers": style_markers,
            "starter_patterns": starter_patterns
        }

    def check_for_issues(self, data):
        """Check for potential data quality issues."""
        print("\n" + "=" * 60)
        print("POTENTIAL ISSUES CHECK")
        print("=" * 60)

        issues = []

        # Check for very short responses
        short_responses = 0
        for example in data:
            messages = example["messages"]
            for msg in messages:
                if msg["role"] == "assistant" and len(msg["content"]) < 50:
                    short_responses += 1

        if short_responses > 0:
            issues.append(f"Found {short_responses} very short assistant responses (<50 chars)")

        # Check for very long responses that might cause memory issues
        long_responses = 0
        for example in data:
            messages = example["messages"]
            for msg in messages:
                if msg["role"] == "assistant" and len(msg["content"]) > 2000:
                    long_responses += 1

        if long_responses > 0:
            issues.append(f"Found {long_responses} very long assistant responses (>2000 chars)")

        # Check for duplicate content
        assistant_contents = []
        for example in data:
            messages = example["messages"]
            for msg in messages:
                if msg["role"] == "assistant":
                    assistant_contents.append(msg["content"])

        unique_responses = len(set(assistant_contents))
        total_responses = len(assistant_contents)
        duplicate_ratio = 1 - (unique_responses / total_responses) if total_responses > 0 else 0

        if duplicate_ratio > 0.1:  # More than 10% duplicates
            issues.append(f"High duplicate content ratio: {duplicate_ratio:.1%}")

        # Report issues
        if issues:
            print("ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("No significant issues detected!")

        return issues

    def create_length_distribution_plot(self, user_lengths, assistant_lengths):
        """Create plots showing length distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # User message lengths
        ax1.hist(user_lengths, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title('User Message Length Distribution')
        ax1.set_xlabel('Characters')
        ax1.set_ylabel('Count')

        # Assistant message lengths
        ax2.hist(assistant_lengths, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_title('Assistant Response Length Distribution')
        ax2.set_xlabel('Characters')
        ax2.set_ylabel('Count')

        plt.tight_layout()
        plt.savefig('data_length_distribution.png', dpi=150, bbox_inches='tight')
        print(f"\nLength distribution plot saved as 'data_length_distribution.png'")

    def show_sample_conversations(self, data, n_samples=3):
        """Show sample conversations from the data."""
        print("\n" + "=" * 60)
        print(f"SAMPLE CONVERSATIONS ({n_samples} examples)")
        print("=" * 60)

        for i, example in enumerate(data[:n_samples]):
            print(f"\n--- EXAMPLE {i+1} ---")
            messages = example["messages"]
            for msg in messages:
                role = msg["role"].upper()
                content = msg["content"]
                # Truncate long content for display
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"{role}: {content}")

if __name__ == "__main__":
    validator = DataValidator()

    # Validate training data
    if Path("train_data.jsonl").exists():
        print("Analyzing training data...")
        train_data = validator.load_jsonl("train_data.jsonl")
        train_stats = validator.analyze_data_quality(train_data)
        validator.check_for_issues(train_data)
        validator.show_sample_conversations(train_data)

        # Create visualization if possible
        try:
            validator.create_length_distribution_plot(
                train_stats["user_lengths"],
                train_stats["assistant_lengths"]
            )
        except Exception as e:
            print(f"Could not create plots: {e}")

    else:
        print("train_data.jsonl not found. Run create_training_data.py first.")

    # Also check validation data
    if Path("val_data.jsonl").exists():
        print("\n" + "=" * 60)
        print("VALIDATION DATA SUMMARY")
        print("=" * 60)
        val_data = validator.load_jsonl("val_data.jsonl")
        print(f"Validation examples: {len(val_data)}")
    else:
        print("\nval_data.jsonl not found.")