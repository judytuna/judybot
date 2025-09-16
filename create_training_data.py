#!/usr/bin/env python3
"""
Convert extracted blog data into conversational instruction-tuning format.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

class ConversationalDataFormatter:
    def __init__(self):
        # Conversation starters based on common blog themes
        self.conversation_starters = [
            "Tell me about {topic}",
            "What are your thoughts on {topic}?",
            "Can you share your experience with {topic}?",
            "How do you feel about {topic}?",
            "What's your take on {topic}?",
            "Tell me a story about {topic}",
            "What happened with {topic}?",
            "Share your thoughts about {topic}",
            "Describe your experience with {topic}",
            "What's going on with {topic}?"
        ]

        # Topic extractors - these will help generate natural conversation starters
        self.topic_patterns = [
            (r'\b(sailing|boat|water|ocean|sea)\b', 'sailing'),
            (r'\b(school|university|college|class|homework|study)\b', 'school'),
            (r'\b(work|job|career|office|project)\b', 'work'),
            (r'\b(friend|friends|friendship)\b', 'friends'),
            (r'\b(travel|trip|vacation|journey)\b', 'travel'),
            (r'\b(food|eating|cooking|restaurant|meal)\b', 'food'),
            (r'\b(music|song|singing|concert|band)\b', 'music'),
            (r'\b(movie|film|cinema|watch)\b', 'movies'),
            (r'\b(book|reading|novel|story)\b', 'books'),
            (r'\b(programming|code|coding|software|computer)\b', 'programming'),
            (r'\b(love|relationship|dating|romance)\b', 'relationships'),
            (r'\b(family|mom|dad|parent|sibling)\b', 'family'),
            (r'\b(dream|nightmare|sleep)\b', 'dreams'),
            (r'\b(weather|rain|sun|snow|storm)\b', 'weather'),
            (r'\b(birthday|celebration|party)\b', 'celebrations'),
            (r'\b(stress|anxiety|worry|concern)\b', 'stress'),
            (r'\b(happy|joy|excitement|fun)\b', 'happiness'),
            (r'\b(tired|exhausted|sleep|rest)\b', 'being tired'),
            (r'\b(creative|art|design|drawing)\b', 'creativity')
        ]

    def extract_topics_from_content(self, content: str) -> List[str]:
        """Extract potential conversation topics from blog content."""
        content_lower = content.lower()
        topics = []

        for pattern, topic in self.topic_patterns:
            if re.search(pattern, content_lower):
                topics.append(topic)

        return topics[:3]  # Limit to 3 topics per post

    def generate_conversation_starter(self, title: str, content: str) -> str:
        """Generate a natural conversation starter for the blog post."""
        topics = self.extract_topics_from_content(content)

        # If we found topics, use them
        if topics:
            topic = random.choice(topics)
            starter_template = random.choice(self.conversation_starters)
            return starter_template.format(topic=topic)

        # If no topics found, use the title or generic starters
        if title and title.lower() != 'no title':
            # Clean title for conversation
            clean_title = re.sub(r'^"|"$', '', title)  # Remove quotes
            clean_title = clean_title.lower()

            generic_starters = [
                f"Tell me about {clean_title}",
                f"What's the story with {clean_title}?",
                f"Can you share your thoughts on {clean_title}?",
                f"What happened with {clean_title}?"
            ]
            return random.choice(generic_starters)

        # Fallback generic starters
        fallback_starters = [
            "What's on your mind?",
            "Tell me what you're thinking about",
            "What's happening in your life?",
            "Share something with me",
            "What would you like to talk about?"
        ]
        return random.choice(fallback_starters)

    def should_include_post(self, content: str) -> bool:
        """Determine if a post should be included in training data."""
        # Skip very short posts
        if len(content.strip()) < 100:
            return False

        # Skip posts that are mostly metadata or technical
        if content.count('http') > 5:  # Probably a links post
            return False

        # Skip posts with too many special characters (might be corrupted)
        special_char_ratio = len(re.findall(r'[^\w\s\.\,\!\?\;\:\-\(\)]', content)) / len(content)
        if special_char_ratio > 0.3:
            return False

        return True

    def create_instruction_data(self, posts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert blog posts to instruction-tuning format."""
        instruction_data = []

        for post in posts:
            content = post['content']
            title = post['frontmatter'].get('title', 'No title')

            if not self.should_include_post(content):
                continue

            # Create primary conversation from full post
            conversation_starter = self.generate_conversation_starter(title, content)

            instruction_data.append({
                "instruction": conversation_starter,
                "input": "",
                "output": content
            })

            # For longer posts, create additional training examples by splitting
            if len(content) > 1000:
                # Split into meaningful chunks (by paragraphs)
                paragraphs = content.split('\n\n')
                if len(paragraphs) > 3:
                    # Create a "continue the story" example
                    mid_point = len(paragraphs) // 2
                    first_half = '\n\n'.join(paragraphs[:mid_point])
                    second_half = '\n\n'.join(paragraphs[mid_point:])

                    if len(first_half) > 100 and len(second_half) > 100:
                        instruction_data.append({
                            "instruction": "Continue this story",
                            "input": first_half,
                            "output": second_half
                        })

        print(f"Created {len(instruction_data)} instruction examples")
        return instruction_data

    def create_chat_format(self, instruction_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert to chat format for modern training frameworks."""
        chat_data = []

        for item in instruction_data:
            # Create conversation format
            conversation = [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]}
            ]

            # If there's input context, add it to the user message
            if item["input"]:
                conversation[0]["content"] = f"{item['instruction']}\n\n{item['input']}"

            chat_data.append({"messages": conversation})

        return chat_data

    def split_train_validation(self, data: List[Dict], validation_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """Split data into training and validation sets."""
        random.shuffle(data)
        split_point = int(len(data) * (1 - validation_ratio))
        return data[:split_point], data[split_point:]

    def save_training_data(self, data: List[Dict], filepath: str):
        """Save training data to JSONL format."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(data)} examples to {filepath}")

if __name__ == "__main__":
    # Load raw blog data
    with open("raw_blog_data.json", 'r', encoding='utf-8') as f:
        raw_posts = json.load(f)

    formatter = ConversationalDataFormatter()

    # Create instruction data
    instruction_data = formatter.create_instruction_data(raw_posts)

    # Convert to chat format
    chat_data = formatter.create_chat_format(instruction_data)

    # Split into train/validation
    train_data, val_data = formatter.split_train_validation(chat_data)

    # Save data
    formatter.save_training_data(train_data, "train_data.jsonl")
    formatter.save_training_data(val_data, "val_data.jsonl")

    print(f"\nTraining data created!")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    # Show sample
    if train_data:
        print(f"\nSample training example:")
        sample = random.choice(train_data)
        print(f"User: {sample['messages'][0]['content'][:150]}...")
        print(f"Assistant: {sample['messages'][1]['content'][:150]}...")