#!/usr/bin/env python3
"""
Extract and preprocess blog content from Jekyll repositories for SLM training.
"""

import os
import re
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

class BlogDataExtractor:
    def __init__(self, jekyll_path: str, private_path: str = None):
        self.jekyll_path = Path(jekyll_path)
        self.private_path = Path(private_path) if private_path else None
        self.posts = []

    def extract_frontmatter_and_content(self, filepath: Path) -> Dict[str, Any]:
        """Extract YAML frontmatter and content from a Jekyll post."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split frontmatter and content
        if content.startswith('---\n'):
            parts = content.split('---\n', 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                post_content = parts[2].strip()
            else:
                frontmatter = {}
                post_content = content.strip()
        else:
            frontmatter = {}
            post_content = content.strip()

        return {
            'frontmatter': frontmatter,
            'content': post_content,
            'filepath': str(filepath)
        }

    def clean_content(self, content: str) -> str:
        """Clean blog content for training."""
        # Remove Jekyll liquid tags
        content = re.sub(r'\{\{.*?\}\}', '', content)
        content = re.sub(r'\{%.*?%\}', '', content)

        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # Clean up markdown image syntax but keep alt text
        content = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', content)

        # Remove excessive whitespace but preserve paragraph breaks
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = content.strip()

        return content

    def extract_posts_from_directory(self, posts_dir: Path) -> List[Dict[str, Any]]:
        """Extract all posts from a _posts directory."""
        posts = []

        if not posts_dir.exists():
            print(f"Posts directory not found: {posts_dir}")
            return posts

        for post_file in posts_dir.rglob('*.md'):
            try:
                post_data = self.extract_frontmatter_and_content(post_file)

                # Skip if content is too short
                if len(post_data['content']) < 50:
                    continue

                # Clean content
                post_data['content'] = self.clean_content(post_data['content'])

                # Skip if cleaned content is too short
                if len(post_data['content']) < 30:
                    continue

                posts.append(post_data)

            except Exception as e:
                print(f"Error processing {post_file}: {e}")
                continue

        return posts

    def extract_all_posts(self) -> List[Dict[str, Any]]:
        """Extract posts from both Jekyll repositories."""
        all_posts = []

        # Extract from main Jekyll repo
        jekyll_posts_dir = self.jekyll_path / '_posts'
        jekyll_posts = self.extract_posts_from_directory(jekyll_posts_dir)
        print(f"Extracted {len(jekyll_posts)} posts from Jekyll repo")
        all_posts.extend(jekyll_posts)

        # Extract from private repo if provided
        if self.private_path:
            private_posts_dir = self.private_path / '_posts'
            private_posts = self.extract_posts_from_directory(private_posts_dir)
            print(f"Extracted {len(private_posts)} posts from private repo")
            all_posts.extend(private_posts)

        # Sort by date if available
        def get_date(post):
            frontmatter = post.get('frontmatter', {})
            date_str = frontmatter.get('date')
            if date_str:
                try:
                    if isinstance(date_str, str):
                        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    return date_str
                except:
                    pass
            # Try to extract date from filename
            filename = Path(post['filepath']).name
            date_match = re.match(r'(\d{4}-\d{2}-\d{2})', filename)
            if date_match:
                try:
                    return datetime.strptime(date_match.group(1), '%Y-%m-%d')
                except:
                    pass
            return datetime.min

        all_posts.sort(key=get_date)
        print(f"Total posts extracted: {len(all_posts)}")
        return all_posts

    def save_raw_data(self, posts: List[Dict[str, Any]], output_path: str):
        """Save raw extracted data to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(posts, f, indent=2, ensure_ascii=False, default=str)
        print(f"Raw data saved to {output_path}")

if __name__ == "__main__":
    # Initialize extractor
    extractor = BlogDataExtractor(
        jekyll_path="../judytuna-jekyll",
        private_path="../judytuna-private"
    )

    # Extract all posts
    posts = extractor.extract_all_posts()

    # Save raw data
    extractor.save_raw_data(posts, "raw_blog_data.json")

    print(f"\nExtraction complete! Found {len(posts)} posts.")

    # Show sample
    if posts:
        print(f"\nSample post:")
        sample = posts[len(posts)//2]  # Middle post
        print(f"Title: {sample['frontmatter'].get('title', 'No title')}")
        print(f"Content preview: {sample['content'][:200]}...")