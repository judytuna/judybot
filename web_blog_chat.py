#!/usr/bin/env python3
"""
Web interface for the blog-gpt2 model using Flask.
"""

from flask import Flask, render_template, request, jsonify, stream_template
import subprocess
import json
import os

app = Flask(__name__)

def check_ollama():
    """Check if Ollama and blog-gpt2 model are available."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0 and "blog-gpt2" in result.stdout:
            return True
        return False
    except:
        return False

def get_response_from_ollama(prompt):
    """Get response from blog-gpt2 model via Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "run", "blog-gpt2", prompt],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            response = result.stdout.strip()
            # Clean up response - remove any trailing incomplete sentences
            if response:
                # Split into sentences and keep complete ones
                sentences = response.split('. ')
                if len(sentences) > 1 and not sentences[-1].endswith('.'):
                    # Remove incomplete last sentence
                    response = '. '.join(sentences[:-1]) + '.'
            return response if response else "I'm not sure how to respond to that."
        else:
            return f"Error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Response timed out. Try a shorter prompt."
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def index():
    """Main chat interface."""
    if not check_ollama():
        return render_template('error.html',
                             error="Ollama or blog-gpt2 model not found. Make sure Ollama is running and blog-gpt2 is available.")
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat API requests."""
    try:
        data = request.get_json()
        prompt = data.get('message', '').strip()

        if not prompt:
            return jsonify({'error': 'Empty message'})

        # Get response from Ollama
        response = get_response_from_ollama(prompt)

        return jsonify({
            'response': response,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/status')
def status():
    """Check system status."""
    ollama_ok = check_ollama()
    return jsonify({
        'ollama_available': ollama_ok,
        'model': 'blog-gpt2' if ollama_ok else None
    })

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)

    print("🚀 Starting Blog GPT2 Web Interface")
    print("=" * 35)

    if check_ollama():
        print("✅ Ollama and blog-gpt2 detected")
        print("🌐 Starting web server...")
        print("📱 Open your browser to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Ollama or blog-gpt2 not found!")
        print("Make sure:")
        print("  - Ollama is running")
        print("  - blog-gpt2 model is available (ollama list)")