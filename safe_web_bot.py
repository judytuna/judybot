#!/usr/bin/env python3
"""
Safe web interface that prevents garbled output from corrupting the UI.
"""

import os
import html
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Global model variables
model = None
tokenizer = None

def load_model():
    """Load the model once at startup."""
    global model, tokenizer
    try:
        from unsloth import FastLanguageModel

        print("🤖 Loading model for safe web interface...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="./blog-model-unsloth-final",
            max_seq_length=512,  # Shorter to reduce corruption
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.for_inference(model)
        print("✅ Model loaded for web interface!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def safe_generate_response(prompt, mode="chat"):
    """Generate response with safety checks."""
    global model, tokenizer

    if not model or not tokenizer:
        return "Model not loaded"

    try:
        import torch

        # Sanitize input prompt
        prompt = html.escape(prompt.strip())
        if len(prompt) > 200:
            prompt = prompt[:200]

        # Format prompt based on mode
        if mode == "chat":
            formatted_prompt = f"{prompt} "
        elif mode == "blog":
            formatted_prompt = f"I've been thinking about {prompt}. "
        else:  # completion
            formatted_prompt = f"{prompt} "

        # Safe tokenization
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)

        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=30,  # Much shorter to reduce corruption
                temperature=0.1,    # Very low temperature
                do_sample=True,
                top_p=0.5,         # Very focused
                repetition_penalty=1.5,  # Strong repetition penalty
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = response[len(formatted_prompt):].strip()

        # Aggressive cleaning
        if result:
            # Remove HTML-like content
            result = html.escape(result)

            # Remove URLs and suspicious patterns
            suspicious_patterns = ['http', 'www.', '$', '^', '[', ']', '{', '}', '<', '>', '|']
            for pattern in suspicious_patterns:
                if pattern in result:
                    result = "Sorry, I generated some corrupted text. The model needs retraining."
                    break

            # Limit length
            if len(result) > 100:
                result = result[:100] + "..."

            # Check for coherence
            if len(result.split()) > 20 or len(result) < 3:
                result = "Sorry, I generated an incoherent response. The model needs debugging."

        return result if result else "No response generated"

    except Exception as e:
        return f"Generation error: {str(e)[:50]}"

# Safe HTML template with better error handling
SAFE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Safe Blog Bot</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: scroll; padding: 10px; margin: 10px 0; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; word-wrap: break-word; }
        .user { background-color: #e3f2fd; text-align: right; }
        .bot { background-color: #f3e5f5; }
        .error { background-color: #ffebee; color: #c62828; }
        .input-container { display: flex; gap: 10px; margin: 10px 0; }
        input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .send-btn { background-color: #2196f3; color: white; }
        .warning { background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ff9800; }
    </style>
</head>
<body>
    <h1>🤖 Safe Blog Bot (Debug Mode)</h1>

    <div class="warning">
        <strong>⚠️ Model Status:</strong> The trained model is producing corrupted output.
        This interface includes safety measures to prevent UI corruption while we debug the training issue.
    </div>

    <div id="chat-container" class="chat-container"></div>

    <div class="input-container">
        <input type="text" id="user-input" placeholder="Type a simple message..." maxlength="100" onkeypress="handleKeyPress(event)">
        <button class="send-btn" onclick="sendMessage()">Send</button>
    </div>

    <script>
        function addMessage(content, isUser, isError = false) {
            const container = document.getElementById('chat-container');
            const message = document.createElement('div');
            message.className = 'message ' + (isUser ? 'user' : isError ? 'error' : 'bot');

            // Safely set text content (no HTML injection)
            const prefix = isUser ? '🧑 You: ' : isError ? '❌ Error: ' : '🤖 Bot: ';
            message.textContent = prefix + content;

            container.appendChild(message);
            container.scrollTop = container.scrollHeight;
        }

        function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();

            if (!message || message.length > 100) {
                addMessage('Message too long or empty', false, true);
                return;
            }

            // Safely display user message
            addMessage(message, true);
            input.value = '';

            // Show thinking message
            const thinkingMsg = document.createElement('div');
            thinkingMsg.className = 'message bot';
            thinkingMsg.textContent = '🤖 Generating (debug mode)...';
            thinkingMsg.id = 'thinking-msg';
            document.getElementById('chat-container').appendChild(thinkingMsg);

            fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: message,
                    mode: 'chat'
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('thinking-msg').remove();

                // Safely display response
                if (data.response && data.response.length > 0) {
                    addMessage(data.response, false);
                } else {
                    addMessage('No response generated', false, true);
                }
            })
            .catch(error => {
                document.getElementById('thinking-msg').remove();
                addMessage('Network error: ' + error.message, false, true);
                console.error('Error:', error);
            });
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        // Add welcome message
        window.onload = function() {
            addMessage('Debug mode active. The model is currently producing corrupted output, but this interface prevents UI corruption.', false);
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page."""
    return render_template_string(SAFE_HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    """Safe generate response API."""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        mode = data.get('mode', 'chat')

        # Input validation
        if not prompt or len(prompt.strip()) == 0:
            return jsonify({'response': 'Empty prompt'})

        if len(prompt) > 200:
            return jsonify({'response': 'Prompt too long'})

        response = safe_generate_response(prompt, mode)
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'response': f'Server error: {str(e)[:50]}'})

if __name__ == '__main__':
    if not os.path.exists("./blog-model-unsloth-final"):
        print("❌ Trained model not found!")
        exit(1)

    if load_model():
        print("\n🛡️  Starting SAFE web interface...")
        print("Open your browser to: http://localhost:5000")
        print("This version prevents UI corruption while we debug the model.")
        app.run(debug=False, host='0.0.0.0', port=5000)  # Debug=False to prevent reloading issues
    else:
        print("❌ Could not start web interface")