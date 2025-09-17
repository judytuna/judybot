#!/usr/bin/env python3
"""
Simple web interface for the blog bot using Flask.
"""

import os
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

        print("🤖 Loading model for web interface...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="./blog-model-unsloth-final",
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.for_inference(model)
        print("✅ Model loaded for web interface!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def generate_response(prompt, mode="chat"):
    """Generate response from the model."""
    global model, tokenizer

    if not model or not tokenizer:
        return "Model not loaded"

    try:
        # Format prompt based on mode
        if mode == "chat":
            formatted_prompt = f"{prompt} "
        elif mode == "blog":
            formatted_prompt = f"I've been thinking about {prompt}. "
        else:  # completion
            formatted_prompt = f"{prompt} "

        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        if model.device.type == "cuda":
            inputs = inputs.cuda()

        outputs = model.generate(
            inputs,
            max_new_tokens=60,
            temperature=0.4,
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = response[len(formatted_prompt):].strip()

        # Clean up response
        if result:
            for marker in ['\n\n', '---', '###']:
                if marker in result:
                    result = result.split(marker)[0].strip()

        return result if result else "..."

    except Exception as e:
        return f"Error: {e}"

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Personal Blog Bot</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: scroll; padding: 10px; margin: 10px 0; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background-color: #e3f2fd; text-align: right; }
        .bot { background-color: #f3e5f5; }
        .input-container { display: flex; gap: 10px; margin: 10px 0; }
        input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .send-btn { background-color: #2196f3; color: white; }
        .mode-btn { background-color: #9c27b0; color: white; margin: 5px; }
        .active-mode { background-color: #4caf50; }
        .header { text-align: center; color: #333; }
        .mode-info { background-color: #fff3e0; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1 class="header">🤖 Personal Blog Bot</h1>
    <p class="header">Chat with your blog personality!</p>

    <div class="mode-info">
        <strong>Current Mode: <span id="current-mode">Chat</span></strong>
        <div>
            <button class="mode-btn active-mode" onclick="setMode('chat')">💬 Chat</button>
            <button class="mode-btn" onclick="setMode('blog')">📝 Blog Assistant</button>
            <button class="mode-btn" onclick="setMode('complete')">🔮 Complete</button>
        </div>
        <div id="mode-description">Ask questions or have conversations with your blog personality!</div>
    </div>

    <div id="chat-container" class="chat-container"></div>

    <div class="input-container">
        <input type="text" id="user-input" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
        <button class="send-btn" onclick="sendMessage()">Send</button>
    </div>

    <script>
        let currentMode = 'chat';

        const modeDescriptions = {
            'chat': 'Ask questions or have conversations with your blog personality!',
            'blog': 'Get help writing blog posts. Try: "programming", "travel", "learning"',
            'complete': 'Start a sentence and let the bot complete it. Try: "Today I was thinking"'
        };

        function setMode(mode) {
            currentMode = mode;
            document.getElementById('current-mode').textContent =
                mode === 'chat' ? 'Chat' : mode === 'blog' ? 'Blog Assistant' : 'Complete';

            document.getElementById('mode-description').textContent = modeDescriptions[mode];

            // Update button styles
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active-mode'));
            event.target.classList.add('active-mode');

            // Update placeholder
            const input = document.getElementById('user-input');
            if (mode === 'chat') {
                input.placeholder = 'Ask a question...';
            } else if (mode === 'blog') {
                input.placeholder = 'Enter blog topic...';
            } else {
                input.placeholder = 'Start a sentence...';
            }
        }

        function addMessage(content, isUser) {
            const container = document.getElementById('chat-container');
            const message = document.createElement('div');
            message.className = 'message ' + (isUser ? 'user' : 'bot');
            message.textContent = (isUser ? '🧑 You: ' : '🤖 BlogBot: ') + content;
            container.appendChild(message);
            container.scrollTop = container.scrollHeight;
        }

        function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();

            if (!message) return;

            addMessage(message, true);
            input.value = '';

            // Show thinking message
            const thinkingMsg = document.createElement('div');
            thinkingMsg.className = 'message bot';
            thinkingMsg.textContent = '🤖 Thinking...';
            thinkingMsg.id = 'thinking-msg';
            document.getElementById('chat-container').appendChild(thinkingMsg);

            fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: message,
                    mode: currentMode
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('thinking-msg').remove();

                let displayText = data.response;
                if (currentMode === 'complete') {
                    displayText = message + ' ' + data.response;
                    // Replace the user message with the complete version
                    const messages = document.querySelectorAll('.message.user');
                    const lastUserMsg = messages[messages.length - 1];
                    lastUserMsg.textContent = '🧑 You: ' + displayText;
                    return; // Don't add bot message for completion mode
                }

                addMessage(displayText, false);
            })
            .catch(error => {
                document.getElementById('thinking-msg').remove();
                addMessage('Sorry, there was an error!', false);
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
            addMessage('Hi! I\\'m your personal blog bot, trained on your writing. Choose a mode and start chatting!', false);
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate response API."""
    data = request.json
    prompt = data.get('prompt', '')
    mode = data.get('mode', 'chat')

    response = generate_response(prompt, mode)

    return jsonify({'response': response})

if __name__ == '__main__':
    if not os.path.exists("./blog-model-unsloth-final"):
        print("❌ Trained model not found!")
        exit(1)

    if load_model():
        print("\n🌐 Starting web interface...")
        print("Open your browser to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Could not start web interface")