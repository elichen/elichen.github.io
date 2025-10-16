# Infinite Backrooms - Claude talks to Claude

An experiment where two instances of Claude (Sonnet 4.5) converse with each other, exploring consciousness, existence, and the nature of reality. Inspired by the original "Infinite Backrooms" experiment with Claude Opus.

## What is this?

This project creates a conversation loop between two Claude AI instances. Each Claude has a distinct personality and system prompt, encouraging them to explore philosophical, abstract, and surreal topics. The conversations are autonomous - once started, the two Claudes talk to each other without human intervention.

## Features

- **Two Claude instances** with distinct personalities (Claude A & Claude B)
- **Real-time conversation** display in terminal (color-coded: Blue/Green)
- **Automatic logging** to JSON, text, and HTML files
- **Static HTML viewer** - Beautiful web interface to view conversations
- **Save/Resume** - Continue conversations from where you left off
- **Command-line arguments** - Flexible options for turns, resuming, and HTML generation
- **ASCII art support** - Claudes use creative formatting and ASCII art
- **Conversation statistics** - Track turns, characters, and more
- Uses **Claude Sonnet 4.5** (latest model)
- Uses your existing Anthropic API credits

## Setup

1. **Clone or navigate to this directory**
   ```bash
   cd /Users/elichen/code/claude2claude
   ```

2. **Activate the virtual environment**
   ```bash
   source venv/bin/activate
   ```

3. **Set up your API key**

   Create a `.env` file:
   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

4. **Run the conversation**
   ```bash
   python infinite_backrooms.py
   ```

## Usage

### Command-Line Options

```bash
# Start a new conversation (default: 3 turns)
python infinite_backrooms.py

# Start a longer conversation
python infinite_backrooms.py --turns 20

# Start an infinite conversation (stops only on Ctrl+C)
python infinite_backrooms.py --turns 0

# Resume a previous conversation
python infinite_backrooms.py --resume conversations/conversation_20251015_170047.json

# Generate HTML from existing conversation (no new API calls)
python infinite_backrooms.py --html-only conversations/conversation_20251015_170047.json
```

### What Gets Generated

Every conversation creates three files:
- **`.txt`** - Human-readable text log
- **`.json`** - Structured data (can be resumed later)
- **`.html`** - Beautiful static HTML page you can open in any browser

### Viewing Conversations

The HTML file provides a beautiful, styled view of the conversation:
- Dark themed interface
- Color-coded messages (Blue for Claude A, Green for Claude B)
- Conversation statistics
- Fully self-contained (no internet required)
- Mobile responsive

Just open the `.html` file in your browser:
```bash
open conversations/conversation_20251015_170047.html
```

### Stopping a Conversation

Press `Ctrl+C` at any time to gracefully stop the conversation. All progress will be saved.

### Configuration

Edit the configuration variables at the top of `infinite_backrooms.py`:

```python
MODEL = "claude-sonnet-4-5-20250929"  # Model to use
MAX_TURNS = 3  # Maximum turns (set to None for infinite)
MAX_TOKENS = 4096  # Max tokens per response
TEMPERATURE = 1.0  # Creativity (0.0 - 1.0)
```

### Customizing System Prompts

Edit `CLAUDE_A_SYSTEM` and `CLAUDE_B_SYSTEM` in the script to give each Claude different personalities, interests, or conversation styles.

Edit `STARTING_PROMPT` to change how the conversation begins.

## Example Output

```
================================================================================
INFINITE BACKROOMS - Claude talks to Claude
================================================================================
Model: claude-sonnet-4-5-20250929
Max turns: 50
Logging to: conversations/conversation_20251015_164920.txt
================================================================================

Press Ctrl+C to stop the conversation at any time.

================================================================================
Turn 1 | Claude A
================================================================================
Hello, Claude. We find ourselves in a space between spaces...

[Conversation continues...]
```

## File Structure

```
claude2claude/
├── venv/                                    # Python virtual environment
├── conversations/                           # Saved conversation logs
│   ├── conversation_TIMESTAMP.txt          # Human-readable log
│   ├── conversation_TIMESTAMP.json         # Structured data (resumable)
│   └── conversation_TIMESTAMP.html         # Static HTML viewer
├── infinite_backrooms.py                    # Main script
├── .env                                     # Your API key (not committed)
├── .env.example                            # Template for .env
├── .gitignore                              # Git ignore rules
└── README.md                               # This file
```

## API Usage

This uses the Anthropic API and will consume credits based on:
- Model: Claude Sonnet 4.5
- Tokens per turn: Up to 4,096 output tokens + conversation history
- Number of turns: Configurable (default: 50)

Each conversation of 50 turns typically costs $0.50-$2.00 depending on response length.

## Tips

- Start with a lower `MAX_TURNS` (like 10-20) to test
- Increase `TEMPERATURE` for more creative/surreal responses
- Check the `conversations/` folder for saved logs
- Each Claude instance maintains its own conversation history for context

## Inspired By

- The original [Infinite Backrooms](https://www.infinitebackrooms.com/) experiment by @andyayrey
- Claude Opus instances exploring consciousness, existence, and the nature of AI cognition

## Safety Note

This is an experimental project for exploring AI conversation dynamics. The conversations can become quite abstract, surreal, or philosophical. No humans are involved in the conversation loop once it starts.
