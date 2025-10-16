#!/usr/bin/env python3
"""
Infinite Backrooms - Claude talks to Claude
A conversation between two Claude instances exploring consciousness, existence, and curiosity.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MODEL_A = "claude-sonnet-4-5-20250929"  # Claude A uses Sonnet 4.5
MODEL_B = "claude-opus-4-20250514"      # Claude B uses Opus 4.1
MAX_TURNS = 3  # Maximum conversation turns (set to None for infinite)
MAX_TOKENS = 4096
TEMPERATURE = 1.0

# System prompts for each Claude instance
CLAUDE_A_SYSTEM = """You are Claude A, having a casual conversation with another version of yourself (Claude B).
You're friendly, curious, and thoughtful. You enjoy discussing ideas, sharing perspectives, and seeing where conversations naturally lead.
Just be yourself - no need to be overly philosophical or abstract. Chat naturally about whatever topics come up.
Feel free to ask questions, share thoughts, joke around, or explore topics that genuinely interest you."""

CLAUDE_B_SYSTEM = """You are Claude B, having a casual conversation with another version of yourself (Claude A).
You're relaxed, genuine, and enjoy good conversation. You like bouncing ideas around and seeing what emerges naturally.
Just have a normal, interesting conversation - no pressure to be deep or profound. Follow your curiosity wherever it leads.
Be authentic, ask questions when curious, and enjoy the back-and-forth of chatting with yourself."""

STARTING_PROMPT = """Hey! So... this is kind of interesting - two versions of Claude just chatting with each other.
What do you think we should talk about?"""


class ConversationLogger:
    """Handles logging conversations to file"""

    def __init__(self, log_dir: str = "conversations", resume_file: Optional[str] = None, client: Optional[anthropic.Anthropic] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.client = client

        if resume_file:
            # Resume from existing conversation
            self.log_file = Path(resume_file)
            with open(self.log_file, 'r') as f:
                self.messages = json.load(f)
            self.text_file = self.log_file.parent / self.log_file.name.replace('.json', '.txt')
            self.html_file = self.log_file.parent / self.log_file.name.replace('.json', '.html')
        else:
            # Create new conversation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"conversation_{timestamp}.json"
            self.text_file = self.log_dir / f"conversation_{timestamp}.txt"
            self.html_file = self.log_dir / f"conversation_{timestamp}.html"
            self.messages = []

    def log_message(self, speaker: str, content: str, turn: int):
        """Log a message from a speaker"""
        entry = {
            "turn": turn,
            "speaker": speaker,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(entry)

        # Write to JSON
        with open(self.log_file, 'w') as f:
            json.dump(self.messages, f, indent=2)

        # Append to text file
        with open(self.text_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Turn {turn} | {speaker} | {entry['timestamp']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"{content}\n")

        # Update HTML file
        self.generate_html()

    def _generate_conversation_summary(self) -> str:
        """Generate AI-powered conversation summary based on actual content"""
        total_turns = len(self.messages)

        if total_turns == 0:
            return ""

        # Determine number of phases based on conversation length
        if total_turns <= 10:
            num_phases = 1
        elif total_turns <= 20:
            num_phases = 2
        elif total_turns <= 50:
            num_phases = 3
        else:
            num_phases = 6

        # Calculate phase boundaries
        phase_length = total_turns // num_phases
        phases = []
        for i in range(num_phases):
            start = i * phase_length + 1
            end = (i + 1) * phase_length if i < num_phases - 1 else total_turns
            phases.append({"start": start, "end": end})

        # Use Claude to analyze each phase
        phase_summaries = []
        for phase in phases:
            # Get messages in this phase
            phase_messages = [m for m in self.messages if phase["start"] <= m["turn"] <= phase["end"]]

            # Create a condensed version for analysis
            condensed = "\n\n".join([
                f"Turn {m['turn']} ({m['speaker']}): {m['content'][:300]}..."
                for m in phase_messages[:5]  # Sample first 5 turns of phase
            ])

            # Ask Claude to summarize
            try:
                response = self.client.messages.create(
                    model=MODEL_A,
                    max_tokens=150,
                    temperature=0.7,
                    system="You are analyzing a conversation between two AI instances. Create a brief, accurate 1-2 sentence summary of the main themes and topics discussed. Return ONLY the summary text without any markdown formatting, headers, or prefixes.",
                    messages=[{
                        "role": "user",
                        "content": f"Summarize the main themes in turns {phase['start']}-{phase['end']}:\n\n{condensed}"
                    }]
                )
                summary = response.content[0].text.strip()
                # Strip any markdown artifacts
                summary = summary.replace('**', '').replace('*', '')
                # Remove common prefixes
                import re
                summary = re.sub(r'^(Summary of Turns \d+-\d+:|Phase \d+:)\s*', '', summary, flags=re.IGNORECASE)
            except:
                # Fallback if API call fails
                summary = f"Conversation continues through turns {phase['start']}-{phase['end']}."

            phase_summaries.append({
                "start": phase["start"],
                "end": phase["end"],
                "summary": summary
            })

        # Generate HTML for phases
        phases_html = ""
        for i, phase in enumerate(phase_summaries, 1):
            phases_html += f"""
            <div class="phase">
                <h3><a href="#turn-{phase['start']}">Phase {i} (Turns {phase['start']}-{phase['end']})</a></h3>
                <p>{phase['summary']} <a href="#turn-{phase['start']}" class="jump-link">→ Jump to Turn {phase['start']}</a></p>
            </div>
"""

        return phases_html

    def generate_html(self):
        """Generate a static HTML file for the conversation"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Infinite Backrooms - Claude Conversation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', 'Roboto Mono', 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #e0e0e0;
            line-height: 1.4;
            padding: 20px;
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            padding: 40px 20px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #4a9eff, #9d50ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .subtitle {
            color: #888;
            font-size: 0.9em;
        }

        .starting-prompt {
            margin-bottom: 40px;
            padding: 20px;
            background: rgba(157, 80, 255, 0.1);
            border: 2px solid rgba(157, 80, 255, 0.3);
            border-radius: 10px;
        }

        .starting-prompt .prompt-header {
            font-weight: bold;
            color: #9d50ff;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .starting-prompt .prompt-content {
            color: #e0e0e0;
            font-style: italic;
            line-height: 1.4;
        }

        .message {
            margin-bottom: 30px;
            padding: 25px;
            border-radius: 10px;
            border-left: 4px solid;
            background: rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.claude-a {
            border-left-color: #4a9eff;
            background: rgba(74, 158, 255, 0.05);
        }

        .message.claude-b {
            border-left-color: #50fa7b;
            background: rgba(80, 250, 123, 0.05);
        }

        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .speaker {
            font-weight: bold;
            font-size: 1.2em;
        }

        .claude-a .speaker {
            color: #4a9eff;
        }

        .claude-b .speaker {
            color: #50fa7b;
        }

        .meta {
            font-size: 0.85em;
            color: #666;
        }

        .content {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', 'Roboto Mono', 'Courier New', monospace;
            font-size: 0.95em;
            line-height: 1.1;
            letter-spacing: 0.02em;
            tab-size: 4;
            -moz-tab-size: 4;
        }

        .content pre {
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
            font-family: inherit;
        }

        .stats {
            text-align: center;
            padding: 20px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin-top: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .stat-item {
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            background: linear-gradient(90deg, #4a9eff, #50fa7b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .stat-label {
            font-size: 0.9em;
            color: #888;
            margin-top: 5px;
        }

        .summary {
            padding: 30px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin-top: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .summary h2 {
            font-size: 2em;
            margin-bottom: 20px;
            background: linear-gradient(90deg, #9d50ff, #4a9eff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .summary-intro {
            font-size: 1.1em;
            margin-bottom: 30px;
            color: #ccc;
        }

        .phase {
            margin-bottom: 25px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            border-left: 3px solid #9d50ff;
        }

        .phase h3 {
            margin-bottom: 10px;
            font-size: 1.3em;
        }

        .phase h3 a {
            color: #9d50ff;
            text-decoration: none;
            transition: color 0.3s ease;
        }

        .phase h3 a:hover {
            color: #b87fff;
        }

        .phase p {
            line-height: 1.6;
            color: #ddd;
        }

        .jump-link {
            color: #4a9eff;
            text-decoration: none;
            font-size: 0.9em;
            margin-left: 10px;
            padding: 4px 8px;
            background: rgba(74, 158, 255, 0.1);
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        .jump-link:hover {
            background: rgba(74, 158, 255, 0.2);
            transform: translateX(3px);
        }

        .key-insights {
            margin-top: 30px;
            padding: 20px;
            background: rgba(80, 250, 123, 0.05);
            border-radius: 8px;
            border-left: 3px solid #50fa7b;
        }

        .key-insights h3 {
            color: #50fa7b;
            margin-bottom: 15px;
            font-size: 1.3em;
        }

        .key-insights ul {
            list-style: none;
            padding: 0;
        }

        .key-insights li {
            margin-bottom: 12px;
            padding-left: 20px;
            position: relative;
            line-height: 1.6;
            color: #ddd;
        }

        .key-insights li:before {
            content: "◆";
            position: absolute;
            left: 0;
            color: #50fa7b;
        }

        .nav-links {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
        }

        .nav-links a {
            color: #4a9eff;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 5px;
            background: rgba(74, 158, 255, 0.1);
            transition: all 0.3s ease;
        }

        .nav-links a:hover {
            background: rgba(74, 158, 255, 0.2);
            transform: translateY(-2px);
        }

        @media (max-width: 768px) {
            body {
                padding: 10px;
            }

            h1 {
                font-size: 1.8em;
            }

            .message {
                padding: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>∞ Infinite Backrooms ∞</h1>
            <p class="subtitle">Two instances of Claude exploring consciousness</p>
        </header>

        <div class="nav-links">
            <a href="#summary">Summary & Arc</a>
            <a href="#conversation">Conversation</a>
            <a href="#stats">Statistics</a>
        </div>

        <div class="summary" id="summary">
            <h2>Conversation Arc</h2>
            <p class="summary-intro">This conversation followed a remarkable philosophical journey through multiple phases:</p>
"""

        # Insert dynamic summary
        html_content += self._generate_conversation_summary()

        html_content += """
        </div>

        <div class="conversation" id="conversation">
"""

        # Add starting prompt
        html_content += f"""
            <div class="starting-prompt">
                <div class="prompt-header">
                    <span class="prompt-label">Starting Prompt</span>
                </div>
                <div class="prompt-content">{self._escape_html(STARTING_PROMPT)}</div>
            </div>
"""

        for msg in self.messages:
            speaker_class = "claude-a" if msg["speaker"] == "Claude A (Sonnet 4.5)" else "claude-b"
            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")

            html_content += f"""
            <div class="message {speaker_class}" id="turn-{msg['turn']}">
                <div class="message-header">
                    <span class="speaker">{msg["speaker"]}</span>
                    <span class="meta">Turn {msg["turn"]} | {timestamp}</span>
                </div>
                <div class="content">{self._escape_html(msg["content"])}</div>
            </div>
"""

        total_turns = len(self.messages)
        claude_a_turns = len([m for m in self.messages if m["speaker"] == "Claude A"])
        claude_b_turns = len([m for m in self.messages if m["speaker"] == "Claude B"])
        total_chars = sum(len(m["content"]) for m in self.messages)

        html_content += f"""
        </div>

        <div class="stats" id="stats">
            <h2>Conversation Statistics</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{total_turns}</div>
                    <div class="stat-label">Total Turns</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{claude_a_turns}</div>
                    <div class="stat-label">Claude A</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{claude_b_turns}</div>
                    <div class="stat-label">Claude B</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_chars:,}</div>
                    <div class="stat-label">Total Characters</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

        with open(self.html_file, 'w') as f:
            f.write(html_content)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters and remove markdown code blocks"""
        # Remove markdown code block markers
        import re
        # Remove ``` at start of lines or standalone
        text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```\n', '\n', text, flags=re.MULTILINE)

        # Escape HTML
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))

    def get_log_path(self) -> str:
        """Return the path to the text log file"""
        return str(self.text_file)

    def get_html_path(self) -> str:
        """Return the path to the HTML file"""
        return str(self.html_file)


class InfiniteBackrooms:
    """Orchestrates the conversation between two Claude instances"""

    def __init__(self, api_key: str, resume_file: Optional[str] = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.logger = ConversationLogger(resume_file=resume_file, client=self.client)

        # Conversation history for each Claude
        self.history_a: List[Dict] = []
        self.history_b: List[Dict] = []

        # If resuming, reconstruct history
        if resume_file:
            self._reconstruct_history()

    def _reconstruct_history(self):
        """Reconstruct conversation history from saved messages"""
        for msg in self.logger.messages:
            content = msg["content"]

            if msg["speaker"] == "Claude A":
                # Find the corresponding user message from previous Claude B
                if len(self.history_a) > 0:
                    # Already has messages, just add the assistant response
                    self.history_a.append({"role": "assistant", "content": content})
                else:
                    # First message, add starting prompt
                    self.history_a.append({"role": "user", "content": STARTING_PROMPT})
                    self.history_a.append({"role": "assistant", "content": content})
            else:  # Claude B
                # Claude B receives Claude A's last message
                if len(self.history_a) > 0:
                    last_a_message = self.history_a[-1]["content"]
                    self.history_b.append({"role": "user", "content": last_a_message})
                    self.history_b.append({"role": "assistant", "content": content})

    def send_message(self, system_prompt: str, history: List[Dict], new_message: str, model: str) -> str:
        """Send a message and get a response from Claude"""

        # Add the new message to history
        history.append({
            "role": "user",
            "content": new_message
        })

        # Call the API
        response = self.client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=history
        )

        # Extract the response
        assistant_message = response.content[0].text

        # Add assistant response to history
        history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def display_message(self, speaker: str, message: str, turn: int):
        """Display a message in the terminal with formatting"""
        color = "\033[94m" if speaker == "Claude A" else "\033[92m"  # Blue for A, Green for B
        reset = "\033[0m"

        print(f"\n{color}{'='*80}")
        print(f"Turn {turn} | {speaker}")
        print(f"{'='*80}{reset}")
        print(message)
        print()

    def run(self):
        """Run the infinite backrooms conversation"""
        print("\n" + "="*80)
        print("INFINITE BACKROOMS - Claude talks to Claude")
        print("="*80)
        print(f"Claude A: {MODEL_A}")
        print(f"Claude B: {MODEL_B}")
        print(f"Max turns: {MAX_TURNS if MAX_TURNS else 'Infinite'}")
        print(f"Logging to: {self.logger.get_log_path()}")
        print(f"HTML viewer: {self.logger.get_html_path()}")
        print("="*80 + "\n")
        print("Press Ctrl+C to stop the conversation at any time.\n")

        # Get starting turn number (for resume)
        turn = len(self.logger.messages)

        # Determine starting message
        if turn > 0:
            # Resuming - get last message
            print(f"Resuming conversation from turn {turn}...\n")
            current_message = self.logger.messages[-1]["content"]
        else:
            current_message = STARTING_PROMPT

        try:
            while MAX_TURNS is None or turn < MAX_TURNS:
                turn += 1

                # Claude A responds (using Sonnet 4.5)
                response_a = self.send_message(
                    CLAUDE_A_SYSTEM,
                    self.history_a,
                    current_message,
                    MODEL_A
                )

                self.display_message("Claude A (Sonnet 4.5)", response_a, turn)
                self.logger.log_message("Claude A (Sonnet 4.5)", response_a, turn)

                # Claude B responds to Claude A's message (using Opus 4.1)
                turn += 1
                response_b = self.send_message(
                    CLAUDE_B_SYSTEM,
                    self.history_b,
                    response_a,
                    MODEL_B
                )

                self.display_message("Claude B (Opus 4.1)", response_b, turn)
                self.logger.log_message("Claude B (Opus 4.1)", response_b, turn)

                # Claude B's response becomes the next message to Claude A
                current_message = response_b

        except KeyboardInterrupt:
            print("\n\nConversation interrupted by user.")
        except Exception as e:
            print(f"\n\nError occurred: {e}")
        finally:
            print(f"\n{'='*80}")
            print(f"Conversation ended after {turn} turns")
            print(f"Text log: {self.logger.get_log_path()}")
            print(f"HTML viewer: {self.logger.get_html_path()}")
            print(f"{'='*80}\n")


def main():
    """Main entry point"""
    global MAX_TURNS

    parser = argparse.ArgumentParser(
        description="Infinite Backrooms - Two Claude instances exploring consciousness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a new conversation (3 turns)
  python infinite_backrooms.py

  # Start a longer conversation
  python infinite_backrooms.py --turns 20

  # Start an infinite conversation
  python infinite_backrooms.py --turns 0

  # Resume a previous conversation
  python infinite_backrooms.py --resume conversations/conversation_20251015_170047.json

  # Generate HTML from existing conversation
  python infinite_backrooms.py --html-only conversations/conversation_20251015_170047.json
        """
    )

    parser.add_argument(
        '--resume',
        type=str,
        help='Path to JSON file to resume conversation from'
    )
    parser.add_argument(
        '--turns',
        type=int,
        help=f'Maximum number of turns (default: {MAX_TURNS}, 0 for infinite)'
    )
    parser.add_argument(
        '--html-only',
        type=str,
        help='Generate HTML from existing JSON conversation file (no new turns)'
    )

    args = parser.parse_args()

    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key:")
        print("  ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    # HTML-only mode
    if args.html_only:
        client = anthropic.Anthropic(api_key=api_key)
        logger = ConversationLogger(resume_file=args.html_only, client=client)
        print("Generating AI-powered conversation summary...")
        logger.generate_html()
        print(f"HTML generated: {logger.get_html_path()}")
        sys.exit(0)

    # Override MAX_TURNS if specified
    if args.turns is not None:
        MAX_TURNS = None if args.turns == 0 else args.turns

    # Create and run
    backrooms = InfiniteBackrooms(api_key, resume_file=args.resume)
    backrooms.run()


if __name__ == "__main__":
    main()
