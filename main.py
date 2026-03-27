"""
Interactive Demo
================
Run this file to chat with the agent interactively.

Usage:
    python main.py

Commands:
    /memory   – print a snapshot of all memory layers
    /clear    – clear short-term memory (simulates a new session)
    /quit     – exit
"""

from agent import GeminiAgent


def main():
    print("=" * 60)
    print("  Gemini 2.5 Flash Agent — 3-Layer Memory System Demo")
    print("=" * 60)
    print("Commands: /memory  /clear  /quit\n")

    agent = GeminiAgent(session_id="demo-session")

    # Seed a few example interactions to showcase memory
    _run_demo_seed(agent)

    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye!")
            break

        if user_input == "/memory":
            summary = agent.memory_summary()
            print("\n📦 Memory Snapshot:")
            print(f"  Short-Term : {summary['short_term_messages']} messages")
            print(f"  Long-Term  : {summary['long_term_facts']}")
            print(f"  Semantic   : {summary['semantic_entries']} entries\n")
            continue

        if user_input == "/clear":
            agent.clear_short_term()
            continue

        agent.chat(user_input)


def _run_demo_seed(agent: GeminiAgent):
    """
    Run a few automatic messages to demonstrate how each memory layer
    gets populated, then drop into the interactive loop.
    """
    print("── Running demo seed messages ──────────────────────────────")

    demo_messages = [
        "Hi! My name is Arjun and I'm a Python developer from Durgapur.",
        "I prefer detailed technical explanations and code examples.",
        "What is the time complexity of binary search?",
        "Can you calculate 2 ** 16 for me?",
    ]

    for msg in demo_messages:
        agent.chat(msg)

    print("\n── Demo seed complete. Now chatting interactively. ─────────")
    print("Try: 'What's my name?' or 'What do you remember about me?'\n")


if __name__ == "__main__":
    main()
