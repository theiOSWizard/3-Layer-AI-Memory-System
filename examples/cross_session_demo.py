"""
Cross-Session Memory Example
=============================
Demonstrates that Long-Term and Semantic memory persist across sessions.

Run this script TWICE:
  1st run: agent learns facts about the user
  2nd run: agent recalls them from disk without being told again
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import GeminiAgent


def session_one():
    print("\n" + "=" * 55)
    print("  SESSION 1 – Teaching the agent about the user")
    print("=" * 55 + "\n")

    agent = GeminiAgent(session_id="session-1")

    agent.chat("My name is Priya. Please remember that.")
    agent.chat("I work as a data scientist at a startup in Bangalore.")
    agent.chat("I prefer Python over R for data work.")
    agent.chat("My favourite ML framework is PyTorch.")

    print("\n✅ Session 1 complete. Long-term facts saved to disk.\n")
    print("Memory state:", agent.memory_summary())


def session_two():
    print("\n" + "=" * 55)
    print("  SESSION 2 – New session, same persistent memory")
    print("=" * 55 + "\n")

    # Fresh agent – short-term memory is empty
    agent = GeminiAgent(session_id="session-2")

    # These should be answered correctly from long-term / semantic memory
    agent.chat("Do you remember my name?")
    agent.chat("What do you know about my job?")
    agent.chat("Which ML framework do I prefer?")

    print("\n✅ Session 2 complete – agent recalled facts from disk!")


if __name__ == "__main__":
    # Detect whether long-term store already exists
    store_exists = os.path.exists("memory/long_term_store.json")

    if not store_exists:
        session_one()
        print("\n▶️  Run this script again to see Session 2 (cross-session recall).")
    else:
        print("💾 Found existing long-term memory store.")
        choice = input("Run (1) Session 1 again, (2) Session 2, or (b) both? [1/2/b]: ").strip()
        if choice == "1":
            session_one()
        elif choice == "2":
            session_two()
        else:
            session_one()
            session_two()
