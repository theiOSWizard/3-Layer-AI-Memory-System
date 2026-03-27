"""
Short-Term Memory
=================
Stores the current conversation history in a Python list (in-memory).
This IS the context window – Gemini sees all of these messages.

Characteristics:
  - Lives only for the current session / process
  - Automatically included in every API call
  - Bounded by max_messages to avoid hitting token limits
  - Oldest messages are dropped when the limit is reached (sliding window)
"""

from typing import Literal


class ShortTermMemory:
    """
    Sliding-window conversation buffer.

    Example:
        mem = ShortTermMemory(max_messages=6)
        mem.add("user",  "Hi, my name is Alice.")
        mem.add("model", "Hello Alice! How can I help?")
        mem.add("user",  "What's 2+2?")
        print(mem.get_history_for_gemini())
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: list[dict] = []

    def add(self, role: Literal["user", "model"], content: str):
        """Add a message; evict oldest if over the limit."""
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            # Always drop in pairs (user + model) to keep conversation coherent
            self.messages = self.messages[2:]

    def get_history_for_gemini(self) -> list[dict]:
        """
        Convert stored messages to Gemini's expected format:
        [{"role": "user"|"model", "parts": [{"text": "..."}]}, ...]
        """
        return [
            {"role": msg["role"], "parts": [{"text": msg["content"]}]}
            for msg in self.messages
        ]

    def clear(self):
        """Wipe all conversation history."""
        self.messages = []

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory("
            f"messages={len(self.messages)}, "
            f"max={self.max_messages})"
        )
