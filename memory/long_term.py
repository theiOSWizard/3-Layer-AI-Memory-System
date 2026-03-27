"""
Long-Term Memory
================
Persists structured key-value facts to a JSON file on disk.
Survives restarts – the agent "remembers" users across sessions.

Characteristics:
  - Stored as a flat JSON dict  { "fact_key": "fact_value", ... }
  - Persists across process restarts
  - Best for: names, preferences, settings, important dates
  - NOT suited for fuzzy / semantic search (use SemanticMemory for that)

Typical usage:
    mem = LongTermMemory("memory/store.json")
    mem.save_fact("user_name", "Alice")
    mem.save_fact("user_city", "Kolkata")
    print(mem.get_fact("user_name"))   # → "Alice"
    print(mem.get_all_facts())         # → {"user_name": "Alice", "user_city": "Kolkata"}
"""

import json
import os
from typing import Dict, Optional


class LongTermMemory:
    """
    File-backed key-value fact store.

    The JSON file is read on init and written on every mutation,
    so facts are always persisted even if the process crashes.
    """

    def __init__(self, storage_file: str = "memory/long_term_store.json"):
        self.storage_file = storage_file
        os.makedirs(os.path.dirname(storage_file), exist_ok=True)
        self._facts: Dict[str, str] = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        """Load facts from disk, or return empty dict if file doesn't exist."""
        if os.path.exists(self.storage_file):
            with open(self.storage_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self):
        """Write current facts to disk."""
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(self._facts, f, indent=2, ensure_ascii=False)

    # ── Public API ────────────────────────────────────────────────────────────

    def save_fact(self, key: str, value: str) -> str:
        """Store or overwrite a fact. Returns a confirmation string."""
        self._facts[key] = value
        self._save()
        return f"Saved: {key} = {value}"

    def get_fact(self, key: str) -> Optional[str]:
        """Retrieve a single fact by key."""
        return self._facts.get(key)

    def get_all_facts(self) -> dict:
        """Return all stored facts."""
        return dict(self._facts)

    def delete_fact(self, key: str) -> str:
        """Remove a fact. Returns confirmation or error."""
        if key in self._facts:
            del self._facts[key]
            self._save()
            return f"Deleted fact: {key}"
        return f"Fact '{key}' not found."

    def clear_all(self):
        """Wipe all facts."""
        self._facts = {}
        self._save()

    def __repr__(self) -> str:
        return f"LongTermMemory(facts={len(self._facts)}, file={self.storage_file})"
