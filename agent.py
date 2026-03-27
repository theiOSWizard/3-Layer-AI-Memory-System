"""
Gemini 2.5 Flash Agent with Full Memory System
================================================
Demonstrates all 3 memory types:
  1. Short-Term Memory  (in-context / conversation history)
  2. Long-Term Memory   (persistent JSON file storage)
  3. Semantic Memory    (vector search via ChromaDB)
"""

import os
import json
import time
import re
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.semantic import SemanticMemory
from tools.tool_executor import execute_tool, get_tool_declarations

load_dotenv()

class GeminiAgent:
    """
    An LLM Agent powered by Gemini 2.5 Flash with a 3-layer memory system.

    Memory Architecture:
    ┌─────────────────────────────────────────────┐
    │  SHORT-TERM   │ Last N messages (in context) │
    │  LONG-TERM    │ Persistent facts (JSON file) │
    │  SEMANTIC     │ Similarity search (ChromaDB) │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.model_name = "gemini-2.5-flash"
        self.api_keys = self._load_api_keys()
        self.active_key_index = 0

        # Initialise all 3 memory layers
        self.short_term = ShortTermMemory(max_messages=20)
        self.long_term = LongTermMemory(storage_file="memory/long_term_store.json")
        self.semantic = SemanticMemory(collection_name="agent_knowledge")

        # Load the Gemini model with tools
        self.model = self._build_model()

        print(f"✅ Agent initialised  |  Session: {session_id}")
        print(
            f"   Gemini keys loaded: {len(self.api_keys)}"
            f"  |  Active key: {self.active_key_index + 1}"
        )
        print("   Memory layers: Short-Term ✓  Long-Term ✓  Semantic ✓\n")

    # ── System Prompt ─────────────────────────────────────────────────────────

    def _load_api_keys(self) -> list[str]:
        """Load Gemini API keys from env vars, including future numbered keys."""
        keys = []

        comma_separated_keys = os.getenv("GEMINI_API_KEYS", "")
        if comma_separated_keys:
            keys.extend(
                key.strip()
                for key in comma_separated_keys.split(",")
                if key.strip()
            )

        indexed_keys: list[tuple[int, str]] = []
        for env_name, value in os.environ.items():
            if not value:
                continue

            if env_name == "GEMINI_API_KEY":
                indexed_keys.append((0, value))
                continue

            match = re.fullmatch(r"GEMINI_API_KEY_(\d+)", env_name)
            if match:
                indexed_keys.append((int(match.group(1)), value))

        indexed_keys.sort(key=lambda item: item[0])
        keys.extend(value for _, value in indexed_keys)

        deduped_keys = []
        seen = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            deduped_keys.append(key)

        if not deduped_keys:
            raise KeyError(
                "No Gemini API keys found. Set GEMINI_API_KEY or GEMINI_API_KEY_2, "
                "GEMINI_API_KEY_3, ... in your environment."
            )

        return deduped_keys

    def _build_model(self):
        """Configure Gemini for the active key and return a model instance."""
        genai.configure(api_key=self.api_keys[self.active_key_index])
        return genai.GenerativeModel(
            model_name=self.model_name,
            tools=get_tool_declarations(),
            system_instruction=self._build_system_prompt(),
        )

    def _advance_to_next_key(self, error: Exception) -> bool:
        """Rotate to the next key when one is exhausted."""
        if self.active_key_index >= len(self.api_keys) - 1:
            return False

        failed_key_number = self.active_key_index + 1
        self.active_key_index += 1
        self.model = self._build_model()
        print(
            "⚠️  Gemini quota/rate limit hit on "
            f"key {failed_key_number}; switching to key {self.active_key_index + 1}."
        )
        return True

    def _start_chat(self, history):
        """Create a chat session from the current model."""
        return self.model.start_chat(history=history)

    def _send_message_with_failover(self, session, message):
        """
        Send a message and automatically retry on the next configured key when
        the current key is rate-limited or out of quota.
        """
        while True:
            try:
                response = session.send_message(message)
                return session, response
            except Exception as error:
                if not self._is_quota_error(error) or not self._advance_to_next_key(error):
                    raise
                session = self._start_chat(session.history)

    def _build_system_prompt(self) -> str:
        return """You are a helpful AI assistant with a 3-layer memory system.

MEMORY LAYERS YOU HAVE ACCESS TO:
1. Short-Term Memory  – the conversation above (automatic, in-context)
2. Long-Term Memory   – persistent facts saved across sessions (tools: save_fact, get_facts)
3. Semantic Memory    – knowledge stored by meaning for fuzzy search (tools: store_knowledge, search_knowledge)

WHEN TO USE EACH LAYER:
- Short-Term  : refer to it naturally – it is already in your context window
- Long-Term   : use save_fact() for important user preferences, names, dates, settings
                use get_facts() to recall them at the start of a session
- Semantic    : use store_knowledge() after learning something substantive
                use search_knowledge() when you need to recall related past knowledge

Always proactively save important information the user shares with you.
"""

    def _is_quota_error(self, error: Exception) -> bool:
        """Return True when the API error is a rate-limit or quota issue."""
        return isinstance(
            error,
            (
                google_exceptions.ResourceExhausted,
                google_exceptions.TooManyRequests,
            ),
        )

    def _fallback_after_tool_use(self, tool_outputs: list[str]) -> str:
        """Create a concise response when tools succeeded but model follow-up failed."""
        saved = [output for output in tool_outputs if output.startswith("Saved:")]
        deleted = [output for output in tool_outputs if output.startswith("Deleted fact:")]

        if saved:
            details = "; ".join(item.replace("Saved: ", "") for item in saved)
            return f"I saved that to memory: {details}."

        if deleted:
            details = "; ".join(item.replace("Deleted fact: ", "") for item in deleted)
            return f"I updated memory successfully: removed {details}."

        if tool_outputs:
            return "I completed the requested tool action, but Gemini hit a temporary quota limit before I could phrase the final reply."

        return "Gemini hit a temporary quota limit. Please wait a bit and try again."

    def _offline_memory_response(self, user_message: str) -> Optional[str]:
        """Answer simple recall questions directly from long-term memory."""
        message = user_message.lower()
        facts = self.long_term.get_all_facts()

        if not facts:
            return None

        if re.search(r"\b(what('?s| is) my name|do you remember my name)\b", message):
            name = facts.get("user_name")
            if name:
                return f"Yes, your name is {name}."

        if "job" in message or "profession" in message or "what do you do" in message:
            job = facts.get("user_profession_location")
            if job:
                return f"You told me that you are a {job}."

        if "framework" in message:
            framework = facts.get("favorite_ml_framework")
            if framework:
                return f"You said your favourite ML framework is {framework}."

        if "prefer" in message and "language" in message:
            language = facts.get("preferred_data_language")
            if language:
                return f"You told me that you prefer {language} for data work."

        if "remember" in message or "know about me" in message:
            remembered = []
            if facts.get("user_name"):
                remembered.append(f"your name is {facts['user_name']}")
            if facts.get("user_profession_location"):
                remembered.append(f"you are a {facts['user_profession_location']}")
            if facts.get("preferred_data_language"):
                remembered.append(f"you prefer {facts['preferred_data_language']} for data work")
            if facts.get("favorite_ml_framework"):
                remembered.append(
                    f"your favourite ML framework is {facts['favorite_ml_framework']}"
                )
            if remembered:
                return "I remember that " + ", ".join(remembered) + "."

        return None

    def _try_local_fact_save(self, user_message: str) -> Optional[str]:
        """Save a few common personal facts without requiring a Gemini round-trip."""
        message = user_message.strip()

        patterns = [
            (
                r"(?i)^my name is\s+([A-Za-z][A-Za-z\s'-]{0,49})$",
                "user_name",
                "Nice to meet you, {value}. I've saved your name.",
            ),
            (
                r"(?i)^my name\s+([A-Za-z][A-Za-z\s'-]{0,49})$",
                "user_name",
                "Nice to meet you, {value}. I've saved your name.",
            ),
            (
                r"(?i)^my mother(?:'s)? name is\s+([A-Za-z][A-Za-z\s'-]{0,49})$",
                "mother_name",
                "I've saved that your mother's name is {value}.",
            ),
            (
                r"(?i)^my mother(?:'s)? name\s+([A-Za-z][A-Za-z\s'-]{0,49})$",
                "mother_name",
                "I've saved that your mother's name is {value}.",
            ),
            (
                r"(?i)^mother(?:'s)? name(?: is)?\s+([A-Za-z][A-Za-z\s'-]{0,49})$",
                "mother_name",
                "I've saved that your mother's name is {value}.",
            ),
            (
                r"(?i)^i am\s+(.{2,80})$",
                "user_profession_location",
                "I've saved that you are {value}.",
            ),
            (
                r"(?i)^i'm\s+(.{2,80})$",
                "user_profession_location",
                "I've saved that you are {value}.",
            ),
        ]

        for pattern, key, reply in patterns:
            match = re.match(pattern, message)
            if not match:
                continue

            value = " ".join(match.group(1).split())
            self.long_term.save_fact(key, value)
            return reply.format(value=value)

        return None

    # ── Memory Context Builder ────────────────────────────────────────────────

    def _build_memory_context(self, user_message: str) -> str:
        """Prepend relevant memory context to the user message."""
        context_parts = []

        # 1. Pull relevant long-term facts
        facts = self.long_term.get_all_facts()
        if facts:
            facts_text = "\n".join(f"  • {k}: {v}" for k, v in facts.items())
            context_parts.append(f"[LONG-TERM MEMORY – Stored Facts]\n{facts_text}")

        # 2. Semantic search for relevant past knowledge
        results = self.semantic.search(user_message, n_results=3)
        if results:
            sem_text = "\n".join(f"  • {r}" for r in results)
            context_parts.append(f"[SEMANTIC MEMORY – Related Knowledge]\n{sem_text}")

        if context_parts:
            memory_block = "\n\n".join(context_parts)
            return f"{memory_block}\n\n[USER MESSAGE]\n{user_message}"
        return user_message

    # ── Agent Loop ────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Main entry point. Runs the full think → act → observe loop.

        Flow:
        1. Enrich message with memory context
        2. Add to short-term memory
        3. Send to Gemini
        4. If model calls a tool → execute → feed back → repeat
        5. Return final text response
        """

        # Add original message to short-term memory
        self.short_term.add("user", user_message)

        local_save_response = self._try_local_fact_save(user_message)
        if local_save_response is not None:
            self.short_term.add("model", local_save_response)
            print(f"👤 User: {user_message}")
            print(f"\n🤖 Agent: {local_save_response}\n")
            return local_save_response

        # Enrich with memory context
        enriched_message = self._build_memory_context(user_message)

        # Build full history for Gemini (short-term memory = conversation history)
        history = self.short_term.get_history_for_gemini()

        # Start a chat session with history (excluding last user message, sent separately)
        chat_session = self._start_chat(history[:-1])  # exclude last user msg

        print(f"👤 User: {user_message}")

        # ── Agentic Loop ─────────────────────────────────────────────────────
        try:
            chat_session, response = self._send_message_with_failover(
                chat_session,
                enriched_message,
            )
        except Exception as error:
            if self._is_quota_error(error):
                final_text = self._offline_memory_response(user_message) or (
                    "Gemini is temporarily over its quota across all configured API keys. "
                    "Please wait a few seconds and retry."
                )
                self.short_term.add("model", final_text)
                print(f"\n🤖 Agent: {final_text}\n")
                return final_text
            raise
        iteration = 0

        while True:
            iteration += 1
            candidate = response.candidates[0]

            # Collect tool calls from this response
            tool_calls = [
                part for part in candidate.content.parts
                if hasattr(part, "function_call") and part.function_call.name
            ]

            if not tool_calls:
                # No more tool calls – extract final text
                final_text = "".join(
                    part.text for part in candidate.content.parts
                    if hasattr(part, "text") and part.text
                )
                break

            # Execute each tool and collect results
            print(f"\n  🔄 Iteration {iteration} – {len(tool_calls)} tool call(s)")
            tool_results = []
            tool_output_texts = []

            for part in candidate.content.parts:
                if not (hasattr(part, "function_call") and part.function_call.name):
                    continue

                fn = part.function_call
                print(f"  🔧 Tool: {fn.name}({dict(fn.args)})")

                result = execute_tool(
                    fn.name,
                    dict(fn.args),
                    long_term=self.long_term,
                    semantic=self.semantic,
                )
                print(f"  📤 Result: {result}")
                tool_output_texts.append(result)

                tool_results.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fn.name,
                            response={"result": result},
                        )
                    )
                )

            # Feed all results back in one turn
            try:
                chat_session, response = self._send_message_with_failover(
                    chat_session,
                    tool_results,
                )
            except Exception as error:
                if self._is_quota_error(error):
                    final_text = self._fallback_after_tool_use(tool_output_texts)
                    break
                raise

        # Save assistant response to short-term memory
        self.short_term.add("model", final_text)

        # Auto-store substantive responses in semantic memory
        if len(final_text) > 100:
            self.semantic.store(
                text=f"Q: {user_message}\nA: {final_text}",
                metadata={"session": self.session_id, "timestamp": time.time()},
            )

        print(f"\n🤖 Agent: {final_text}\n")
        return final_text

    # ── Utility ───────────────────────────────────────────────────────────────

    def memory_summary(self) -> dict:
        """Return a snapshot of all memory layers."""
        return {
            "short_term_messages": len(self.short_term.messages),
            "long_term_facts": self.long_term.get_all_facts(),
            "semantic_entries": self.semantic.count(),
        }

    def clear_short_term(self):
        """Clear conversation history (simulate new session)."""
        self.short_term.clear()
        print("🗑️  Short-term memory cleared.")
