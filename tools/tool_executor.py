"""
Tool Executor
=============
Defines the tools the agent can call and routes execution to the correct handler.

Tools exposed to the model:
  Memory tools:
    - save_fact(key, value)       → LongTermMemory.save_fact()
    - get_facts()                 → LongTermMemory.get_all_facts()
    - store_knowledge(text)       → SemanticMemory.store()
    - search_knowledge(query)     → SemanticMemory.search()

  Utility tools:
    - calculate(expression)       → safe math eval
    - get_current_time()          → current date/time string
"""

import math
import datetime
from typing import Any

import google.generativeai as genai

from memory.long_term import LongTermMemory
from memory.semantic import SemanticMemory


# ── Tool Declarations (sent to Gemini) ────────────────────────────────────────

def get_tool_declarations() -> list:
    """Return the list of tool declarations in Gemini's function-calling format."""

    return [
        # ── Long-Term Memory ──────────────────────────────────────────────────
        genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="save_fact",
                    description=(
                        "Save an important fact to long-term persistent memory. "
                        "Use for user name, preferences, settings, important dates, etc. "
                        "Facts survive across sessions."
                    ),
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "key": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="Short identifier, e.g. 'user_name', 'preferred_language'",
                            ),
                            "value": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="The value to store",
                            ),
                        },
                        required=["key", "value"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="get_facts",
                    description=(
                        "Retrieve all facts stored in long-term memory. "
                        "Call this at the start of a session to recall what you know about the user."
                    ),
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="delete_fact",
                    description="Delete a specific fact from long-term memory by its key.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "key": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="The key of the fact to delete",
                            ),
                        },
                        required=["key"],
                    ),
                ),
            ]
        ),

        # ── Semantic Memory ───────────────────────────────────────────────────
        genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="store_knowledge",
                    description=(
                        "Store a piece of knowledge in semantic (vector) memory for future fuzzy recall. "
                        "Use for long-form facts, summaries, Q&A pairs, or anything you want to find "
                        "by meaning rather than exact key."
                    ),
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "text": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="The knowledge text to store",
                            ),
                        },
                        required=["text"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="search_knowledge",
                    description=(
                        "Search semantic memory for knowledge related to the query. "
                        "Returns the most relevant stored entries by meaning (not exact match)."
                    ),
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "query": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="Natural language search query",
                            ),
                            "n_results": genai.protos.Schema(
                                type=genai.protos.Type.INTEGER,
                                description="Number of results to return (default 3)",
                            ),
                        },
                        required=["query"],
                    ),
                ),
            ]
        ),

        # ── Utility Tools ─────────────────────────────────────────────────────
        genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="calculate",
                    description="Evaluate a safe mathematical expression and return the result.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "expression": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="Math expression to evaluate, e.g. '2 ** 10 + sqrt(144)'",
                            ),
                        },
                        required=["expression"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="get_current_time",
                    description="Return the current date and time.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={},
                    ),
                ),
            ]
        ),
    ]


# ── Tool Executor (routes calls to Python functions) ──────────────────────────

def execute_tool(
    name: str,
    args: dict[str, Any],
    long_term: LongTermMemory,
    semantic: SemanticMemory,
) -> str:
    """
    Route a tool call from the model to the correct Python function.

    Args:
        name:       The tool name returned by the model.
        args:       The arguments dict from the model.
        long_term:  Injected LongTermMemory instance.
        semantic:   Injected SemanticMemory instance.

    Returns:
        A string result that is fed back to the model.
    """

    # ── Long-Term Memory Tools ────────────────────────────────────────────────
    if name == "save_fact":
        return long_term.save_fact(args["key"], args["value"])

    if name == "get_facts":
        facts = long_term.get_all_facts()
        if not facts:
            return "No facts stored yet."
        return "\n".join(f"{k}: {v}" for k, v in facts.items())

    if name == "delete_fact":
        return long_term.delete_fact(args["key"])

    # ── Semantic Memory Tools ─────────────────────────────────────────────────
    if name == "store_knowledge":
        return semantic.store(args["text"])

    if name == "search_knowledge":
        n = args.get("n_results", 3)
        results = semantic.search(args["query"], n_results=n)
        if not results:
            return "No relevant knowledge found."
        return "\n---\n".join(results)

    # ── Utility Tools ─────────────────────────────────────────────────────────
    if name == "calculate":
        return _safe_calculate(args["expression"])

    if name == "get_current_time":
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"Unknown tool: {name}"


def _safe_calculate(expression: str) -> str:
    """Evaluate a math expression with a safe subset of builtins."""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round})

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"
