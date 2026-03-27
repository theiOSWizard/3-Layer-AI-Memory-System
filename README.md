# AI 3-Layer Memory System

A fully working Python agent built on **Gemini 2.5 Flash** that demonstrates all three
types of LLM agent memory.

---

## Memory Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     AGENT MEMORY LAYERS                      │
├────────────────┬─────────────────────────────────────────────┤
│  SHORT-TERM    │ In-context conversation history              │
│                │ • Python list (in-memory)                   │
│                │ • Sliding window (last 20 messages)         │
│                │ • Lost when process exits                   │
├────────────────┼─────────────────────────────────────────────┤
│  LONG-TERM     │ Persistent key-value facts                  │
│                │ • Stored in JSON file on disk               │
│                │ • Survives restarts                         │
│                │ • Best for: names, prefs, settings          │
├────────────────┼─────────────────────────────────────────────┤
│  SEMANTIC      │ Vector similarity search                    │
│                │ • ChromaDB + SentenceTransformers           │
│                │ • Fuzzy recall by meaning                   │
│                │ • Best for: long facts, Q&A, summaries      │
└────────────────┴─────────────────────────────────────────────┘
```

---

## Project Structure

```
gemini_agent/
├── agent.py                    # Main GeminiAgent class + agentic loop
├── main.py                     # Interactive chat demo
├── requirements.txt
├── .env.example
│
├── memory/
│   ├── __init__.py
│   ├── short_term.py           # Sliding-window conversation buffer
│   ├── long_term.py            # File-backed JSON fact store
│   └── semantic.py             # ChromaDB vector memory
│
├── tools/
│   ├── __init__.py
│   └── tool_executor.py        # Tool declarations + execution router
│
└── examples/
    └── cross_session_demo.py   # Demonstrates cross-session memory recall
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Semantic memory** requires `chromadb` and `sentence-transformers`.
> If you skip them the agent falls back to simple keyword search automatically.

### 2. Configure API key(s)

```bash
cp .env.example .env
# Edit .env and add one or more Gemini API keys
```

Get a free key at: https://aistudio.google.com/

The agent supports automatic key failover. You can configure keys in either format:

```bash
GEMINI_API_KEY=primary_key
GEMINI_API_KEY_2=backup_key
GEMINI_API_KEY_3=another_backup_key
```

or

```bash
GEMINI_API_KEYS=primary_key,backup_key,another_backup_key
```

If one key hits quota or rate limits, the agent automatically switches to the next configured key.

### 3. Run the interactive demo

```bash
python main.py
```

### 4. Run the cross-session demo

```bash
python examples/cross_session_demo.py   # run twice to see persistence
```

---

## How the Agentic Loop Works

```
User message
     │
     ▼
Enrich with memory context
(long-term facts + semantic search results prepended)
     │
     ▼
Send to Gemini 2.5 Flash
     │
     ├──► Model returns tool_call?
     │         │ YES
     │         ▼
     │    Execute tool (save_fact / search_knowledge / calculate / ...)
     │         │
     │    Feed result back to model
     │         │
     │    Loop ◄──────────────────────────────┐
     │                                         │
     │    (repeat until no more tool calls) ───┘
     │
     │ NO more tool calls
     ▼
Return final text response
Auto-store response in semantic memory
```

---

## Tools Available to the Agent

| Tool | Memory Layer | Description |
|------|-------------|-------------|
| `save_fact(key, value)` | Long-Term | Persist a key-value fact to disk |
| `get_facts()` | Long-Term | Retrieve all stored facts |
| `delete_fact(key)` | Long-Term | Remove a specific fact |
| `store_knowledge(text)` | Semantic | Store text as a vector embedding |
| `search_knowledge(query)` | Semantic | Find related knowledge by meaning |
| `calculate(expression)` | Utility | Safe math evaluation |
| `get_current_time()` | Utility | Current date and time |

---

## Example Interaction

```
You: Hi, my name is Arjun and I love Python.
🤖 Agent: [calls save_fact("user_name", "Arjun")]
         [calls save_fact("user_language", "Python")]
         Nice to meet you, Arjun! I've saved that you love Python.

--- (new session / process restart) ---

You: Do you remember my name?
🤖 Agent: [facts loaded from disk at startup]
         Yes! Your name is Arjun and you love Python.
```

---

## Commands (interactive mode)

| Command | Action |
|---------|--------|
| `/memory` | Print snapshot of all memory layers |
| `/clear` | Clear short-term memory (simulate new session) |
| `/quit` | Exit |
