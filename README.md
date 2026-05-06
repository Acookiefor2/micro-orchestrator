# micro-orchestrator

A lightweight, **local-first** multi-agent orchestration framework built from
first principles in Python — no CrewAI, no AutoGen, no LangGraph.

```
┌─────────────────────────────────────────────────────┐
│                     Orchestrator                     │
│                                                      │
│  ┌──────────┐     ┌──────────────────────────────┐  │
│  │  Manager │────▶│  TaskPlan (Pydantic-validated)│  │
│  │   LLM    │     └──────────────┬───────────────┘  │
│  └──────────┘                    │                   │
│                      ┌───────────▼────────────┐      │
│                      │   Sequential Dispatch   │      │
│                      └──┬────────┬────────┬───┘      │
│                ┌────────▼─┐  ┌───▼────┐  ┌▼──────┐  │
│                │Researcher│  │ Coder  │  │Writer │  │
│                └────────┬─┘  └───┬────┘  └┬──────┘  │
│                         └────────┴─────────┘         │
│                              │ MissionState           │
│                     ┌────────▼────────┐               │
│                     │  Critic / Judge │               │
│                     └────────┬────────┘               │
│                              │ approved / retry        │
└──────────────────────────────┼─────────────────────────┘
                               ▼
                         Final Output
```

## Stack

| Concern | Choice |
|---|---|
| LLM backend | Ollama (OpenAI-compatible at `http://localhost:11434/v1`) |
| LLM client | `openai` Python SDK |
| Schema validation | `pydantic` v2 |
| Concurrency | `threading.RLock` (state is thread-safe) |
| Language | Python 3.10+ |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running with a model pulled
ollama pull llama3:8b

# 3. Run the built-in demo
python main.py

# 4. Run with a custom goal
python main.py --goal "Write a Python script that monitors CPU usage every second."

# 5. Save output to a file
python main.py --output result.md

# 6. Verbose debug logging
python main.py --verbose
```

## File Structure

```
micro-orchestrator/
├── main.py                 # CLI entry point + demo goal
├── orchestrator.py         # Manager LLM, task routing, Critic
├── state.py                # Thread-safe MissionState
├── requirements.txt
└── agents/
    ├── __init__.py
    ├── base_agent.py       # Abstract base + retry logic
    └── specialists.py      # CoderAgent, ResearcherAgent, WriterAgent
```

## Configuration

Edit the constants at the top of `agents/base_agent.py`:

| Constant | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `WORKER_MODEL` | `llama3:8b` | Model used by specialist agents |
| `MANAGER_MODEL` | `llama3:8b` | Model used by Manager + Critic |
| `MAX_RETRIES` | `3` | Per-call LLM retry budget |
| `REQUEST_TIMEOUT` | `120.0` | Seconds before a request times out |

Edit `orchestrator.py` to change:

| Constant | Default | Description |
|---|---|---|
| `MAX_PIPELINE_RETRIES` | `2` | How many full replans the Critic can trigger |

## Extending

To add a new specialist agent:

1. Subclass `BaseAgent` in `agents/specialists.py`.
2. Set `ROLE = "your_role"`.
3. Implement `system_prompt` and `execute()`.
4. Register it in `_AGENT_REGISTRY` in `orchestrator.py`.
5. Add `"your_role"` to the `AgentRole` `Literal` type.
