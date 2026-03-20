# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses `uv` with `pyproject.toml` for dependency management (Python ≥ 3.11 required):

```bash
uv venv
uv pip install -e ".[dev]"
```

The `.venv` directory is already present. Activate it with:

```bash
source .venv/bin/activate
```

Required environment variables (create a `.env` file in the project root):

```env
OPENAI_API_KEY=...
GROQ_API_KEY=...
SERPER_API_KEY=...   # Only needed for Research Assistant
```

## Running Notebooks

```bash
jupyter notebook 01_basics/stateful_culinary_assistant.ipynb
jupyter notebook 02_multi_agent/research_assistant.ipynb
jupyter notebook 02_multi_agent/logistics_analysis.ipynb
jupyter notebook 03_flows/workflow_with_flows.ipynb
jupyter notebook 04_comprehensive/complete_tutorial.ipynb
```

## Architecture Overview

This is a **Jupyter notebook-based tutorial repository** for the CrewAI framework. All substantive code lives in concept-level subfolders. There are no importable application modules except `helpers/utils.py`.

### `helpers/utils.py` — LLM Factory

Central utility providing `get_llm()`, `get_openai_llm()`, `get_groq_llm()`, `get_databricks_llm()`, `get_embeddings()`. `get_llm()` auto-selects provider/model by platform:

- **macOS (`darwin`)**: Databricks (`databricks-claude-opus-4-6`)
- **Windows (`win32`)**: Groq (`openai/gpt-oss-120b`)

Notebooks currently bypass `helpers/utils.py` and initialize their LLM directly using `crewai.LLM` pointed at Groq's OpenAI-compatible endpoint.

### Notebook Progression

| Folder | Notebook | Pattern | Key Concepts |
| --- | --- | --- | --- |
| `01_basics/` | stateful_culinary_assistant | Single agent, multiple tasks | `context=[task]` for stateful task chaining, `planning=True` |
| `02_multi_agent/` | research_assistant | Two agents, sequential | Agent tools (`SerperDevTool`), multi-agent collaboration |
| `02_multi_agent/` | logistics_analysis | Two agents, `Process.sequential` | Parameterized tasks via `{variable}` + `kickoff(inputs={})`, `allow_delegation=False` |
| `03_flows/` | workflow_with_flows | `Flow` class, decorators | `@start`, `@router`, `@listen`, Pydantic state model, `flow.plot()` |
| `04_comprehensive/` | complete_tutorial | All patterns combined | Includes `crew.usage_metrics`, unified playground |

### Two Execution Paradigms

**Crew** (`01_basics`, `02_multi_agent`, `04_comprehensive`): LLM-driven agents execute tasks. State flows via `context=[task]` or sequential ordering. Entry point: `crew.kickoff()` or `crew.kickoff(inputs={})`.

**Flow** (`03_flows`, `04_comprehensive`): Deterministic Python logic with typed Pydantic state. Decorators define the execution graph. Entry point: `flow.kickoff()`. `nest_asyncio.apply()` is required in Jupyter.

### LLM Configuration Pattern

Notebooks use CrewAI's native `LLM` wrapper to call Groq's OpenAI-compatible API:

```python
llm = LLM(
    model="openai/gpt-oss-20b",
    provider="openai",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)
```

The `provider="openai"` is required so CrewAI does not strip the `"openai/"` prefix from the model name.
