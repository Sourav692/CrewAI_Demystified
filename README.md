# CrewAI Demystified

A comprehensive learning repository for understanding and building AI agents using CrewAI framework. This project contains hands-on examples demonstrating various CrewAI capabilities including stateful applications, research assistants, and workflow management.

## 📚 Overview

CrewAI is a powerful framework for orchestrating role-playing, autonomous AI agents. This repository demystifies CrewAI concepts through practical examples and tutorials based on the [Analytics Vidhya course](https://courses.analyticsvidhya.com/courses/take/building-your-first-ai-agent-with-crew-ai/lessons/60042706-building-a-workflow-with-crewai).

## 🗂️ Project Structure

```
CrewAI_Demystified/
├── 01_basics/
│   └── stateful_culinary_assistant.ipynb
├── 02_multi_agent/
│   ├── research_assistant.ipynb
│   └── logistics_analysis.ipynb
├── 03_flows/
│   └── workflow_with_flows.ipynb
├── 04_comprehensive/
│   └── complete_tutorial.ipynb
├── helpers/
│   └── utils.py
├── reference_docs/
│   └── link.md
├── pyproject.toml
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- API keys for your chosen LLM provider (Groq recommended)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/CrewAI_Demystified.git
cd CrewAI_Demystified
```

2. Install dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
```

3. Set up your environment variables — create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
SERPER_API_KEY=your_serper_api_key_here   # Only needed for research_assistant.ipynb
```

## 📖 Examples

### 01_basics — Stateful Culinary Assistant

**File:** `01_basics/stateful_culinary_assistant.ipynb`

A single-agent culinary assistant that finds recipes and provides step-by-step cooking instructions. Demonstrates task context chaining (`context=[task]`) and `planning=True`.

```bash
jupyter notebook 01_basics/stateful_culinary_assistant.ipynb
```

### 02_multi_agent — Research Assistant

**File:** `02_multi_agent/research_assistant.ipynb`

A two-agent market research pipeline: a **Market Researcher** searches the web with `SerperDevTool`, then a **Product Strategist** synthesizes findings into a positioning strategy.

```bash
jupyter notebook 02_multi_agent/research_assistant.ipynb
```

### 02_multi_agent — Logistics Analysis

**File:** `02_multi_agent/logistics_analysis.ipynb`

A sequential two-agent pipeline using `Process.sequential` and parameterized task inputs (`{variable}` placeholders + `kickoff(inputs={})`). Demonstrates `allow_delegation=False`.

```bash
jupyter notebook 02_multi_agent/logistics_analysis.ipynb
```

### 03_flows — Workflow with Flows

**File:** `03_flows/workflow_with_flows.ipynb`

Demonstrates CrewAI's `Flow` system with `@start`, `@router`, and `@listen` decorators for deterministic, event-driven workflows with typed Pydantic state.

```bash
jupyter notebook 03_flows/workflow_with_flows.ipynb
```

### 04_comprehensive — Complete Tutorial

**File:** `04_comprehensive/complete_tutorial.ipynb`

A unified notebook combining all three patterns (stateful tasks, multi-agent, flows) with token usage tracking via `crew.usage_metrics`.

```bash
jupyter notebook 04_comprehensive/complete_tutorial.ipynb
```

## 🔑 Key Concepts

### Agents
Autonomous entities with specific roles, goals, and backstories. They can use tools and work collaboratively.

### Tasks
Specific assignments given to agents with clear descriptions and expected outputs.

### Crew
Orchestrates multiple agents and tasks, managing the workflow and planning.

### Flow
Advanced workflow management with routers, listeners, and state management for complex decision trees.

### Tools
External capabilities agents can use (e.g., web search, file operations, API calls).

## 🛠️ Technologies Used

- **CrewAI** (v0.80.0): Core framework for AI agents
- **crewai-tools**: Additional tools for agents
- **OpenAI GPT-4/GPT-4o-mini**: Language models
- **Pydantic**: Data validation and settings management
- **python-dotenv**: Environment variable management
- **nest-asyncio**: Async support for Jupyter notebooks

## 📝 Learning Path

1. **Start with Basics**: `01_basics/stateful_culinary_assistant.ipynb` — single-agent, stateful task chaining
2. **Multi-Agent Systems**: `02_multi_agent/research_assistant.ipynb` — agent collaboration with tools
3. **Parameterized Pipelines**: `02_multi_agent/logistics_analysis.ipynb` — reusable crews with runtime inputs
4. **Advanced Workflows**: `03_flows/workflow_with_flows.ipynb` — deterministic flows with routing
5. **Everything Together**: `04_comprehensive/complete_tutorial.ipynb` — unified playground

## 💡 Use Cases

- **Recipe Assistants**: Personalized cooking guidance
- **Market Research**: Automated trend analysis and strategy development
- **Content Creation**: Multi-agent content generation pipelines
- **Decision Systems**: Complex routing and conditional workflows
- **Research Automation**: Web scraping and information synthesis

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new examples
- Improve documentation
- Fix bugs
- Share use cases

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🔗 Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [Analytics Vidhya Course](https://courses.analyticsvidhya.com/courses/take/building-your-first-ai-agent-with-crew-ai/)
- [OpenAI API](https://platform.openai.com/)

## 🙋 Support

For questions or issues:
- Check the code examples in this repository
- Review CrewAI official documentation
- Open an issue for bugs or feature requests

---

**Happy Learning! 🚀**
