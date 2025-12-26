# CrewAI Demystified

A comprehensive learning repository for understanding and building AI agents using CrewAI framework. This project contains hands-on examples demonstrating various CrewAI capabilities including stateful applications, research assistants, and workflow management.

## 📚 Overview

CrewAI is a powerful framework for orchestrating role-playing, autonomous AI agents. This repository demystifies CrewAI concepts through practical examples and tutorials based on the [Analytics Vidhya course](https://courses.analyticsvidhya.com/courses/take/building-your-first-ai-agent-with-crew-ai/lessons/60042706-building-a-workflow-with-crewai).

## 🗂️ Project Structure

```
CrewAI_Demystified/
├── Building your First AI Agent with CrewAI/
│   ├── 1. Stateful_Applications_with_CrewAI.py
│   ├── 2. Research Assistant in CrewAI.py
│   ├── 3. Code Notebook.ipynb
│   ├── 4. Workflow_with_CrewAI.py
│   └── link.md
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (or other LLM provider API key)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CrewAI_Demystified.git
cd CrewAI_Demystified
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## 📖 Examples

### 1. Stateful Applications with CrewAI

**File:** `1. Stateful_Applications_with_CrewAI.py`

A culinary assistant agent that:
- Finds recipes based on main ingredients
- Filters recipes according to dietary restrictions
- Provides step-by-step cooking instructions

**Key Features:**
- Agent with specific role and backstory
- Multiple sequential tasks
- Planning capabilities
- LLM integration (GPT-4)

**Usage:**
```bash
python "Building your First AI Agent with CrewAI/1. Stateful_Applications_with_CrewAI.py"
```

### 2. Research Assistant in CrewAI

**File:** `2. Research Assistant in CrewAI.py`

A multi-agent system for market research:
- **Market Researcher Agent**: Analyzes market trends using web search
- **Product Strategist Agent**: Creates positioning strategies

**Key Features:**
- Multiple agents with different roles
- Tool integration (SerperDevTool for web search)
- Agent collaboration
- Market intelligence gathering

**Usage:**
```bash
python "Building your First AI Agent with CrewAI/2. Research Assistant in CrewAI.py"
```

### 3. Code Notebook

**File:** `3. Code Notebook.ipynb`

Comprehensive Jupyter notebook covering:
- Recipe recommendation system
- Marketing research workflow
- Game flow with routing logic

**Topics Covered:**
- Agent creation and configuration
- Task definition and execution
- Crew orchestration
- Flow control with routers and listeners
- Token usage tracking

**Usage:**
```bash
jupyter notebook "Building your First AI Agent with CrewAI/3. Code Notebook.ipynb"
```

### 4. Workflow with CrewAI

**File:** `4. Workflow_with_CrewAI.py`

Demonstrates advanced flow control:
- Game session management
- Router-based decision making
- Event listeners for different outcomes
- State management with Pydantic models

**Key Concepts:**
- `@start()` decorator for entry points
- `@router()` for conditional branching
- `@listen()` for event-driven actions
- State persistence with BaseModel

**Usage:**
```bash
python "Building your First AI Agent with CrewAI/4. Workflow_with_CrewAI.py"
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

1. **Start with Basics**: Begin with `1. Stateful_Applications_with_CrewAI.py` to understand single-agent systems
2. **Multi-Agent Systems**: Move to `2. Research Assistant in CrewAI.py` for agent collaboration
3. **Interactive Learning**: Explore `3. Code Notebook.ipynb` for hands-on experiments
4. **Advanced Workflows**: Study `4. Workflow_with_CrewAI.py` for complex flow patterns

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
