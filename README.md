# Dynamic Multi-Agent Research Assistant with LangGraph

This repository implements a dynamic, LLM-powered multi-agent system using [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). It features a modular architecture that combines multiple specialized agents orchestrated by a Coordinator node, enabling intelligent, iterative processing of complex user queries.

---

## 1. Overview

This project demonstrates a collaborative research assistant composed of:

* **Coordinator Agent**: Determines whether the system should directly respond or invoke deeper analysis.
* **Researcher Agent**: Gathers structured, comprehensive information about a given query.
* **Critic Agent**: Reviews the research output, highlighting strengths, gaps, and potential bias.
* **Writer Agent**: Synthesizes the validated information into a clear and well-written response.
* **Output Node**: Returns the final response to the user.

The system supports dynamic routing, allowing multiple iterations and agent loops when needed.

---

## 2. Features

* Built with the latest LangGraph and LangChain APIs
* Fully dynamic execution using conditional graph edges
* Powered by OpenAI’s GPT-4o through the langchain-openai interface
* Uses modern TypedDict for strongly typed shared state
* Modular components: Coordinator, Researcher, Critic, Writer
* Prompt engineering support with configurable system prompts
* Easily extendable to more agents, tools, or memory layers

---

## 3. Installation

### 3.1 Using `requirements.txt`

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

```
langgraph>=0.0.32
langchain>=0.1.16
langchain-openai>=0.1.6
openai>=1.30.1
python-dotenv>=1.0.1
```

---

### 3.2 Using Conda

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
2. Create a new environment:

```bash
conda create -n langgraph_agents python=3.10
```

3. Activate the environment:

```bash
conda activate langgraph_agents
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_key_here
```

---

## 4. Usage

1. Clone the repository:

```bash
git clone https://github.com/willierhart/langgraph-multi-agent.git
cd langgraph-multi-agent
```

2. Activate the Conda environment:

```bash
conda activate langgraph_agents
```

3. Run the assistant:

```bash
python multi_agent_assistant.py
```

The system will:

* Start at the Coordinator Agent to analyze the query.
* Decide whether to return an immediate answer or trigger a full multi-agent workflow.
* If needed, route the query through Researcher → Critic → Writer.
* Return to the Coordinator for another cycle or finalize the output.
* Output a complete, polished response.

---

## 5. Example

```python
initial_state = {
    "messages": [HumanMessage(content="What are the implications of AI in education?")],
    "next": ""
}
```

Example output:

```
HumanMessage: What are the implications of AI in education?
AIMessage: AI has a significant impact on education, including personalized learning, automated feedback systems...
```

---

## 6. Project Structure

```
langgraph-multi-agent/
├── multi_agent_assistant.py     # Main execution script
├── requirements.txt             # Python dependencies
├── .env                         # API key configuration
├── README.md                    # Documentation
```

---

## 7. Contributing

Feel free to open an issue or submit a pull request if you find bugs or have suggestions for additional features

---

## 8. Acknowledgments

Architecture and implementation are based on the article "Building Multi-Agent Systems with LangGraph" (Clearwater Analytics, June 2025):
[https://medium.com/cwan-engineering/building-multi-agent-systems-with-langgraph-04f90f312b8e](https://medium.com/cwan-engineering/building-multi-agent-systems-with-langgraph-04f90f312b8e)

