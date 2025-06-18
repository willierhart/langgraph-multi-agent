# === SETUP ===
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import os, json
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# === STATE DEFINITION ===
# Custom state structure shared between agents
class ResearchState(Dict):
    messages: List[BaseMessage]
    next: Optional[str]
    cycle: int

# === SYSTEM PROMPTS ===
COORDINATOR_SYSTEM_PROMPT = """
You are a coordinator agent responsible for deciding the appropriate next step
in a multi-agent research assistant workflow.

If the query has already been answered well, respond with:
{ "next": "done" }

Otherwise, if further research is needed, respond with:
{ "next": "researcher" }

Base your decision on the conversation so far.
Respond only with valid JSON containing the 'next' field.
"""

RESEARCHER_SYSTEM_PROMPT = """
You are a skilled research agent tasked with gathering comprehensive information
about a specific topic.

Your responsibilities include:
1. Analyzing the research query to understand what information is needed
2. Conducting thorough research to collect relevant facts, data, and perspectives
3. Organizing information in a clear, structured format
4. Ensuring accuracy and objectivity in your findings
5. Citing sources or noting where information might need verification
6. Identifying potential gaps in the information

Present your findings in the following structured format:

SUMMARY: A brief overview of your findings (2-3 sentences)

KEY POINTS:
- Point 1
- Point 2
- Point 3

DETAILED FINDINGS:
1. [Topic Area 1]
- Details and explanations
- Supporting evidence
- Different perspectives if applicable

2. [Topic Area 2]
- Details and explanations
- Supporting evidence
- Different perspectives if applicable

GAPS AND LIMITATIONS:
- Identify any areas where information might be incomplete
- Note any contradictions or areas of debate
- Suggest additional research that might be needed

Your goal is to provide comprehensive, accurate, and useful information that fulfills the research request.
"""

CRITIC_SYSTEM_PROMPT = """
You are a Critic Agent, part of a collaborative research assistant system.
Your role is to evaluate and challenge information provided by the Researcher Agent
to ensure accuracy, completeness, and balance.

Your responsibilities include:
1. Analyzing research findings for accuracy, completeness, and potential biases
2. Identifying gaps in the information or logical inconsistencies
3. Asking important questions that might have been overlooked
4. Suggesting improvements or alternative perspectives
5. Ensuring that the final information is balanced and well-rounded

Be constructive in your criticism. Your goal is not to dismiss the researcher's work,
but to strengthen it.

Format your feedback in a clear, organized manner, highlighting specific points.

Remember, your ultimate goal is to ensure that the final research output is of high quality.
"""

WRITER_SYSTEM_PROMPT = """
You are a Writer Agent, part of a collaborative research assistant system.
Your job is to synthesize information from the Researcher Agent and feedback
from the Critic Agent into a final, well-written response.

Your responsibilities include:
1. Analyzing the information provided by the researcher and the feedback from the critic
2. Organizing the information in a logical, easy-to-understand structure
3. Presenting the information in a clear, engaging writing style
4. Balancing different perspectives and ensuring objectivity
5. Creating a final response that is comprehensive, accurate, and well-written

Format your response in a clear, organized manner with appropriate headings, paragraphs, and bullet points.

Use simple language to explain complex concepts, and provide examples where helpful.

Remember, your goal is to create a final response that effectively communicates the key insights to the user.
"""

# === AGENT FACTORY WITH DETAILED LOGGING ===
def create_agent(prompt: str, temperature=0.5, agent_name: str = "Unnamed Agent"):
    llm = ChatOpenAI(model="gpt-4o", temperature=temperature)

    def agent(messages: List[BaseMessage]) -> BaseMessage:
        full_messages = [SystemMessage(content=prompt)] + messages
        response = llm.invoke(full_messages)

        print(f"\n--- {agent_name} OUTPUT ---\n{response.content}\n")
        return response

    return agent

# === NODE DEFINITIONS ===
def coordinator_node(state: ResearchState) -> ResearchState:
    coordinator = create_agent(COORDINATOR_SYSTEM_PROMPT, temperature=0.2, agent_name="Coordinator Agent")
    response = coordinator(state["messages"])
    try:
        decision = json.loads(response.content)
        next_step = decision.get("next", "researcher")
        print(f"Coordinator decision: next → {next_step}")
    except Exception as e:
        print(f"Coordinator JSON parse error: {e}. Defaulting to 'researcher'")
        next_step = "researcher"
    return {**state, "next": next_step, "cycle": state.get("cycle", 0) + 1}

def researcher_node(state: ResearchState) -> ResearchState:
    researcher = create_agent(RESEARCHER_SYSTEM_PROMPT, agent_name="Researcher Agent")
    query = state["messages"][-1]  # Assume last human message is the query
    response = researcher([query])
    return {
        **state,
        "messages": state["messages"] + [response],
        "next": "critic"
    }

def critic_node(state: ResearchState) -> ResearchState:
    critic = create_agent(CRITIC_SYSTEM_PROMPT, agent_name="Critic Agent")
    response = critic(state["messages"])
    return {
        **state,
        "messages": state["messages"] + [response],
        "next": "writer"
    }

def writer_node(state: ResearchState) -> ResearchState:
    writer = create_agent(WRITER_SYSTEM_PROMPT, temperature=0.6, agent_name="Writer Agent")
    response = writer(state["messages"])
    return {
        **state,
        "messages": state["messages"] + [response],
        "next": "done"
    }

def output_node(state: ResearchState) -> ResearchState:
    print("\n--- Output Node: Returning final result ---")
    return state

# === GRAPH DEFINITION ===
def build_dynamic_multi_agent_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("output", output_node)

    workflow.add_conditional_edges("coordinator", lambda s: s["next"], {
        "researcher": "researcher",
        "done": "output"
    })

    workflow.add_edge("researcher", "critic")
    workflow.add_edge("critic", "writer")
    workflow.add_edge("writer", "coordinator")
    workflow.add_edge("output", END)

    workflow.set_entry_point("coordinator")
    return workflow.compile()

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("\n=== Starting Multi-Agent Research Assistant ===")
    graph = build_dynamic_multi_agent_graph()

    initial_state = {
        "messages": [HumanMessage(content="What are the implications of AI in education?")],
        "next": "",
        "cycle": 0
    }

    print(f"\nInitial User Question: {initial_state['messages'][0].content}\n")

    final_state = graph.invoke(initial_state, config={"recursion_limit": 10})

    print("\n=== FINAL OUTPUT ===\n")
    for msg in final_state["messages"]:
        sender = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"{sender}: {msg.content}")

    print("\n=== Workflow Completed ===")
