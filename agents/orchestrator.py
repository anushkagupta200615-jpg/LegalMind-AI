import os
import sys
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pdf_parser_tool import pdf_parser
from agents.analysis_agent import analyze_legal_issue
from agents.drafting_agent import DraftingAgent

class AgentState(TypedDict):
    user_input: str
    file_path: str
    task_type: str
    extracted_text: str
    analysis_result: str
    draft_output: str
    messages: list
    
def classify_task(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    template = """Classify the user input into exactly one of the following task types:
['analyze_document', 'research_law', 'draft_document', 'general_query']

If a document path is provided and the user wants to analyze it, choose 'analyze_document'.
If the user is asking about specific laws, case laws, or legal research, choose 'research_law'.
If the user wants to draft a legal notice, reply, or summarize a contract, choose 'draft_document'.
Otherwise, choose 'general_query'.

User Input: {user_input}
Has File: {has_file}

Output ONLY the category name."""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    task_type = chain.invoke({
        "user_input": state["user_input"],
        "has_file": str(bool(state.get("file_path")))
    }).strip()
    
    # Fallback to prevent invalid outputs
    valid_tasks = ['analyze_document', 'research_law', 'draft_document', 'general_query']
    if task_type not in valid_tasks:
        task_type = 'general_query'
        
    return {"task_type": task_type}

def run_document_agent(state: AgentState) -> AgentState:
    file_path = state.get("file_path")
    if file_path:
        result = pdf_parser.invoke({"file_path": file_path})
        if isinstance(result, dict) and 'text' in result:
            return {"extracted_text": result['text']}
        elif isinstance(result, str):
            return {"extracted_text": result}
    return {"extracted_text": ""}

def run_analysis_agent(state: AgentState) -> AgentState:
    input_text = state["user_input"]
    if state.get("extracted_text"):
        input_text += f"\n\nDocument Text:\n{state['extracted_text']}"
        
    result = analyze_legal_issue(input_text)
    return {"analysis_result": result}

def run_drafting_agent(state: AgentState) -> AgentState:
    agent = DraftingAgent()
    input_text = state["user_input"]
    if state.get("extracted_text"):
        input_text += f"\n\nDocument Text:\n{state['extracted_text']}"
        
    # Heuristic based routing within drafting
    lower_input = input_text.lower()
    if "summar" in lower_input:
        result = agent.summarize_contract(input_text)
    elif "reply" in lower_input:
        result = agent.draft_reply(input_text)
    else:
        # Dummy arguments for notice generation, in real world we'd extract these
        result = agent.draft_notice("Opposite Party", "Our Client", input_text, "Relevant Sections")
        
    return {"draft_output": result}

def final_response(state: AgentState) -> AgentState:
    task_type = state["task_type"]
    response = ""
    
    if task_type == 'analyze_document' or task_type == 'research_law':
        response = state.get("analysis_result", "No analysis result.")
    elif task_type == 'draft_document':
        response = state.get("draft_output", "No draft generated.")
    else:
        # general query
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke(f"Answer this general legal query: {state['user_input']}").content
        
    # Append any extracted text info
    if state.get("extracted_text"):
        response = f"**Document Parsed Successfully!**\n\n{response}"
        
    return {"messages": [response]}

# Conditional routing logic
def route_task(state: AgentState) -> str:
    task_type = state.get("task_type", "general_query")
    if task_type == "analyze_document":
        return "run_document_agent"
    elif task_type == "research_law":
        return "run_analysis_agent"
    elif task_type == "draft_document":
        return "run_document_agent" if state.get("file_path") else "run_drafting_agent"
    else:
        return "final_response"

def route_after_document(state: AgentState) -> str:
    task_type = state.get("task_type")
    if task_type == "analyze_document":
        return "run_analysis_agent"
    else:
        return "run_drafting_agent"

# Build Graph
builder = StateGraph(AgentState)

builder.add_node("classify_task", classify_task)
builder.add_node("run_document_agent", run_document_agent)
builder.add_node("run_analysis_agent", run_analysis_agent)
builder.add_node("run_drafting_agent", run_drafting_agent)
builder.add_node("final_response", final_response)

builder.add_edge(START, "classify_task")

builder.add_conditional_edges(
    "classify_task",
    route_task,
    {
        "run_document_agent": "run_document_agent",
        "run_analysis_agent": "run_analysis_agent",
        "run_drafting_agent": "run_drafting_agent",
        "final_response": "final_response"
    }
)

builder.add_conditional_edges(
    "run_document_agent",
    route_after_document,
    {
        "run_analysis_agent": "run_analysis_agent",
        "run_drafting_agent": "run_drafting_agent"
    }
)

builder.add_edge("run_analysis_agent", "final_response")
builder.add_edge("run_drafting_agent", "final_response")
builder.add_edge("final_response", END)

graph = builder.compile()

def run_legalmind(user_input: str, file_path: str = None) -> str:
    initial_state = {
        "user_input": user_input,
        "file_path": file_path,
        "task_type": "",
        "extracted_text": "",
        "analysis_result": "",
        "draft_output": "",
        "messages": []
    }
    
    final_state = graph.invoke(initial_state)
    messages = final_state.get("messages", [])
    return messages[-1] if messages else "No response generated."
