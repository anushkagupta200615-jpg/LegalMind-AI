import os
import sys
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.vector_search_tool import legal_vector_search
from tools.web_search_tool import web_search, indian_kanoon_search

def analyze_legal_issue(text: str) -> str:
    """
    Runs the legal analysis agent on the given text and returns the final answer.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [legal_vector_search, web_search, indian_kanoon_search]
    
    # System prompt for the ReAct agent
    template = '''You are a legal analysis expert. Given a legal issue or document text, identify the applicable Indian laws, IPC sections, and relevant precedents. Always cite your sources.

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

New input: {input}
{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        max_iterations=8,
        handle_parsing_errors=True
    )
    
    try:
        response = agent_executor.invoke({"input": text})
        return response.get("output", "Analysis failed.")
    except Exception as e:
        return f"Error during analysis: {str(e)}"
