import os
from dotenv import load_dotenv
from typing import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()

# Access OPENAI and LANGSMITH api key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Define state
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define model
model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

def make_default_graph():
    """ Make a simple LLM Agent. """ 
    
    def call_model(state: State):
        return {'messages': [model.invoke(state['messages'])]}
    
    graph_workflow = StateGraph(State)
    graph_workflow.add_node('agent', call_model)
    graph_workflow.add_edge(START, 'call_model')
    graph_workflow.add_edge('call_model', END)
    graph = graph_workflow.compile()
    return graph

def make_alternate_graph():
    """ Make a tool calling agent. """
    
    @tool
    def add(a: float, b: float) -> float:
        """ Adds number a and b """
        return a + b
    
    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])
    
    def call_model(state: State):
        return {'messages': [model_with_tools.invoke(state['messages'])]}
    
    def should_continue(state: State):
        if state['messages'][-1].tool_calls:
            return 'tools'
        else:
            return END
    
    graph_workflow = StateGraph(State)    
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", tools_condition)
    graph_workflow.add_edge("tools", "agent")
    
    graph = graph_workflow.compile()
    return graph

agent = make_alternate_graph()
