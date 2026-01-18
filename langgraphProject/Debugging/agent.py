from langgraph.graph import START, END
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from IPython.display import display, Image
from langchain_core.tools import tool

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_WORKSPACE_ID'] = os.getenv('WORKSPACE')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = "LANGRAPG_DEBUG"

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)

class State(TypedDict):
    messages:Annotated[list, add_messages]

def make_tools():
    @tool
    def multiply(x: float, y: float) -> float:
        """
        Useful for multiplying two numbers.

        Args:
            x (float): The first number.
            y (float): The second number.

        Returns:
            float: The product of x and y.
        """
        return x * y

    toolNode = ToolNode([multiply])

    llm_with_tools = llm.bind_tools([multiply])

    def call_llm(state:State):
        return {"messages":[llm_with_tools.invoke(state["messages"])]}##it gives the all stored messages to the llm

    graph_builder = StateGraph(State)
    graph_builder.add_node("tool_calling_llm", call_llm)
    graph_builder.add_node("tools",toolNode)

    graph_builder.add_edge(START,"tool_calling_llm")
    graph_builder.add_conditional_edges("tool_calling_llm",tools_condition)

    graph_builder.add_edge("tools",END)

    graph = graph_builder.compile()
    
    return graph

graph = make_tools()



