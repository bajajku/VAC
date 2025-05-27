import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from tools.retriever_tool import RetrieverTool
from scripts.data_collection import JsonParser

'''
    So this step ensures that AI agent manages its state (so it knows what to do next)
    And respond appropriately.
    -> This is a TypedDict that defines the structure of the state.
    -> This state will hold messages, including input from users and output from agents or tools.
'''

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] 