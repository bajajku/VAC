import operator
from typing import Annotated, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

'''
    So this step ensures that AI agent manages its state (so it knows what to do next)
    And respond appropriately.
    -> This is a TypedDict that defines the structure of the state.
    -> This state will hold messages, including input from users and output from agents or tools.
    -> Extended to support fallback mechanism with violation context.
'''

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Fallback-related fields
    validation_failed: Optional[bool]
    violation_category: Optional[str] 
    violation_context: Optional[Dict[str, Any]] 