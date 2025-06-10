from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from models.tools.retriever_tool import retrieve_information
from models.state import State
from langchain_core.chat_history import InMemoryChatMessageHistory
'''
    Helper functions to handle tool errors and execute tools.
'''

import re
from typing import List

def extract_sources_from_toolmessage(content: str) -> List[str]:
    """
    Extracts all unique sources from a ToolMessage content string.
    
    Args:
        content (str): The full content string from ToolMessage.
    
    Returns:
        List[str]: A list of unique sources extracted from the content.
    """
    # Match patterns like (from: ...some/path.html)
    pattern = r'\(from:\s*(.*?)\)'
    matches = re.findall(pattern, content)
    
    # Deduplicate and strip extra whitespace
    unique_sources = list({match.strip() for match in matches})
    
    return unique_sources


def handle_tool_error(state) -> dict:
    """
    Function to handle errors that occur during tool execution.
    
    Args:
        state (dict): The current state of the AI agent, which includes messages and tool call details.
    
    Returns:
        dict: A dictionary containing error messages for each tool that encountered an issue.
    """
    # Retrieve the error from the current state
    error = state.get("error")
    
    # Access the tool calls from the last message in the state's message history
    tool_calls = state["messages"][-1].tool_calls
    
    # Return a list of ToolMessages with error details, linked to each tool call ID
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",  # Format the error message for the user
                tool_call_id=tc["id"],  # Associate the error message with the corresponding tool call ID
            )
            for tc in tool_calls  # Iterate over each tool call to produce individual error messages
        ]
    }
def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Function to create a tool node with fallback error handling.
    
    Args:
        tools (list): A list of tools to be included in the node.
    
    Returns:
        dict: A tool node that uses fallback behavior in case of errors.
    """
    # Create a ToolNode with the provided tools and attach a fallback mechanism
    # If an error occurs, it will invoke the handle_tool_error function to manage the error
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],  # Use a lambda function to wrap the error handler
        exception_key="error"  # Specify that this fallback is for handling errors
    )
tools_names = {
    "retrieve_information": retrieve_information
}
def execute_tools(state: State):
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:

      if not t['name'] in tools_names:
        result = "Error: There's no such tool, please try again"
      else:
        result = tools_names[t['name']].invoke(t['args'])

        results.append(
          ToolMessage(
            tool_call_id=t['id'],
            name=t['name'],
            content=str(result)
          )
        )

    return {'messages': results}

def tool_exists(state: State):
    result = state['messages'][-1]
    return len(result.tool_calls) > 0

def get_chat_history(session_id: str, chats_by_session_id: dict, max_tokens: int = 4000) -> InMemoryChatMessageHistory:
    """Get chat history with token limit."""
    chat_history = chats_by_session_id.get(session_id)
    if chat_history is None:
        chat_history = InMemoryChatMessageHistory()
        chats_by_session_id[session_id] = chat_history
    
    # Truncate if too many tokens
    if len(chat_history.messages) > 20:  # Simple heuristic
        # Keep only recent messages
        chat_history.messages = chat_history.messages[-20:]
    
    return chat_history
