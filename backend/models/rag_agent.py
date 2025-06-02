from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from models.state import State
from models.llm import LLM
from models.tools.retriever_tool import retrieve_information
from utils.graph_image import GraphImage
class RAGAgent:
    """
    A RAG (Retrieval-Augmented Generation) agent using langgraph.
    Handles user queries by first retrieving relevant information and then generating responses.
    """
    
    def __init__(self, llm: LLM, **kwargs):
        self.llm = llm.create_chat()
        self.tools = [retrieve_information]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the langgraph state graph for the RAG agent."""
        
        # Create the graph
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )
        workflow.add_edge("tools", "agent")
        
        graph = workflow.compile()
        GraphImage.create_graph_image(graph)
        return graph
    
    def _call_model(self, state: State):
        """Call the LLM with the current state."""
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: State) -> Literal["continue", "end"]:
        """Determine whether to continue to tools or end the conversation."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM makes a tool call, continue to tools
        if last_message.tool_calls:
            return "continue"
        # Otherwise, end
        return "end"
    
    def invoke(self, user_input: str) -> str:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            user_input: The user's question or query
            
        Returns:
            str: The agent's response
        """
        # Create the initial state with the user message
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Extract the final AI message
        final_message = result["messages"][-1]
        if isinstance(final_message, AIMessage):
            return final_message.content
        else:
            return "I apologize, but I couldn't generate a proper response."
    
    async def ainvoke(self, user_input: str) -> str:
        """Async version of invoke."""
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        result = await self.graph.ainvoke(initial_state)
        
        final_message = result["messages"][-1]
        if isinstance(final_message, AIMessage):
            return final_message.content
        else:
            return "I apologize, but I couldn't generate a proper response."
    
    def stream(self, user_input: str):
        """Stream the agent's response."""
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        for chunk in self.graph.stream(initial_state):
            if "agent" in chunk and "messages" in chunk["agent"]:
                message = chunk["agent"]["messages"][-1]
                if isinstance(message, AIMessage) and message.content:
                    yield message.content

    async def astream(self, user_input: str):
        """Async version of stream."""
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }

        async for chunk in self.graph.astream(initial_state):
            if "agent" in chunk and "messages" in chunk["agent"]:
                message = chunk["agent"]["messages"][-1]
                if isinstance(message, AIMessage) and message.content:
                    yield message.content

# Example usage and testing functions
def create_rag_agent(provider: str = "openai", model_name: str = "gpt-3.5-turbo", **kwargs) -> RAGAgent:
    """
    Create a RAG agent with the specified LLM configuration.
    
    Args:
        provider: LLM provider ('openai', 'openrouter', 'huggingface_pipeline', 'huggingface_endpoint')
        model_name: Model identifier
        **kwargs: Additional LLM configuration
        
    Returns:
        RAGAgent: Configured RAG agent
    """
    llm = LLM(provider=provider, model_name=model_name, **kwargs)
    return RAGAgent(llm)

def test_rag_agent():
    """Test function for the RAG agent."""
    # This would be used for testing once the retriever is properly initialized
    try:
        agent = create_rag_agent()
        response = agent.invoke("What information do you have about machine learning?")
        print(f"Agent response: {response}")
    except Exception as e:
        print(f"Error testing RAG agent: {e}")

if __name__ == "__main__":
    test_rag_agent() 