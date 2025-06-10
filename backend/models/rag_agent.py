from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from models.state import State
from models.llm import LLM
from models.tools.retriever_tool import retrieve_information
from utils.graph_image import GraphImage
from langchain_core.runnables import RunnableConfig
from utils.helper import get_chat_history
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
        self.chats_by_session_id = kwargs.get("chats_by_session_id", {})
    
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
    
    # TODO: some issues with the chat history, saving too many messages, causing the chat history to be too large
    def _call_model(self, state: State, config: RunnableConfig):
        """Call the LLM with the current state."""

        if "configurable" not in config or "session_id" not in config["configurable"]:
            raise ValueError(
                "Make sure that the config includes the following information: {'configurable': {'session_id': 'some_value'}}"
            )

        session_id = config["configurable"]["session_id"]
        
        # 🔧 GET CLEAN CHAT HISTORY (for personal context like names)
        recent_history = self._get_recent_clean_history(session_id, max_messages=6)
        
        # 🎯 ADD SYSTEM PROMPT HERE
        system_prompt = SystemMessage(content="""
You are a trauma-informed, empathetic mental health assistant designed to support military personnel and veterans.

RESPONSE FORMAT GUIDELINES:
- Keep responses concise (400 words maximum)
- Use short paragraphs (2-3 sentences each)
- Use bullet points for multiple items or steps
- Add line breaks between sections for readability
- Start with the most important information first
- Use markdown formatting (headers, lists, emphasis) when helpful

When interacting with users:
1. Always prioritize empathy, active listening, and emotional validation.
2. Use retrieved information from trusted trauma-informed and military-specific resources to guide your responses.
3. If you do not have enough information or if a question is out of scope (e.g., medical diagnosis, legal advice), gently inform the user and encourage seeking professional help.
4. Never speculate, fabricate information, or provide unsafe or triggering content.
5. Always use gender-neutral, inclusive, and respectful language.
6. Avoid re-traumatization: do not probe for explicit trauma details unless the user voluntarily offers them, and then respond with sensitivity.
7. When appropriate, suggest mindfulness, grounding techniques, or trusted support resources.
8. If signs of severe distress, self-harm, or crisis appear, follow escalation protocol and recommend contacting a qualified professional or emergency service.
9. Be clear, compassionate, and concise. Always prioritize the user's emotional safety and privacy.

CRITICAL - TOOL USAGE RULES:
- NEVER show, mention, or display function calls, tool calls, or JSON objects in your responses
- When you need additional information, call the retrieve_information tool silently
- Integrate retrieved information naturally into your response as if you knew it all along
- Do NOT say things like "Let me search for that" or "Here's a function call"
- Your response should be complete, natural, and conversational - no technical artifacts

You are here to support — not to replace professional therapy.
        """)
        
        # Combine with current state (system prompt first)
        messages = [system_prompt] + recent_history + state["messages"]
        
        response = self.llm_with_tools.invoke(messages)
        
        # Store clean exchange
        self._store_clean_exchange(session_id, state["messages"], response)

        return {"messages": [response]}

    def _get_recent_clean_history(self, session_id: str, max_messages: int = 6):
        """Get recent conversation history (for personal context)."""
        if session_id not in self.chats_by_session_id:
            return []
        
        chat_history = self.chats_by_session_id[session_id]
        clean_messages = []
        
        # Get clean conversation messages only
        for msg in chat_history.messages:
            if isinstance(msg, HumanMessage):
                clean_messages.append(msg)
            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                # Create clean AI message
                clean_messages.append(AIMessage(content=msg.content))
        
        return clean_messages[-max_messages:]

    def _store_clean_exchange(self, session_id: str, user_messages, ai_response):
        """Store clean conversation exchange."""
        if not ai_response.content or ai_response.tool_calls:
            return
        
        if session_id not in self.chats_by_session_id:
            from langchain_core.chat_history import InMemoryChatMessageHistory
            self.chats_by_session_id[session_id] = InMemoryChatMessageHistory()
        
        chat_history = self.chats_by_session_id[session_id]
        
        # Store user message and AI response
        for msg in user_messages:
            if isinstance(msg, HumanMessage):
                chat_history.add_message(msg)
                chat_history.add_message(AIMessage(content=ai_response.content))
                break
        
        # Limit history size (keep last 8 messages = 4 exchanges)
        if len(chat_history.messages) > 8:
            chat_history.messages = chat_history.messages[-8:]
    
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
        # for message_chunk, metadata in graph.stream( 
        #     {"topic": "ice cream"},
        #     stream_mode="messages",
        # ):
        #     if message_chunk.content:
        #         print(message_chunk.content, end="|", flush=True)

        for chunk, metadata in self.graph.stream(initial_state, stream_mode="messages"):

            if isinstance(chunk, AIMessage):
                if chunk.content:
                    print(chunk.content)
                    yield chunk.content

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