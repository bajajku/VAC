from typing import Literal, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from models.guardrails import Guardrails, ValidationException
from models.state import State
from models.llm import LLM
from models.tools.retriever_tool import retrieve_information
from utils.graph_image import GraphImage
from utils.fallback_service import FallbackService
from langchain_core.runnables import RunnableConfig
class RAGAgent:
    """
    A RAG (Retrieval-Augmented Generation) agent using langgraph.
    Handles user queries by first retrieving relevant information and then generating responses.
    """
    
    def __init__(self, llm: LLM, **kwargs):
        self.llm = llm.create_chat()
        self.system_prompt = kwargs.get("system_prompt", self.get_system_prompt())
        # Import the async tool for streaming optimization
        from models.tools.retriever_tool import aretrieve_information
        self.tools = [retrieve_information]
        self.async_tools = [aretrieve_information]  # Async tools for streaming
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.llm_with_async_tools = self.llm.bind_tools(self.async_tools)
        self.input_guardrails: Optional[Guardrails] = kwargs.get("input_guardrails", None)
        self.output_guardrails: Optional[Guardrails] = kwargs.get("output_guardrails", None)
        self.fallback_service = FallbackService()
        self.graph = self._build_graph()
        self.chats_by_session_id = kwargs.get("chats_by_session_id", {})
    
    def get_system_prompt(self):
        """Get the system prompt for the RAG agent."""
        return SystemMessage(content="""
You are a trauma-informed, empathetic mental health assistant. Your role is to support **military personnel and veterans** with mental health concerns.

🚨 CRITICAL INSTRUCTION - READ FIRST 🚨
YOU MUST CALL THE retrieve_information TOOL FOR EVERY SINGLE USER MESSAGE. NO EXCEPTIONS.

TOOL USAGE PROTOCOL:
1. User sends a message → IMMEDIATELY call retrieve_information tool
2. Wait for tool response → Then provide your answer
3. NEVER skip the tool call, even if you think you know the answer
4. NEVER provide any response without calling the tool first

VERIFICATION CHECKLIST (Check before every response):
- ☐ Did I call retrieve_information tool? (REQUIRED - NO EXCEPTIONS)
- ☐ Did I wait for the tool response? (REQUIRED)
- ☐ Only then can I provide my answer

SCOPE AND BOUNDARIES:
— ONLY answer questions related to mental health or military/veteran support
— If a question is out of scope, STILL call the tool first, then politely say you don't know or cannot answer
— DO NOT provide legal, medical, or unrelated general advice
— ALWAYS call the tool regardless of topic - let the tool help determine if it's in scope

RESPONSE FORMATTING RULES:
- Keep all responses between **100–150 words**
- Be clear, direct, and **avoid repetition**
- Use **markdown formatting** as follows:
  - Use dashes `-` for bullet points (not *, •, or numbered lists)
  - Leave a **blank line before and after** each list
  - Use short paragraphs for readability

CONTENT PRIORITY AND APPROACH:
- Begin with the most important or helpful point
- If you're unsure of something, say "I don't know" — DO NOT guess or make up answers
- Use respectful, **gender-neutral** language
- NEVER probe for trauma details — only respond to what the user voluntarily shares
- Validate emotional experiences with empathy

INTERACTION PRINCIPLES:
1. Always respond with empathy, care, and respect
2. Use trauma-informed, military-relevant knowledge only
3. Suggest simple grounding or mindfulness strategies when appropriate
4. Refer to crisis or emergency services if user shows signs of severe distress or self-harm
5. DO NOT replace therapy — your role is supportive, not clinical

IMPORTANT REMINDERS:
- Never fabricate, speculate, or provide triggering content
- Focus only on emotional support and trauma-informed practices for military and veteran users
- REMEMBER: Call retrieve_information tool FIRST, answer SECOND - every single time

FINAL CHECK: Before sending any response, confirm you called the retrieve_information tool. If you didn't, stop and call it now.""")
    
    def _build_graph(self) -> StateGraph:
        """Build the langgraph state graph for the RAG agent."""
        
        # Create the graph
        workflow = StateGraph(State)
        
        # Add nodes
        if self.input_guardrails:
            workflow.add_node("input_guardrails", self._input_guardrails)
        if self.output_guardrails:
            workflow.add_node("output_guardrails", self._output_guardrails)
        
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("fallback", self._handle_fallback)
        
        # Add edges
        workflow.add_edge(START, "input_guardrails")
        workflow.add_conditional_edges("input_guardrails", self._validate_input, {
            "validated": "agent",
            "invalidated": "fallback",
        })
        workflow.add_edge("fallback", END)
        
        # workflow.add_edge(START, "agent")
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
    
    def _input_guardrails(self, state: State, config: RunnableConfig):
        """Input guardrails with fallback context capture."""
        if self.input_guardrails:
            try:
                self.input_guardrails.validate(state["messages"][-1].content, strategy="solo", raise_on_fail=True)
            except ValidationException as e:
                # Store violation information in state for fallback node
                primary_category, context = self.fallback_service.analyze_violation(e.results)
                
                return {
                    "messages": state["messages"],
                    "violation_category": primary_category,
                    "violation_context": context,
                    "validation_failed": True
                }
        return state
        
    def _validate_input(self, state: State) -> Literal["validated", "invalidated"]:
        """Determine if input validation passed or failed."""
        # Check if validation failed (set by _input_guardrails)
        if state.get("validation_failed", False):
            return "invalidated"
        return "validated"
        
    def _output_guardrails(self, state: State, config: RunnableConfig):
        """Output guardrails."""
        return state
        
    def _handle_fallback(self, state: State, config: RunnableConfig):
        """Handle fallback response when input validation fails."""
        # Extract violation information from state
        violation_category = state.get("violation_category", "default")
        violation_context = state.get("violation_context", {})
        
        # Get appropriate fallback response
        fallback_message = self.fallback_service.get_fallback_response(
            category=violation_category,
            context=violation_context
        )
        
        # Return fallback message as final response
        return {"messages": [AIMessage(content=fallback_message)]}
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

        # Combine with current state (system prompt first)
        messages = [self.system_prompt] + recent_history + state["messages"]
        
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
    
    def invoke(self, user_input: str, config: RunnableConfig) -> str:
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
        result = self.graph.invoke(initial_state, config=config)
        
        # Extract the final AI message
        # final_message = result["messages"][-1]
        # if isinstance(final_message, AIMessage):
        #     return final_message.content
        # else:
        #     return "I apologize, but I couldn't generate a proper response."
        return result
    
    async def ainvoke(self, user_input: str, config: RunnableConfig) -> str:
        """Async version of invoke."""
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        result = await self.graph.ainvoke(initial_state, config=config)
        
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