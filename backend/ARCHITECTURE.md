# RAG System Architecture with LangGraph

## Overview

This document describes the architecture of our Retrieval-Augmented Generation (RAG) system built with LangGraph. The system provides intelligent question-answering by combining document retrieval with large language models in a stateful, graph-based workflow.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG System                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   FastAPI   │    │ RAG Agent   │    │ Vector DB   │         │
│  │   Server    │◄──►│ (LangGraph) │◄──►│  (Chroma)   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                   │                   │               │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Config    │    │ LLM Models  │    │ Embeddings  │         │
│  │  Manager    │    │(Multi-Prov) │    │(HuggingFace)│         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. LangGraph Agent (`models/rag_agent.py`)

The heart of our system is a LangGraph-based agent that orchestrates the RAG workflow:

```python
# Simplified LangGraph workflow
START → Agent (LLM) → Should Continue? → Tools (Retriever) → Agent → END
                           ↓
                          END
```

**Key Features:**
- **Stateful Conversations**: Maintains conversation history using LangGraph's state management
- **Tool Integration**: Seamlessly integrates retrieval tools with LLM reasoning
- **Conditional Routing**: Intelligently decides when to retrieve information vs. respond directly
- **Async Support**: Full async/await support for scalable applications

**Workflow Steps:**
1. **Agent Node**: Processes user input and decides whether to use tools
2. **Tools Node**: Retrieves relevant documents from vector database
3. **Conditional Routing**: Determines whether to continue with tools or end

### 2. Vector Database (`models/vector_database.py`)

Manages document storage and retrieval using Chroma:

**Features:**
- **Flexible Embeddings**: Supports any HuggingFace embedding model
- **Persistent Storage**: Documents persist across sessions
- **Metadata Support**: Rich metadata for document filtering and organization
- **Similarity Search**: Efficient semantic search capabilities

### 3. LLM Integration (`models/llm.py`)

Comprehensive LLM support with factory pattern:

**Supported Providers:**
- **OpenAI**: GPT models via OpenAI API
- **OpenRouter**: Access to open-source models
- **HuggingFace Pipeline**: Local model inference
- **HuggingFace Endpoint**: Hosted model endpoints

**Features:**
- **Provider Abstraction**: Unified interface across all providers
- **Configuration Management**: Flexible parameter configuration
- **Quantization Support**: Memory-efficient local model loading

### 4. Application Manager (`core/app.py`)

Central orchestrator that ties all components together:

**Responsibilities:**
- **Component Initialization**: Sets up vector DB, retriever, and agent
- **Data Management**: Handles document loading and processing
- **Query Processing**: Provides unified query interface
- **Statistics**: System monitoring and health checks

### 5. Configuration System (`config/settings.py`)

Environment-based configuration management:

**Features:**
- **Environment Variables**: Configurable via env vars
- **Validation**: Configuration validation and error reporting
- **Defaults**: Sensible defaults for quick setup
- **Type Safety**: Dataclass-based configuration with type hints

### 6. API Layer (`api/routes.py`)

FastAPI-based REST API for external integration:

**Endpoints:**
- `POST /query`: Query the RAG system
- `POST /documents`: Add documents to knowledge base
- `GET /search`: Search documents without generation
- `GET /stats`: System statistics and health

## Data Flow

### 1. Document Ingestion
```
Raw Data → JSON Parser → Document Objects → Vector DB → Embeddings
```

### 2. Query Processing
```
User Query → RAG Agent → LLM Decision → Retrieval Tool → Vector Search → Context → LLM Generation → Response
```

### 3. State Management
```
Initial State → Agent Processing → Tool Calls → State Updates → Final Response
```

## LangGraph Design Patterns

### 1. State Schema
```python
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

Our state is minimal but powerful:
- **Messages**: Conversation history with automatic message aggregation
- **Type Safety**: TypedDict ensures type checking
- **Extensible**: Easy to add new state fields

### 2. Node Functions
```python
def _call_model(self, state: State):
    """Call the LLM with the current state."""
    messages = state["messages"]
    response = self.llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

Each node is a pure function that:
- Takes current state as input
- Performs specific operation
- Returns state updates

### 3. Conditional Edges
```python
def _should_continue(self, state: State) -> Literal["continue", "end"]:
    """Determine whether to continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "continue"
    return "end"
```

Smart routing based on:
- **Tool Calls**: If LLM requests tools, continue to tools node
- **Direct Response**: If LLM provides direct answer, end workflow

### 4. Tool Integration
```python
from langgraph.prebuilt import ToolNode

workflow.add_node("tools", ToolNode(self.tools))
```

Seamless tool integration:
- **Automatic Tool Calling**: LangGraph handles tool invocation
- **Result Processing**: Tool results automatically added to state
- **Error Handling**: Built-in error handling for tool failures

## Extensibility

### Adding New Tools
```python
@tool
def web_search_tool(query: str) -> str:
    """Search the web for current information."""
    # Implementation here
    return search_results

# Add to agent
class ExtendedRAGAgent(RAGAgent):
    def __init__(self, llm):
        super().__init__(llm)
        self.tools.append(web_search_tool)
```

### Adding New Nodes
```python
def custom_processing_node(state: State):
    """Custom processing logic."""
    # Your logic here
    return {"messages": [processed_message]}

# Add to workflow
workflow.add_node("custom", custom_processing_node)
workflow.add_edge("agent", "custom")
```

### Multi-Agent Patterns
```python
# Future extension: Multiple specialized agents
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("writing_agent", writing_agent_node)
workflow.add_conditional_edges(
    "router",
    route_to_specialist,
    ["research_agent", "writing_agent"]
)
```

## Performance Considerations

### 1. Async Processing
- All major operations support async/await
- Non-blocking I/O for database and LLM calls
- Concurrent request handling

### 2. Caching
- Vector database persistence reduces recomputation
- LLM response caching (can be added)
- Embedding caching for repeated documents

### 3. Memory Management
- Streaming responses for large outputs
- Quantization for local models
- Configurable batch sizes

### 4. Scalability
- Stateless design (except for conversation state)
- Horizontal scaling via load balancing
- Database connection pooling

## Security Considerations

### 1. API Key Management
- Environment variable configuration
- No hardcoded credentials
- Provider-specific key validation

### 2. Input Validation
- Pydantic models for API validation
- Query length limits
- Metadata sanitization

### 3. Access Control
- CORS configuration
- Rate limiting (can be added)
- Authentication middleware (can be added)

## Monitoring and Observability

### 1. Logging
- Structured logging throughout the system
- Error tracking and reporting
- Performance metrics

### 2. Health Checks
- System status endpoints
- Component health monitoring
- Database connectivity checks

### 3. Metrics
- Query response times
- Document retrieval accuracy
- System resource usage

## Future Enhancements

### 1. Multi-Modal Support
- Image and document processing
- Audio transcription integration
- Video content analysis

### 2. Advanced RAG Techniques
- Hierarchical retrieval
- Query rewriting
- Result re-ranking

### 3. Agent Collaboration
- Multiple specialized agents
- Agent-to-agent communication
- Workflow orchestration

### 4. Real-time Features
- Live document updates
- Streaming responses
- WebSocket support

This architecture provides a solid foundation for building sophisticated RAG applications while maintaining flexibility for future enhancements. 