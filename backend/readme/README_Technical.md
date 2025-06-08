# RAG System with LangGraph - Technical Overview

## Architecture

Our system implements a **Retrieval-Augmented Generation (RAG)** architecture using cutting-edge AI technologies to provide intelligent document-based question answering.

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   RAG Agent     │    │  Vector DB      │
│   (API Layer)   │◄──►│  (LangGraph)    │◄──►│  (Chroma)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │   LLM Models    │    │  Embeddings     │
│                 │    │ (Multiple)      │    │ (HuggingFace)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Technical Stack

**Core Framework**: LangGraph for stateful AI workflows  
**Vector Database**: ChromaDB for semantic document storage  
**Embeddings**: HuggingFace Transformers (sentence-transformers/all-MiniLM-L6-v2)  
**LLM Support**: OpenAI GPT, OpenRouter, HuggingFace Pipeline  
**API Framework**: FastAPI with async support  
**Language**: Python 3.12+  

## Key Features

### 1. Stateful Conversation Management
- **LangGraph workflow**: Maintains conversation context across interactions
- **Memory persistence**: Conversation history retained during session
- **State management**: TypedDict-based state schema for type safety

### 2. Multi-Provider LLM Support
```python
Supported Providers:
├── OpenAI (GPT-3.5, GPT-4)
├── OpenRouter (Open source models)
├── HuggingFace Pipeline (Local inference)
└── HuggingFace Endpoints (Hosted models)
```

### 3. Advanced Retrieval
- **Semantic search**: Vector similarity-based document retrieval
- **Metadata filtering**: Rich document categorization and filtering
- **Configurable retrieval**: Adjustable k-value for result count
- **Persistent storage**: ChromaDB for scalable vector operations

## System Workflow

### Document Processing Pipeline
```
Raw Documents → Text Extraction → Chunking → Embedding Generation → Vector Storage
```

### Query Processing Pipeline
```
User Query → LangGraph Agent → Similarity Search → Context Assembly → LLM Response
```

### LangGraph Workflow
```python
START → Agent (LLM) → Decision Node → Retrieval Tools → Response Generation → END
                           ↓
                    Direct Response → END
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/query` | POST | Submit questions to the RAG system |
| `/documents` | POST | Add documents to knowledge base |
| `/search` | GET | Search documents without generation |
| `/stats` | GET | System health and statistics |

## Configuration

### Environment Variables
```bash
LLM_PROVIDER=openai                    # LLM provider selection
LLM_MODEL=gpt-3.5-turbo               # Model specification
OPENAI_API_KEY=your_key_here          # API authentication
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_PERSIST_DIR=./chroma_db     # Database persistence
RETRIEVAL_K=4                         # Retrieved document count
```

### Programmatic Configuration
```python
from core.app import RAGApplication

app = RAGApplication()
config = {
    "llm_provider": "openai",
    "llm_model": "gpt-4",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "api_key": "your-api-key"
}
app.initialize(**config)
```

## Performance Characteristics

**Async Operations**: Full async/await support for scalability  
**Vector Search**: Sub-second similarity search on 10k+ documents  
**Memory Efficiency**: Quantization support for local models  
**Caching**: Persistent embeddings reduce recomputation overhead  

## Deployment Options

### 1. Local Development
```bash
python main.py  # Interactive CLI
uvicorn api:app --reload  # API server
```

### 2. Production Deployment
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Docker Deployment
```dockerfile
FROM python:3.12-slim
COPY . /app
WORKDIR /app
RUN pip install -e .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0"]
```

## Integration Examples

### Python SDK
```python
from core.app import RAGApplication

app = RAGApplication()
app.initialize(llm_provider="openai", api_key="key")
app.add_documents_from_text(["Document content..."])
response = app.query("What is machine learning?")
```

### REST API
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is our policy?", "k": 3}'
```

## Scalability & Monitoring

**Horizontal Scaling**: Stateless design enables load balancing  
**Database Scaling**: ChromaDB supports distributed deployments  
**Monitoring**: Built-in statistics and health check endpoints  
**Error Handling**: Comprehensive exception handling and logging  

---

**Technical Contact**: For implementation details, architecture questions, or custom integrations. 