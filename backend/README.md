# RAG System with LangGraph

A comprehensive Retrieval-Augmented Generation (RAG) system built with LangGraph, LangChain, and FastAPI. This system provides intelligent question-answering capabilities by combining vector-based document retrieval with large language models.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   RAG Agent     │    │  Vector DB      │
│   (API Layer)   │◄──►│  (LangGraph)    │◄──►│  (Chroma)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │   LLM Models    │    │  Embeddings     │
│   (Frontend)    │    │ (OpenAI/Local)  │    │ (HuggingFace)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

1. **LangGraph Agent**: Orchestrates the RAG workflow with stateful conversation management
2. **Vector Database**: Stores and retrieves document embeddings using Chroma
3. **LLM Integration**: Supports multiple providers (OpenAI, OpenRouter, HuggingFace)
4. **FastAPI Server**: Provides REST API endpoints for integration
5. **Data Pipeline**: Web crawling and document processing capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- API key for your chosen LLM provider (OpenAI, OpenRouter, etc.)

### Installation

1. **Clone and navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   # or with uv
   uv sync
   ```

3. **Set up environment variables:**
   ```bash
   # Create a .env file or export these variables
   export OPENAI_API_KEY="your-api-key-here"
   # or for OpenRouter
   export OPENROUTER_API_KEY="your-openrouter-key"
   export LLM_PROVIDER="openai"  # or "openrouter"
   export LLM_MODEL="gpt-3.5-turbo"
   ```

### Running the System

#### Option 1: Interactive CLI Demo
```bash
python main.py
```

#### Option 2: FastAPI Server
```bash
# Start the API server
uvicorn api.routes:app_api --host 0.0.0.0 --port 8000 --reload

# Or run directly
python api/routes.py
```

#### Option 3: Programmatic Usage
```python
from core.app import RAGApplication

# Initialize the system
app = RAGApplication()
app.initialize(
    llm_provider="openai",
    llm_model="gpt-3.5-turbo",
    api_key="your-api-key"
)

# Add some documents
app.add_documents_from_text([
    "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
    "RAG combines information retrieval with text generation."
])

# Query the system
response = app.query("What is LangGraph?")
print(response)
```

## 📖 Usage Examples

### API Endpoints

Once the FastAPI server is running, you can interact with these endpoints:

#### Query the RAG System
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is machine learning?", "k": 3}'
```

#### Add Documents
```bash
curl -X POST "http://localhost:8000/documents" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": ["Machine learning is a subset of AI..."],
       "metadatas": [{"source": "ml_guide", "topic": "ml"}]
     }'
```

#### Search Documents
```bash
curl "http://localhost:8000/search?query=machine%20learning&k=3"
```

#### System Status
```bash
curl "http://localhost:8000/"
```

### Python SDK Usage

```python
from core.app import RAGApplication
from models.llm import LLM

# Initialize with different LLM providers
app = RAGApplication()

# OpenAI
app.initialize(
    llm_provider="openai",
    llm_model="gpt-4",
    api_key="your-openai-key"
)

# OpenRouter (for open source models)
app.initialize(
    llm_provider="openrouter",
    llm_model="microsoft/wizardlm-2-8x22b",
    api_key="your-openrouter-key",
    base_url="https://openrouter.ai/api/v1"
)

# HuggingFace Pipeline (local)
app.initialize(
    llm_provider="huggingface_pipeline",
    llm_model="microsoft/DialoGPT-medium"
)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai, openrouter, huggingface_pipeline) | openai |
| `LLM_MODEL` | Model name | gpt-3.5-turbo |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENROUTER_API_KEY` | OpenRouter API key | - |
| `LLM_TEMPERATURE` | Model temperature | 0.7 |
| `EMBEDDING_MODEL` | HuggingFace embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| `VECTOR_DB_PERSIST_DIR` | Vector database directory | ./chroma_db |
| `RETRIEVAL_K` | Number of documents to retrieve | 4 |

### Programmatic Configuration

```python
from config.settings import Settings

settings = Settings()
print(settings.config.llm.provider)  # Current LLM provider
print(settings.config.vector_db.embedding_model)  # Embedding model
```

## 📊 Data Management

### Loading Data from JSON
```python
# From web crawler results
app.load_data_from_json("scripts/data_collection/crawl_results/data.json")
```

### Adding Documents Directly
```python
app.add_documents_from_text(
    texts=["Document content here..."],
    metadatas=[{"source": "manual", "date": "2024-01-01"}]
)
```

### Web Crawling
```python
from scripts.data_collection.web_crawler import WebCrawler

crawler = WebCrawler()
crawler.crawl_urls(["https://example.com"])
```

## 🧪 Testing

### Run Component Tests
```bash
python main.py
# Choose option 2: Test components only
```

### Test Individual Components
```python
# Test vector database
from models.vector_database import VectorDatabase
vdb = VectorDatabase()
vdb.create_vector_database("sentence-transformers/all-MiniLM-L6-v2")

# Test LLM
from models.llm import LLM
llm = LLM("openai", "gpt-3.5-turbo", api_key="your-key")
chat = llm.create_chat()
```

## 🔍 LangGraph Workflow

The RAG agent uses LangGraph to create a stateful workflow:

```python
# Simplified workflow
START → Agent (LLM) → Should Continue? → Tools (Retriever) → Agent → END
                           ↓
                          END
```

1. **Agent Node**: Processes user input and decides whether to use tools
2. **Tools Node**: Retrieves relevant documents from vector database
3. **Conditional Routing**: Determines whether to continue with tools or end

## 🛠️ Extending the System

### Adding New Tools
```python
from langchain_core.tools import tool

@tool
def custom_tool(query: str) -> str:
    """Custom tool description."""
    # Your tool logic here
    return "Tool result"

# Add to RAG agent
class CustomRAGAgent(RAGAgent):
    def __init__(self, llm):
        super().__init__(llm)
        self.tools.append(custom_tool)
```

### Adding New LLM Providers
```python
from models.llm import BaseLLM, LLMFactory

class CustomLLM(BaseLLM):
    def create_llm(self):
        # Your LLM implementation
        pass

# Register the new provider
LLMFactory.register_implementation("custom", CustomLLM)
```

### Adding New Vector Databases
```python
# Extend VectorDatabase class
class VectorDatabase:
    def create_vector_database(self, embedding_model, type: str = "chroma", **kwargs):
        match type:
            case "chroma":
                # Existing Chroma implementation
            case "pinecone":
                # Add Pinecone implementation
            case "weaviate":
                # Add Weaviate implementation
```

## 📈 Performance Tips

1. **Embedding Model**: Use smaller models for faster inference
2. **Chunk Size**: Optimize document chunking for your use case
3. **Retrieval K**: Balance between context and performance
4. **Caching**: Enable LLM response caching for repeated queries
5. **Async**: Use async methods for better concurrency

## 🐛 Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```bash
   export OPENAI_API_KEY="your-key"
   # or add to .env file
   ```

2. **Import Errors**
   ```bash
   # Make sure you're in the backend directory
   cd backend
   pip install -e .
   ```

3. **Vector Database Issues**
   ```bash
   # Clear the database
   rm -rf ./chroma_db
   ```

4. **Memory Issues with Local Models**
   ```python
   # Use quantization for HuggingFace models
   llm = LLM("huggingface_pipeline", "model-name", quantization=True)
   ```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the configuration options
