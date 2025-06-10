# Enhanced RAG System with LangGraph

A comprehensive Retrieval-Augmented Generation (RAG) system built with LangGraph, LangChain, and FastAPI. This system provides intelligent question-answering capabilities by combining advanced document processing, multiple retrieval strategies, and feedback mechanisms with large language models.

## 🚀 Key Features

### ✨ **Enhanced Document Processing**
- **Offline Preprocessing**: LLM-based data cleaning with 90% faster API startup
- **Recursive Text Splitting**: Smart chunking using LangChain's RecursiveCharacterTextSplitter
- **Content Cleaning**: Remove ads, navigation, footers from scraped content
- **Rich Metadata**: Enhanced document metadata with processing statistics
- **Flexible Processing**: Choose between enhanced (LLM-cleaned) or basic processing

### 🎯 **Advanced Retrieval Strategies**
- **Similarity Search**: Traditional semantic similarity using embeddings
- **MMR (Maximal Marginal Relevance)**: Reduces redundancy, increases diversity
- **Hybrid Retrieval**: Combines semantic + keyword search approaches
- **Ensemble Retrieval**: Aggregates multiple strategies for maximum accuracy
- **Re-ranking**: Advanced result scoring and intelligent ordering
- **Result Fusion**: Combines and deduplicates results from multiple methods

### 📊 **Feedback & Analytics System**
- **MongoDB Integration**: Persistent feedback storage and analytics
- **Real-time Feedback**: Collect user feedback on AI responses
- **Feedback Dashboard**: Statistics and insights on system performance
- **Session Tracking**: Track feedback across conversation sessions
- **REST API**: Complete CRUD operations for feedback management

### 🏗️ **Production-Ready Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   RAG Agent     │    │  Vector DB      │
│   (Enhanced)    │◄──►│  (LangGraph)    │◄──►│  (Chroma)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Feedback      │    │   LLM Models    │    │  Embeddings     │
│   System        │    │ (Multi-Prov)    │    │ (HuggingFace)   │
│   (MongoDB)     │    │ + Preprocessing │    │ + Advanced      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

1. **LangGraph Agent**: Orchestrates the RAG workflow with stateful conversation management
2. **Enhanced Vector Database**: Advanced chunking and retrieval with multiple strategies
3. **LLM Integration**: Supports multiple providers (OpenAI, OpenRouter, HuggingFace, Together AI)
4. **Preprocessing Pipeline**: Offline data processing with LLM-based cleaning
5. **Feedback System**: MongoDB-based feedback collection and analytics
6. **FastAPI Server**: Enhanced REST API with advanced endpoints

## ⚡ Quick Start

### Prerequisites

- Python 3.12+
- API key for your chosen LLM provider
- MongoDB (optional, for feedback system)

### 🚀 Production Setup (Recommended)

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -e .
   # or with uv
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env and set:
   # TOGETHER_API_KEY=your_api_key_here
   # SKIP_AUTO_PROCESSING=true
   # MONGODB_URL=your_mongodb_url (optional)
   ```

3. **Preprocess data (one-time):**
   ```bash
   # Enhanced processing with LLM cleaning (recommended)
   python preprocess_data.py --auto
   
   # Or basic processing (faster, good quality)
   python preprocess_data.py --auto --basic
   ```

4. **Start API (fast startup!):**
   ```bash
   python api.py
   # ✅ Ready in 30 seconds vs 5+ minutes!
   ```

### 🧪 Development Setup

```bash
# Configure for development (auto-processing on startup)
echo "SKIP_AUTO_PROCESSING=false" >> .env

# Start API with auto-processing
python api.py
```

## 📖 Enhanced API Usage

### New Enhanced Endpoints

#### **Enhanced Query** - Better Retrieval
```bash
curl -X POST "http://localhost:8000/query-enhanced" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is machine learning?",
       "retrieval_strategy": "hybrid",
       "k": 5,
       "enable_reranking": true,
       "similarity_threshold": 0.7
     }'
```

**Retrieval Strategies:**
- `similarity`: Fast semantic similarity (default)
- `mmr`: Diverse results with reduced redundancy
- `hybrid`: Balanced approach combining multiple methods
- `ensemble`: Highest quality, aggregates multiple strategies

#### **Enhanced Document Upload** - Smart Processing
```bash
curl -X POST "http://localhost:8000/documents" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": ["Document with ads and noise..."],
       "metadatas": [{"title": "Clean Doc", "source": "web"}],
       "use_enhanced_processing": true
     }'
```

#### **Feedback System** - User Analytics
```bash
# Create feedback
curl -X POST "http://localhost:8000/feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "session-123",
       "question": "What is AI?",
       "answer": "AI is...",
       "feedback_type": "positive",
       "rating": 5,
       "feedback_text": "Great explanation!"
     }'

# Get feedback statistics
curl "http://localhost:8000/feedback-stats"
```

#### **Preprocessing & Configuration**
```bash
# Load preprocessed JSON with enhanced processing
curl -X POST "http://localhost:8000/load-json-enhanced" \
     -d '{"file_path": "data.json", "use_enhanced_processing": true}'

# Configure retriever dynamically
curl -X POST "http://localhost:8000/configure-retriever" \
     -d '{"retrieval_strategy": "ensemble", "max_results": 3}'

# Get retrieval performance statistics
curl "http://localhost:8000/retrieval-stats"
```

### Backward Compatible Endpoints

All original endpoints still work:
- `POST /query` - Basic querying
- `POST /documents` - Basic document upload
- `POST /load-json` - Basic JSON loading
- `GET /search` - Document search
- `GET /stats` - System statistics

## 🔧 Advanced Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TOGETHER_API_KEY` | Together AI API key | - |
| `LLM_PROVIDER` | Provider (openai, openrouter, chatopenai, huggingface_pipeline) | chatopenai |
| `LLM_MODEL` | Model name | meta-llama/Llama-3.3-70B-Instruct-Turbo-Free |
| `SKIP_AUTO_PROCESSING` | Skip expensive startup processing | false |
| `CHUNK_SIZE` | Text chunk size for processing | 800 |
| `CHUNK_OVERLAP` | Overlap between chunks | 100 |
| `MONGODB_URL` | MongoDB connection URL (for feedback) | - |
| `MONGODB_DATABASE` | MongoDB database name | vac_feedback |
| `EMBEDDING_MODEL` | HuggingFace embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| `VECTOR_DB_PERSIST_DIR` | Vector database directory | ./chroma_db |
| `RETRIEVAL_K` | Number of documents to retrieve | 4 |

### MongoDB Setup (Optional - for Feedback System)

```bash
# Local MongoDB
brew install mongodb-community
brew services start mongodb/brew/mongodb-community

# Or use MongoDB Atlas (cloud)
# Set MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/
```

### Preprocessing Options

```bash
# Enhanced processing (best quality, takes 2-5 minutes)
python preprocess_data.py --auto

# Basic processing (faster, good quality)
python preprocess_data.py --auto --basic

# Custom chunk sizes
python preprocess_data.py --auto --chunk-size 1000 --chunk-overlap 200

# Process specific file
python preprocess_data.py --input path/to/data.json --output ./cleaned/
```

## 📊 Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **API Startup Time** | 5+ minutes | 30 seconds | **90% faster** |
| **Data Processing** | Every startup | Once offline | **Cost reduction** |
| **Retrieval Quality** | Basic similarity | Multi-strategy | **Better answers** |
| **Chunk Quality** | Basic splitting | Recursive smart | **Better context** |
| **User Feedback** | None | Full analytics | **Quality insights** |

## 🛠️ Programmatic Usage

### Enhanced Python SDK
```python
from core.app import RAGApplication

# Initialize with enhanced features
app = RAGApplication()
app.initialize(
    llm_provider="chatopenai",
    llm_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    api_key="your-api-key"
)

# Load preprocessed data (fast!)
app.load_preprocessed_data("scripts/data_cleaning/cleaned_data/")

# Query with different strategies
response = app.query_enhanced(
    "What is machine learning?",
    retrieval_strategy="hybrid",
    k=5
)

# Add documents with enhanced processing
app.add_documents_from_text(
    texts=["Document with noise and ads..."],
    metadatas=[{"source": "web", "cleaned": True}],
    use_enhanced_processing=True
)
```

### Multiple LLM Providers
```python
# Together AI (recommended)
app.initialize(
    llm_provider="chatopenai",
    llm_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    api_key="your-together-key"
)

# OpenAI
app.initialize(
    llm_provider="openai",
    llm_model="gpt-4",
    api_key="your-openai-key"
)

# OpenRouter (open source models)
app.initialize(
    llm_provider="openrouter",
    llm_model="microsoft/wizardlm-2-8x22b",
    api_key="your-openrouter-key"
)

# HuggingFace (local inference)
app.initialize(
    llm_provider="huggingface_pipeline",
    llm_model="microsoft/DialoGPT-medium",
    quantization=True  # For memory efficiency
)
```

## 🧪 Testing & Validation

### Run Enhanced Demos
```bash
# Complete enhanced RAG demonstration
python examples/enhanced_rag_demo.py

# Workflow demonstration
python examples/workflow_demonstration.py

# Test enhanced API features
python tests/test_enhanced_api.py
```

### Interactive CLI Demo
```bash
python main.py
# Choose enhanced features in the menu
```

### Component Testing
```python
# Test different retrieval strategies
from models.advanced_retriever import AdvancedRetriever

retriever = AdvancedRetriever(vector_db)
results = retriever.retrieve(
    "test query",
    strategy="ensemble",
    k=5
)

# Test preprocessing
from scripts.data_cleaning.data_cleaner import DataCleaner

cleaner = DataCleaner(data, use_advanced_processing=True)
processed_docs = cleaner.clean_data()
```

## 🔍 LangGraph Workflow

Enhanced workflow with multiple strategies:

```python
# Enhanced workflow with retrieval strategies
START → Agent (LLM) → Strategy Selection → Multi-Retrieval → Re-ranking → Response → END
                           ↓
                    Direct Response → END
```

1. **Agent Node**: Processes user input and selects retrieval strategy
2. **Strategy Selection**: Chooses optimal retrieval method based on query
3. **Multi-Retrieval**: Executes selected strategy (similarity/MMR/hybrid/ensemble)
4. **Re-ranking**: Intelligent result ordering and fusion
5. **Response Generation**: Context-aware response with enhanced metadata

## 📈 Production Deployment

### Deployment Checklist
- [ ] Set `SKIP_AUTO_PROCESSING=true` in production
- [ ] Run preprocessing offline: `python preprocess_data.py --auto`
- [ ] Configure MongoDB for feedback system
- [ ] Set up monitoring for `/stats` and `/feedback-stats`
- [ ] Test enhanced endpoints: `/query-enhanced`, `/feedback`
- [ ] Configure load balancing for FastAPI server

### Docker Deployment
```dockerfile
FROM python:3.12-slim
COPY . /app
WORKDIR /app
RUN pip install -e .

# Preprocess data in container
RUN python preprocess_data.py --auto

ENV SKIP_AUTO_PROCESSING=true
EXPOSE 8000
CMD ["uvicorn", "api:app_api", "--host", "0.0.0.0", "--port", "8000"]
```

### Health Monitoring
```bash
# System health
curl "http://localhost:8000/stats"

# Feedback system status
curl "http://localhost:8000/feedback-service-status"

# Retrieval performance
curl "http://localhost:8000/retrieval-stats"
```

## 🛠️ Extending the System

### Adding New Retrieval Strategies
```python
from models.advanced_retriever import AdvancedRetriever

class CustomRetriever(AdvancedRetriever):
    def custom_strategy(self, query: str, k: int):
        # Your custom retrieval logic
        return results

# Register new strategy
retriever.register_strategy("custom", custom_strategy)
```

### Adding New Tools
```python
from langchain_core.tools import tool

@tool
def custom_analysis_tool(query: str) -> str:
    """Custom analysis tool for specialized queries."""
    # Your tool logic here
    return "Analysis result"

# Add to RAG agent
class CustomRAGAgent(RAGAgent):
    def __init__(self, llm):
        super().__init__(llm)
        self.tools.append(custom_analysis_tool)
```

### Adding New Feedback Types
```python
# Extend feedback system
class CustomFeedbackService(FeedbackService):
    async def create_detailed_feedback(self, feedback_data):
        # Add custom feedback processing
        return await super().create_feedback(feedback_data)
```

## 🐛 Troubleshooting

### API Won't Start Fast
```bash
# Enable fast startup
export SKIP_AUTO_PROCESSING=true
python api.py
```

### Poor Answer Quality
```bash
# Use enhanced preprocessing
python preprocess_data.py --auto  # (not --basic)

# Try ensemble retrieval for critical queries
curl -X POST "http://localhost:8000/query-enhanced" \
  -d '{"question": "query", "retrieval_strategy": "ensemble"}'
```

### Memory Issues
```bash
# Use smaller chunk sizes
python preprocess_data.py --auto --chunk-size 600

# Enable quantization for local models
export USE_QUANTIZATION=true
```

### Feedback System Issues
```bash
# Check MongoDB connection
curl "http://localhost:8000/feedback-service-status"

# Verify MongoDB is running
mongosh --eval "db.runCommand({ connectionStatus: 1 })"
```

## 📚 Documentation

For detailed information, see the comprehensive documentation in the `readme/` directory:

- **[Quick Start Guide](readme/QUICK_START_GUIDE.md)** - 2-minute setup
- **[Enhanced RAG Features](readme/README_Enhanced_RAG.md)** - Complete feature overview
- **[Preprocessing Guide](readme/PREPROCESSING_GUIDE.md)** - Data processing details
- **[API Enhanced Features](readme/API_Enhanced_Features.md)** - API documentation
- **[Feedback Setup](readme/FEEDBACK_SETUP.md)** - MongoDB feedback system
- **[Architecture](readme/ARCHITECTURE.md)** - Technical architecture
- **[Implementation Summary](readme/IMPLEMENTATION_SUMMARY.md)** - Feature summary
- **[Non-Technical Overview](readme/README_NonTechnical.md)** - Business overview

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Update documentation
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the comprehensive documentation in `readme/`
- Review troubleshooting section above
- Create an issue in the repository
- Test with the demo scripts in `examples/`
