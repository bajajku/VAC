# Enhanced RAG System with Recursive Text Splitting

This enhanced RAG (Retrieval-Augmented Generation) system includes advanced document processing, recursive text splitting from LangChain, and multiple retrieval strategies for robust information retrieval.

## 🚀 Key Features

### 📄 Advanced Document Processing
- **Recursive Text Splitting**: Uses LangChain's `RecursiveCharacterTextSplitter` for intelligent text chunking
- **Content-Aware Splitting**: Automatically detects content type (markdown, code, structured text) and applies appropriate splitting strategies
- **Smart Preprocessing**: Removes noise patterns (ads, navigation, boilerplate) while preserving important content
- **Metadata Enhancement**: Rich metadata tracking including chunk relationships, content types, and processing statistics

### 🔍 Multiple Retrieval Strategies
- **Similarity Search**: Traditional cosine similarity-based retrieval
- **MMR (Maximum Marginal Relevance)**: Balances relevance and diversity to avoid redundant results
- **Hybrid Retrieval**: Combines multiple strategies for optimal results
- **Ensemble Retrieval**: Uses multiple retrievers and fuses results
- **Threshold Filtering**: Only returns results above a configurable similarity threshold

### 🎯 Advanced Features
- **Re-ranking**: Post-retrieval re-ranking using multiple relevance signals
- **Result Fusion**: Combines results from different retrieval methods
- **Contextual Compression**: Optional content compression using LLM
- **Multi-Query Retrieval**: Generates multiple query variations for comprehensive results
- **BM25 Integration**: Traditional keyword-based search combined with semantic search

### 📊 Performance Monitoring
- **Retrieval Statistics**: Detailed metrics on retrieval performance
- **Processing Analytics**: Document processing and chunking statistics
- **Dynamic Configuration**: Runtime configuration updates for optimization

## 🏗️ Architecture

```
Enhanced RAG System
├── Document Processing Layer
│   ├── AdvancedDocumentProcessor
│   │   ├── Recursive Text Splitting
│   │   ├── Content Type Detection
│   │   ├── Smart Preprocessing
│   │   └── Metadata Enhancement
│   └── DataCleaner (Enhanced)
│       ├── LLM-based Cleaning
│       ├── Integration with Document Processor
│       └── Processing Statistics
├── Retrieval Layer
│   ├── AdvancedRetriever
│   │   ├── Multiple Strategies
│   │   ├── Re-ranking
│   │   ├── Result Fusion
│   │   └── Performance Monitoring
│   └── Enhanced Retriever (Updated)
│       ├── Legacy Compatibility
│       ├── Advanced Features
│       └── Dynamic Configuration
└── Tool Layer
    ├── Enhanced Retriever Tool
    ├── Configuration Tool
    └── Statistics Tool
```

## 📦 Installation and Setup

### Prerequisites
All required packages are already included in the existing `requirements.txt`:
- `langchain` and related packages
- `tiktoken` for token counting
- `chromadb` for vector storage
- `sentence-transformers` for embeddings

### Quick Start

1. **Initialize the Enhanced System**:
```python
from scripts.data_cleaning.data_cleaner import DataCleaner
from models.vector_database import VectorDatabase
from utils.retriever import global_retriever

# Create enhanced data cleaner
cleaner = DataCleaner(
    your_data, 
    use_advanced_processing=True,
    chunk_size=1000,
    chunk_overlap=200
)

# Process documents with recursive splitting
processed_docs = cleaner.clean_data()

# Setup vector database
vector_db = VectorDatabase()
vector_db.create_vector_database(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db.add_documents(processed_docs)

# Initialize enhanced retriever
global_retriever.initialize(vector_db)
```

2. **Use Advanced Retrieval**:
```python
from models.tools.retriever_tool import retrieve_information, configure_retriever

# Configure retrieval strategy
configure_retriever(
    retrieval_strategy="hybrid",
    similarity_threshold=0.7,
    max_results=5,
    enable_reranking=True
)

# Retrieve with enhanced features
result = retrieve_information(
    query="Your question here",
    max_results=5,
    retrieval_strategy="hybrid",
    use_reranking=True
)
```

## 🔧 Configuration Options

### Document Processing
```python
processor = AdvancedDocumentProcessor(
    chunk_size=1000,                    # Characters per chunk
    chunk_overlap=200,                  # Overlap between chunks
    separators=["\n\n", "\n", ". "],   # Custom splitting separators
    model_name="gpt-3.5-turbo"         # For token counting
)
```

### Retrieval Strategies
- `"similarity"`: Basic cosine similarity
- `"mmr"`: Maximum Marginal Relevance for diversity
- `"hybrid"`: Combined approach with multiple techniques
- `"ensemble"`: Multiple retrievers with result fusion
- `"threshold"`: Similarity with minimum score filtering

### Advanced Parameters
```python
retriever_config = {
    "retrieval_strategy": "hybrid",
    "similarity_threshold": 0.7,        # 0.0-1.0
    "mmr_diversity_bias": 0.3,          # 0.0 (similarity) to 1.0 (diversity)
    "max_results": 10,
    "enable_reranking": True,
    "enable_compression": False
}
```

## 📈 Usage Examples

### Basic Enhanced Retrieval
```python
# Simple enhanced retrieval
result = retrieve_information("What is machine learning?")
print(result)
```

### Advanced Retrieval with Configuration
```python
# Configure for maximum diversity
configure_retriever(
    retrieval_strategy="mmr",
    similarity_threshold=0.6,
    max_results=8,
    enable_reranking=True
)

result = retrieve_information(
    query="Explain neural networks",
    max_results=5,
    retrieval_strategy="mmr"
)
```

### Performance Analysis
```python
from models.tools.retriever_tool import get_retrieval_stats

# Get detailed statistics
stats = get_retrieval_stats("Your query here")
print(stats)
```

### Custom Document Processing
```python
# Process documents with custom settings
cleaner = DataCleaner(
    data,
    use_advanced_processing=True,
    chunk_size=800,      # Smaller chunks
    chunk_overlap=100    # Less overlap
)

docs = cleaner.clean_data()

# Get processing statistics
stats = cleaner.get_statistics()
print(f"Processed {stats['processed_chunks']} chunks")
print(f"Average chunk size: {stats['average_chunk_size']} chars")
```

## 🧪 Testing and Demonstration

Run the comprehensive demonstration:
```bash
cd backend
python examples/enhanced_rag_demo.py
```

This will:
1. Create sample knowledge base
2. Process documents with recursive splitting
3. Test different retrieval strategies
4. Compare performance metrics
5. Demonstrate advanced features

## 📊 Performance Benefits

### Improved Chunking
- **Smart Boundaries**: Respects sentence and paragraph boundaries
- **Content Awareness**: Different strategies for different content types
- **Overlap Optimization**: Maintains context while avoiding redundancy
- **Metadata Tracking**: Enhanced metadata for better retrieval

### Enhanced Retrieval
- **Higher Accuracy**: Multiple strategies improve relevance
- **Reduced Redundancy**: MMR and deduplication eliminate similar results
- **Better Ranking**: Multi-factor scoring considers various relevance signals
- **Adaptive Performance**: Dynamic configuration for different use cases

### Robust Processing
- **Error Handling**: Graceful fallbacks for various failure modes
- **Monitoring**: Comprehensive statistics and performance tracking
- **Scalability**: Efficient processing of large document collections
- **Flexibility**: Configurable parameters for different domains

## 🔍 Advanced Features Details

### Recursive Text Splitting
- Uses LangChain's `RecursiveCharacterTextSplitter`
- Hierarchical splitting: paragraphs → sentences → phrases → words
- Preserves semantic boundaries
- Configurable separators for different content types

### Maximum Marginal Relevance (MMR)
- Balances relevance and diversity
- Reduces redundant information
- Configurable diversity bias
- Optimal for exploratory queries

### Result Re-ranking
- Multi-factor scoring system
- Considers term frequency, metadata, content type
- Position-aware ranking
- Length normalization

### Ensemble Retrieval
- Combines multiple retrieval methods
- Fusion scoring based on agreement
- Source tracking and transparency
- Fallback mechanisms

## 🛠️ Customization

### Custom Content Processors
```python
# Extend AdvancedDocumentProcessor
class CustomProcessor(AdvancedDocumentProcessor):
    def detect_content_type(self, text, metadata=None):
        # Custom content type detection
        return super().detect_content_type(text, metadata)
    
    def preprocess_text(self, text):
        # Custom preprocessing
        return super().preprocess_text(text)
```

### Custom Retrieval Strategies
```python
# Extend AdvancedRetriever
class CustomRetriever(AdvancedRetriever):
    def _calculate_document_scores(self, documents, query):
        # Custom scoring logic
        return super()._calculate_document_scores(documents, query)
```

## 📝 Best Practices

1. **Chunk Size Selection**:
   - Smaller chunks (500-800 chars): Better for specific questions
   - Larger chunks (1000-1500 chars): Better for comprehensive answers
   - Consider your embedding model's context window

2. **Strategy Selection**:
   - Use `"similarity"` for precise, focused queries
   - Use `"mmr"` for exploratory or broad queries
   - Use `"hybrid"` for balanced performance
   - Use `"ensemble"` for maximum coverage

3. **Threshold Tuning**:
   - Higher thresholds (0.8+): More precise but fewer results
   - Lower thresholds (0.5-0.7): More comprehensive but may include noise
   - Monitor and adjust based on your data quality

4. **Performance Optimization**:
   - Enable re-ranking for better quality
   - Use appropriate max_results for your use case
   - Monitor retrieval statistics for optimization

## 🐛 Troubleshooting

### Common Issues

1. **ImportError for AdvancedDocumentProcessor**:
   - Ensure all LangChain packages are installed
   - Check Python path configuration

2. **Poor Retrieval Quality**:
   - Adjust similarity threshold
   - Try different retrieval strategies
   - Check document processing quality

3. **Slow Performance**:
   - Reduce chunk overlap
   - Limit max_results
   - Disable re-ranking for speed

4. **Memory Issues**:
   - Process documents in batches
   - Reduce chunk size
   - Use compression if available

## 📚 Additional Resources

- [LangChain Text Splitters Documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Vector Database Best Practices](https://docs.trychroma.com/usage-guide)
- [Retrieval Strategy Comparison](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/retrievers)

## 🤝 Contributing

To contribute to the enhanced RAG system:

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure backward compatibility
5. Add performance benchmarks for new strategies

## 📄 License

This enhanced RAG system follows the same license as the main project. 