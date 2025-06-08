import time
import os
from contextlib import contextmanager
from models.vector_database import VectorDatabase
from models.advanced_retriever import AdvancedRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

@contextmanager
def timing(description: str):
    """Context manager to time operations."""
    start = time.time()
    print(f"🕐 Starting: {description}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"⏱️  {description} took {elapsed:.2f} seconds")

def debug_vector_database_creation():
    """Debug vector database creation performance."""
    print("🔍 Debugging Vector Database Creation...")
    
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Test 1: HuggingFace Embeddings initialization
    with timing("HuggingFace Embeddings initialization"):
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Test 2: Chroma initialization
    with timing("Chroma vector database creation"):
        vector_db = Chroma(
            embedding_function=embeddings,
            persist_directory="./test_chroma_db",
            collection_name="test_collection"
        )
    
    # Test 3: VectorDatabase class initialization
    with timing("VectorDatabase class initialization"):
        vdb = VectorDatabase()
        vdb.create_vector_database(
            embedding_model=embedding_model,
            type="chroma",
            persist_directory="./test_chroma_db2",
            collection_name="test_collection2"
        )
    
    return vdb

def debug_advanced_retriever_initialization(vector_db):
    """Debug advanced retriever initialization performance."""
    print("\n🔍 Debugging Advanced Retriever Initialization...")
    
    # Test individual components
    with timing("Basic similarity retriever creation"):
        similarity_retriever = vector_db.vector_database.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}
        )
    
    with timing("MMR retriever creation"):
        mmr_retriever = vector_db.vector_database.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "lambda_mult": 0.7}
        )
    
    with timing("Threshold retriever creation"):
        threshold_retriever = vector_db.vector_database.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7, "k": 10}
        )
    
    # Test the expensive _get_all_documents operation
    with timing("_get_all_documents (similarity search for 1000 docs)"):
        try:
            docs = vector_db.vector_database.similarity_search("", k=1000)
            print(f"📄 Retrieved {len(docs)} documents for BM25 initialization")
        except Exception as e:
            print(f"❌ Error getting all documents: {e}")
            docs = []
    
    # Test BM25 creation with documents
    if docs:
        with timing("BM25Retriever creation from documents"):
            try:
                from langchain_community.retrievers import BM25Retriever
                bm25_retriever = BM25Retriever.from_documents(docs)
                print("✅ BM25 retriever created successfully")
            except Exception as e:
                print(f"❌ BM25 creation failed: {e}")
    
    # Test full AdvancedRetriever initialization (without LLM)
    with timing("AdvancedRetriever initialization (no LLM)"):
        retriever = AdvancedRetriever(
            vector_store=vector_db.vector_database,
            retrieval_strategy="similarity",
            max_results=10,
            enable_reranking=True,
            enable_compression=False,  # Disable compression to avoid LLM requirement
            llm=None
        )
    
    return retriever

def debug_with_minimal_setup():
    """Debug with minimal document setup to isolate the issue."""
    print("\n🔍 Testing with Minimal Document Setup...")
    
    from langchain_core.documents import Document
    
    # Create minimal test documents
    test_docs = [
        Document(page_content="This is test document 1", metadata={"id": 1}),
        Document(page_content="This is test document 2", metadata={"id": 2}),
        Document(page_content="This is test document 3", metadata={"id": 3}),
    ]
    
    with timing("Vector database with minimal docs"):
        vdb = VectorDatabase()
        vdb.create_vector_database(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            type="chroma",
            persist_directory="./minimal_test_db",
            collection_name="minimal_test"
        )
        vdb.add_documents(test_docs)
    
    with timing("AdvancedRetriever with minimal docs"):
        retriever = AdvancedRetriever(
            vector_store=vdb.vector_database,
            retrieval_strategy="similarity",
            max_results=5,
            enable_reranking=False,
            enable_compression=False,
            llm=None
        )
    
    return vdb, retriever

def test_different_embedding_models():
    """Test performance of different embedding models."""
    print("\n🔍 Testing Different Embedding Models...")
    
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2"
    ]
    
    for model in models:
        try:
            with timing(f"Loading embedding model: {model}"):
                embeddings = HuggingFaceEmbeddings(model_name=model)
                # Test a small embedding to ensure it works
                test_embedding = embeddings.embed_query("test")
                print(f"✅ Model {model} loaded successfully (embedding dim: {len(test_embedding)})")
        except Exception as e:
            print(f"❌ Model {model} failed: {e}")

def main():
    """Main debugging function."""
    print("🚀 Starting RAG Performance Debugging...")
    print("=" * 60)
    
    # Set environment variable to reduce transformers logging
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    try:
        # Test 1: Vector database creation
        with timing("Complete vector database setup"):
            vector_db = debug_vector_database_creation()
        
        # Test 2: Advanced retriever initialization
        with timing("Complete advanced retriever setup"):
            retriever = debug_advanced_retriever_initialization(vector_db)
        
        # Test 3: Minimal setup
        with timing("Complete minimal setup"):
            minimal_vdb, minimal_retriever = debug_with_minimal_setup()
        
        # Test 4: Different embedding models
        test_different_embedding_models()
        
        print("\n📊 Performance Summary:")
        print("=" * 60)
        print("Check the individual timings above to identify bottlenecks.")
        print("Common issues:")
        print("1. HuggingFace model download on first run")
        print("2. Large document retrieval for BM25 initialization")
        print("3. Multiple retriever initialization overhead")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 