import os
import sys
from pathlib import Path
from utils.prompt import Prompt
from scripts.data_cleaning.data_cleaner import DataCleaner
import json

from dotenv import load_dotenv
load_dotenv('.env')
# Add the backend directory to the Python path
TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from core.app import RAGApplication

PROMPT_TEMPLATE = """
You are a helpful assistant that can answer questions about the given documents.

{context}

Question: {input}
"""
def demo_rag_agent_system():
    """Demonstrate the RAG system with sample data."""
    
    # Initialize the RAG application
    app = RAGApplication()
    
    # Example configuration - you'll need to provide your API key
    config = {
        "app_type": "rag_agent",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_provider": "chatopenai",  # or "openrouter" for open source models
        "llm_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "persist_directory": "./chroma_db",
        "collection_name": "demo_collection",
        "api_key": TOGETHER_API_KEY  # Uncomment and add your API key
    }
    
    try:
        # Initialize the system
        app.initialize(**config)
        
        # Add some sample documents if no existing data
        sample_texts = [
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.",
            "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It first retrieves relevant documents from a knowledge base, then uses that information to generate more accurate and contextual responses.",
            "Vector databases store high-dimensional vectors that represent semantic meaning of text. They enable efficient similarity search, which is crucial for RAG systems to find relevant documents quickly.",
            "LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, data augmented generation, agents, and memory."
        ]
        
        sample_metadata = [
            {"source": "langgraph_docs", "topic": "langgraph"},
            {"source": "rag_guide", "topic": "rag"},
            {"source": "vector_db_guide", "topic": "vector_databases"},
            {"source": "langchain_docs", "topic": "langchain"}
        ]
        
        # Add sample documents
        app.add_documents_from_text(sample_texts, sample_metadata)
        
        # Check if there's existing JSON data to load
        json_files = list(Path("scripts/data_collection/crawl_results").glob("*.json"))
        if json_files:
            print(f"Found {len(json_files)} JSON files. Loading the most recent one...")
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            app.load_data_from_json(str(latest_json))
        
        # Display system stats
        stats = app.get_stats()
        print(f"\n📊 System Stats: {stats}")
        
        # Interactive query loop
        print("\n🤖 RAG System Ready! Ask me anything (type 'quit' to exit)")
        print("Example questions:")
        print("- What is LangGraph?")
        print("- How does RAG work?")
        print("- Tell me about vector databases")
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            try:
                print("\n🔍 Searching knowledge base...")
                response = app.query(question)
                print(f"\n🤖 Response: {response}")
                
                # Show retrieved documents for transparency
                docs = app.search_documents(question, k=2)
                print(f"\n📚 Retrieved {len(docs)} relevant documents:")
                for i, doc in enumerate(docs, 1):
                    print(f"  {i}. {doc.page_content[:100]}...")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Make sure you have set up your API key correctly.")
    
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have set your API key")
        print("2. Check your internet connection")
        print("3. Verify all dependencies are installed")

def demo_rag_chain_system():
    """Demonstrate the RAG system with sample data."""
    app = RAGApplication()
    
    # Example configuration - you'll need to provide your API key
    config = {
        "app_type": "rag_chain",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_provider": "chatopenai",  # or "openrouter" for open source models
        "llm_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "persist_directory": "./chroma_db",
        "collection_name": "demo_collection",
        "api_key": TOGETHER_API_KEY  # Uncomment and add your API key
        ,"prompt": Prompt(template=PROMPT_TEMPLATE)

    }
    
    try:
        # Initialize the system
        app.initialize(**config)
        
        # Add some sample documents if no existing data
        sample_texts = [
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.",
            "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It first retrieves relevant documents from a knowledge base, then uses that information to generate more accurate and contextual responses.",
            "Vector databases store high-dimensional vectors that represent semantic meaning of text. They enable efficient similarity search, which is crucial for RAG systems to find relevant documents quickly.",
            "LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, data augmented generation, agents, and memory."
        ]
        
        sample_metadata = [
            {"source": "langgraph_docs", "topic": "langgraph"},
            {"source": "rag_guide", "topic": "rag"},
            {"source": "vector_db_guide", "topic": "vector_databases"},
            {"source": "langchain_docs", "topic": "langchain"}
        ]
        
        # Add sample documents
        app.add_documents_from_text(sample_texts, sample_metadata)
        
        # Check if there's existing JSON data to load
        json_files = list(Path("scripts/data_collection/crawl_results").glob("*.json"))
        if json_files:
            print(f"Found {len(json_files)} JSON files. Loading the most recent one...")
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            app.load_data_from_json(str(latest_json))
        
        # Display system stats
        stats = app.get_stats()
        print(f"\n📊 System Stats: {stats}")
        
        # Interactive query loop
        print("\n🤖 RAG Chain System Ready! Ask me anything (type 'quit' to exit)")
        print("Example questions:")
        print("- What is LangGraph?")
        print("- How does RAG work?")
        print("- Tell me about vector databases")
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            try:
                print("\n🔍 Searching knowledge base...")
                response = app.query(question)
                print(f"\n🤖 Response: {response}")
                
                # Show retrieved documents for transparency
                docs = app.search_documents(question, k=2)
                print(f"\n📚 Retrieved {len(docs)} relevant documents:")
                for i, doc in enumerate(docs, 1):
                    print(f"  {i}. {doc.page_content[:100]}...")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Make sure you have set up your API key correctly.")
    
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have set your API key")
        print("2. Check your internet connection")
        print("3. Verify all dependencies are installed")
    
def test_data_cleaning():
    """Test the data cleaning process."""
    data_cleaner = DataCleaner(unclean_data=json.load(open('scripts/data_collection/crawl_results/crawl_results_20250526_133954.json')))
    data_cleaner.clean_data()
    data_cleaner.save_cleaned_documents('data/cleaned_data.json')
    print(data_cleaner.get_statistics())

def test_components():
    """Test individual components of the RAG system."""
    print("🧪 Testing RAG System Components...")
    
    try:
        from models.vector_database import VectorDatabase
        from utils.retriever import global_retriever
        from langchain_core.documents import Document
        
        # Test vector database
        print("Testing vector database...")
        vdb = VectorDatabase()
        vdb.create_vector_database("sentence-transformers/all-MiniLM-L6-v2")
        
        # Test adding documents
        test_docs = [
            Document(page_content="This is a test document about AI.", metadata={"source": "test"}),
            Document(page_content="This is another test document about machine learning.", metadata={"source": "test"})
        ]
        vdb.add_documents(test_docs)
        
        # Test retriever
        print("Testing retriever...")
        global_retriever.initialize(vdb)
        results = global_retriever.get_relevant_documents("AI", k=1)
        print(results)
        print(f"Retrieved {len(results)} documents")
        
        print("✅ All components working correctly!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")

def main():
    """Main entry point."""
    print("🚀 Welcome to the RAG System!")
    print("Choose an option:")
    print("1. Run full demo")
    print("2. Run RAG Chain demo")
    print("3. Test components only")
    print("4. Test data cleaning")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        demo_rag_agent_system()
    elif choice == "2":
        demo_rag_chain_system()
    elif choice == "3":
        test_components()
    elif choice == "4":
        test_data_cleaning()
    elif choice == "5":
        print("Goodbye!")
    else:
        print("Invalid choice. Please run again.")

if __name__ == "__main__":
    main()
