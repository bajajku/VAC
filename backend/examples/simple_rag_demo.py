#!/usr/bin/env python3
"""
Simple RAG System Demo

This script demonstrates how to use the RAG system with minimal setup.
Make sure to set your API key before running.
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from core.app import RAGApplication

def main():
    """Run a simple RAG demonstration."""
    
    print("🚀 Simple RAG System Demo")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No API key found!")
        print("Please set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        # Initialize the RAG application
        print("🔧 Initializing RAG system...")
        app = RAGApplication()
        
        # Configure based on available API key
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            model = "gpt-3.5-turbo"
            key = os.getenv("OPENAI_API_KEY")
        else:
            provider = "openrouter"
            model = "microsoft/wizardlm-2-8x22b"  # Free model on OpenRouter
            key = os.getenv("OPENROUTER_API_KEY")
        
        app.initialize(
            llm_provider=provider,
            llm_model=model,
            api_key=key,
            temperature=0.7
        )
        
        print(f"✅ Initialized with {provider}/{model}")
        
        # Add sample knowledge base
        print("📚 Adding sample documents to knowledge base...")
        
        sample_docs = [
            {
                "text": "LangGraph is a library for building stateful, multi-actor applications with LLMs, used by thousands of developers in production. It extends LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.",
                "metadata": {"source": "langgraph_docs", "topic": "langgraph", "type": "definition"}
            },
            {
                "text": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It works by first retrieving relevant documents from a knowledge base, then using that information to generate more accurate and contextual responses.",
                "metadata": {"source": "rag_guide", "topic": "rag", "type": "definition"}
            },
            {
                "text": "Vector databases store high-dimensional vectors that represent semantic meaning of text. They enable efficient similarity search, which is crucial for RAG systems to find relevant documents quickly. Popular vector databases include Chroma, Pinecone, and Weaviate.",
                "metadata": {"source": "vector_db_guide", "topic": "vector_databases", "type": "definition"}
            },
            {
                "text": "LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, data augmented generation, agents, and memory. LangChain makes it easy to build complex LLM applications.",
                "metadata": {"source": "langchain_docs", "topic": "langchain", "type": "definition"}
            },
            {
                "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions.",
                "metadata": {"source": "ml_guide", "topic": "machine_learning", "type": "definition"}
            }
        ]
        
        texts = [doc["text"] for doc in sample_docs]
        metadatas = [doc["metadata"] for doc in sample_docs]
        
        app.add_documents_from_text(texts, metadatas)
        print(f"✅ Added {len(sample_docs)} documents to knowledge base")
        
        # Show system stats
        stats = app.get_stats()
        print(f"📊 System Status: {stats}")
        
        # Demo queries
        demo_questions = [
            "What is LangGraph and how is it used?",
            "How does RAG work?",
            "What are vector databases?",
            "Explain machine learning in simple terms",
            "What's the difference between LangChain and LangGraph?"
        ]
        
        print("\n🤖 Demo Questions and Answers:")
        print("=" * 50)
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n❓ Question {i}: {question}")
            print("-" * 40)
            
            try:
                # Get the answer
                answer = app.query(question)
                print(f"🤖 Answer: {answer}")
                
                # Show retrieved documents
                docs = app.search_documents(question, k=2)
                print(f"\n📚 Sources ({len(docs)} documents):")
                for j, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source', 'unknown')
                    topic = doc.metadata.get('topic', 'general')
                    preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"  {j}. [{source}] {preview}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("\n" + "="*50)
        
        # Interactive mode
        print("\n🎯 Interactive Mode")
        print("Ask your own questions! (type 'quit' to exit)")
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', '']:
                    break
                
                print("🔍 Searching knowledge base...")
                answer = app.query(question)
                print(f"🤖 Answer: {answer}")
                
                # Show sources
                docs = app.search_documents(question, k=2)
                if docs:
                    print(f"\n📚 Sources:")
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get('source', 'unknown')
                        print(f"  {i}. [{source}] {doc.page_content[:80]}...")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Thanks for trying the RAG system!")
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your API key is correct")
        print("2. Ensure you have internet connection")
        print("3. Try running: pip install -e .")

if __name__ == "__main__":
    main() 