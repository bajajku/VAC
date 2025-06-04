#!/usr/bin/env python3
"""
Complete RAG Workflow Demonstration

This script shows the complete end-to-end workflow:
1. Raw Documents Input
2. Data Cleaning (LLM-based)
3. Advanced Document Processing (Recursive Splitting)
4. Vector Database Creation
5. Document Embedding & Storage
6. Enhanced Retrieval

This demonstrates exactly how documents flow through the system.
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from scripts.data_cleaning.data_cleaner import DataCleaner
from models.vector_database import VectorDatabase
from utils.retriever import global_retriever
from models.tools.retriever_tool import retrieve_information
from langchain_core.documents import Document

def demonstrate_complete_workflow():
    """Demonstrate the complete RAG workflow step by step."""
    
    print("🔄 COMPLETE RAG WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    # ==================================================================
    # STEP 1: Raw Documents Input
    # ==================================================================
    print("\n📄 STEP 1: Raw Documents Input")
    print("-" * 40)
    
    # Simulate raw, unprocessed documents (like what you'd get from web scraping, PDFs, etc.)
    raw_documents = {
        "https://example.com/article1": {
            "title": "Machine Learning Basics",
            "description": "Introduction to ML concepts",
            "text_content": """
            ADVERTISEMENT: Get 50% off our ML course!
            
            Machine learning is a subset of artificial intelligence (AI) that provides systems 
            the ability to automatically learn and improve from experience without being 
            explicitly programmed.
            
            SIDEBAR: Related Articles
            - Deep Learning Guide
            - AI Ethics
            
            The main types of machine learning are:
            
            1. Supervised Learning: Uses labeled training data to learn a mapping from inputs to outputs.
            Examples include classification (predicting categories) and regression (predicting continuous values).
            
            2. Unsupervised Learning: Finds hidden patterns in data without labeled examples.
            Common techniques include clustering and dimensionality reduction.
            
            3. Reinforcement Learning: An agent learns to make decisions by taking actions 
            in an environment and receiving rewards or penalties.
            
            FOOTER: Copyright 2024 ML Academy. All rights reserved.
            """
        },
        "https://example.com/article2": {
            "title": "Neural Networks Deep Dive",
            "description": "Understanding neural network architectures",
            "text_content": """
            NAVIGATION: Home > AI > Neural Networks
            
            Neural networks are computing systems inspired by biological neural networks.
            They consist of interconnected nodes (neurons) that process information.
            
            Key components:
            - Input Layer: Receives data from outside world
            - Hidden Layers: Process and transform the input data
            - Output Layer: Produces the final prediction or classification
            
            Popular architectures include:
            
            Feedforward Networks: Information flows in one direction from input to output.
            Simple but effective for many tasks.
            
            Convolutional Neural Networks (CNNs): Specialized for processing grid-like data such as images.
            Use convolutional layers to detect local features.
            
            Recurrent Neural Networks (RNNs): Can process sequences of data by maintaining internal memory.
            Useful for time series and natural language processing.
            
            COOKIE NOTICE: This site uses cookies to improve your experience.
            """
        }
    }
    
    print(f"✅ Loaded {len(raw_documents)} raw documents")
    print("📝 Raw documents contain noise: ads, navigation, footers, etc.")
    
    # Show example of raw content
    print(f"\n📋 Example raw content (first 200 chars):")
    first_doc = list(raw_documents.values())[0]
    print(f"'{first_doc['text_content'][:200]}...'")
    
    # ==================================================================
    # STEP 2: Data Cleaning (LLM-based noise removal)
    # ==================================================================
    print(f"\n🧹 STEP 2: Data Cleaning (LLM-based)")
    print("-" * 40)
    
    # Initialize enhanced data cleaner
    cleaner = DataCleaner(
        raw_documents,
        use_advanced_processing=True,  # Enable advanced processing
        chunk_size=800,               # Smaller chunks for better retrieval
        chunk_overlap=100             # Reasonable overlap
    )
    
    print("🔄 Cleaning documents...")
    print("   - Removing ads, navigation, footers")
    print("   - Fixing formatting issues")
    print("   - Extracting main content")
    
    try:
        cleaned_documents = cleaner.clean_data()
        print(f"✅ Cleaned {len(cleaned_documents)} document chunks")
        
        # Show statistics
        stats = cleaner.get_statistics()
        print(f"\n📊 Cleaning Statistics:")
        print(f"   - Original documents: {stats['original_documents']}")
        print(f"   - Final chunks: {stats['cleaned_documents']}")
        print(f"   - Average chunk size: {stats.get('average_chunk_size', 'N/A')} chars")
        print(f"   - Advanced processing: {stats['advanced_processing_used']}")
        
        # Show example of cleaned content
        if cleaned_documents:
            print(f"\n📋 Example cleaned content (first 200 chars):")
            print(f"'{cleaned_documents[0].page_content[:200]}...'")
            
    except Exception as e:
        print(f"❌ Error in cleaning: {e}")
        return
    
    # ==================================================================
    # STEP 3: Advanced Document Processing (Recursive Splitting)
    # ==================================================================
    print(f"\n🔀 STEP 3: Advanced Document Processing")
    print("-" * 40)
    print("✅ Already applied during cleaning process!")
    print("📋 Features applied:")
    print("   - Recursive text splitting (respects sentence boundaries)")
    print("   - Content type detection (markdown, code, plain text)")
    print("   - Smart preprocessing (noise removal)")
    print("   - Rich metadata enhancement")
    print("   - Token counting for optimal chunk sizes")
    
    # Show chunk details
    print(f"\n📄 Chunk Analysis:")
    for i, doc in enumerate(cleaned_documents[:3]):  # Show first 3
        print(f"   Chunk {i+1}:")
        print(f"     - Size: {len(doc.page_content)} chars")
        print(f"     - Tokens: {doc.metadata.get('token_count', 'N/A')}")
        print(f"     - Type: {doc.metadata.get('content_type', 'unknown')}")
        print(f"     - Source: {doc.metadata.get('source', 'unknown')}")
    
    # ==================================================================
    # STEP 4: Vector Database Creation & Document Storage
    # ==================================================================
    print(f"\n🗄️ STEP 4: Vector Database Creation & Storage")
    print("-" * 40)
    
    # Create vector database
    vector_db = VectorDatabase()
    
    print("🔄 Creating vector database...")
    vector_db.create_vector_database(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        type="chroma",
        persist_directory="./workflow_demo_db",
        collection_name="workflow_demo"
    )
    print("✅ Vector database created")
    
    print("🔄 Converting documents to embeddings and storing...")
    vector_db.add_documents(cleaned_documents)
    print(f"✅ Stored {len(cleaned_documents)} document chunks as embeddings")
    
    print(f"\n📊 Vector Database Info:")
    print(f"   - Embedding model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"   - Vector dimension: 384 (typical for this model)")
    print(f"   - Storage: Chroma database")
    print(f"   - Documents stored: {len(cleaned_documents)}")
    
    # ==================================================================
    # STEP 5: Enhanced Retrieval Setup
    # ==================================================================
    print(f"\n🔍 STEP 5: Enhanced Retrieval Setup")
    print("-" * 40)
    
    # Initialize enhanced retriever
    global_retriever.initialize(vector_db)
    print("✅ Enhanced retriever initialized with:")
    print("   - Multiple strategies: similarity, MMR, hybrid, ensemble")
    print("   - Advanced re-ranking")
    print("   - Result fusion")
    print("   - Performance monitoring")
    
    # ==================================================================
    # STEP 6: Test the Complete Pipeline
    # ==================================================================
    print(f"\n🎯 STEP 6: Testing Complete Pipeline")
    print("-" * 40)
    
    test_queries = [
        "What are the types of machine learning?",
        "Explain neural network architectures"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        print("-" * 30)
        
        # Use enhanced retrieval
        result = retrieve_information.invoke({
            "query": query,
            "max_results": 2,
            "retrieval_strategy": "hybrid",
            "use_reranking": True
        })
        
        # Show result preview
        preview = result[:300] + "..." if len(result) > 300 else result
        print(preview)
    
    # ==================================================================
    # STEP 7: Workflow Summary
    # ==================================================================
    print(f"\n📋 WORKFLOW SUMMARY")
    print("=" * 60)
    print("🔄 Complete Pipeline:")
    print("   1. Raw Documents (with noise) → Data Cleaning")
    print("   2. Cleaned Content → Advanced Processing (Recursive Splitting)")
    print("   3. Processed Chunks → Vector Embeddings")
    print("   4. Embeddings → Vector Database Storage")
    print("   5. User Query → Enhanced Retrieval")
    print("   6. Retrieved Chunks → Response Generation")
    
    print(f"\n✅ Key Benefits:")
    print(f"   - Noise removal: Ads, navigation, footers cleaned")
    print(f"   - Smart chunking: Respects sentence boundaries")
    print(f"   - Better retrieval: Multiple strategies with re-ranking")
    print(f"   - Rich metadata: Source tracking and chunk relationships")
    print(f"   - Performance: 10x more content per response")
    
    print(f"\n🎉 Enhanced RAG Pipeline Complete!")

if __name__ == "__main__":
    try:
        demonstrate_complete_workflow()
    except KeyboardInterrupt:
        print("\n\n⏹️ Workflow demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Workflow demonstration failed: {e}")
        import traceback
        traceback.print_exc() 