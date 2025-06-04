#!/usr/bin/env python3
"""
Enhanced RAG System Demonstration

This script demonstrates the improved RAG functionality with:
- Recursive text splitting with LangChain
- Advanced retrieval strategies (MMR, ensemble, re-ranking)
- Robust document processing and chunking
- Multiple retrieval techniques and result fusion
"""

import os
import sys
import json
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from scripts.data_cleaning.data_cleaner import DataCleaner
from models.vector_database import VectorDatabase
from utils.retriever import global_retriever
from models.tools.retriever_tool import retrieve_information, get_retrieval_stats, configure_retriever
from langchain_core.documents import Document

def create_sample_knowledge_base():
    """Create a comprehensive sample knowledge base for testing."""
    sample_data = {
        "https://example.com/ml-intro": {
            "title": "Introduction to Machine Learning",
            "description": "A comprehensive guide to machine learning fundamentals",
            "text_content": """
Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.

Types of Machine Learning:

1. Supervised Learning
Supervised learning algorithms build a mathematical model of training data that contains both inputs and desired outputs. The data is known as training data and consists of a set of training examples. Each training example has one or more inputs and a desired output, also known as a supervisory signal.

2. Unsupervised Learning
Unsupervised learning algorithms take a set of data that contains only inputs, and find structure in the data, like grouping or clustering of data points. The algorithms, therefore, learn from test data that has not been labeled, classified or categorized.

3. Reinforcement Learning
Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward.

Applications of machine learning include computer vision, speech recognition, email filtering, agriculture, and medicine, where it is too costly to develop algorithms to perform the needed tasks.
"""
        },
        "https://example.com/deep-learning": {
            "title": "Deep Learning Neural Networks",
            "description": "Understanding neural networks and deep learning architectures",
            "text_content": """
Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

Neural Network Basics:
A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. Neural networks can adapt to changing input; so the network generates the best possible result without needing to redesign the output criteria.

Layers in Neural Networks:
- Input Layer: The input layer receives data from the outside world
- Hidden Layers: These layers perform computations and transfer information from the input layer to the output layer
- Output Layer: The output layer produces the final result

Deep Learning Architectures:

1. Convolutional Neural Networks (CNNs)
CNNs are particularly effective for image recognition tasks. They use convolutional layers to detect local features in images.

2. Recurrent Neural Networks (RNNs)
RNNs are designed to work with sequence data and have memory capabilities, making them suitable for tasks like language modeling and time series prediction.

3. Long Short-Term Memory (LSTM)
LSTMs are a special kind of RNN capable of learning long-term dependencies, solving the vanishing gradient problem of traditional RNNs.

4. Transformer Networks
Transformers use self-attention mechanisms and have revolutionized natural language processing tasks.

Deep learning has achieved remarkable success in various domains including computer vision, natural language processing, speech recognition, and game playing.
"""
        },
        "https://example.com/nlp-guide": {
            "title": "Natural Language Processing Guide",
            "description": "Comprehensive guide to NLP techniques and applications",
            "text_content": """
Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.

Key NLP Tasks:

1. Text Classification
Text classification involves assigning predefined categories to text documents. Examples include sentiment analysis, spam detection, and topic classification.

2. Named Entity Recognition (NER)
NER involves identifying and classifying named entities (such as person names, organizations, locations) in text.

3. Part-of-Speech Tagging
This involves marking up words in a text as corresponding to particular parts of speech (nouns, verbs, adjectives, etc.).

4. Machine Translation
Machine translation is the task of automatically converting text from one language to another.

5. Question Answering
Question answering systems automatically answer questions posed by humans in natural language.

6. Text Summarization
Text summarization involves creating a shorter version of a text document while preserving its key information.

Modern NLP Techniques:

Word Embeddings: Dense vector representations of words that capture semantic relationships.

Attention Mechanisms: Allow models to focus on relevant parts of the input when making predictions.

Transfer Learning: Pre-training models on large corpora and fine-tuning for specific tasks.

Large Language Models: Models like GPT, BERT, and T5 that have achieved state-of-the-art performance on many NLP tasks.

NLP applications are everywhere in modern technology, from search engines and virtual assistants to language translation services and content recommendation systems.
"""
        },
        "https://example.com/data-science": {
            "title": "Data Science Methodology",
            "description": "Data science process and best practices",
            "text_content": """
Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.

The Data Science Process:

1. Problem Definition
Clearly define the business problem and determine how data science can help solve it. This involves understanding stakeholder requirements and success criteria.

2. Data Collection
Gather relevant data from various sources including databases, APIs, web scraping, sensors, and surveys. Consider data quality, completeness, and accessibility.

3. Data Exploration and Analysis
Perform exploratory data analysis (EDA) to understand data patterns, distributions, correlations, and anomalies. Use visualization tools and statistical techniques.

4. Data Cleaning and Preprocessing
Clean the data by handling missing values, outliers, duplicates, and inconsistencies. Transform data into suitable formats for analysis.

5. Feature Engineering
Create new features or modify existing ones to improve model performance. This may involve domain expertise and creativity.

6. Model Building and Selection
Choose appropriate algorithms based on the problem type (classification, regression, clustering, etc.). Train multiple models and compare their performance.

7. Model Evaluation and Validation
Assess model performance using appropriate metrics and validation techniques such as cross-validation, holdout sets, and statistical tests.

8. Model Deployment and Monitoring
Deploy the model to production environment and continuously monitor its performance. Implement feedback loops for model improvement.

Key Skills for Data Scientists:
- Programming (Python, R, SQL)
- Statistics and Mathematics
- Machine Learning
- Data Visualization
- Domain Expertise
- Communication Skills

Data science is transforming industries by enabling data-driven decision making and creating intelligent systems that can learn from data.
"""
        }
    }
    return sample_data

def demonstrate_enhanced_rag():
    """Main demonstration function."""
    print("🚀 Enhanced RAG System Demonstration")
    print("=" * 60)
    
    # Step 1: Create and clean data
    print("\n📝 Step 1: Creating and Cleaning Sample Data")
    sample_data = create_sample_knowledge_base()
    
    # Initialize data cleaner with advanced processing
    cleaner = DataCleaner(
        sample_data, 
        use_advanced_processing=True,
        chunk_size=800,  # Smaller chunks for better retrieval
        chunk_overlap=100
    )
    
    try:
        cleaned_docs = cleaner.clean_data()
        print(f"✅ Successfully processed {len(cleaned_docs)} document chunks")
        
        # Show statistics
        stats = cleaner.get_statistics()
        print("\n📊 Processing Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error in data cleaning: {e}")
        print("Creating basic documents as fallback...")
        
        # Fallback: create basic documents
        cleaned_docs = []
        for key, item in sample_data.items():
            doc = Document(
                page_content=f"Title: {item['title']}\n\n{item['text_content']}",
                metadata={'source': key, 'title': item['title']}
            )
            cleaned_docs.append(doc)
    
    # Step 2: Initialize vector database
    print("\n🗄️ Step 2: Setting up Vector Database")
    try:
        vector_db = VectorDatabase()
        vector_db.create_vector_database(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            type="chroma",
            persist_directory="./demo_chroma_db",
            collection_name="enhanced_demo"
        )
        
        # Add documents to vector database
        vector_db.add_documents(cleaned_docs)
        print(f"✅ Added {len(cleaned_docs)} documents to vector database")
        
        # Initialize global retriever
        global_retriever.initialize(vector_db)
        print("✅ Global retriever initialized")
        
    except Exception as e:
        print(f"❌ Error setting up vector database: {e}")
        return
    
    # Step 3: Demonstrate different retrieval strategies
    print("\n🔍 Step 3: Testing Different Retrieval Strategies")
    
    test_queries = [
        "What is machine learning?",
        "Explain deep learning neural networks",
        "How does natural language processing work?",
        "What are the steps in data science methodology?"
    ]
    
    strategies = ["similarity", "mmr", "hybrid", "ensemble"]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        print("-" * 50)
        
        for strategy in strategies:
            try:
                print(f"\n📋 Using {strategy} strategy:")
                
                # Configure retriever for this strategy
                config_result = configure_retriever.invoke({
                    "retrieval_strategy": strategy,
                    "similarity_threshold": 0.6,
                    "max_results": 3,
                    "enable_reranking": True
                })
                
                # Retrieve information
                result = retrieve_information.invoke({
                    "query": query,
                    "max_results": 3,
                    "retrieval_strategy": strategy,
                    "use_reranking": True
                })
                
                print(result)
                
                # Get retrieval stats
                stats = get_retrieval_stats.invoke({"query": query})
                print(f"\n📈 {stats}")
                
            except Exception as e:
                print(f"❌ Error with {strategy} strategy: {e}")
        
        print("\n" + "="*60)
    
    # Step 4: Performance comparison
    print("\n📊 Step 4: Performance Analysis")
    
    comparison_query = "What are the different types of machine learning?"
    print(f"Comparing strategies for: '{comparison_query}'")
    
    for strategy in strategies:
        try:
            configure_retriever.invoke({
                "retrieval_strategy": strategy,
                "max_results": 2
            })
            result = retrieve_information.invoke({
                "query": comparison_query,
                "max_results": 2,
                "retrieval_strategy": strategy
            })
            
            print(f"\n{strategy.upper()} Strategy Results:")
            print("-" * 30)
            # Show first 200 characters
            preview = result[:200] + "..." if len(result) > 200 else result
            print(preview)
            
        except Exception as e:
            print(f"Error with {strategy}: {e}")
    
    # Step 5: Advanced features demonstration
    print("\n🔧 Step 5: Advanced Features")
    
    # Test with different parameters
    print("\nTesting with different similarity thresholds:")
    thresholds = [0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        configure_retriever.invoke({
            "retrieval_strategy": "threshold",
            "similarity_threshold": threshold,
            "max_results": 3
        })
        
        result = retrieve_information.invoke({
            "query": "What is supervised learning?",
            "max_results": 3,
            "retrieval_strategy": "threshold"
        })
        
        print(f"\nThreshold {threshold}:")
        result_preview = result[:150] + "..." if len(result) > 150 else result
        print(result_preview)
    
    print("\n🎉 Enhanced RAG Demonstration Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Recursive text splitting with LangChain")
    print("✅ Multiple retrieval strategies (similarity, MMR, hybrid, ensemble)")
    print("✅ Advanced document processing and chunking")
    print("✅ Re-ranking and result fusion")
    print("✅ Configurable similarity thresholds")
    print("✅ Rich metadata and source tracking")
    print("✅ Intelligent content truncation")
    print("✅ Performance monitoring and statistics")

if __name__ == "__main__":
    try:
        demonstrate_enhanced_rag()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc() 