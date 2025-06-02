#!/usr/bin/env python3
"""
Example usage of the DataCleaner class.

This script demonstrates how to use the DataCleaner to clean unstructured data
and convert it into clean Document objects suitable for a RAG system.
"""

import json
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from scripts.data_cleaning.data_cleaner import DataCleaner

def create_sample_data():
    """Create sample unclean data for demonstration."""
    sample_data = {
        "https://example.com/article1": {
            "title": "Introduction to Machine Learning",
            "description": "A comprehensive guide to ML basics",
            "text_content": """
            ADVERTISEMENT: Buy our premium course now! 50% off!
            
            Machine learning is a subset of artificial intelligence (AI) that provides systems 
            the ability to automatically learn and improve from experience without being 
            explicitly programmed. Machine learning focuses on the development of computer 
            programs that can access data and use it to learn for themselves.
            
            NAVIGATION: Home | About | Contact | Privacy Policy
            
            The process of learning begins with observations or data, such as examples, 
            direct experience, or instruction, in order to look for patterns in data and 
            make better decisions in the future based on the examples that we provide.
            
            FOOTER: Copyright 2024 Example Corp. All rights reserved.
            """
        },
        "https://example.com/article2": {
            "title": "Deep Learning Fundamentals",
            "description": "Understanding neural networks",
            "text_content": """
            SIDEBAR: Related Articles | Popular Posts | Subscribe to Newsletter
            
            Deep learning is part of a broader family of machine learning methods based on 
            artificial neural networks with representation learning. Learning can be 
            supervised, semi-supervised or unsupervised.
            
            ADVERTISEMENT: Click here for amazing deals!
            
            Deep learning architectures such as deep neural networks, deep belief networks, 
            deep reinforcement learning, recurrent neural networks and convolutional neural 
            networks have been applied to fields including computer vision, speech recognition, 
            natural language processing, machine translation, bioinformatics and drug design.
            
            COOKIE NOTICE: This website uses cookies to improve your experience.
            """
        },
        "https://example.com/spam": {
            "title": "Buy Now! Amazing Deals!",
            "description": "Limited time offer",
            "text_content": """
            CLICK HERE NOW! AMAZING DEALS! BUY BUY BUY!
            
            Limited time offer! Don't miss out! Call 1-800-SPAM-NOW!
            
            This is clearly spam content with no educational value.
            """
        }
    }
    return sample_data

def main():
    """Main function to demonstrate DataCleaner usage."""
    print("🧹 DataCleaner Example Usage")
    print("=" * 50)
    
    # Create sample data
    print("📝 Creating sample unclean data...")
    sample_data = create_sample_data()
    print(f"✅ Created {len(sample_data)} sample documents")
    
    # Initialize DataCleaner
    print("\n🔧 Initializing DataCleaner...")
    try:
        cleaner = DataCleaner(sample_data)
        print("✅ DataCleaner initialized successfully")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set the TOGETHER_API_KEY environment variable.")
        return
    
    # Clean the data
    print("\n🧹 Cleaning documents...")
    try:
        cleaned_documents = cleaner.clean_data()
        print(f"✅ Successfully cleaned {len(cleaned_documents)} documents")
    except Exception as e:
        print(f"❌ Error during cleaning: {e}")
        return
    
    # Display results
    print("\n📊 Cleaning Results:")
    stats = cleaner.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n📄 Cleaned Documents:")
    for i, doc in enumerate(cleaned_documents, 1):
        print(f"\n--- Document {i} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Original Title: {doc.metadata.get('original_title', 'N/A')}")
        print(f"Content Preview: {doc.page_content[:200]}...")
        print(f"Content Length: {len(doc.page_content)} characters")
    
    # Save cleaned documents
    output_path = "cleaned_documents.json"
    print(f"\n💾 Saving cleaned documents to {output_path}...")
    try:
        cleaner.save_cleaned_documents(output_path)
        print("✅ Documents saved successfully")
    except Exception as e:
        print(f"❌ Error saving documents: {e}")
    
    print("\n🎉 DataCleaner example completed!")
    print("\nNext steps:")
    print("1. Review the cleaned documents in cleaned_documents.json")
    print("2. Use these Document objects in your RAG system")
    print("3. Add them to a vector database for retrieval")

if __name__ == "__main__":
    main() 