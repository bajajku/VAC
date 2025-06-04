#!/usr/bin/env python3
"""
Test script for Enhanced RAG API
Tests both basic and enhanced endpoints to verify functionality
"""

import requests
import json
import time
from typing import Dict, Any

class EnhancedAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_connection(self) -> bool:
        """Test if API is running"""
        try:
            response = self.session.get(f"{self.base_url}/")
            print(f"✅ API Connection: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ API Connection failed: {e}")
            return False
    
    def test_basic_query(self, question: str = "What is machine learning?") -> Dict[str, Any]:
        """Test basic query endpoint"""
        print(f"\n🔍 Testing Basic Query: '{question}'")
        try:
            response = self.session.post(
                f"{self.base_url}/query",
                json={"question": question, "k": 3}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Basic Query Success")
                print(f"📝 Answer Length: {len(result.get('answer', ''))}")
                print(f"📚 Sources Found: {len(result.get('sources', []))}")
                return result
            else:
                print(f"❌ Basic Query Failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Basic Query Error: {e}")
            return {}
    
    def test_enhanced_query(self, question: str = "What is machine learning?") -> Dict[str, Any]:
        """Test enhanced query endpoint"""
        print(f"\n🚀 Testing Enhanced Query: '{question}'")
        try:
            response = self.session.post(
                f"{self.base_url}/query-enhanced",
                json={
                    "question": question,
                    "k": 3,
                    "retrieval_strategy": "hybrid",
                    "enable_reranking": True,
                    "similarity_threshold": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Enhanced Query Success")
                print(f"📝 Answer Length: {len(result.get('answer', ''))}")
                print(f"📚 Sources Found: {len(result.get('sources', []))}")
                
                # Show enhanced metadata
                sources = result.get('sources', [])
                if sources:
                    first_source = sources[0]
                    metadata = first_source.get('metadata', {})
                    print(f"🔍 Enhanced Features:")
                    print(f"   - Retrieval Strategy: {metadata.get('retrieval_strategy', 'N/A')}")
                    print(f"   - Content Type: {metadata.get('content_type', 'N/A')}")
                    print(f"   - Chunk Info: {metadata.get('chunk_index', 'N/A')}/{metadata.get('total_chunks', 'N/A')}")
                
                return result
            else:
                print(f"❌ Enhanced Query Failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Enhanced Query Error: {e}")
            return {}
    
    def test_document_upload(self) -> bool:
        """Test enhanced document upload"""
        print(f"\n📄 Testing Enhanced Document Upload")
        try:
            test_docs = [
                "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans.",
                "Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed."
            ]
            
            test_metadata = [
                {"title": "AI Introduction", "description": "Basic AI concepts"},
                {"title": "ML Introduction", "description": "Basic ML concepts"}
            ]
            
            response = self.session.post(
                f"{self.base_url}/documents",
                json={
                    "texts": test_docs,
                    "metadatas": test_metadata,
                    "use_enhanced_processing": True
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Document Upload Success")
                print(f"📊 Processing Type: {result.get('processing_type', 'N/A')}")
                print(f"📈 Expansion Ratio: {result.get('expansion_ratio', 'N/A')}")
                print(f"📄 Final Count: {result.get('final_count', 'N/A')}")
                return True
            else:
                print(f"❌ Document Upload Failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Document Upload Error: {e}")
            return False
    
    def test_retrieval_stats(self) -> Dict[str, Any]:
        """Test retrieval statistics endpoint"""
        print(f"\n📊 Testing Retrieval Statistics")
        try:
            response = self.session.get(f"{self.base_url}/retrieval-stats")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Retrieval Stats Success")
                print(f"📈 Stats Available: {len(result)}")
                return result
            else:
                print(f"❌ Retrieval Stats Failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Retrieval Stats Error: {e}")
            return {}
    
    def test_configure_retriever(self) -> bool:
        """Test retriever configuration"""
        print(f"\n⚙️ Testing Retriever Configuration")
        try:
            response = self.session.post(
                f"{self.base_url}/configure-retriever",
                json={
                    "max_results": 5,
                    "retrieval_strategy": "ensemble",
                    "enable_reranking": True,
                    "similarity_threshold": 0.8
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Retriever Configuration Success")
                return True
            else:
                print(f"❌ Retriever Configuration Failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Retriever Configuration Error: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🧪 ENHANCED RAG API COMPREHENSIVE TEST")
        print("=" * 50)
        
        # Test connection
        if not self.test_connection():
            print("❌ Cannot connect to API. Make sure the server is running.")
            return
        
        # Wait a moment for initialization
        print("⏳ Waiting for API initialization...")
        time.sleep(2)
        
        # Test document upload first
        self.test_document_upload()
        
        # Wait for documents to be processed
        time.sleep(1)
        
        # Test queries
        self.test_basic_query()
        self.test_enhanced_query()
        
        # Test configuration and stats
        self.test_configure_retriever()
        self.test_retrieval_stats()
        
        print("\n🎉 COMPREHENSIVE TESTING COMPLETE!")
        print("=" * 50)

def main():
    """Main test function"""
    tester = EnhancedAPITester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main() 