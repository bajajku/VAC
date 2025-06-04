#!/usr/bin/env python3
"""
Offline Data Preprocessing Script for Enhanced RAG

This script processes and cleans raw data offline, saving the results
for fast loading during API startup. This avoids expensive LLM calls
every time the API starts.

Usage:
    python preprocess_data.py --input <input_file> --output <output_dir>
    python preprocess_data.py --auto  # Auto-process latest crawl results
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import sys

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.data_cleaning.data_cleaner import DataCleaner
from langchain_core.documents import Document

class DataPreprocessor:
    """Offline data preprocessing for RAG system"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_docs = []
        
    def process_json_file(self, input_file: Path, use_enhanced: bool = True) -> List[Document]:
        """
        Process a single JSON file and return cleaned documents
        
        Args:
            input_file: Path to input JSON file
            use_enhanced: Whether to use enhanced processing
            
        Returns:
            List of processed documents
        """
        print(f"📄 Processing file: {input_file}")
        
        try:
            with open(input_file, 'r') as f:
                raw_data = f.read()
            
            if use_enhanced:
                print("🧹 Using enhanced LLM-based cleaning...")
                cleaner = DataCleaner(
                    raw_data,
                    use_advanced_processing=True,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                cleaned_docs = cleaner.clean_data()
                
                # Get processing statistics
                if cleaner.doc_processor:
                    stats = cleaner.doc_processor.get_processing_stats([], cleaned_docs)
                    print(f"📊 Processing Stats:")
                    print(f"   - Final chunks: {stats['processed_chunks']}")
                    print(f"   - Average chunk size: {stats['average_chunk_size']:.0f} chars")
                    print(f"   - Average tokens per chunk: {stats['average_chunk_tokens']:.0f}")
                
            else:
                print("📝 Using basic processing...")
                # Basic processing without LLM cleaning
                data = json.loads(raw_data)
                cleaned_docs = []
                for key, item in data.items():
                    doc = Document(
                        page_content=item['text_content'],
                        metadata={
                            'url': key,
                            'title': item['title'],
                            'description': item['description'],
                            'processing_type': 'basic'
                        }
                    )
                    cleaned_docs.append(doc)
            
            print(f"✅ Processed {len(cleaned_docs)} document chunks")
            return cleaned_docs
            
        except Exception as e:
            print(f"❌ Error processing {input_file}: {e}")
            return []
    
    def save_processed_data(self, documents: List[Document], output_file: Path):
        """
        Save processed documents to JSON file
        
        Args:
            documents: List of processed documents
            output_file: Path to output JSON file
        """
        print(f"💾 Saving processed data to: {output_file}")
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert documents to serializable format
        serializable_docs = []
        for doc in documents:
            serializable_docs.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        
        # Add processing metadata
        processing_info = {
            'processed_at': datetime.now().isoformat(),
            'total_documents': len(documents),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'version': '1.0'
        }
        
        output_data = {
            'processing_info': processing_info,
            'documents': serializable_docs
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(serializable_docs, f, indent=2)  # Save just documents for easy loading
            
            # Also save full info with metadata
            info_file = output_file.parent / f"{output_file.stem}_info.json"
            with open(info_file, 'w') as f:
                json.dump(processing_info, f, indent=2)
            
            print(f"✅ Saved {len(documents)} processed documents")
            print(f"📊 Processing info saved to: {info_file}")
            
        except Exception as e:
            print(f"❌ Error saving processed data: {e}")
    
    def auto_process_latest(self, crawl_results_dir: Path = None, output_dir: Path = None):
        """
        Automatically process the latest crawl results
        
        Args:
            crawl_results_dir: Directory containing crawl results
            output_dir: Directory to save processed results
        """
        if crawl_results_dir is None:
            crawl_results_dir = Path("scripts/data_collection/crawl_results")
        
        if output_dir is None:
            output_dir = Path("scripts/data_cleaning/cleaned_data")
        
        print("🔍 Auto-processing latest crawl results...")
        
        # Find latest crawl file
        json_files = list(crawl_results_dir.glob("*.json"))
        if not json_files:
            print("❌ No JSON files found in crawl results directory")
            return
        
        latest_file = max(json_files, key=os.path.getctime)
        print(f"📁 Latest file: {latest_file}")
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"cleaned_data_{timestamp}.json"
        
        # Process the file
        documents = self.process_json_file(latest_file, use_enhanced=True)
        
        if documents:
            self.save_processed_data(documents, output_file)
            print(f"🎉 Auto-processing complete!")
            print(f"💡 Next time API starts, it will load: {output_file}")
        else:
            print("❌ No documents were processed")

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for Enhanced RAG API")
    parser.add_argument("--input", "-i", type=str, help="Input JSON file path")
    parser.add_argument("--output", "-o", type=str, help="Output directory path")
    parser.add_argument("--auto", "-a", action="store_true", help="Auto-process latest crawl results")
    parser.add_argument("--basic", "-b", action="store_true", help="Use basic processing (no LLM cleaning)")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap for text splitting")
    
    args = parser.parse_args()
    
    print("🚀 ENHANCED RAG DATA PREPROCESSOR")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    if args.auto:
        # Auto-process latest crawl results
        preprocessor.auto_process_latest()
        
    elif args.input:
        # Process specific file
        input_file = Path(args.input)
        if not input_file.exists():
            print(f"❌ Input file not found: {input_file}")
            return
        
        # Determine output file
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path("scripts/data_cleaning/cleaned_data")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"cleaned_{input_file.stem}_{timestamp}.json"
        
        # Process the file
        use_enhanced = not args.basic
        documents = preprocessor.process_json_file(input_file, use_enhanced=use_enhanced)
        
        if documents:
            preprocessor.save_processed_data(documents, output_file)
            print("🎉 Processing complete!")
        else:
            print("❌ No documents were processed")
    
    else:
        # Show help if no arguments provided
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python preprocess_data.py --auto")
        print("   python preprocess_data.py -i path/to/file.json")

if __name__ == "__main__":
    main() 