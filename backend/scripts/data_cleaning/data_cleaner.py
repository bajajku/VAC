import json
import os
import sys
from typing import List, Dict, Any
from langchain_core.documents import Document
import tqdm
from models.llm import LLM
from models.rag_chain import NormalChain
from utils.prompt import Prompt
from dotenv import load_dotenv

# Add utils to path for document processor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from utils.document_processor import AdvancedDocumentProcessor
except ImportError:
    print("Warning: AdvancedDocumentProcessor not available, using basic processing")
    AdvancedDocumentProcessor = None

# Load environment variables
load_dotenv('.env')

class DataCleaner:
    def __init__(self, unclean_data: json, 
                 use_advanced_processing: bool = True,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        self.unclean_data = unclean_data
        self.chain = None
        self.cleaned_documents = []
        self.use_advanced_processing = use_advanced_processing and AdvancedDocumentProcessor is not None
        
        # Initialize document processor if available
        if self.use_advanced_processing:
            self.doc_processor = AdvancedDocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            print("✅ Using advanced document processing with recursive text splitting")
        else:
            self.doc_processor = None
            print("⚠️ Using basic document processing")
        
        self.data_cleaning_prompt = self._create_cleaning_prompt()

    def _create_cleaning_prompt(self) -> Prompt:
        """Create a prompt template for data cleaning."""
        template = """
You are a data cleaning specialist. Your task is to analyze the given document and clean it for relevance and quality.

Please perform the following tasks:
1. Remove any irrelevant information (ads, navigation elements, boilerplate text)
2. Don't change the original content, just clean it.
3. Fix any obvious formatting issues
4. Extract only the main content that would be useful for a knowledge base
5. Ensure the text is coherent and well-structured

Document to clean:
{context}

Please return only the cleaned, relevant content. If the document contains no useful information, return "NO_USEFUL_CONTENT".
"""
        return Prompt(template=template)

    def _initialize_chain(self):
        """Initialize the NormalChain for data cleaning."""
        if self.chain is None:
            # Get API key from environment
            api_key = os.environ.get('TOGETHER_API_KEY')
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable not set. Please set it to use the data cleaner.")
            
            llm = LLM(
                provider="ollama", 
                model_name="llama3.1:70b",
                api_key=api_key
            )
            self.chain = NormalChain(llm=llm, prompt=self.data_cleaning_prompt)
            print(self.chain)

    def clean_data(self) -> List[Document]:
        """
        Clean the unclean data and return a list of cleaned Document objects.
        
        Returns:
            List[Document]: List of cleaned documents in LangChain Document format
        """
        self._initialize_chain()
        
        # Process the JSON data
        if isinstance(self.unclean_data, dict):
            data_dict = self.unclean_data
        else:
            # If it's a JSON string, parse it
            data_dict = json.loads(self.unclean_data) if isinstance(self.unclean_data, str) else self.unclean_data
        
        raw_documents = []
        
        # First pass: Create raw documents from the data
        for key, item in tqdm.tqdm(data_dict.items(), desc="Processing raw data"):
            try:
                # Extract content from the item
                if isinstance(item, dict):
                    # Assume structure like {'text_content': '...', 'title': '...', 'description': '...'}
                    content = item.get('text_content', '')
                    title = item.get('title', '')
                    description = item.get('description', '')
                    
                    # Combine content for cleaning
                    full_content = f"Title: {title}\nDescription: {description}\nContent: {content}"
                else:
                    full_content = str(item)
                
                # Skip if content is empty or too short
                if not full_content.strip() or len(full_content.strip()) < 10:
                    continue
                
                # Create metadata
                metadata = {
                    'source': key,
                    'original_title': title if isinstance(item, dict) else '',
                    'original_description': description if isinstance(item, dict) else '',
                    'raw_document': True
                }
                
                # Create raw Document object
                raw_doc = Document(
                    page_content=full_content.strip(),
                    metadata=metadata
                )
                
                raw_documents.append(raw_doc)
                
            except Exception as e:
                print(f"Error processing item {key}: {str(e)}")
                continue
        
        print(f"📄 Created {len(raw_documents)} raw documents")
        
        # Second pass: Clean the documents
        cleaned_documents = []
        for doc in tqdm.tqdm(raw_documents, desc="Cleaning documents"):
            try:
                # Clean the content using the chain
                cleaned_content = self.chain.invoke({'context': doc.page_content})
                cleaned_content = cleaned_content.content
                
                # Skip if no useful content
                if cleaned_content.strip() == "NO_USEFUL_CONTENT" or len(cleaned_content.strip()) < 20:
                    continue
                
                # Update metadata
                enhanced_metadata = doc.metadata.copy()
                enhanced_metadata.update({
                    'cleaned': True,
                    'cleaning_method': 'llm_chain',
                    'raw_document': False
                })
                
                # Create cleaned Document object
                cleaned_doc = Document(
                    page_content=cleaned_content.strip(),
                    metadata=enhanced_metadata
                )
                
                cleaned_documents.append(cleaned_doc)
                
            except Exception as e:
                print(f"Error cleaning document: {str(e)}")
                continue
        
        print(f"🧹 Cleaned {len(cleaned_documents)} documents")
        
        # Third pass: Advanced processing with recursive text splitting
        if self.use_advanced_processing and self.doc_processor:
            try:
                print("🔀 Applying advanced document processing...")
                processed_documents = self.doc_processor.process_documents(
                    cleaned_documents, 
                    use_semantic_chunking=False
                )
                
                # Get processing statistics
                stats = self.doc_processor.get_processing_stats(cleaned_documents, processed_documents)
                print(f"📊 Processing Statistics:")
                print(f"   Original documents: {stats['original_documents']}")
                print(f"   Final chunks: {stats['processed_chunks']}")
                print(f"   Expansion ratio: {stats['expansion_ratio']:.2f}")
                print(f"   Average chunk size: {stats['average_chunk_size']:.0f} chars")
                print(f"   Average tokens per chunk: {stats['average_chunk_tokens']:.0f}")
                
                self.cleaned_documents = processed_documents
                return processed_documents
                
            except Exception as e:
                print(f"⚠️ Advanced processing failed: {e}")
                print("Falling back to basic cleaned documents")
                self.cleaned_documents = cleaned_documents
                return cleaned_documents
        else:
            self.cleaned_documents = cleaned_documents
            return cleaned_documents

    def get_cleaned_documents(self) -> List[Document]:
        """
        Get the cleaned documents. If not already cleaned, perform cleaning first.
        
        Returns:
            List[Document]: List of cleaned documents
        """
        if not self.cleaned_documents:
            return self.clean_data()
        return self.cleaned_documents

    def save_cleaned_documents(self, output_path: str):
        """
        Save cleaned documents to a JSON file.
        
        Args:
            output_path (str): Path to save the cleaned documents
        """
        if not self.cleaned_documents:
            self.clean_data()
        
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert documents to serializable format
        serializable_docs = []
        for doc in self.cleaned_documents:
            serializable_docs.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(self.cleaned_documents)} cleaned documents to {output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the cleaning process.
        
        Returns:
            Dict[str, Any]: Statistics about original vs cleaned data
        """
        original_count = len(self.unclean_data) if isinstance(self.unclean_data, dict) else 0
        cleaned_count = len(self.cleaned_documents)
        
        stats = {
            'original_documents': original_count,
            'cleaned_documents': cleaned_count,
            'documents_removed': original_count - cleaned_count,
            'cleaning_efficiency': f"{(cleaned_count/original_count)*100:.1f}%" if original_count > 0 else "0%",
            'advanced_processing_used': self.use_advanced_processing
        }
        
        # Add document processor statistics if available
        if self.use_advanced_processing and self.cleaned_documents:
            # Calculate some additional stats
            total_chars = sum(len(doc.page_content) for doc in self.cleaned_documents)
            avg_chunk_size = total_chars / len(self.cleaned_documents) if self.cleaned_documents else 0
            
            content_types = {}
            for doc in self.cleaned_documents:
                content_type = doc.metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            stats.update({
                'total_characters': total_chars,
                'average_chunk_size': f"{avg_chunk_size:.0f}",
                'content_types': content_types
            })
        
        return stats