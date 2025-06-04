from typing import List, Dict, Any, Optional, Tuple
import re
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    EmbeddingsClusteringFilter
)
import tiktoken

class AdvancedDocumentProcessor:
    """
    Advanced document processor with recursive text splitting and multiple chunking strategies.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None,
                 model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: Custom separators for text splitting
            model_name: Model name for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Default separators for recursive splitting
        self.separators = separators or [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "! ",    # Exclamation sentences
            "? ",    # Question sentences
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Spaces
            ""       # Fallback to character splitting
        ]
        
        # Initialize different text splitters
        self._init_splitters()
    
    def _init_splitters(self):
        """Initialize various text splitters."""
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=True,
            is_separator_regex=False,
            length_function=len
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size // 4,  # Approximate 4 chars per token
            chunk_overlap=self.chunk_overlap // 4,
            model_name=self.model_name
        )
        
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.python_splitter = PythonCodeTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the initialized tokenizer."""
        return len(self.tokenizer.encode(text))
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing it.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not text.strip():
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common noise patterns
        noise_patterns = [
            r'ADVERTISEMENT:.*?(?=\n|\Z)',
            r'SIDEBAR:.*?(?=\n|\Z)',
            r'NAVIGATION:.*?(?=\n|\Z)',
            r'FOOTER:.*?(?=\n|\Z)',
            r'COOKIE NOTICE:.*?(?=\n|\Z)',
            r'PRIVACY POLICY:.*?(?=\n|\Z)',
            r'Terms of Service.*?(?=\n|\Z)',
            r'Copyright \d{4}.*?(?=\n|\Z)',
            r'All rights reserved.*?(?=\n|\Z)',
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up remaining whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def detect_content_type(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Detect the type of content to choose appropriate splitting strategy.
        
        Args:
            text: Text content to analyze
            metadata: Document metadata
            
        Returns:
            Content type ('markdown', 'code', 'structured', 'plain')
        """
        if metadata and metadata.get('source', '').endswith(('.md', '.markdown')):
            return 'markdown'
        
        # Check for code patterns
        code_indicators = [
            r'def\s+\w+\s*\(',
            r'class\s+\w+\s*[\(:]',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'function\s+\w+\s*\(',
            r'var\s+\w+\s*=',
            r'const\s+\w+\s*=',
            r'#include\s*<',
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return 'code'
        
        # Check for markdown patterns
        markdown_indicators = [
            r'^#+\s',  # Headers
            r'\*\*.*?\*\*',  # Bold
            r'\*.*?\*',  # Italic
            r'^\s*[-*+]\s',  # Lists
            r'^\s*\d+\.\s',  # Numbered lists
            r'```',  # Code blocks
        ]
        
        for pattern in markdown_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return 'markdown'
        
        # Check for structured content
        if re.search(r'^\s*\w+:\s*\w+', text, re.MULTILINE):
            return 'structured'
        
        return 'plain'
    
    def smart_split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Smart document splitting using different strategies based on content type.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents with enhanced metadata
        """
        split_documents = []
        
        for doc in documents:
            # Preprocess the content
            cleaned_content = self.preprocess_text(doc.page_content)
            
            if not cleaned_content:
                continue
            
            # Detect content type
            content_type = self.detect_content_type(cleaned_content, doc.metadata)
            
            # Choose appropriate splitter
            if content_type == 'markdown':
                splitter = self.markdown_splitter
            elif content_type == 'code':
                splitter = self.python_splitter
            else:
                splitter = self.recursive_splitter
            
            # Create temporary document for splitting
            temp_doc = Document(page_content=cleaned_content, metadata=doc.metadata.copy())
            
            # Split the document
            chunks = splitter.split_documents([temp_doc])
            
            # Enhance metadata for each chunk
            for i, chunk in enumerate(chunks):
                enhanced_metadata = chunk.metadata.copy()
                enhanced_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'content_type': content_type,
                    'chunk_size': len(chunk.page_content),
                    'token_count': self.count_tokens(chunk.page_content),
                    'splitting_method': splitter.__class__.__name__,
                    'is_first_chunk': i == 0,
                    'is_last_chunk': i == len(chunks) - 1
                })
                
                # Add contextual information
                if i > 0:
                    enhanced_metadata['previous_chunk_preview'] = chunks[i-1].page_content[:100]
                if i < len(chunks) - 1:
                    enhanced_metadata['next_chunk_preview'] = chunks[i+1].page_content[:100]
                
                chunk.metadata = enhanced_metadata
                split_documents.append(chunk)
        
        return split_documents
    
    def create_semantic_chunks(self, documents: List[Document], 
                             similarity_threshold: float = 0.8) -> List[Document]:
        """
        Create semantic chunks by grouping similar content together.
        
        Args:
            documents: List of documents to process
            similarity_threshold: Threshold for similarity grouping
            
        Returns:
            List of semantically grouped documents
        """
        # This is a placeholder for semantic chunking
        # In a full implementation, you would use embeddings to group similar content
        
        semantic_chunks = []
        current_chunk = None
        current_length = 0
        
        for doc in documents:
            if (current_chunk is None or 
                current_length + len(doc.page_content) > self.chunk_size):
                
                if current_chunk:
                    semantic_chunks.append(current_chunk)
                
                # Start new chunk
                current_chunk = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        'semantic_chunk': True,
                        'combined_sources': [doc.metadata.get('source', 'unknown')]
                    }
                )
                current_length = len(doc.page_content)
            else:
                # Combine with current chunk
                current_chunk.page_content += "\n\n" + doc.page_content
                current_chunk.metadata['combined_sources'].append(
                    doc.metadata.get('source', 'unknown')
                )
                current_length += len(doc.page_content)
        
        if current_chunk:
            semantic_chunks.append(current_chunk)
        
        return semantic_chunks
    
    def process_documents(self, documents: List[Document], 
                         use_semantic_chunking: bool = False) -> List[Document]:
        """
        Main method to process documents with advanced splitting techniques.
        
        Args:
            documents: Raw documents to process
            use_semantic_chunking: Whether to use semantic chunking
            
        Returns:
            Processed and split documents ready for vector storage
        """
        # First, apply smart splitting
        split_docs = self.smart_split_documents(documents)
        
        # Optional semantic chunking
        if use_semantic_chunking:
            split_docs = self.create_semantic_chunks(split_docs)
        
        # Filter out very small chunks
        min_chunk_size = 50
        filtered_docs = [
            doc for doc in split_docs 
            if len(doc.page_content.strip()) >= min_chunk_size
        ]
        
        return filtered_docs
    
    def get_processing_stats(self, original_docs: List[Document], 
                           processed_docs: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the document processing.
        
        Args:
            original_docs: Original documents before processing
            processed_docs: Documents after processing
            
        Returns:
            Processing statistics
        """
        original_total_chars = sum(len(doc.page_content) for doc in original_docs)
        processed_total_chars = sum(len(doc.page_content) for doc in processed_docs)
        
        original_tokens = sum(self.count_tokens(doc.page_content) for doc in original_docs)
        processed_tokens = sum(self.count_tokens(doc.page_content) for doc in processed_docs)
        
        return {
            'original_documents': len(original_docs),
            'processed_chunks': len(processed_docs),
            'expansion_ratio': len(processed_docs) / len(original_docs) if original_docs else 0,
            'original_total_characters': original_total_chars,
            'processed_total_characters': processed_total_chars,
            'original_total_tokens': original_tokens,
            'processed_total_tokens': processed_tokens,
            'average_chunk_size': processed_total_chars / len(processed_docs) if processed_docs else 0,
            'average_chunk_tokens': processed_tokens / len(processed_docs) if processed_docs else 0
        } 