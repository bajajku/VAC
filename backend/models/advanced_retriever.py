from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.vectorstores import VectorStore
from langchain.retrievers import (
    EnsembleRetriever,
    MultiQueryRetriever,
    ContextualCompressionRetriever
)
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline
)
from langchain_community.retrievers import (
    BM25Retriever,
    TFIDFRetriever,
    SVMRetriever
)
import logging
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class AdvancedRetriever(BaseRetriever):
    """
    Advanced retriever with multiple retrieval strategies, re-ranking, and result fusion.
    """
    
    # Declare fields to be compatible with Pydantic
    vector_store: Any
    retrieval_strategy: str
    similarity_threshold: float
    mmr_diversity_bias: float
    max_results: int
    enable_reranking: bool
    enable_compression: bool
    llm: Optional[Any]
    
    # Declare dynamic attributes that will be set during initialization
    similarity_retriever: Optional[Any] = None
    mmr_retriever: Optional[Any] = None
    threshold_retriever: Optional[Any] = None
    bm25_retriever: Optional[Any] = None
    multi_query_retriever: Optional[Any] = None
    compression_retriever: Optional[Any] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

    def __init__(self,
                 vector_store,
                 retrieval_strategy: str = "hybrid",
                 similarity_threshold: float = 0.7,
                 mmr_diversity_bias: float = 0.3,
                 max_results: int = 10,
                 enable_reranking: bool = True,
                 enable_compression: bool = False,
                 llm=None):
        """
        Initialize the advanced retriever.
        
        Args:
            vector_store: The vector store to retrieve from
            retrieval_strategy: Strategy to use ('similarity', 'mmr', 'hybrid', 'ensemble')
            similarity_threshold: Minimum similarity score for results
            mmr_diversity_bias: MMR diversity parameter (0=similarity only, 1=diversity only)
            max_results: Maximum number of results to return
            enable_reranking: Whether to enable result re-ranking
            enable_compression: Whether to enable contextual compression
            llm: Language model for compression and multi-query retrieval
        """
        # Initialize with field values
        super().__init__(
            vector_store=vector_store,
            retrieval_strategy=retrieval_strategy,
            similarity_threshold=similarity_threshold,
            mmr_diversity_bias=mmr_diversity_bias,
            max_results=max_results,
            enable_reranking=enable_reranking,
            enable_compression=enable_compression,
            llm=llm
        )
        
        # Initialize different retrievers
        self._init_retrievers()
    
    def _init_retrievers(self):
        """Initialize different types of retrievers."""
        try:
            # Vector similarity retriever
            self.similarity_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.max_results * 2}  # Get more for filtering
            )
            
            # MMR retriever for diversity
            self.mmr_retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.max_results,
                    "lambda_mult": 1 - self.mmr_diversity_bias  # LangChain uses inverse
                }
            )
            
            # Similarity with score threshold
            self.threshold_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": self.similarity_threshold,
                    "k": self.max_results
                }
            )
            
            # Try to create BM25 retriever if documents are available
            try:
                docs = self._get_all_documents()
                if docs:
                    self.bm25_retriever = BM25Retriever.from_documents(docs)
                    self.bm25_retriever.k = self.max_results
                else:
                    self.bm25_retriever = None
            except Exception as e:
                logger.warning(f"Could not create BM25 retriever: {e}")
                self.bm25_retriever = None
            
            # Multi-query retriever if LLM is available
            if self.llm:
                try:
                    self.multi_query_retriever = MultiQueryRetriever.from_llm(
                        retriever=self.similarity_retriever,
                        llm=self.llm
                    )
                except Exception as e:
                    logger.warning(f"Could not create multi-query retriever: {e}")
                    self.multi_query_retriever = None
            else:
                self.multi_query_retriever = None
            
            # Compression retriever if enabled
            if self.enable_compression and self.llm:
                try:
                    compressor = LLMChainExtractor.from_llm(self.llm)
                    self.compression_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor,
                        base_retriever=self.similarity_retriever
                    )
                except Exception as e:
                    logger.warning(f"Could not create compression retriever: {e}")
                    self.compression_retriever = None
            else:
                self.compression_retriever = None
                
        except Exception as e:
            logger.error(f"Error initializing retrievers: {e}")
            raise
    
    def _get_all_documents(self) -> List[Document]:
        """Get all documents from the vector store for BM25 initialization."""
        try:
            # This is a workaround since there's no standard way to get all docs
            # We'll do a broad search to get a representative sample
            docs = self.vector_store.similarity_search("", k=1000)
            return docs
        except Exception as e:
            logger.warning(f"Could not retrieve documents for BM25: {e}")
            return []
    
    def _calculate_document_scores(self, documents: List[Document], 
                                 query: str) -> List[Tuple[Document, float]]:
        """
        Calculate relevance scores for documents using multiple factors.
        
        Args:
            documents: List of documents to score
            query: Original query
            
        Returns:
            List of (document, score) tuples
        """
        scored_docs = []
        query_terms = set(query.lower().split())
        
        for doc in documents:
            score = 0.0
            content = doc.page_content.lower()
            
            # Term frequency score
            tf_score = sum(content.count(term) for term in query_terms)
            tf_score = tf_score / len(content.split()) if content.split() else 0
            
            # Exact phrase bonus
            if query.lower() in content:
                phrase_bonus = 0.2
            else:
                phrase_bonus = 0.0
            
            # Metadata relevance
            metadata_score = 0.0
            if 'title' in doc.metadata:
                title = doc.metadata['title'].lower()
                metadata_score += sum(0.1 for term in query_terms if term in title)
            
            # Content type bonus
            content_type = doc.metadata.get('content_type', 'plain')
            type_bonus = {
                'structured': 0.05,
                'markdown': 0.03,
                'code': 0.02,
                'plain': 0.0
            }.get(content_type, 0.0)
            
            # Chunk position penalty (prefer first chunks)
            chunk_index = doc.metadata.get('chunk_index', 0)
            position_penalty = -0.01 * chunk_index
            
            # Length normalization
            length_factor = min(1.0, len(doc.page_content) / 500)  # Prefer reasonable length
            
            # Combine scores
            total_score = (tf_score * 0.4 + phrase_bonus + metadata_score * 0.3 + 
                         type_bonus + position_penalty) * length_factor
            
            scored_docs.append((doc, total_score))
        
        return sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    def _rerank_documents(self, documents: List[Document], 
                         query: str) -> List[Document]:
        """
        Re-rank documents using advanced scoring techniques.
        
        Args:
            documents: Documents to re-rank
            query: Original query
            
        Returns:
            Re-ranked documents
        """
        if not self.enable_reranking or not documents:
            return documents
        
        try:
            scored_docs = self._calculate_document_scores(documents, query)
            return [doc for doc, score in scored_docs[:self.max_results]]
        except Exception as e:
            logger.warning(f"Error in re-ranking: {e}")
            return documents[:self.max_results]
    
    def _deduplicate_documents(self, documents: List[Document], 
                              similarity_threshold: float = 0.9) -> List[Document]:
        """
        Remove duplicate or very similar documents.
        
        Args:
            documents: Documents to deduplicate
            similarity_threshold: Threshold for considering documents similar
            
        Returns:
            Deduplicated documents
        """
        if not documents:
            return documents
        
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            # Simple deduplication based on content hash
            content_hash = hash(doc.page_content.strip())
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
            
            # Stop if we have enough unique documents
            if len(unique_docs) >= self.max_results:
                break
        
        return unique_docs
    
    def _ensemble_retrieval(self, query: str) -> List[Document]:
        """
        Perform ensemble retrieval combining multiple strategies.
        
        Args:
            query: Query string
            
        Returns:
            Combined results from multiple retrievers
        """
        all_results = {}  # Use dict to track documents and their sources
        
        # Get results from different retrievers
        retrievers = [
            ("similarity", self.similarity_retriever),
            ("mmr", self.mmr_retriever),
            ("threshold", self.threshold_retriever)
        ]
        
        if self.bm25_retriever:
            retrievers.append(("bm25", self.bm25_retriever))
        
        if self.multi_query_retriever:
            retrievers.append(("multi_query", self.multi_query_retriever))
        
        for retriever_name, retriever in retrievers:
            try:
                results = retriever.get_relevant_documents(query)
                for doc in results:
                    doc_key = hash(doc.page_content)
                    if doc_key not in all_results:
                        # Add retriever source to metadata
                        enhanced_metadata = doc.metadata.copy()
                        enhanced_metadata['retriever_sources'] = [retriever_name]
                        enhanced_metadata['retrieval_rank'] = {retriever_name: len(all_results)}
                        
                        doc_copy = Document(
                            page_content=doc.page_content,
                            metadata=enhanced_metadata
                        )
                        all_results[doc_key] = doc_copy
                    else:
                        # Update existing document with additional source
                        existing_doc = all_results[doc_key]
                        existing_doc.metadata['retriever_sources'].append(retriever_name)
                        existing_doc.metadata['retrieval_rank'][retriever_name] = len([
                            d for d in all_results.values() 
                            if retriever_name not in d.metadata['retriever_sources']
                        ])
                        
            except Exception as e:
                logger.warning(f"Error in {retriever_name} retrieval: {e}")
        
        # Convert back to list and sort by number of sources (more sources = higher relevance)
        combined_docs = list(all_results.values())
        combined_docs.sort(
            key=lambda doc: len(doc.metadata['retriever_sources']),
            reverse=True
        )
        
        return combined_docs
    
    def _get_relevant_documents(self, query: str, 
                               run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Main retrieval method implementing the configured strategy.
        
        Args:
            query: Query string
            run_manager: Callback manager
            
        Returns:
            Retrieved and processed documents
        """
        try:
            # Choose retrieval strategy
            if self.retrieval_strategy == "similarity":
                documents = self.similarity_retriever.get_relevant_documents(query)
            elif self.retrieval_strategy == "mmr":
                documents = self.mmr_retriever.get_relevant_documents(query)
            elif self.retrieval_strategy == "threshold":
                documents = self.threshold_retriever.get_relevant_documents(query)
            elif self.retrieval_strategy == "ensemble" or self.retrieval_strategy == "hybrid":
                documents = self._ensemble_retrieval(query)
            elif self.retrieval_strategy == "multi_query" and self.multi_query_retriever:
                documents = self.multi_query_retriever.get_relevant_documents(query)
            elif self.retrieval_strategy == "compression" and self.compression_retriever:
                documents = self.compression_retriever.get_relevant_documents(query)
            else:
                # Fallback to similarity
                documents = self.similarity_retriever.get_relevant_documents(query)
            
            # Post-processing steps
            if documents:
                # Deduplicate
                documents = self._deduplicate_documents(documents)
                
                # Re-rank if enabled
                documents = self._rerank_documents(documents, query)
                
                # Ensure we don't exceed max_results
                documents = documents[:self.max_results]
            
            logger.info(f"Retrieved {len(documents)} documents using {self.retrieval_strategy} strategy")
            return documents
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            # Fallback to basic similarity search
            try:
                return self.similarity_retriever.get_relevant_documents(query)[:self.max_results]
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {fallback_error}")
                return []
    
    def get_retrieval_stats(self, query: str) -> Dict[str, Any]:
        """
        Get statistics about retrieval performance for analysis.
        
        Args:
            query: Query to analyze
            
        Returns:
            Retrieval statistics
        """
        stats = {
            'query': query,
            'strategy': self.retrieval_strategy,
            'similarity_threshold': self.similarity_threshold,
            'max_results': self.max_results,
            'retrievers_available': []
        }
        
        # Check which retrievers are available
        if hasattr(self, 'similarity_retriever'):
            stats['retrievers_available'].append('similarity')
        if hasattr(self, 'mmr_retriever'):
            stats['retrievers_available'].append('mmr')
        if hasattr(self, 'bm25_retriever') and self.bm25_retriever:
            stats['retrievers_available'].append('bm25')
        if hasattr(self, 'multi_query_retriever') and self.multi_query_retriever:
            stats['retrievers_available'].append('multi_query')
        if hasattr(self, 'compression_retriever') and self.compression_retriever:
            stats['retrievers_available'].append('compression')
        
        return stats
    
    def update_config(self, **kwargs):
        """
        Update retriever configuration dynamically.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
        
        # Reinitialize retrievers with new config
        self._init_retrievers() 