from typing import Dict, List, Any, Optional
from langchain_core.documents import Document
from .advanced_retriever import AdvancedRetriever

'''
    Enhanced Retriever class with advanced techniques for robust RAG applications.
    Now includes recursive text splitting, MMR, re-ranking, and ensemble retrieval.
'''

class Retriever:
    def __init__(self, vector_database, 
                 retrieval_strategy: str = "hybrid",
                 similarity_threshold: float = 0.7,
                 mmr_diversity_bias: float = 0.3,
                 max_results: int = 10,
                 enable_reranking: bool = True,
                 enable_compression: bool = False,
                 llm=None):
        """
        Initialize the enhanced retriever.
        
        Args:
            vector_database: Vector database instance
            retrieval_strategy: Strategy ('similarity', 'mmr', 'hybrid', 'ensemble')
            similarity_threshold: Minimum similarity score for results
            mmr_diversity_bias: MMR diversity parameter (0=similarity only, 1=diversity only)
            max_results: Maximum number of results to return
            enable_reranking: Whether to enable result re-ranking
            enable_compression: Whether to enable contextual compression
            llm: Language model for advanced features
        """
        self.vector_database = vector_database
        self.retrieval_strategy = retrieval_strategy
        self.similarity_threshold = similarity_threshold
        self.mmr_diversity_bias = mmr_diversity_bias
        self.max_results = max_results
        self.enable_reranking = enable_reranking
        self.enable_compression = enable_compression
        self.llm = llm

        
        # Create both legacy and advanced retrievers
        self.retriever = self.create_retriever()
        self.advanced_retriever = None
    
        print(f"advanced_retriever: {self.advanced_retriever}")
    def _create_advanced_retriever(self):
        """Create the advanced retriever with enhanced capabilities."""
        try:
            return AdvancedRetriever(
                vector_store=self.vector_database,
                retrieval_strategy=self.retrieval_strategy,
                similarity_threshold=self.similarity_threshold,
                mmr_diversity_bias=self.mmr_diversity_bias,
                max_results=self.max_results,
                enable_reranking=self.enable_reranking,
                enable_compression=self.enable_compression,
                llm=self.llm
            )
        except Exception as e:
            print(f"Warning: Could not create advanced retriever: {e}")
            print("Falling back to basic retriever")
            return None

    def create_retriever(self, **kwargs):
        """Create a basic retriever from the vector database with custom parameters."""
        default_kwargs = {
            'search_type': 'similarity',
            'search_kwargs': {'k': self.max_results}
        }
        default_kwargs.update(kwargs)
        return self.vector_database.as_retriever(**default_kwargs)
    
    def get_relevant_documents(self, query: str, k: int = None, 
                             use_advanced: bool = True) -> List[Document]:
        """
        Get relevant documents for a query using enhanced retrieval.
        
        Args:
            query: Query string
            k: Number of documents to return (overrides max_results if provided)
            use_advanced: Whether to use advanced retriever features
            
        Returns:
            List of relevant documents
        """
        if use_advanced and self.advanced_retriever:
            try:
                # Update max_results if k is provided
                if k is not None:
                    original_max = self.advanced_retriever.max_results
                    self.advanced_retriever.max_results = k
                
                documents = self.advanced_retriever._get_relevant_documents(query, None)
                
                # Restore original max_results
                if k is not None:
                    self.advanced_retriever.max_results = original_max
                
                return documents
            except Exception as e:
                print(f"Advanced retrieval failed: {e}, falling back to basic retrieval")
        
        # Fallback to basic retrieval
        if k is not None:
            # Update basic retriever
            search_kwargs = self.retriever.search_kwargs.copy()
            search_kwargs['k'] = k
            temp_retriever = self.vector_database.as_retriever(
                search_type=self.retriever.search_type,
                search_kwargs=search_kwargs
            )
            return temp_retriever.get_relevant_documents(query)
        
        return self.retriever.get_relevant_documents(query)
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Perform similarity search directly on the vector database."""
        k = k or self.max_results
        return self.vector_database.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Perform similarity search with scores."""
        k = k or self.max_results
        return self.vector_database.similarity_search_with_score(query, k=k)
    
    def mmr_search(self, query: str, k: int = None, 
                   lambda_mult: float = None) -> List[Document]:
        """
        Perform Maximum Marginal Relevance search for diverse results.
        
        Args:
            query: Query string
            k: Number of documents to return
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            
        Returns:
            List of diverse relevant documents
        """
        k = k or self.max_results
        lambda_mult = lambda_mult or (1 - self.mmr_diversity_bias)
        
        try:
            return self.vector_database.max_marginal_relevance_search(
                query, k=k, lambda_mult=lambda_mult
            )
        except AttributeError:
            # Fallback if MMR is not available
            print("MMR not available, falling back to similarity search")
            return self.similarity_search(query, k)
    
    def threshold_search(self, query: str, threshold: float = None) -> List[Document]:
        """
        Search with similarity threshold filtering.
        
        Args:
            query: Query string
            threshold: Minimum similarity threshold
            
        Returns:
            Documents above the similarity threshold
        """
        threshold = threshold or self.similarity_threshold
        
        try:
            docs_with_scores = self.similarity_search_with_score(query, k=self.max_results * 2)
            filtered_docs = [
                doc for doc, score in docs_with_scores 
                if score >= threshold
            ]
            return filtered_docs[:self.max_results]
        except Exception as e:
            print(f"Threshold search failed: {e}, falling back to similarity search")
            return self.similarity_search(query)
    
    def get_retrieval_stats(self, query: str) -> Dict[str, Any]:
        """Get retrieval statistics and performance metrics."""
        stats = {
            'query': query,
            'retrieval_strategy': self.retrieval_strategy,
            'similarity_threshold': self.similarity_threshold,
            'max_results': self.max_results,
            'advanced_retriever_available': self.advanced_retriever is not None
        }
        
        if self.advanced_retriever:
            advanced_stats = self.advanced_retriever.get_retrieval_stats(query)
            stats.update(advanced_stats)
        
        return stats
    
    def update_config(self, **kwargs):
        """
        Update retriever configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update advanced retriever if available
        if self.advanced_retriever:
            self.advanced_retriever.update_config(**kwargs)
        
        # Recreate basic retriever with new config
        self.retriever = self.create_retriever()
        
        print(f"Updated retriever configuration: {kwargs}")
