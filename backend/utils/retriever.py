from typing import Optional, List
from langchain_core.documents import Document
from models.vector_database import VectorDatabase
from models.retriever import Retriever

class GlobalRetriever:
    """Global retriever instance for the RAG system."""
    
    def __init__(self):
        self._retriever: Optional[Retriever] = None
        self._vector_db: Optional[VectorDatabase] = None
    
    def initialize(self, vector_database: VectorDatabase):
        """Initialize the global retriever with a vector database."""
        self._vector_db = vector_database
        print(f"vector_database: {self._vector_db}")
        self._retriever = Retriever(vector_database.vector_database)
    
    def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for a query."""
        if self._retriever is None:
            raise ValueError("Global retriever not initialized. Call initialize() first.")
        
        # Use the retriever's get_relevant_documents method
        return self._retriever.retriever.get_relevant_documents(query, k=k)
    
    async def aget_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """Async retrieve relevant documents for a query."""
        if self._retriever is None:
            raise ValueError("Global retriever not initialized. Call initialize() first.")
        
        # Use the retriever's async method if available, otherwise fall back to sync
        if hasattr(self._retriever.retriever, 'aget_relevant_documents'):
            return await self._retriever.retriever.aget_relevant_documents(query, k=k)
        else:
            # Import asyncio to run sync method in thread
            import asyncio
            import functools
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                functools.partial(self._retriever.retriever.get_relevant_documents, query, k=k)
            )
    
    def is_initialized(self) -> bool:
        """Check if the retriever is initialized."""
        return self._retriever is not None

# Global instance
global_retriever = GlobalRetriever() 