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
    
    def is_initialized(self) -> bool:
        """Check if the retriever is initialized."""
        return self._retriever is not None

# Global instance
global_retriever = GlobalRetriever() 