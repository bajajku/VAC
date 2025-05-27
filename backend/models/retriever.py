from typing import Dict, List, Any
from langchain_core.documents import Document

'''
    Retriever class to create and manage retrievers.
'''

class Retriever:
    def __init__(self, vector_database):
        self.vector_database = vector_database
        self.retriever = self.create_retriever()

    def create_retriever(self, **kwargs):
        """Create a retriever from the vector database with custom parameters."""
        default_kwargs = {
            'search_type': 'similarity',
            'search_kwargs': {'k': 4}
        }
        default_kwargs.update(kwargs)
        return self.vector_database.as_retriever(**default_kwargs)
    
    def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """Get relevant documents for a query."""
        return self.retriever.get_relevant_documents(query)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search directly on the vector database."""
        return self.vector_database.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with scores."""
        return self.vector_database.similarity_search_with_score(query, k=k)
