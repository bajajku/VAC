import os
from typing import Optional
from models.vector_database import VectorDatabase
from models.rag_agent import RAGAgent, create_rag_agent
from utils.retriever import global_retriever
from scripts.data_collection.json_parser import JsonParser

class RAGApplication:
    """
    Main application class that manages the RAG system components.
    Handles initialization, configuration, and provides a unified interface.
    """
    
    def __init__(self):
        self.vector_db: Optional[VectorDatabase] = None
        self.rag_agent: Optional[RAGAgent] = None
        self.is_initialized = False
    
    def initialize(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "openai",
        llm_model: str = "gpt-3.5-turbo",
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_collection",
        **llm_kwargs
    ):
        """
        Initialize the RAG application with vector database and LLM.
        
        Args:
            embedding_model: HuggingFace embedding model name
            llm_provider: LLM provider ('openai', 'openrouter', 'huggingface_pipeline', etc.)
            llm_model: LLM model name
            persist_directory: Directory to persist vector database
            collection_name: Name for the vector database collection
            **llm_kwargs: Additional LLM configuration (api_key, temperature, etc.)
        """
        print("🚀 Initializing RAG Application...")
        
        # Initialize vector database
        print("📊 Setting up vector database...")
        self.vector_db = VectorDatabase()
        self.vector_db.create_vector_database(
            embedding_model=embedding_model,
            type="chroma",
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # Initialize global retriever
        print("🔍 Setting up retriever...")
        global_retriever.initialize(self.vector_db)
        
        # Initialize RAG agent
        print(f"🤖 Setting up RAG agent with {llm_provider}/{llm_model}...")
        self.rag_agent = create_rag_agent(
            provider=llm_provider,
            model_name=llm_model,
            **llm_kwargs
        )
        
        self.is_initialized = True
        print("✅ RAG Application initialized successfully!")
    
    def load_data_from_json(self, json_file_path: str) -> int:
        """
        Load documents from a JSON file into the vector database.
        
        Args:
            json_file_path: Path to the JSON file containing crawled data
            
        Returns:
            int: Number of documents loaded
        """
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        print(f"📚 Loading data from {json_file_path}...")
        num_docs = self.vector_db.load_documents_from_json(json_file_path)
        print(f"✅ Loaded {num_docs} documents into vector database")
        return num_docs
    
    def add_documents_from_text(self, texts: list[str], metadatas: list[dict] = None):
        """
        Add documents directly from text content.
        
        Args:
            texts: List of text content
            metadatas: Optional list of metadata dictionaries
        """
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        from langchain_core.documents import Document
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        
        self.vector_db.add_documents(documents)
        print(f"✅ Added {len(documents)} documents to vector database")
    
    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            
        Returns:
            str: Agent's response
        """
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        return self.rag_agent.invoke(question)
    
    async def aquery(self, question: str) -> str:
        """Async version of query."""
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        return await self.rag_agent.ainvoke(question)
    
    def stream_query(self, question: str):
        """Stream the response to a query."""
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        return self.rag_agent.stream(question)
    
    def search_documents(self, query: str, k: int = 4):
        """
        Search for relevant documents without generating a response.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant documents
        """
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        return global_retriever.get_relevant_documents(query, k=k)
    
    def get_stats(self) -> dict:
        """Get statistics about the RAG system."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        # Get collection info if available
        try:
            collection = self.vector_db.vector_database._collection
            doc_count = collection.count()
        except:
            doc_count = "unknown"
        
        return {
            "status": "initialized",
            "document_count": doc_count,
            "retriever_initialized": global_retriever.is_initialized(),
            "vector_db_type": "chroma"
        }


# Global application instance
app = RAGApplication()

def get_app() -> RAGApplication:
    """Get the global application instance."""
    return app 